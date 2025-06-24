#if defined(_MSC_VER)
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "common.h"
#include "log.h"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
#include "json-schema-to-grammar.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstring>
#include <csignal>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <thread>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(LLAMA_USE_CURL)
#include <curl/curl.h>
#include <curl/easy.h>
#include <future>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined(LLAMA_USE_CURL)
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#define PATH_MAX MAX_PATH
#else
#include <sys/syslimits.h>
#endif
#define LLAMA_CURL_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083
#endif // LLAMA_USE_CURL

#if defined(USE_HIGHS)
#include "Highs.h"
#endif

using json = nlohmann::ordered_json;

constexpr int GIGABYTE = 1024 * 1024 * 1024;

struct HiGHSException {
    int signal;
    const char * message;
};

[[noreturn]] static void highs_handler(int signal) {
    HiGHSException e{signal, "HiGHS terminated due to signal"};
    throw e;
}

//
// CPU utils
//

int32_t cpu_get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#endif
    return cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        LOG_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (!setpriority(PRIO_PROCESS, 0, p)) {
        LOG_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(cpu_params& cpuparams, const cpu_params* role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = cpu_get_num_math();
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        LOG_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        LOG_ERR("Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            LOG_ERR("Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            LOG_ERR("End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    if (num_digits > 128) num_digits = 128;

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            LOG_ERR("Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void gpt_init() {
    llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
        if (LOG_DEFAULT_LLAMA <= gpt_log_verbosity_thold) {
            gpt_log_add(gpt_log_main(), level, "%s", text);
        }
    }, NULL);

#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif

    LOG_INF("build: %d (%s) with %s for %s%s\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT, LLAMA_COMPILER, LLAMA_BUILD_TARGET, build_type);
}

std::string gpt_params_get_system_info(const gpt_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::vector<std::string> string_split(std::string input, char separator) {
    std::vector<std::string> parts;
    size_t separator_pos = input.find(separator);
    while (separator_pos != std::string::npos) {
        std::string part = input.substr(0, separator_pos);
        parts.emplace_back(part);
        input = input.substr(separator_pos + 1);
        separator_pos = input.find(separator);
    }
    parts.emplace_back(input);
    return parts;
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string string_from(bool value) {
    return value ? "true" : "false";
}

std::string string_from(const std::vector<int> & values) {
    std::stringstream buf;

    buf << "[ ";
    bool first = true;
    for (auto e : values) {
        if (first) {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = llama_token_to_piece(ctx, token);

        detokenized.erase(
            std::remove_if(
                detokenized.begin(),
                detokenized.end(),
                [](const unsigned char c) { return !std::isprint(c); }),
            detokenized.end());

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = llama_token_to_piece(ctx, batch.token[i]);

        detokenized.erase(
                std::remove_if(
                    detokenized.begin(),
                    detokenized.end(),
                    [](const unsigned char c) { return !std::isprint(c); }),
                detokenized.end());

        buf << "\n" << std::to_string(i)
            << ":token '" << detokenized << "'"
            << ":pos " << std::to_string(batch.pos[i])
            << ":n_seq_id  " << std::to_string(batch.n_seq_id[i])
            << ":seq_id " << std::to_string(batch.seq_id[i][0])
            << ":logits " << std::to_string(batch.logits[i]);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        LOG_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            LOG_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            LOG_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        LOG_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    std::u32string filename_utf32;
    try {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        filename_utf32 = converter.from_bytes(filename);

        // If the reverse conversion mismatches, it means overlong UTF-8 sequences were used,
        // or invalid encodings were encountered. Reject such attempts
        std::string filename_reencoded = converter.to_bytes(filename_utf32);
        if (filename_reencoded != filename) {
            return false;
        }
    } catch (const std::exception &) {
        return false;
    }

    // Check for forbidden codepoints:
    // - Control characters
    // - Unicode equivalents of illegal characters
    // - UTF-16 surrogate pairs
    // - UTF-8 replacement character
    // - Byte order mark (BOM)
    // - Illegal characters: / \ : * ? " < > |
    for (char32_t c : filename_utf32) {
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == '/' || c == '\\' || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

// returns true if successful, false otherwise
bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wpath = converter.from_bytes(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);
        const wchar_t * test = subpath.c_str();

        const bool success = CreateDirectoryW(test, NULL);
        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#ifdef __linux__
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#endif // __linux__
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

std::string fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    std::string cache_directory = fs_get_cache_directory();
    const bool success = fs_create_directory_with_parents(cache_directory);
    if (!success) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory);
    }
    return cache_directory + filename;
}

template <typename T>
static std::string vec_to_str(const std::vector<T> & vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

static bool assign_layers_to_device(
                                uint32_t   n_world,
                       const device_info * dev_info_set, 
                                uint32_t * n_layer_window, 
                                uint32_t * n_gpu_layers,
                      struct llama_model * model,
       const struct llama_context_params   cparams,
                                   float   min_disk_read_speed = 0.1f) { // minimum disk I/O speed: 100 MB/s
    GGML_ASSERT(dev_info_set != nullptr);
    GGML_ASSERT(n_layer_window != nullptr);

    const uint32_t n_layer = llama_model_n_layers(model);
    std::vector<int>   w(n_world, 0);
    std::vector<int>   n(n_world, 0);
    std::vector<float> mem_budget(n_world, 0.0f);

    // model-specific constants
    const int n_embd_k_gqa = llama_model_n_embd_k_gqa(model);
    const int n_embd_v_gqa = llama_model_n_embd_v_gqa(model);
    if (n_embd_k_gqa <= 0 || n_embd_v_gqa <= 0) {
        LOG_ERR("Invalid model parameters,n_embd_k_gqa and n_embd_v_gqa cannot be less than 0");
        return false;
    }
    const int n_kv         = cparams.n_ctx;

    const int64_t b        = dev_info_set[0].model_bytes.nb_layer;
    const int64_t bo       = dev_info_set[0].model_bytes.nb_output;
    const int64_t b_prime  = b + 2 * (n_embd_k_gqa + n_embd_v_gqa) * n_kv;

#if defined(USE_HIGHS) 
    const device_info &master = dev_info_set[0];
    const int n_vocab = llama_n_vocab(model);
    const int64_t bi  = dev_info_set[0].model_bytes.nb_input;

    // device-specific constants
    std::vector<float> alpha(n_world, 0.0f);
    std::vector<float> beta(n_world, 0.0f);
    std::vector<float> xi(n_world, 0.0f);
    float kappa = 0.0f;

    // -------- Compute alpha[m], beta[m], xi[m] --------
    for (uint32_t m = 0; m < n_world; ++m) {
        // alpha[m]
        const device_info & dev = dev_info_set[m];
        float t_read_ram_cpu = 0.0f;

        float t_calc_cpu = (
            master.model_flops.layer_f32_f32   / (dev.cpu_props.flops_f32_f32   * 1e9 + EPS) +
            master.model_flops.layer_f16_f32   / (dev.cpu_props.flops_f16_f32   * 1e9 + EPS) +
            master.model_flops.layer_q2k_f32   / (dev.cpu_props.flops_q2k_f32   * 1e9 + EPS) +
            master.model_flops.layer_q4k_f32   / (dev.cpu_props.flops_q4k_f32   * 1e9 + EPS) +
            master.model_flops.layer_q5k_f32   / (dev.cpu_props.flops_q5k_f32   * 1e9 + EPS) +
            master.model_flops.layer_q6k_f32   / (dev.cpu_props.flops_q6k_f32   * 1e9 + EPS) +
            master.model_flops.layer_iq2xxs_f32/ (dev.cpu_props.flops_iq2xxs_f32* 1e9 + EPS) +
            master.model_flops.layer_q50_f32   / (dev.cpu_props.flops_q50_f32   * 1e9 + EPS) +
            master.model_flops.layer_q80_f32   / (dev.cpu_props.flops_q80_f32   * 1e9 + EPS) +
            master.model_flops.layer_iq1s_f32  / (dev.cpu_props.flops_iq1s_f32  * 1e9 + EPS) +
            master.model_flops.layer_iq4nl_f32 / (dev.cpu_props.flops_iq4nl_f32 * 1e9 + EPS) +
            master.model_flops.layer_iq1m_f32  / (dev.cpu_props.flops_iq1m_f32  * 1e9 + EPS) ) * 1000; // in ms

        float t_kv_cpy_cpu = dev.memory.mem_cpy_delay; // in ms
        // t_read_ram_cpu = b_prime / (dev.memory.cpu_read_ram_bw * 1e9) * 1000; // in ms

        alpha[m] = t_calc_cpu + t_kv_cpy_cpu + t_read_ram_cpu; // in ms

        // beta[m]
        if (dev.gpu_support.metal || dev.gpu_support.cuda) {
            float t_calc_gpu     = 0.0;
            float t_kv_cpy_gpu   = 0.0;
            float t_read_ram_gpu = 0.0;

            if (dev.gpu_support.metal) {
                t_calc_gpu = (
                    master.model_flops.layer_f32_f32    / (dev.gpu_props.metal_flops_f32_f32    * 1e9 + EPS) +
                    master.model_flops.layer_f16_f32    / (dev.gpu_props.metal_flops_f16_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q2k_f32    / (dev.gpu_props.metal_flops_q2k_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q4k_f32    / (dev.gpu_props.metal_flops_q4k_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q5k_f32    / (dev.gpu_props.metal_flops_q5k_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q6k_f32    / (dev.gpu_props.metal_flops_q6k_f32    * 1e9 + EPS) +
                    master.model_flops.layer_iq2xxs_f32 / (dev.gpu_props.metal_flops_iq2xxs_f32 * 1e9 + EPS) +
                    master.model_flops.layer_q50_f32    / (dev.gpu_props.metal_flops_q50_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q80_f32    / (dev.gpu_props.metal_flops_q80_f32    * 1e9 + EPS) +
                    master.model_flops.layer_iq1s_f32   / (dev.gpu_props.metal_flops_iq1s_f32   * 1e9 + EPS) +
                    master.model_flops.layer_iq4nl_f32  / (dev.gpu_props.metal_flops_iq4nl_f32  * 1e9 + EPS) +
                    master.model_flops.layer_iq1m_f32   / (dev.gpu_props.metal_flops_iq1m_f32   * 1e9 + EPS) ) * 1000; // in ms

                t_kv_cpy_gpu = dev.gpu_props.metal_mem_cpy_delay; // in ms
                // t_read_ram_gpu = b_prime / (dev.gpu_props.metal_read_vram_bw * 1e9) * 1000; // in ms
            } else {
                t_calc_gpu = (
                    master.model_flops.layer_f32_f32    / (dev.gpu_props.cuda_flops_f32_f32    * 1e9 + EPS) +
                    master.model_flops.layer_f16_f32    / (dev.gpu_props.cuda_flops_f16_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q2k_f32    / (dev.gpu_props.cuda_flops_q2k_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q4k_f32    / (dev.gpu_props.cuda_flops_q4k_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q5k_f32    / (dev.gpu_props.cuda_flops_q5k_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q6k_f32    / (dev.gpu_props.cuda_flops_q6k_f32    * 1e9 + EPS) +
                    master.model_flops.layer_iq2xxs_f32 / (dev.gpu_props.cuda_flops_iq2xxs_f32 * 1e9 + EPS) +
                    master.model_flops.layer_q50_f32    / (dev.gpu_props.cuda_flops_q50_f32    * 1e9 + EPS) +
                    master.model_flops.layer_q80_f32    / (dev.gpu_props.cuda_flops_q80_f32    * 1e9 + EPS) +
                    master.model_flops.layer_iq1s_f32   / (dev.gpu_props.cuda_flops_iq1s_f32   * 1e9 + EPS) +
                    master.model_flops.layer_iq4nl_f32  / (dev.gpu_props.cuda_flops_iq4nl_f32  * 1e9 + EPS) +
                    master.model_flops.layer_iq1m_f32   / (dev.gpu_props.cuda_flops_iq1m_f32   * 1e9 + EPS) ) * 1000; // in ms

                t_kv_cpy_gpu = dev.gpu_props.cuda_mem_cpy_delay; // in ms
                // t_read_ram_gpu = b_prime / (dev.gpu_props.cuda_read_vram_bw * 1e9) * 1000; // in ms
            }
            beta[m] = t_calc_gpu - t_calc_cpu + t_kv_cpy_gpu - t_kv_cpy_cpu + t_read_ram_gpu - t_read_ram_cpu; // in ms
        }
        
        // xi[m]
        // the ram-vram and vram-ram transfer time and the communication time are less than 1 ms
        xi[m] = 0.0;
    }

    // we adopt an iterative optimization approach. Initially, $w_m$ is set proportionally 
    // based on the available memory budget and $n_m$ is initialized to 0. 
    for (uint32_t m = 0; m < n_world; ++m) {
        const device_info & dev = dev_info_set[m];
        GGML_ASSERT(dev.device_os != nullptr);

        bool is_macos   = strcmp(dev.device_os, "macOS") == 0;
        bool is_linux   = strcmp(dev.device_os, "Linux") == 0;
        bool is_android = strcmp(dev.device_os, "Android") == 0;
        bool is_windows = strcmp(dev.device_os, "Windows") == 0;
        GGML_ASSERT(!is_windows && "Windows is not tested yet\n");
        
        if ((is_macos && !dev.gpu_support.metal) || is_linux) {
            mem_budget[m] = dev.memory.available_physical;
        } else if (is_macos && dev.gpu_support.metal) {
            mem_budget[m] = dev.gpu_props.memory_free;
        } else if (is_android) {
            mem_budget[m] = dev.memory.available_physical + dev.memory.used_can_swap;
        } else {
            // todo: add support for other OS such as Windows
            GGML_ASSERT(false && "Unsupported OS\n");
        }
    }

    // initialize w_m proportionally to memory budget and n_m to 0
    float total_mem_budget = std::accumulate(mem_budget.begin(), mem_budget.end(), 0.0f);
    for (uint32_t m = 0; m < n_world; ++m) {
        w[m] = std::round(mem_budget[m] / total_mem_budget * n_layer);
        n[m] = 0;
    }
    // adjust w[m] to ensure L mod W = 0
    int diff = n_layer - std::accumulate(w.begin(), w.end(), 0);
    auto device = (diff > 0) ? std::max_element(mem_budget.begin(), mem_budget.end()) 
                             : std::min_element(mem_budget.begin(), mem_budget.end());
    w[std::distance(mem_budget.begin(), device)] += diff;

    // stores the actual read bandwidth (GB/s) for each device
    std::vector<float> disk_speed(n_world, 0.0f);
    for (uint32_t m = 0; m < n_world; ++m) {
        const device_info & dev = dev_info_set[m];
        GGML_ASSERT(dev.device_os != nullptr);
        bool is_linux = strcmp(dev.device_os, "Linux") == 0;

        if (is_linux) {
            disk_speed[m] = dev.disk.read_seq_bw;
        } else {
            disk_speed[m] = dev.disk.read_rnd_bw;
        }
    }

    // helper function to find valid factors for a given n_layers
    auto find_factors = [&](int n_layers) {
        std::vector<int> factors;
        for (int k = 1; k <= n_layers / 2; ++k) {
            if (n_layers % k == 0) {
                factors.push_back(k);
            }
        }
        return factors;
    };

    // get valid factors
    std::vector<int> valid_k = cparams.n_cycles > 0 ? std::vector<int>{static_cast<int>(cparams.n_cycles)} : find_factors(n_layer);

    // assign devices to sets M1, M2, M3, and M4
    // M1: devices running on macOS without Metal, and with insufficient memory
    // M2: devices running on macOS with Metal and insufficient memory
    // M3: devices running on Linux or Android and with insufficient memory
    // M4: devices with sufficient memory or very slow disk I/O (slower than min_disk_io_speed)
    std::vector<uint32_t> M1, M2, M3, M4, M1_prev, M2_prev, M3_prev, M4_prev;
    std::vector<bool> M4_force(n_world, false);
    std::vector<int64_t> c_cpu(n_world, 0), c_gpu(n_world, 0);

    // helper function to check if a device is in a specific set
    auto in_set = [&](uint32_t m, const std::vector<uint32_t> & M) {
        return (std::find(M.begin(), M.end(), m) != M.end());
    };

    auto assign_sets = [&](int k) -> bool {
        M1.clear(), M2.clear(), M3.clear(), M4.clear();

        for (uint32_t m = 0; m < n_world; ++m) {
            const device_info & dev = dev_info_set[m];

            GGML_ASSERT(dev.device_os != nullptr);
            bool is_macos   = strcmp(dev.device_os, "macOS") == 0;
            bool is_linux   = strcmp(dev.device_os, "Linux") == 0;
            bool is_android = strcmp(dev.device_os, "Android") == 0;
            bool is_windows = strcmp(dev.device_os, "Windows") == 0;
            GGML_ASSERT(!is_windows && "Windows is not tested yet\n");

            bool use_gpu = dev.gpu_support.metal || dev.gpu_support.cuda;
            llama_model_compute_buf_size(&c_cpu[m], &c_gpu[m], model, cparams, use_gpu, m == 0, w[m] * k, n[m] * k);

            int  l_m          = w[m] * k;  // total number of layers assigned to device m
            int  l_m_gpu      = n[m] * k;  // number of layers assigned to device m that run on GPU
            bool condition1   = l_m * b + (bi / n_vocab + bo) * int(m == 0) + 2 * (n_embd_k_gqa + n_embd_v_gqa) * n_kv * l_m + c_cpu[m] > mem_budget[m] * GIGABYTE;
            bool condition2   = l_m * b + (bi / n_vocab + bo) * int(m == 0) + 2 * (n_embd_k_gqa + n_embd_v_gqa) * n_kv * l_m + c_cpu[m] + c_gpu[m] > mem_budget[m] * GIGABYTE;
            bool condition3   = (l_m - l_m_gpu) * b_prime + (bi / n_vocab + bo) * int(m == 0) + c_cpu[m] > mem_budget[m] * GIGABYTE;
            bool is_slow_disk = disk_speed[m] < min_disk_read_speed;

            if (M4_force[m] || is_slow_disk) {
                M4.push_back(m); // case 4: devices with very slow disk or force to be in M4
            } else if (is_macos && !dev.gpu_support.metal && condition1) {
                M1.push_back(m); // case 1: macOS without Metal, and with insufficient memory
            } else if (is_macos && dev.gpu_support.metal && condition2) {
                M2.push_back(m); // case 2: macOS with Metal, and with insufficient memory
            } else if ((is_linux || is_android) && condition3) {
                M3.push_back(m); // case 3: Linux with insufficient memory
            } else {
                M4.push_back(m); // case 4: devices with sufficient memory
            }
        }

        // check whether the sets are changed
        bool sets_changed = (M1 != M1_prev || M2 != M2_prev || M3 != M3_prev || M4 != M4_prev);

        // update the previous sets
        M1_prev = M1, M2_prev = M2, M3_prev = M3, M4_prev = M4;

        return sets_changed;
    };

    // helper function to print a matrix
    auto print_matrix = [](const std::vector<std::vector<double>>& matrix) {
        for (const auto& row : matrix) {
            for (const auto& elem : row) {
                printf("%.3f ", elem);
            }
            printf("\n");
        }
    };
    (void)print_matrix;

    std::vector<double> final_solution, rollback_solution;
    int final_k = -1, rollback_k = -1;

    // iterative optimization to find a valid set assignment (M1, M2, M3, M4)
    while (true) {
        int W = std::accumulate(w.begin(), w.end(), 0);
        int cur_k = (int)n_layer / W;

        if (W <= 1 || (int)n_layer % W != 0) {
            LOG_INF("Constraint: L = k * W must hold, but W = %d, L = %d\n", W, n_layer);
            fflush(stdout);
            fflush(stderr);
            return false;
        }

        if (!assign_sets(cur_k)) break;

        LOG_INF("Set assignment: M1: %s, M2: %s, M3: %s, M4: %s\n", 
                vec_to_str(M1).c_str(), vec_to_str(M2).c_str(), vec_to_str(M3).c_str(), vec_to_str(M4).c_str());

        // update kappa
        for (uint32_t m = 0; m < n_world; ++m) {
            const device_info & dev = dev_info_set[m];
            GGML_ASSERT(dev.device_os != nullptr);
            bool is_android = strcmp(dev.device_os, "Android") == 0;

            if (m == 0) {
                kappa = (
                    dev.model_flops.layer_f32_f32    / (dev.cpu_props.flops_f32_f32    * 1e9 + EPS) +
                    dev.model_flops.layer_f16_f32    / (dev.cpu_props.flops_f16_f32    * 1e9 + EPS) +
                    dev.model_flops.layer_q2k_f32    / (dev.cpu_props.flops_q2k_f32    * 1e9 + EPS) +
                    dev.model_flops.layer_q4k_f32    / (dev.cpu_props.flops_q4k_f32    * 1e9 + EPS) +
                    dev.model_flops.layer_q5k_f32    / (dev.cpu_props.flops_q5k_f32    * 1e9 + EPS) +
                    dev.model_flops.layer_q6k_f32    / (dev.cpu_props.flops_q6k_f32    * 1e9 + EPS) +
                    dev.model_flops.layer_iq2xxs_f32 / (dev.cpu_props.flops_iq2xxs_f32 * 1e9 + EPS) +
                    dev.model_flops.layer_q50_f32    / (dev.cpu_props.flops_q50_f32    * 1e9 + EPS) +
                    dev.model_flops.layer_q80_f32    / (dev.cpu_props.flops_q80_f32    * 1e9 + EPS) +
                    dev.model_flops.layer_iq1s_f32   / (dev.cpu_props.flops_iq1s_f32   * 1e9 + EPS) +
                    dev.model_flops.layer_iq4nl_f32  / (dev.cpu_props.flops_iq4nl_f32  * 1e9 + EPS) +
                    dev.model_flops.layer_iq1m_f32   / (dev.cpu_props.flops_iq1m_f32   * 1e9 + EPS) ) * 1000; // in ms
                // kappa += (bi / n_vocab + bo) / (dev.memory.cpu_read_ram_bw * 1e9) * 1000; // in ms

                kappa += (bi / n_vocab) / (disk_speed[m] * 1e9) * 1000; // in ms
                if (!in_set(m, M4)) {
                    kappa += bo / (disk_speed[m] * 1e9) * 1000; // in ms
                }
            }

            if (in_set(m, M1) || in_set(m, M3)) {  
                kappa += (c_cpu[m] - dev.memory.available_physical * GIGABYTE - dev.memory.used_can_swap * GIGABYTE * int(is_android)) / (disk_speed[m] * 1e9) * 1000; // in ms
            }
        }

        std::vector<int> dev_gpu(n_world, 0);
        for (uint32_t m = 0; m < n_world; ++m) {
            const device_info & dev = dev_info_set[m];
            if (dev.gpu_support.cuda || dev.gpu_support.metal) {
                dev_gpu[m] = 1;
            }
        }

        // -------------------------------------------------------------
        // Construct vectors va, vb, vc
        // -------------------------------------------------------------
        std::vector<float> vec_a(n_world, 0.0f), vec_b(n_world, 0.0f), vec_c(n_world, 0.0f);
        
        for (uint32_t m = 0; m < n_world; ++m) {
            if (in_set(m, M1)) {
                vec_a[m] = alpha[m] + b_prime / (disk_speed[m] * 1e9) * 1000; // in ms
            } else if (in_set(m, M2)) {
                vec_a[m] = alpha[m] + b / (disk_speed[m] * 1e9) * 1000; // in ms
                vec_b[m] = beta[m];
            } else if (in_set(m, M3)) {
                vec_a[m] = alpha[m] + b_prime / (disk_speed[m] * 1e9) * 1000; // in ms
                if (dev_gpu[m]) vec_b[m] = beta[m] - b_prime / (disk_speed[m] * 1e9) * 1000; // in ms
            } else {
                vec_a[m] = alpha[m];
                if (dev_gpu[m]) vec_b[m] = beta[m];
            }
            vec_c[m] = xi[m];
        }

        // -------------------------------------------------------------
        // Construct vectors vz, vz_gpu
        // -------------------------------------------------------------
        std::vector<float> vec_z(n_world, 0.0f), vec_z_gpu(n_world, 0.0f);

        for (uint32_t m = 0; m < n_world; ++m) {
            const device_info & dev = dev_info_set[m];

            GGML_ASSERT(dev.device_os != nullptr);
            bool is_macos   = strcmp(dev.device_os, "macOS") == 0;
            bool is_android = strcmp(dev.device_os, "Android") == 0;
            bool is_windows = strcmp(dev.device_os, "Windows") == 0;
            GGML_ASSERT(!is_windows && "Windows is not tested yet\n");

            int64_t b_cio = (bi / n_vocab + bo) * int(m == 0) + c_cpu[m];

            if (in_set(m, M1)) {
                vec_z[m] = (double)(dev.memory.available_physical * GIGABYTE - b_cio) / (double)(n_layer * b_prime);
            } else if (in_set(m, M2)) {
                vec_z[m] = (double)(dev.gpu_props.memory_free * GIGABYTE - b_cio - c_gpu[m]) / (double)(n_layer * b_prime);
            } else if (in_set(m, M3)) {
                vec_z[m] = (double)(dev.memory.available_physical * GIGABYTE + dev.memory.used_can_swap * GIGABYTE * int(is_android) - b_cio) / (double)(n_layer * b_prime);
            } else {
                if (is_macos && !dev.gpu_support.metal) {
                    vec_z[m] = - (double)(dev.memory.available_physical * GIGABYTE - b_cio) / (double)(n_layer * b_prime);
                } else if (is_macos && dev.gpu_support.metal) {
                    vec_z[m] = - (double)(dev.gpu_props.memory_free * GIGABYTE - b_cio - c_gpu[m]) / (double)(n_layer * b_prime);
                } else {
                    vec_z[m] = - (double)((dev.memory.available_physical + dev.memory.used_can_swap * int(is_android)) * GIGABYTE - b_cio) / (double)(n_layer * b_prime);
                }
            }

            if (dev_gpu[m]) {
                vec_z_gpu[m] = (double)(dev.gpu_props.memory_free * GIGABYTE - c_gpu[m]) / (double)(n_layer * b_prime);
                if (dev.gpu_support.metal && m == 0 && cparams.keep_out_in_metal) {
                    vec_z_gpu[m] -= (double)bo / (double)(n_layer * b_prime);
                }
            }
        }

        // -------------------------------------------------------------
        // Build and solve the optimization model
        // -------------------------------------------------------------
        double best_objective = 1.0e30;
        std::vector<double> best_solution;
        int best_k = -1;

        // iterate over all possible values of k to find the best solution
        for (int k : valid_k) {
            GGML_ASSERT(n_layer % k == 0 && "Constraint: L = k * W must hold\n");
            int W = n_layer / k;

            HighsModel model;

            // define the number of decision variables and constraints
            model.lp_.num_col_ = n_world * 2; // number of decision variables
            model.lp_.num_row_ = 1 + 3 * n_world; // number of constraints

            // define the objective: k * sum(a[m] * w[m] + b[m] * n[m]) + kappa + k * sum(c[m])
            model.lp_.sense_  = ObjSense::kMinimize;
            model.lp_.offset_ = k * std::accumulate(vec_c.begin(), vec_c.end(), 0.0f) + kappa;
            model.lp_.col_cost_.clear();
            std::copy(vec_a.begin(), vec_a.end(), std::back_inserter(model.lp_.col_cost_));
            std::copy(vec_b.begin(), vec_b.end(), std::back_inserter(model.lp_.col_cost_));
            std::transform(
                model.lp_.col_cost_.begin(), 
                model.lp_.col_cost_.end(), 
                model.lp_.col_cost_.begin(), [k](double cost) {
                    return cost * k;
                }
            );

            // define the variable bounds
            model.lp_.col_lower_ = std::vector<double>(n_world * 2, 0.0);
            std::fill(model.lp_.col_lower_.begin(), model.lp_.col_lower_.begin() + n_world, 1.0);
            model.lp_.col_upper_ = std::vector<double>(n_world * 2, n_layer);

            // define the constraint bounds
            int constraint_idx = 0;
            model.lp_.row_lower_ = std::vector<double>(model.lp_.num_row_, -1.0e30); // initialize to a large negative value
            model.lp_.row_upper_ = std::vector<double>(model.lp_.num_row_,  1.0e30); // initialize to a large positive value
            
            // constraint bound 1: sum(w[m]) = W
            model.lp_.row_lower_[constraint_idx] = {(double)W}; 
            model.lp_.row_upper_[constraint_idx] = {(double)W};
            constraint_idx++;

            // constraint bound 2: n[m] <= w[m], m = 1, 2, ..., n_world
            std::fill_n(model.lp_.row_upper_.begin() + constraint_idx, n_world, 0.0); // constraint: -w[m] + n[m] <= 0.0
            constraint_idx += n_world;

            // constraint bound 3: RAM constraint for each device
            for (uint32_t m = 0; m < n_world; ++m) {
                model.lp_.row_upper_[constraint_idx + m] = -W * vec_z[m];
            }
            constraint_idx += n_world;

            // constraint bound 4: CUDA/shared memory constraint for CUDA/Metal devices
            for (uint32_t m = 0; m < n_world; ++m) {
                double upper_bound = W * vec_z_gpu[m];
                model.lp_.row_upper_[constraint_idx] = std::max(upper_bound, 0.0);
                constraint_idx++;
            }

            // define the constraint matrix
            const int n_rows = model.lp_.num_row_;
            const int n_cols = model.lp_.num_col_;
            std::vector<std::vector<double>> A(n_rows, std::vector<double>(n_cols, 0.0));
            constraint_idx = 0;

            // constraint coefficients 1: sum(w[m]) = W
            std::fill_n(A[constraint_idx].begin(), n_world, 1.0);
            constraint_idx++;

            // constraint coefficients 2: n[m] <= w[m], m = 1, 2, ..., n_world
            for (uint32_t m = 0; m < n_world; ++m) {
                A[constraint_idx + m][m] = -1.0; // coefficient for w[m]
                A[constraint_idx + m][m + n_world] = 1.0; // coefficient for n[m]
            }
            constraint_idx += n_world;
            
            // constraint coefficients 3: RAM constraint for each device
            for (uint32_t m = 0; m < n_world; ++m) {
                const device_info & dev = dev_info_set[m];
                GGML_ASSERT(dev.device_os != nullptr);
                bool is_macos = strcmp(dev.device_os, "macOS") == 0;
                int cons_row = constraint_idx + m;

                if (in_set(m, M1) || in_set(m, M2)) { // in sets M1 and M2
                    A[cons_row][m] = -1.0; // coefficient for w[m]
                } else if (in_set(m, M3)) { // in set M3
                    A[cons_row][m] = -1.0; // coefficient for w[m]
                    if (dev_gpu[m]) {
                        A[cons_row][m + n_world] = 1.0; // coefficient for n[m]
                    }
                } else { // in set M4
                    A[cons_row][m] = 1.0; // coefficient for w[m]
                    if (!is_macos && dev_gpu[m]) {
                        A[cons_row][m + n_world] = -1.0; // coefficient for n[m]
                    }
                }
            }
            constraint_idx += n_world;

            // constraint coefficients 4: CUDA/shared memory constraint for CUDA/Metal devices
            for (uint32_t m = 0; m < n_world; ++m) {
                A[constraint_idx][m + n_world] = 1.0; // coefficient for n[m]
                constraint_idx++;
            }

            // translate the constraint matrix A into the LP model
            model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
            model.lp_.a_matrix_.start_.resize(n_cols + 1);
            model.lp_.a_matrix_.index_.clear();
            model.lp_.a_matrix_.value_.clear();

            int nnz_count = 0; // number of non-zero elements
            for (int j = 0; j < n_cols; ++j) {
                model.lp_.a_matrix_.start_[j] = nnz_count;
                for (int i = 0; i < n_rows; ++i) {
                    if (A[i][j] != 0.0) {
                        model.lp_.a_matrix_.index_.push_back(i);
                        model.lp_.a_matrix_.value_.push_back(A[i][j]);
                        nnz_count++;
                    }
                }
            }
            model.lp_.a_matrix_.start_[n_cols] = nnz_count;

            // integer constraints
            model.lp_.integrality_ = std::vector<HighsVarType>(n_world * 2, HighsVarType::kInteger);

            // solve the optimization problem
            Highs highs;
            highs.setOptionValue("log_to_console", false); // disable logging

            HighsStatus return_status = highs.passModel(model);
            GGML_ASSERT(return_status == HighsStatus::kOk && "Failed to pass model\n");
            
            // run the solver
            try {
                std::signal(SIGABRT, highs_handler);
                return_status = highs.run();
                GGML_ASSERT(return_status == HighsStatus::kOk && "Failed to run the solver\n");
            } catch (const HiGHSException &e) {
                LOG_INF("Failed to run the solver when k = %d: unknown exception\n", k);
                continue;
            }

            // get the solution
            const HighsModelStatus& model_status = highs.getModelStatus();

            if (model_status != HighsModelStatus::kOptimal) {
                bool is_all_in_M4 = true;
                for (uint32_t m = 0; m < n_world; ++m) {
                    if (!in_set(m, M4)) {
                        is_all_in_M4 = false;
                        break;
                    }
                }
                if (!is_all_in_M4) continue;
            }

            // record the best solution
            const HighsSolution& solution = highs.getSolution();
            double objective_value = highs.getInfo().objective_function_value;

            if (solution.value_valid) {
                if (objective_value < best_objective) {
                    best_objective = objective_value;
                    best_k = k;
                    best_solution = solution.col_value;
                }
                LOG_INF("k = %2d, obj = %7.1f, solution: %s | best_k = %2d, best_obj = %7.1f, best_solution: %s\n", 
                    k, objective_value, vec_to_str(solution.col_value).c_str(), best_k, best_objective, vec_to_str(best_solution).c_str());
            }
        }

        if (best_solution.empty()) {
            LOG_INF("No feasible solution found for this set assignment, rolling back to previous sets.\n");

            final_solution = rollback_solution;
            final_k = rollback_k;

            // update w[m] and n[m]
            GGML_ASSERT(final_solution.size() == n_world * 2 && "Invalid solution\n");
            std::copy(final_solution.begin(), final_solution.begin() + n_world, w.begin());
            std::copy(final_solution.begin() + n_world, final_solution.end(), n.begin());

            break;
        } else {
            rollback_solution = best_solution;
            rollback_k = best_k;
        }

        // check the solution
        bool has_free_gpu_memory = false, has_gpu_overload = false, has_cpu_overload = false;
        for (uint32_t m = 0; m < n_world; ++m) {
            // if (!dev_gpu[m]) continue;
            uint32_t w_m = best_solution[m], n_m = best_solution[m + n_world];

            if (dev_gpu[m]) {
                if (n_m < static_cast<uint32_t>(std::floor(W * vec_z_gpu[m]))) {
                    // if there is still free GPU memory
                    has_free_gpu_memory = true;
                } else if (w_m > n_m) {
                    // if the GPU is overloaded
                    has_gpu_overload = true;
                }
            } else if (!in_set(m, M4)) {
                // if the CPU is overloaded
                has_cpu_overload = true;
            }
        }

        if (has_free_gpu_memory && (has_gpu_overload || has_cpu_overload)) {
            int worst_device = -1;
            float worst_speed = std::numeric_limits<float>::max();

            // find the device with slowest disk speed but was not in M4 yet
            for (uint32_t m = 0; m < n_world; ++m) {
                if (!in_set(m, M4) && disk_speed[m] < worst_speed) {
                    worst_speed = disk_speed[m];
                    worst_device = m;
                }
            }

            if (worst_device != -1) {
                M4_force[worst_device] = true;
                LOG_INF("Forcing device %d (disk speed %.2f GB/s) into M4\n", worst_device, worst_speed);
            } else {
                LOG_INF("Infeasible solution detected but no device can be forced into M4\n");
            }

            continue;
        }

        // update w[m] and n[m]
        GGML_ASSERT(best_solution.size() == n_world * 2 && "Invalid solution\n");
        std::copy(best_solution.begin(), best_solution.begin() + n_world, w.begin());
        std::copy(best_solution.begin() + n_world, best_solution.end(), n.begin());

        bool solution_unchanged = (final_solution == best_solution);

        // update the global best solution
        final_k = best_k;
        final_solution = best_solution;

        if (solution_unchanged) break;
    }

    LOG_INF("\n----- Allocation Strategy (by HiGHS) -----\n");
    LOG_INF("\nParameters:\n");
    LOG_INF("  - k = %d\n", final_k);
    LOG_INF("  - W = %d\n", n_layer / final_k);
    for (uint32_t m = 0; m < n_world; ++m) {
        const char * device_name = dev_info_set[m].device_name;
        GGML_ASSERT(final_solution[m] == w[m] && final_solution[m + n_world] == n[m]);
        LOG_INF("\n%s:\n", device_name);
        LOG_INF("  - Device Index   : %d\n", m);
        LOG_INF("  - Assignment Set : %s\n", in_set(m, M1) ? "M1" : in_set(m, M2) ? "M2" : in_set(m, M3) ? "M3" : "M4");
        LOG_INF("  - N Layer Window : %d\n", w[m]);
        LOG_INF("  - N GPU Layers   : %d\n", n[m]);
    }
    LOG_INF("\n");

    // copy value from w and n to n_layer_window and n_gpu_layers, respectively
    std::copy(w.begin(), w.end(), n_layer_window);
    std::copy(n.begin(), n.end(), n_gpu_layers);

#else
    (void)min_disk_read_speed;

    // assign layers according to RAM/VRAM
    for (uint32_t m = 0; m < n_world; ++m) {
        const device_info & dev = dev_info_set[m];
        if (dev.gpu_support.metal || dev.gpu_support.cuda) {
            mem_budget[m] = dev.gpu_props.memory_free;
        } else {
            mem_budget[m] = dev.memory.available_physical;
        } 
    }

    // initialize w_m proportionally to memory budget and n_m to 0
    float total_mem_budget = std::accumulate(mem_budget.begin(), mem_budget.end(), 0.0f);
    for (uint32_t m = 0; m < n_world; ++m) {
        w[m] = std::round(mem_budget[m] / total_mem_budget * n_layer);
        n[m] = 0;
    }
    // adjust w[m] to ensure L mod W = 0
    int diff = n_layer - std::accumulate(w.begin(), w.end(), 0);
    auto device = (diff > 0) ? std::max_element(mem_budget.begin(), mem_budget.end()) 
                            : std::min_element(mem_budget.begin(), mem_budget.end());
    w[std::distance(mem_budget.begin(), device)] += diff;
    std::copy(w.begin(), w.end(), n_layer_window);

    std::vector<float> vec_z_gpu(n_world, 0.0f);
    std::vector<int64_t> c_cpu(n_world, 0), c_gpu(n_world, 0); 

    for (uint32_t m = 0; m < n_world; ++m) {
        const device_info & dev = dev_info_set[m];
        bool use_gpu = dev.gpu_support.metal || dev.gpu_support.cuda;
        llama_model_compute_buf_size(&c_cpu[m], &c_gpu[m], model, cparams, use_gpu, m == 0, w[m], n[m]);

        if (dev.gpu_support.cuda || dev.gpu_support.metal) {
            int64_t required_mem = w[m] * b_prime;
            int64_t available_mem = dev.gpu_props.memory_free * GIGABYTE - c_gpu[m];
            if (dev.gpu_support.metal && m == 0 && cparams.keep_out_in_metal) {
                available_mem -= bo;
            }

            if (required_mem <= available_mem) {
                n_gpu_layers[m] = w[m];
            } else {
                n_gpu_layers[m] = available_mem / b_prime;
            }
        }
    }

#endif

    return true;
}

static bool assign_layers_and_select_devices(
                                uint32_t   n_world,
                std::vector<device_info>   dev_infos,
                                uint32_t * n_layer_window, 
                                uint32_t * n_gpu_layers,
                      struct llama_model * model,
       const struct llama_context_params   cparams) {
    memset(n_layer_window, 0, n_world * sizeof(uint32_t));
    memset(n_gpu_layers,   0, n_world * sizeof(uint32_t));

    std::vector<device_info> dev_infos_temp = dev_infos;
    std::vector<uint32_t> n_layer_windows_temp, n_gpu_layers_temp;
    
    while (n_world > 0) {
        std::vector<device_info> dev_infos_ = dev_infos_temp;
        std::vector<uint32_t> n_layer_windows_(n_world, 0), n_gpu_layers_(n_world, 0);
        
        if (!assign_layers_to_device(n_world, dev_infos_.data(), 
                                     n_layer_windows_.data(), n_gpu_layers_.data(), model, cparams)) {
            return false;
        }

        dev_infos_temp.clear();
        n_layer_windows_temp.clear();
        n_gpu_layers_temp.clear();

        for (uint32_t i = 0; i < n_world; i++) {
            if (n_layer_windows_[i] > 1 || i == 0 ) {
                dev_infos_temp.push_back(dev_infos_[i]);
                n_layer_windows_temp.push_back(n_layer_windows_[i]);
                n_gpu_layers_temp.push_back(n_gpu_layers_[i]);
            } else {
                // remove this device
                LOG_INF("Remove device %s (rank %d) with only %d layer assigned.\n", 
                        dev_infos_[i].device_name, dev_infos_[i].rank, n_layer_windows_[i]);
            }
        }

        if(dev_infos_temp.size() == n_world) {
            // no device be removed
            break;
        }

        n_world = dev_infos_temp.size();

        LOG_INF("Reassign layers to the remaining %d device(s).\n\n", n_world);
    }

    uint32_t i = 0 , j = 0;
    while (j < n_world) {
        if (dev_infos[i].rank == dev_infos_temp[j].rank) {
            n_layer_window[i] = n_layer_windows_temp[j];
            n_gpu_layers[i]   = n_gpu_layers_temp[j];
            j++;
        } else {
            n_layer_window[i] = 0;
            n_gpu_layers[i] = 0;
        }
        i++;
    }

    return true;
}

//
// Model utils
//

struct llama_init_result llama_init_from_gpt_params(gpt_params & params) {

#if !(defined(GGML_USE_METAL) || defined(GGML_USE_CUDA))
    // reset n_gpu_layers to 0 if GPU is not used
    params.n_gpu_layers  = 0;
#endif

    llama_init_result iparams;
    auto mparams = llama_model_params_from_gpt_params(params);

    struct llama_model * model = nullptr;

    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        model = llama_load_model_from_hf(params.hf_repo.c_str(), params.hf_file.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else if (!params.model_url.empty()) {
        model = llama_load_model_from_url(params.model_url.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else {
        model = llama_load_model_from_file(params.model.c_str(), mparams);
    }

    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.c_str());
        return iparams;
    }

    llama_model_loader * ml = llama_model_load(params.model.c_str(), model, &mparams);

    if (params.reranking) {
        bool ok = true;

        if (llama_token_bos(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have a  BOS token, reranking will not work\n", __func__);
            ok = false;
        }

        if (llama_token_eos(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have an EOS token, reranking will not work\n", __func__);
            ok = false;
        }

        if (llama_token_sep(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have a  SEP token, reranking will not work\n", __func__);
            ok = false;
        }

        if (!ok) {
            llama_free_model(model);
            return iparams;
        }
    }

    device_info dev_info;
    uint32_t n_world   = params.n_world;
    uint32_t my_rank   = params.rank;
    bool auto_schedule = params.n_layer_window[0] == 0;
    
    // create llama context
    struct llama_context_params cparams = llama_context_params_from_gpt_params(params);
    llama_context * lctx                = llama_new_context_with_model(model, cparams);

    if (n_world == 1) {
        uint32_t n_layers = llama_model_n_layers(model);
        
        // assign all layers to this device
        params.n_layer_window[0]  = n_layers;
        cparams.n_layer_window[0] = n_layers;
        mparams.n_layer_window[0] = n_layers;
        llama_context_n_layer_window(lctx)[0] = n_layers;

#if defined(GGML_USE_METAL) || defined(GGML_USE_CUDA)
        params.n_gpu_layers = std::min((int32_t)n_layers, params.n_gpu_layers);
        cparams.n_gpu_layers = params.n_gpu_layers;
        mparams.n_gpu_layers = params.n_gpu_layers;
#endif

    } else {
        uint32_t n_layer_window[32] = {0}, n_gpu_layers[32] = {0};

        // initialize sockets
        llama_init_sockets(lctx, n_world, my_rank);

        // broadcast startup args
        struct startup_args args;
        if (my_rank == 0){
            args.should_profile = auto_schedule;
            args.n_ctx          = params.n_ctx;
        }

        llama_bcast_startup_args(lctx, my_rank, &args);

        if (my_rank > 0) {
            // receive startup args
            auto_schedule = args.should_profile;
            params.n_ctx  = args.n_ctx;
            cparams.n_ctx = args.n_ctx;
        }

        // if n_world > 1 and need auto schdule, then prifile
        if (auto_schedule){
            // get device profile
            LOG_INF("\nstart profiling this device, this may take some seconds ...\n");
            dev_info.rank = params.rank;
            dev_info.next_ip = params.next_node_ip.c_str();
            if (n_world > 1) {
                llama_profile_device(&dev_info, model, ml, params.gpu_mem, params.n_predict, params.n_ctx, params.cpuparams.n_threads, params.flash_attn);
            }
        }

        // sychronize device profile to the master node
        if (my_rank == 0) {
            if (auto_schedule) {
                std::vector<device_info> dev_info_set(n_world);
                dev_info_set[0] = dev_info;

                llama_gather_device_info(lctx, dev_info_set.data());
                device_print_props(dev_info_set.data(), n_world, model, cparams);

                // assign layers to devices and remove weak devices
                if (!assign_layers_and_select_devices(n_world, dev_info_set, n_layer_window, n_gpu_layers, model, cparams)) {
                    LOG_ERR("%s: Invalid allocation by HiGHS solver\n", __func__);
                    llama_free(lctx);
                    llama_free_model(model);
                    return iparams;
                }
                llama_bcast_layer_setup(lctx, n_layer_window, n_gpu_layers);
                llama_rebuild_topo(lctx, n_layer_window, dev_info_set.data());
            } else {
                // use the user-defined n_layer_window
                std::copy(std::begin(params.n_layer_window), std::end(params.n_layer_window), n_layer_window);
                llama_bcast_layer_setup(lctx, n_layer_window, nullptr);
            }
        } else {
            if (auto_schedule){
                llama_send_device_info(lctx, &dev_info);
                llama_recv_layer_setup(lctx, n_layer_window, n_gpu_layers);
                llama_rebuild_topo    (lctx, n_layer_window, nullptr);
            } else {
                llama_recv_layer_setup(lctx, n_layer_window, n_gpu_layers);
            }
        }

        // if this is a weak device, then exit
        if (n_layer_window[my_rank] <= 0) {
            LOG_INF("No layer is assigned to me, exit.\n");
            llama_free(lctx);
            llama_free_model(model);
            exit(0);
        }

        // update my rank and n_world
        uint32_t update_rank = 0, update_n_world = 1;
        std::vector<uint32_t> n_layer_window_temp = {n_layer_window[0]}, n_gpu_layers_temp = {n_gpu_layers[0]};

        for (uint32_t i = 1; i < n_world; i++) {
            if (n_layer_window[i] <= 0) {
                continue;
            }
            if (i <= my_rank) {
                update_rank++;
            }
            update_n_world++;
            n_layer_window_temp.push_back(n_layer_window[i]);
            n_gpu_layers_temp.push_back(n_gpu_layers[i]);
        }

        memset(n_layer_window, 0, n_world * sizeof(uint32_t));
        memset(n_gpu_layers,   0, n_world * sizeof(uint32_t));

        for (uint32_t i = 0; i < update_n_world; i++) {
            n_layer_window[i] = n_layer_window_temp[i];
            n_gpu_layers[i]   = n_gpu_layers_temp[i];
        }

        // update my rank
        cparams.rank = update_rank;
        mparams.rank = update_rank;
        params.rank  = update_rank;
        my_rank      = update_rank;

        // update n_world
        cparams.n_world = update_n_world;
        mparams.n_world = update_n_world;
        params.n_world  = update_n_world;
        n_world         = update_n_world;

        llama_update_context_with_rankworld(lctx, update_rank, update_n_world);
        
        // update n_layer_window and n_gpu_layers
        std::copy(std::begin(n_layer_window), std::end(n_layer_window), params.n_layer_window);
        std::copy(std::begin(n_layer_window), std::end(n_layer_window), cparams.n_layer_window);
        std::copy(std::begin(n_layer_window), std::end(n_layer_window), mparams.n_layer_window);
        std::copy(std::begin(n_layer_window), std::end(n_layer_window), llama_context_n_layer_window(lctx));

        if (params.n_gpu_layers == 0) { // if -ngl not set
            params.n_gpu_layers  = n_gpu_layers[my_rank];
            cparams.n_gpu_layers = n_gpu_layers[my_rank];
            mparams.n_gpu_layers = n_gpu_layers[my_rank];
            llama_model_set_n_gpu_layers(model, n_gpu_layers[my_rank]);
        } else { // -ngl is set
            params.n_gpu_layers  = std::min(params.n_gpu_layers, (int32_t)n_layer_window[my_rank]);
            cparams.n_gpu_layers = params.n_gpu_layers;
            mparams.n_gpu_layers = params.n_gpu_layers;
            llama_model_set_n_gpu_layers(model, params.n_gpu_layers);
        }
    }

    LOG_INF("\nUsing window size: %d, GPU layers: %d\n\n", cparams.n_layer_window[my_rank], cparams.n_gpu_layers);

    if (!mparams.vocab_only && llm_load_tensors(ml, model, mparams) < 0) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.c_str());
        llama_free(lctx);
        llama_free_model(model);
        return iparams;
    }

    llama_perf_context_sync(lctx, model);

    if (llama_context_setup_backend(model, cparams, lctx) == nullptr) {
        LOG_ERR("%s: failed to setup context with model '%s'\n", __func__, params.model.c_str());
        llama_free(lctx);
        llama_free_model(model);
        return iparams;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_n_layer(model);

        const auto cvec = llama_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }

        int err = llama_control_vector_apply(lctx,
                                             cvec.data.data(),
                                             cvec.data.size(),
                                             cvec.n_embd,
                                             params.control_vector_layer_start,
                                             params.control_vector_layer_end);
        if (err) {
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_lora_adapter_container loaded_la;
        loaded_la.path = la.path;
        loaded_la.scale = la.scale;
        loaded_la.adapter = llama_lora_adapter_init(model, la.path.c_str());
        if (loaded_la.adapter == nullptr) {
            LOG_ERR("%s: failed to apply lora adapter '%s'\n", __func__, la.path.c_str());
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }
        iparams.lora_adapters.push_back(loaded_la); // copy to list of loaded adapters
    }
    if (!params.lora_init_without_apply) {
        llama_lora_adapters_apply(lctx, iparams.lora_adapters);
    }

    if (params.sparams.ignore_eos && llama_token_eos(model) == LLAMA_TOKEN_NULL) {
        LOG_WRN("%s: warning: model does not have an EOS token, ignoring --ignore-eos\n", __func__);
        params.sparams.ignore_eos = false;
    }

    if (params.warmup) {
        LOG_WRN("%s: warming up the model with an empty run - please wait ...\n", __func__);

        const uint32_t my_rank = cparams.rank;
        std::vector<llama_token> tmp;

        if (my_rank == 0) {
            llama_token bos = llama_token_bos(model);
            llama_token eos = llama_token_eos(model);
            // some models (e.g. T5) don't have a BOS token
            if (bos != LLAMA_TOKEN_NULL) {
                tmp.push_back(bos);
            }
            if (eos != LLAMA_TOKEN_NULL) {
                tmp.push_back(eos);
            }
            if (tmp.empty()) {
                tmp.push_back(0);
            }

            if (llama_model_has_encoder(model)) {
                throw std::runtime_error("this model is currently not supported");

                llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size(), 0, 0));
                llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
                if (decoder_start_token_id == -1) {
                    decoder_start_token_id = bos;
                }
                tmp.clear();
                tmp.push_back(decoder_start_token_id);
            }
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch), 0, 0));
        }
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);
    }

    iparams.model   = model;
    iparams.context = lctx;

    return iparams;
}

void llama_lora_adapters_apply(struct llama_context * ctx, std::vector<llama_lora_adapter_container> & lora_adapters) {
    llama_lora_adapter_clear(ctx);
    for (auto & la : lora_adapters) {
        if (la.scale != 0.0f) {
            llama_lora_adapter_set(ctx, la.adapter, la.scale);
        }
    }
}

struct llama_model_params llama_model_params_from_gpt_params(const gpt_params & params) {
    auto mparams = llama_model_default_params();

    if (params.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.n_gpu_layers;
    }
    mparams.n_world         = params.n_world;
    mparams.rank            = params.rank;
    mparams.rpc_servers     = params.rpc_servers.c_str();
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.use_mmap        = params.use_mmap;
    mparams.use_mlock       = params.use_mlock;
    mparams.check_tensors   = params.check_tensors;
    mparams.keep_out_in_metal = params.keep_out_in_metal;
    std::copy(std::begin(params.n_layer_window), std::end(params.n_layer_window), mparams.n_layer_window);
    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    return mparams;
}

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16, // Added BF16 data type support
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_world           = params.n_world;
    cparams.rank              = params.rank;
    cparams.prefetch          = params.prefetch;
    cparams.force             = params.force;
    cparams.keep_out_in_metal = params.keep_out_in_metal;
    cparams.n_gpu_layers      = params.n_gpu_layers;
    cparams.n_cycles          = params.n_cycles;
    cparams.is_perplexity_eval= params.is_perplexity_eval;
    std::copy(std::begin(params.n_layer_window), std::end(params.n_layer_window), cparams.n_layer_window);

    if (cparams.master_ip != nullptr) {
        delete[] cparams.master_ip;
    }
    cparams.master_ip         = new char[params.master_ip.length() + 1];
    std::strcpy(cparams.master_ip, params.master_ip.c_str());

    if (cparams.next_node_ip != nullptr) {
        delete[] cparams.next_node_ip;
    }
    cparams.next_node_ip      = new char[params.next_node_ip.length() + 1];
    std::strcpy(cparams.next_node_ip, params.next_node_ip.c_str());

    cparams.n_ctx             = params.n_ctx;
    cparams.n_predict         = params.n_predict;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                    params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.logits_all        = params.logits_all;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.defrag_thold      = params.defrag_thold;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.flash_attn        = params.flash_attn;
    cparams.no_perf           = params.no_perf;

    if (params.reranking) {
        cparams.embeddings    = true;
        cparams.pooling_type  = LLAMA_POOLING_TYPE_RANK;
    }

    cparams.type_k = kv_cache_type_from_str(params.cache_type_k);
    cparams.type_v = kv_cache_type_from_str(params.cache_type_v);

    return cparams;
}

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

#ifdef LLAMA_USE_CURL

#define CURL_MAX_RETRY 3
#define CURL_RETRY_DELAY_SECONDS 2


static bool starts_with(const std::string & str, const std::string & prefix) {
    // While we wait for C++20's std::string::starts_with...
    return str.rfind(prefix, 0) == 0;
}

static bool curl_perform_with_retry(const std::string& url, CURL* curl, int max_attempts, int retry_delay_seconds) {
    int remaining_attempts = max_attempts;

    while (remaining_attempts > 0) {
        LOG_INF("%s: Trying to download from %s (attempt %d of %d)...\n", __func__ , url.c_str(), max_attempts - remaining_attempts + 1, max_attempts);

        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
            return true;
        }

        int exponential_backoff_delay = std::pow(retry_delay_seconds, max_attempts - remaining_attempts) * 1000;
        LOG_WRN("%s: curl_easy_perform() failed: %s, retrying after %d milliseconds...\n", __func__, curl_easy_strerror(res), exponential_backoff_delay);

        remaining_attempts--;
        std::this_thread::sleep_for(std::chrono::milliseconds(exponential_backoff_delay));
    }

    LOG_ERR("%s: curl_easy_perform() failed after %d attempts\n", __func__, max_attempts);

    return false;
}

static bool llama_download_file(const std::string & url, const std::string & path, const std::string & hf_token) {

    // Initialize libcurl
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
    if (!curl) {
        LOG_ERR("%s: error initializing libcurl\n", __func__);
        return false;
    }

    bool force_download = false;

    // Set the URL, allow to follow http redirection
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);

    // Check if hf-token or bearer-token was specified
    if (!hf_token.empty()) {
      std::string auth_header = "Authorization: Bearer ";
      auth_header += hf_token.c_str();
      struct curl_slist *http_headers = NULL;
      http_headers = curl_slist_append(http_headers, auth_header.c_str());
      curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, http_headers);
    }

#if defined(_WIN32)
    // CURLSSLOPT_NATIVE_CA tells libcurl to use standard certificate store of
    //   operating system. Currently implemented under MS-Windows.
    curl_easy_setopt(curl.get(), CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
#endif

    // Check if the file already exists locally
    struct stat model_file_info;
    auto file_exists = (stat(path.c_str(), &model_file_info) == 0);

    // If the file exists, check its JSON metadata companion file.
    std::string metadata_path = path + ".json";
    nlohmann::json metadata;
    std::string etag;
    std::string last_modified;

    if (file_exists) {
        // Try and read the JSON metadata file (note: stream autoclosed upon exiting this block).
        std::ifstream metadata_in(metadata_path);
        if (metadata_in.good()) {
            try {
                metadata_in >> metadata;
                LOG_INF("%s: previous metadata file found %s: %s\n", __func__, metadata_path.c_str(), metadata.dump().c_str());
                if (metadata.contains("url") && metadata.at("url").is_string()) {
                    auto previous_url = metadata.at("url").get<std::string>();
                    if (previous_url != url) {
                        LOG_ERR("%s: Model URL mismatch: %s != %s\n", __func__, url.c_str(), previous_url.c_str());
                        return false;
                    }
                }
                if (metadata.contains("etag") && metadata.at("etag").is_string()) {
                    etag = metadata.at("etag");
                }
                if (metadata.contains("lastModified") && metadata.at("lastModified").is_string()) {
                    last_modified = metadata.at("lastModified");
                }
            } catch (const nlohmann::json::exception & e) {
            LOG_ERR("%s: error reading metadata file %s: %s\n", __func__, metadata_path.c_str(), e.what());
                return false;
            }
        }
    } else {
        LOG_INF("%s: no previous model file found %s\n", __func__, path.c_str());
    }

    // Send a HEAD request to retrieve the etag and last-modified headers
    struct llama_load_model_from_url_headers {
        std::string etag;
        std::string last_modified;
    };
    llama_load_model_from_url_headers headers;
    {
        typedef size_t(*CURLOPT_HEADERFUNCTION_PTR)(char *, size_t, size_t, void *);
        auto header_callback = [](char * buffer, size_t /*size*/, size_t n_items, void * userdata) -> size_t {
            llama_load_model_from_url_headers *headers = (llama_load_model_from_url_headers *) userdata;

            static std::regex header_regex("([^:]+): (.*)\r\n");
            static std::regex etag_regex("ETag", std::regex_constants::icase);
            static std::regex last_modified_regex("Last-Modified", std::regex_constants::icase);

            std::string header(buffer, n_items);
            std::smatch match;
            if (std::regex_match(header, match, header_regex)) {
                const std::string & key = match[1];
                const std::string & value = match[2];
                if (std::regex_match(key, match, etag_regex)) {
                    headers->etag = value;
                } else if (std::regex_match(key, match, last_modified_regex)) {
                    headers->last_modified = value;
                }
            }
            return n_items;
        };

        curl_easy_setopt(curl.get(), CURLOPT_NOBODY, 1L); // will trigger the HEAD verb
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 1L); // hide head request progress
        curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, static_cast<CURLOPT_HEADERFUNCTION_PTR>(header_callback));
        curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &headers);

        bool was_perform_successful = curl_perform_with_retry(url, curl.get(), CURL_MAX_RETRY, CURL_RETRY_DELAY_SECONDS);
        if (!was_perform_successful) {
            return false;
        }

        long http_code = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 200) {
            // HEAD not supported, we don't know if the file has changed
            // force trigger downloading
            force_download = true;
            LOG_ERR("%s: HEAD invalid http status code received: %ld\n", __func__, http_code);
        }
    }

    bool should_download = !file_exists || force_download;
    if (!should_download) {
        if (!etag.empty() && etag != headers.etag) {
            LOG_WRN("%s: ETag header is different (%s != %s): triggering a new download\n", __func__, etag.c_str(), headers.etag.c_str());
            should_download = true;
        } else if (!last_modified.empty() && last_modified != headers.last_modified) {
            LOG_WRN("%s: Last-Modified header is different (%s != %s): triggering a new download\n", __func__, last_modified.c_str(), headers.last_modified.c_str());
            should_download = true;
        }
    }
    if (should_download) {
        std::string path_temporary = path + ".downloadInProgress";
        if (file_exists) {
            LOG_WRN("%s: deleting previous downloaded file: %s\n", __func__, path.c_str());
            if (remove(path.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path.c_str());
                return false;
            }
        }

        // Set the output file

        struct FILE_deleter {
            void operator()(FILE * f) const {
                fclose(f);
            }
        };

        std::unique_ptr<FILE, FILE_deleter> outfile(fopen(path_temporary.c_str(), "wb"));
        if (!outfile) {
            LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path.c_str());
            return false;
        }

        typedef size_t(*CURLOPT_WRITEFUNCTION_PTR)(void * data, size_t size, size_t nmemb, void * fd);
        auto write_callback = [](void * data, size_t size, size_t nmemb, void * fd) -> size_t {
            return fwrite(data, size, nmemb, (FILE *)fd);
        };
        curl_easy_setopt(curl.get(), CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, static_cast<CURLOPT_WRITEFUNCTION_PTR>(write_callback));
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, outfile.get());

        //  display download progress
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);

        // helper function to hide password in URL
        auto llama_download_hide_password_in_url = [](const std::string & url) -> std::string {
            std::size_t protocol_pos = url.find("://");
            if (protocol_pos == std::string::npos) {
                return url;  // Malformed URL
            }

            std::size_t at_pos = url.find('@', protocol_pos + 3);
            if (at_pos == std::string::npos) {
                return url;  // No password in URL
            }

            return url.substr(0, protocol_pos + 3) + "********" + url.substr(at_pos);
        };

        // start the download
        LOG_INF("%s: trying to download model from %s to %s (server_etag:%s, server_last_modified:%s)...\n", __func__,
            llama_download_hide_password_in_url(url).c_str(), path.c_str(), headers.etag.c_str(), headers.last_modified.c_str());
        bool was_perform_successful = curl_perform_with_retry(url, curl.get(), CURL_MAX_RETRY, CURL_RETRY_DELAY_SECONDS);
        if (!was_perform_successful) {
            return false;
        }

        long http_code = 0;
        curl_easy_getinfo (curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code < 200 || http_code >= 400) {
            LOG_ERR("%s: invalid http status code received: %ld\n", __func__, http_code);
            return false;
        }

        // Causes file to be closed explicitly here before we rename it.
        outfile.reset();

        // Write the updated JSON metadata file.
        metadata.update({
            {"url", url},
            {"etag", headers.etag},
            {"lastModified", headers.last_modified}
        });
        std::ofstream(metadata_path) << metadata.dump(4);
        LOG_INF("%s: file metadata saved: %s\n", __func__, metadata_path.c_str());

        if (rename(path_temporary.c_str(), path.c_str()) != 0) {
            LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
            return false;
        }
    }

    return true;
}

struct llama_model * llama_load_model_from_url(
        const char * model_url,
        const char * path_model,
        const char * hf_token,
        const struct llama_model_params & params) {
    // Basic validation of the model_url
    if (!model_url || strlen(model_url) == 0) {
        LOG_ERR("%s: invalid model_url\n", __func__);
        return NULL;
    }

    if (!llama_download_file(model_url, path_model, hf_token)) {
        return NULL;
    }

    // check for additional GGUFs split to download
    int n_split = 0;
    {
        struct gguf_init_params gguf_params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ NULL,
        };
        auto * ctx_gguf = gguf_init_from_file(path_model, gguf_params);
        if (!ctx_gguf) {
            LOG_ERR("\n%s:  failed to load input GGUF from %s\n", __func__, path_model);
            return NULL;
        }

        auto key_n_split = gguf_find_key(ctx_gguf, LLM_KV_SPLIT_COUNT);
        if (key_n_split >= 0) {
            n_split = gguf_get_val_u16(ctx_gguf, key_n_split);
        }

        gguf_free(ctx_gguf);
    }

    if (n_split > 1) {
        char split_prefix[PATH_MAX] = {0};
        char split_url_prefix[LLAMA_CURL_MAX_URL_LENGTH] = {0};

        // Verify the first split file format
        // and extract split URL and PATH prefixes
        {
            if (!llama_split_prefix(split_prefix, sizeof(split_prefix), path_model, 0, n_split)) {
                LOG_ERR("\n%s: unexpected model file name: %s n_split=%d\n", __func__, path_model, n_split);
                return NULL;
            }

            if (!llama_split_prefix(split_url_prefix, sizeof(split_url_prefix), model_url, 0, n_split)) {
                LOG_ERR("\n%s: unexpected model url: %s n_split=%d\n", __func__, model_url, n_split);
                return NULL;
            }
        }

        // Prepare download in parallel
        std::vector<std::future<bool>> futures_download;
        for (int idx = 1; idx < n_split; idx++) {
            futures_download.push_back(std::async(std::launch::async, [&split_prefix, &split_url_prefix, &n_split, hf_token](int download_idx) -> bool {
                char split_path[PATH_MAX] = {0};
                llama_split_path(split_path, sizeof(split_path), split_prefix, download_idx, n_split);

                char split_url[LLAMA_CURL_MAX_URL_LENGTH] = {0};
                llama_split_path(split_url, sizeof(split_url), split_url_prefix, download_idx, n_split);

                return llama_download_file(split_url, split_path, hf_token);
            }, idx));
        }

        // Wait for all downloads to complete
        for (auto & f : futures_download) {
            if (!f.get()) {
                return NULL;
            }
        }
    }

    return llama_load_model_from_file(path_model, params);
}

struct llama_model * llama_load_model_from_hf(
        const char * repo,
        const char * model,
        const char * path_model,
        const char * hf_token,
        const struct llama_model_params & params) {
    // construct hugging face model url:
    //
    //  --repo ggml-org/models --file tinyllama-1.1b/ggml-model-f16.gguf
    //    https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf
    //
    //  --repo TheBloke/Mixtral-8x7B-v0.1-GGUF --file mixtral-8x7b-v0.1.Q4_K_M.gguf
    //    https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q4_K_M.gguf
    //

    std::string model_url = "https://huggingface.co/";
    model_url += repo;
    model_url += "/resolve/main/";
    model_url += model;

    return llama_load_model_from_url(model_url.c_str(), path_model, hf_token, params);
}

#else

struct llama_model * llama_load_model_from_url(
        const char * /*model_url*/,
        const char * /*path_model*/,
        const char * /*hf_token*/,
        const struct llama_model_params & /*params*/) {
    LOG_WRN("%s: llama.cpp built without libcurl, downloading from an url not supported.\n", __func__);
    return nullptr;
}

struct llama_model * llama_load_model_from_hf(
        const char * /*repo*/,
        const char * /*model*/,
        const char * /*path_model*/,
        const char * /*hf_token*/,
        const struct llama_model_params & /*params*/) {
    LOG_WRN("%s: llama.cpp built without libcurl, downloading from Hugging Face not supported.\n", __func__);
    return nullptr;
}

#endif // LLAMA_USE_CURL

//
// Batch utils
//

void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    return llama_tokenize(llama_get_model(ctx), text, add_special, parse_special);
}

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string llama_detokenize(llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Chat template utils
//

bool llama_chat_verify_template(const std::string & tmpl) {
    llama_chat_message chat[] = {{"user", "test"}};
    int res = llama_chat_apply_template(nullptr, tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string llama_chat_apply_template(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & msgs,
        bool add_ass) {
    int alloc_size = 0;
    bool fallback = false; // indicate if we must fallback to default chatml
    std::vector<llama_chat_message> chat;
    for (auto & msg : msgs) {
        chat.push_back({msg.role.c_str(), msg.content.c_str()});
        alloc_size += (msg.role.size() + msg.content.size()) * 1.25;
    }

    const char * ptr_tmpl = tmpl.empty() ? nullptr : tmpl.c_str();
    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    int32_t res = llama_chat_apply_template(model, ptr_tmpl, chat.data(), chat.size(), add_ass, buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        if (ptr_tmpl != nullptr) {
            // if the custom "tmpl" is not supported, we throw an error
            // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
            throw std::runtime_error("this custom template is not supported");
        } else {
            // If the built-in template is not supported, we default to chatml
            res = llama_chat_apply_template(nullptr, "chatml", chat.data(), chat.size(), add_ass, buf.data(), buf.size());
            fallback = true;
        }
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(
            fallback ? nullptr : model,
            fallback ? "chatml" : ptr_tmpl,
            chat.data(), chat.size(), add_ass, buf.data(), buf.size());
    }

    std::string formatted_chat(buf.data(), res);
    return formatted_chat;
}

std::string llama_chat_format_single(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & past_msg,
        const llama_chat_msg & new_msg,
        bool add_ass) {
    std::ostringstream ss;
    auto fmt_past_msg = past_msg.empty() ? "" : llama_chat_apply_template(model, tmpl, past_msg, false);
    std::vector<llama_chat_msg> chat_new(past_msg);
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    chat_new.push_back(new_msg);
    auto fmt_new_msg = llama_chat_apply_template(model, tmpl, chat_new, add_ass);
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string llama_chat_format_example(const struct llama_model * model,
        const std::string & tmpl) {
    std::vector<llama_chat_msg> msgs = {
        {"system",    "You are a helpful assistant"},
        {"user",      "Hello"},
        {"assistant", "Hi there"},
        {"user",      "How are you?"},
    };
    return llama_chat_apply_template(model, tmpl, msgs, true);
}

//
// KV cache utils
//

void llama_kv_cache_dump_view(const llama_kv_cache_view & view, int row_size) {
    static const char slot_chars[] = ".123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+";

    printf("=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache %d, largest empty slot=%d @ %d",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    llama_kv_cache_view_cell * c_curr = view.cells;
    llama_seq_id * cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        int seq_count = 0;
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) { seq_count++; }
        }
        putchar(slot_chars[std::min(sizeof(slot_chars) - 2, size_t(seq_count))]);
    }

    printf("\n=== Done dumping\n");
}

void llama_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size) {
    static const char slot_chars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    printf("=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache %d, largest empty slot=%d @ %d\n",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    std::unordered_map<llama_seq_id, size_t> seqs;
    llama_kv_cache_view_cell * c_curr = view.cells;
    llama_seq_id * cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] < 0) { continue; }
            if (seqs.find(cs_curr[j]) == seqs.end()) {
                if (seqs.size() + 1 >= sizeof(slot_chars)) { break; }
                const size_t sz = seqs.size();
                seqs[cs_curr[j]] = sz;
            }
        }
        if (seqs.size() + 1 >= sizeof(slot_chars)) { break; }
    }

    printf("=== Sequence legend: ");
    for (const auto & it : seqs) {
        printf("%zu=%d, ", it.second, it.first);
    }
    printf("'+'=other sequence ids");

    c_curr = view.cells;
    cs_curr = view.cells_sequences;
    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) {
                const auto & it = seqs.find(cs_curr[j]);
                putchar(it != seqs.end() ? int(slot_chars[it->second]) : '+');
            } else {
                putchar('.');
            }
        }
        putchar(' ');
    }

    printf("\n=== Done dumping\n");
}

//
// Embedding utils
//

void llama_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) sum = std::abs(inp[i]);
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float llama_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static llama_control_vector_data llama_control_vector_load_one(const llama_control_vector_load_info & load_info) {
    llama_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to load control vector file from %s\n", __func__, load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        LOG_WRN("%s: no direction tensors found in %s\n", __func__, load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            LOG_ERR("%s: invalid/unparsable direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            LOG_ERR("%s: invalid (zero) direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            LOG_ERR("%s: invalid (non-F32) direction tensor type in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            LOG_ERR("%s: invalid (non-1D) direction tensor shape in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            LOG_ERR("%s: direction tensor in %s does not match previous dimensions\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        LOG_WRN("%s: skipping %s due to invalid direction tensors\n", __func__, load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

llama_control_vector_data llama_control_vector_load(const std::vector<llama_control_vector_load_info> & load_infos) {
    llama_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = llama_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            LOG_ERR("%s: control vectors in %s does not match previous dimensions\n", __func__, info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        LOG_ERR("%s: no valid control vector files passed\n", __func__);
        result.data.clear();
    }

    return result;
}

//
// YAML utils
//

void yaml_dump_vector_float(FILE * stream, const char * prop_name, const std::vector<float> & data) {
    if (data.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    fprintf(stream, "%s: [", prop_name);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        fprintf(stream, "%e, ", data[i]);
    }
    fprintf(stream, "%e]\n", data.back());
}

void yaml_dump_vector_int(FILE * stream, const char * prop_name, const std::vector<int> & data) {
    if (data.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    fprintf(stream, "%s: [", prop_name);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        fprintf(stream, "%d, ", data[i]);
    }
    fprintf(stream, "%d]\n", data.back());
}

void yaml_dump_string_multiline(FILE * stream, const char * prop_name, const char * data) {
    std::string data_str(data == NULL ? "" : data);

    if (data_str.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    size_t pos_start = 0;
    size_t pos_found = 0;

    if (std::isspace(data_str[0]) || std::isspace(data_str.back())) {
        data_str = std::regex_replace(data_str, std::regex("\n"), "\\n");
        data_str = std::regex_replace(data_str, std::regex("\""), "\\\"");
        data_str = std::regex_replace(data_str, std::regex(R"(\\[^n"])"), R"(\$&)");
        data_str = "\"" + data_str + "\"";
        fprintf(stream, "%s: %s\n", prop_name, data_str.c_str());
        return;
    }

    if (data_str.find('\n') == std::string::npos) {
        fprintf(stream, "%s: %s\n", prop_name, data_str.c_str());
        return;
    }

    fprintf(stream, "%s: |\n", prop_name);
    while ((pos_found = data_str.find('\n', pos_start)) != std::string::npos) {
        fprintf(stream, "  %s\n", data_str.substr(pos_start, pos_found-pos_start).c_str());
        pos_start = pos_found + 1;
    }
}

void yaml_dump_non_result_info(FILE * stream, const gpt_params & params, const llama_context * lctx,
                               const std::string & timestamp, const std::vector<int> & prompt_tokens, const char * model_desc) {
    const auto & sparams = params.sparams;

    fprintf(stream, "build_commit: %s\n",        LLAMA_COMMIT);
    fprintf(stream, "build_number: %d\n",        LLAMA_BUILD_NUMBER);
    fprintf(stream, "cpu_has_arm_fma: %s\n",     ggml_cpu_has_arm_fma()     ? "true" : "false");
    fprintf(stream, "cpu_has_avx: %s\n",         ggml_cpu_has_avx()         ? "true" : "false");
    fprintf(stream, "cpu_has_avx_vnni: %s\n",    ggml_cpu_has_avx_vnni()    ? "true" : "false");
    fprintf(stream, "cpu_has_avx2: %s\n",        ggml_cpu_has_avx2()        ? "true" : "false");
    fprintf(stream, "cpu_has_avx512: %s\n",      ggml_cpu_has_avx512()      ? "true" : "false");
    fprintf(stream, "cpu_has_avx512_vbmi: %s\n", ggml_cpu_has_avx512_vbmi() ? "true" : "false");
    fprintf(stream, "cpu_has_avx512_vnni: %s\n", ggml_cpu_has_avx512_vnni() ? "true" : "false");
    fprintf(stream, "cpu_has_cuda: %s\n",        ggml_cpu_has_cuda()        ? "true" : "false");
    fprintf(stream, "cpu_has_vulkan: %s\n",      ggml_cpu_has_vulkan()      ? "true" : "false");
    fprintf(stream, "cpu_has_kompute: %s\n",     ggml_cpu_has_kompute()     ? "true" : "false");
    fprintf(stream, "cpu_has_fma: %s\n",         ggml_cpu_has_fma()         ? "true" : "false");
    fprintf(stream, "cpu_has_gpublas: %s\n",     ggml_cpu_has_gpublas()     ? "true" : "false");
    fprintf(stream, "cpu_has_neon: %s\n",        ggml_cpu_has_neon()        ? "true" : "false");
    fprintf(stream, "cpu_has_sve: %s\n",         ggml_cpu_has_sve()         ? "true" : "false");
    fprintf(stream, "cpu_has_f16c: %s\n",        ggml_cpu_has_f16c()        ? "true" : "false");
    fprintf(stream, "cpu_has_fp16_va: %s\n",     ggml_cpu_has_fp16_va()     ? "true" : "false");
    fprintf(stream, "cpu_has_riscv_v: %s\n",     ggml_cpu_has_riscv_v()     ? "true" : "false");
    fprintf(stream, "cpu_has_wasm_simd: %s\n",   ggml_cpu_has_wasm_simd()   ? "true" : "false");
    fprintf(stream, "cpu_has_blas: %s\n",        ggml_cpu_has_blas()        ? "true" : "false");
    fprintf(stream, "cpu_has_sse3: %s\n",        ggml_cpu_has_sse3()        ? "true" : "false");
    fprintf(stream, "cpu_has_vsx: %s\n",         ggml_cpu_has_vsx()         ? "true" : "false");
    fprintf(stream, "cpu_has_matmul_int8: %s\n", ggml_cpu_has_matmul_int8() ? "true" : "false");

#ifdef NDEBUG
    fprintf(stream, "debug: false\n");
#else
    fprintf(stream, "debug: true\n");
#endif // NDEBUG

    fprintf(stream, "model_desc: %s\n", model_desc);
    fprintf(stream, "n_vocab: %d  # output size of the final layer, 32001 for some models\n", llama_n_vocab(llama_get_model(lctx)));

#ifdef __OPTIMIZE__
    fprintf(stream, "optimize: true\n");
#else
    fprintf(stream, "optimize: false\n");
#endif // __OPTIMIZE__

    fprintf(stream, "time: %s\n", timestamp.c_str());

    fprintf(stream, "\n");
    fprintf(stream, "###############\n");
    fprintf(stream, "# User Inputs #\n");
    fprintf(stream, "###############\n");
    fprintf(stream, "\n");

    fprintf(stream, "alias: %s # default: unknown\n", params.model_alias.c_str());
    fprintf(stream, "batch_size: %d # default: 512\n", params.n_batch);
    fprintf(stream, "chunks: %d # default: -1 (unlimited)\n", params.n_chunks);
    fprintf(stream, "color: %s # default: false\n", params.use_color ? "true" : "false");
    fprintf(stream, "ctx_size: %d # default: 512\n", params.n_ctx);
    fprintf(stream, "escape: %s # default: false\n", params.escape ? "true" : "false");
    fprintf(stream, "file: # never logged, see prompt instead. Can still be specified for input.\n");
    fprintf(stream, "frequency_penalty: %f # default: 0.0 \n", sparams.penalty_freq);
    yaml_dump_string_multiline(stream, "grammar", sparams.grammar.c_str());
    fprintf(stream, "grammar-file: # never logged, see grammar instead. Can still be specified for input.\n");
    fprintf(stream, "hellaswag: %s # default: false\n", params.hellaswag ? "true" : "false");
    fprintf(stream, "hellaswag_tasks: %zu # default: 400\n", params.hellaswag_tasks);
    fprintf(stream, "ignore_eos: %s # default: false\n", sparams.ignore_eos ? "true" : "false");

    yaml_dump_string_multiline(stream, "in_prefix", params.input_prefix.c_str());
    fprintf(stream, "in_prefix_bos: %s # default: false\n", params.input_prefix_bos ? "true" : "false");
    yaml_dump_string_multiline(stream, "in_suffix", params.input_prefix.c_str());
    fprintf(stream, "interactive: %s # default: false\n", params.interactive ? "true" : "false");
    fprintf(stream, "interactive_first: %s # default: false\n", params.interactive_first ? "true" : "false");
    fprintf(stream, "keep: %d # default: 0\n", params.n_keep);
    fprintf(stream, "logdir: %s # default: unset (no logging)\n", params.logdir.c_str());

    fprintf(stream, "logit_bias:\n");
    for (const auto & logit_bias : sparams.logit_bias) {
        fprintf(stream, "  %d: %f", logit_bias.token, logit_bias.bias);
    }

    fprintf(stream, "lora:\n");
    for (auto & la : params.lora_adapters) {
        if (la.scale == 1.0f) {
            fprintf(stream, "  - %s\n", la.path.c_str());
        }
    }
    fprintf(stream, "lora_scaled:\n");
    for (auto & la : params.lora_adapters) {
        if (la.scale != 1.0f) {
            fprintf(stream, "  - %s: %f\n", la.path.c_str(), la.scale);
        }
    }
    fprintf(stream, "lora_init_without_apply: %s # default: false\n", params.lora_init_without_apply ? "true" : "false");
    fprintf(stream, "main_gpu: %d # default: 0\n", params.main_gpu);
    fprintf(stream, "min_keep: %d # default: 0 (disabled)\n", sparams.min_keep);
    fprintf(stream, "mirostat: %d # default: 0 (disabled)\n", sparams.mirostat);
    fprintf(stream, "mirostat_ent: %f # default: 5.0\n", sparams.mirostat_tau);
    fprintf(stream, "mirostat_lr: %f # default: 0.1\n", sparams.mirostat_eta);
    fprintf(stream, "mlock: %s # default: false\n", params.use_mlock ? "true" : "false");
    fprintf(stream, "model: %s # default: %s\n", params.model.c_str(), DEFAULT_MODEL_PATH);
    fprintf(stream, "model_draft: %s # default:\n", params.model_draft.c_str());
    fprintf(stream, "multiline_input: %s # default: false\n", params.multiline_input ? "true" : "false");
    fprintf(stream, "n_gpu_layers: %d # default: -1\n", params.n_gpu_layers);
    fprintf(stream, "n_predict: %d # default: -1 (unlimited)\n", params.n_predict);
    fprintf(stream, "n_probs: %d # only used by server binary, default: 0\n", sparams.n_probs);
    fprintf(stream, "no_mmap: %s # default: false\n", !params.use_mmap ? "true" : "false");
    fprintf(stream, "penalize_nl: %s # default: false\n", sparams.penalize_nl ? "true" : "false");
    fprintf(stream, "ppl_output_type: %d # default: 0\n", params.ppl_output_type);
    fprintf(stream, "ppl_stride: %d # default: 0\n", params.ppl_stride);
    fprintf(stream, "presence_penalty: %f # default: 0.0\n", sparams.penalty_present);
    yaml_dump_string_multiline(stream, "prompt", params.prompt.c_str());
    fprintf(stream, "prompt_cache: %s\n", params.path_prompt_cache.c_str());
    fprintf(stream, "prompt_cache_all: %s # default: false\n", params.prompt_cache_all ? "true" : "false");
    fprintf(stream, "prompt_cache_ro: %s # default: false\n", params.prompt_cache_ro ? "true" : "false");
    yaml_dump_vector_int(stream, "prompt_tokens", prompt_tokens);
    fprintf(stream, "repeat_penalty: %f # default: 1.1\n", sparams.penalty_repeat);

    fprintf(stream, "reverse_prompt:\n");
    for (std::string ap : params.antiprompt) {
        size_t pos = 0;
        while ((pos = ap.find('\n', pos)) != std::string::npos) {
            ap.replace(pos, 1, "\\n");
            pos += 1;
        }

        fprintf(stream, "  - %s\n", ap.c_str());
    }

    fprintf(stream, "rope_freq_base: %f # default: 10000.0\n", params.rope_freq_base);
    fprintf(stream, "rope_freq_scale: %f # default: 1.0\n", params.rope_freq_scale);
    fprintf(stream, "simple_io: %s # default: false\n", params.simple_io ? "true" : "false");
    fprintf(stream, "cont_batching: %s # default: false\n", params.cont_batching ? "true" : "false");
    fprintf(stream, "flash_attn: %s # default: false\n", params.flash_attn ? "true" : "false");
    fprintf(stream, "temp: %f # default: 0.8\n", sparams.temp);

    const std::vector<float> tensor_split_vector(params.tensor_split, params.tensor_split + llama_max_devices());
    yaml_dump_vector_float(stream, "tensor_split", tensor_split_vector);

    fprintf(stream, "tfs: %f # default: 1.0\n", sparams.tfs_z);
    fprintf(stream, "threads: %d # default: %u\n", params.cpuparams.n_threads, std::thread::hardware_concurrency());
    fprintf(stream, "top_k: %d # default: 40\n", sparams.top_k);
    fprintf(stream, "top_p: %f # default: 0.95\n", sparams.top_p);
    fprintf(stream, "min_p: %f # default: 0.0\n", sparams.min_p);
    fprintf(stream, "typ_p: %f # default: 1.0\n", sparams.typ_p);
    fprintf(stream, "verbose_prompt: %s # default: false\n", params.verbose_prompt ? "true" : "false");
    fprintf(stream, "display_prompt: %s # default: true\n", params.display_prompt ? "true" : "false");
}

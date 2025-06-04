#include "log.h"
#include "profiler.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include <sys/stat.h>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#elif defined(__linux__)
    #include <unistd.h>
    #include <sys/sysinfo.h>
    #include <sys/types.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <dirent.h>
#elif defined(__APPLE__) && defined(__MACH__)
    #include <sys/sysctl.h>
    #include <sys/param.h>
    #include <sys/mount.h>
    #include <mach/mach.h>
    #include <unistd.h>
#endif

#ifdef GGML_USE_METAL
    #include "ggml-metal.h"
#endif

#ifdef GGML_USE_CUDA
    #include "ggml-cuda.h"
    #include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <sys/types.h>
#include <vector>
#include <inttypes.h>
#include <thread>
#include <random>
#include <regex>
#include <unordered_map>
#include <dirent.h>


static int gcd_int(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}


static size_t get_page_size() {
    size_t page_size = 0;

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    page_size = si.dwPageSize;
#elif defined(__APPLE__) || defined(__linux__)
    page_size = sysconf(_SC_PAGESIZE);
#endif

    return page_size;
}

static const char * get_uname_os() {
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("uname -o", "r"), pclose);
    if (!pipe) {
        return "Unknown"; 
    }

    static char buffer[16]; 
    if (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        buffer[strcspn(buffer, "\n")] = '\0';
        return buffer; 
    }
    return "Unknown"; 
}

const char * device_name() {
    static char device_name[256];

#if defined(_WIN32) || defined(_WIN64)
    DWORD size = sizeof(device_name);
    if (GetComputerNameA(device_name, &size) == 0) {
        strncpy(device_name, "Unknown Windows Device", sizeof(device_name));
    }
#elif defined(__linux__)
    const char * os = get_uname_os(); 
    if (strstr(os, "Android") != nullptr) {
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("getprop ro.product.model", "r"), pclose);
        if (pipe) {
            if (fgets(device_name, sizeof(device_name), pipe.get()) != nullptr) {
                device_name[strcspn(device_name, "\n")] = '\0';
                return device_name;
            }
        }
        strncpy(device_name, "Unknown Device", sizeof(device_name));
    } else {
        if (gethostname(device_name, sizeof(device_name)) != 0) {
            strncpy(device_name, "Unknown Device", sizeof(device_name));
        }
    }
#elif defined(__APPLE__) && defined(__MACH__)
    if (gethostname(device_name, sizeof(device_name)) != 0) {
        strncpy(device_name, "Unknown Mac Device", sizeof(device_name));
    }
#else
    strncpy(device_name, "Unknown Device", sizeof(device_name));
#endif

    return device_name;
}

const char * device_os() {
#ifdef _WIN32
    return "Windows";
#elif __linux__
    // const char * os = get_uname_os();
    // if (strstr(os, "Android") != nullptr) {
    //     return "Android";
    // }
    return "Linux";
#elif __APPLE__ || __MACH__
    return "macOS";
#endif
}

uint32_t device_cpu_cores() {
    unsigned int core_count = 1; // default to 1 in case of failure

#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    core_count = sysinfo.dwNumberOfProcessors;
#elif defined(__linux__)
    core_count = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(__APPLE__) && defined(__MACH__)
    int mib[4];
    size_t len = sizeof(core_count);

    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;

    if (sysctl(mib, 2, &core_count, &len, NULL, 0) != 0 || core_count < 1) {
        mib[1] = HW_NCPU; // total number of cpus
        if (sysctl(mib, 2, &core_count, &len, NULL, 0) != 0 || core_count < 1) {
            core_count = 1; // default to 1 if sysctl fails
        }
    }
#endif

    return core_count;
}

static float device_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t, enum profiler_backend_type btype, int n_threads) {
    int n_repeat = 1;
    int n_embd   = std::min(llama_n_embd(model), 4096);

    // simulate small tensor calculation on cpu
    if (btype == PROFILER_BACKEND_TYPE_CPU) n_embd /= 8;

    // ensure that the block sizes of the tensors are compatible
    int bs0 = ggml_blck_size(src0t);
    int bs1 = ggml_blck_size(src1t);
    int gcd = gcd_int(bs0, bs1);
    int lcm = bs0 / gcd * bs1;

    if (n_embd % bs0 != 0 || n_embd % bs1 != 0) {
        if (n_embd < lcm) {
            n_embd = 2 * lcm;
        } else {
            n_embd = 2 * (n_embd / lcm) * lcm;
        }
    }

    std::vector<float> matrix_A(n_embd * n_embd, 1.0f); 
    std::vector<float> matrix_B(n_embd * n_embd, 1.0f / n_embd);

    ggml_backend_t backend = NULL;
    switch (btype) {
        case PROFILER_BACKEND_TYPE_CPU:
            backend = ggml_backend_cpu_init();
            break;
        case PROFILER_BACKEND_TYPE_METAL:
#ifdef GGML_USE_METAL
            backend = ggml_backend_metal_init();
#endif
            break;
        case PROFILER_BACKEND_TYPE_CUDA:
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
#endif
            break;
    }

    if (!backend) {
        LOG_INF("%s: ggml backend init failed\n", __func__);
        return 0.0f;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, src0t, n_embd, n_embd);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, src1t, n_embd, n_embd);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(tensor_a, matrix_A.data(), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, matrix_B.data(), 0, ggml_nbytes(tensor_b));

    struct ggml_cgraph  * gf         = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    struct ggml_tensor  * cur        = NULL;
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (n_repeat + 2) + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph(ctx_cgraph);
        
        cur = ggml_mul_mat(ctx_cgraph, tensor_a, tensor_b);
        for (int i = 0; i < n_repeat - 1; i++) {
            cur = ggml_mul_mat(ctx_cgraph, tensor_a, cur);
        }

        ggml_build_forward_expand(gf, cur);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    // use scheduler
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<ggml_backend_t> backends = {backend};
    if (!ggml_backend_is_cpu(backend)) {
        backends.push_back(ggml_backend_cpu_init());
    }

    for (ggml_backend_t bak : backends) {
        if (ggml_backend_is_cpu(bak)) {
            backend_buft.push_back(ggml_backend_cpu_buffer_type());
        } else {
            backend_buft.push_back(ggml_backend_get_default_buffer_type(bak));
        }
    }
    
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), backend_buft.data(), backends.size(), 256, false);

    bool ok = ggml_backend_sched_reserve(sched, gf);
    if (!ok) {
        LOG_INF("%s: failed to allocate compute buffers\n", __func__);
        ggml_free(ctx_cgraph);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return 0.0f;
    }

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    // warm-up
    ggml_backend_graph_compute(backend, gf);

    const int64_t t_start = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    const int64_t t_end = ggml_time_us();

    double elapsed_seconds = ((double)t_end - (double)t_start) / 1e6; // convert to seconds
    double flops = (2.0 * (double)n_embd * (double)n_embd * (double)n_embd * n_repeat) / elapsed_seconds / 1e9; // convert to GFLOPS

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return (float)flops;
}

float device_cpu_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t, int n_threads) {
    return device_flops(model, src0t, src1t, PROFILER_BACKEND_TYPE_CPU, n_threads);
}

float device_metal_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t) {
    float flops = 0.0f;

#ifdef GGML_USE_METAL
    flops = device_flops(model, src0t, src1t, PROFILER_BACKEND_TYPE_METAL, 4);
#endif

    (void)model;
    (void)src0t;
    (void)src1t;
    return flops;
}

float device_cuda_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t) {
    float flops = 0.0f;

#ifdef GGML_USE_CUDA
    flops = device_flops(model, src0t, src1t, PROFILER_BACKEND_TYPE_CUDA, 4);
#endif

    (void)model;
    (void)src0t;
    (void)src1t;
    return flops;
}

float device_inp_embd_delay(struct llama_model * model, enum ggml_type src0t, int n_tokens, int n_threads) {
    const int n_vocab = llama_n_vocab(model);
    const int n_embd  = llama_n_embd(model);
    
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG_INF("%s: ggml backend init failed\n", __func__);
        return 0.0f;
    }

    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * tok_embd   = ggml_new_tensor_2d(ctx, src0t, n_embd, n_vocab);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<int32_t> matrix_A(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        matrix_A[i] = i % n_vocab;
    }

    const size_t embd_size = n_vocab * n_embd;
    void * matrix_B = nullptr;

    // quantization and dequantization functions
    ggml_type_traits_t qfns = ggml_internal_get_type_traits(src0t);
    if (!qfns.from_float || !qfns.to_float) {
        LOG_INF("Unsupported or uninitialized quantization type: %d\n", src0t);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return 0.0f;
    }

    switch (src0t) {
        case GGML_TYPE_F32: {
            matrix_B = malloc(embd_size * sizeof(float));
            float * matrix_B_f32 = static_cast<float *>(matrix_B);
            for (size_t i = 0; i < embd_size; ++i) {
                matrix_B_f32[i] = static_cast<float>(rand() / RAND_MAX);
            }
            break;
        }
        case GGML_TYPE_F16: {
            matrix_B = malloc(embd_size * sizeof(ggml_fp16_t));
            std::vector<float> temp_f32(embd_size);
            for (size_t i = 0; i < embd_size; ++i) {
                temp_f32[i] = static_cast<float>(rand() / RAND_MAX);
            }
            ggml_fp32_to_fp16_row(temp_f32.data(), static_cast<ggml_fp16_t *>(matrix_B), embd_size);
            break;
        }
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ1_M:
            matrix_B = malloc((embd_size / ggml_blck_size(src0t) * ggml_type_size(src0t))); // The quantization block sizes are inconsistent for different quantization methods
            break;
        default:
            LOG_INF("Unsupported type: %d\n", src0t);
            ggml_free(ctx);
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            return 0.0f;
    }

    ggml_backend_tensor_set(inp_tokens, matrix_A.data(), 0, ggml_nbytes(inp_tokens));
    ggml_backend_tensor_set(tok_embd, matrix_B, 0, ggml_nbytes(tok_embd));

    struct ggml_cgraph  * gf         = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 3 + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph(ctx_cgraph);
        struct ggml_tensor * cur = ggml_get_rows(ctx_cgraph, tok_embd, inp_tokens);
        ggml_build_forward_expand(gf, cur);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    // warm-up
    ggml_backend_graph_compute(backend, gf);

    const int64_t t_start = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    const int64_t t_end = ggml_time_us();

    double elapsed_ms = ((double)t_end - (double)t_start) / 1e3; // convert to ms

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return (float)elapsed_ms;
}

static bool device_is_docker_container() {
#if defined(__linux__)
    struct stat buffer;
    if (stat("/.dockerenv", &buffer) == 0) {
        return true;
    }

    std::ifstream cgroup_file("/proc/1/cgroup");
    std::string line;
    while (std::getline(cgroup_file, line)) {
        if (line.find("docker") != std::string::npos || 
            line.find("containerd") != std::string::npos) {
            return true;
        }
    }
    cgroup_file.close();
#endif

    return false;
}

static int is_uma_arch() {
#if defined(__APPLE__) && defined(__MACH__)
    int is_arm64 = 0;
    size_t size = sizeof(is_arm64);

    // check whether it is Apple Silicon (ARM64)
    if (sysctlbyname("hw.optional.arm64", &is_arm64, &size, NULL, 0) != 0) {
        return 0;
    }

    return is_arm64;
#else
    return 0;
#endif
}

static uint64_t device_host_physical_memory(bool available) {
    uint64_t memory = 0;

#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    if (available) {
        memory = status.ullAvailPhys;
    } else {
        memory = status.ullTotalPhys;
    }

#elif defined(__linux__)
    if (available) {
        // read available memory from /proc/meminfo
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        if (meminfo.is_open()) {
            while (std::getline(meminfo, line)) {
                if (line.find("MemAvailable:") == 0) {
                    std::istringstream iss(line);
                    std::string key;
                    uint64_t kb;
                    iss >> key >> kb;
                    memory = kb * 1024 * 0.8;
                    break;
                }
            }
            meminfo.close();
        }
    } else {
        // get total memory using sysinfo
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            memory = info.totalram * info.mem_unit;
        }
    }

#elif defined(__APPLE__) && defined(__MACH__)
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    uint64_t total_memory = 0;
    size_t len = sizeof(total_memory);
    int mib[2] = {CTL_HW, HW_MEMSIZE};

    if (sysctl(mib, 2, &total_memory, &len, NULL, 0) != 0) {
        LOG_INF("sysctl failed\n");
        return 0;
    }

    if (available) {
        if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
            size_t page_size = get_page_size();
            memory = (vm_stats.free_count + vm_stats.inactive_count + vm_stats.purgeable_count) * page_size;

            // active pages compression has higher priority than releasing the clean mmap-ed pages
            // some of the active pages can be compressed to save memory for our mmap-ed model weights
            if (is_uma_arch()) {
                // assume 10% of active pages can be compressed on macOS UMA (an empirical value) 
                // because GPU is more likely to use the inactive memory
                memory += vm_stats.active_count * 0.1 * page_size;
            } else {
                // assume 50% of active pages can be compressed on macOS NUMA (an empirical value)
                memory += vm_stats.active_count * 0.5 * page_size;
            }

            if (!is_uma_arch()) {
                memory += (vm_stats.speculative_count + vm_stats.compressor_page_count) * page_size;
            } else {
// #ifndef GGML_USE_METAL
//                 memory += vm_stats.speculative_count * page_size;
// #endif
            }
        } else {
            LOG_INF("host_statistics64 failed\n");
        }
    } else {
        memory = total_memory;
    }
#endif

    return memory;
}

static uint64_t read_value_from_file(const char * path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }
    std::string line;
    if (!std::getline(file, line)) {
        return 0;
    }
    try {
        return std::stoull(line);
    } catch (...) {
        return 0;
    }
}

static std::unordered_map<std::string, uint64_t> read_memory_stat() {
    std::unordered_map<std::string, uint64_t> stats;
    std::ifstream file("/sys/fs/cgroup/memory.stat");
    if (!file.is_open()) {
        return stats;
    }
    std::string line;
    while (std::getline(file, line)) {
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string key = line.substr(0, space_pos);
            std::string val_str = line.substr(space_pos + 1);
            try {
                uint64_t val = std::stoull(val_str);
                stats[key] = val;
            } catch (...) {
                return stats;
            }
        }
    }
    return stats;
}

static uint64_t device_cgroup_physical_memory(bool available) {
    const char * file_path = nullptr;

    bool is_cgroup_v2 = false;
    {
        std::ifstream cgroup_file("/proc/cgroups");
        if (cgroup_file.is_open()) {
            std::string line;
            while (std::getline(cgroup_file, line)) {
                if (line.find("0") != std::string::npos) {
                    is_cgroup_v2 = true;
                    break;
                }
            }
        }
    }

    if (!available) {
        if (is_cgroup_v2) {
            file_path = "/sys/fs/cgroup/memory.max";
        } else {
            file_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes";
        }
        return read_value_from_file(file_path);
    } else {
        if (is_cgroup_v2) {
            uint64_t mem_max     = read_value_from_file("/sys/fs/cgroup/memory.max");
            uint64_t mem_current = read_value_from_file("/sys/fs/cgroup/memory.current");

            if (mem_max == UINT64_MAX) {
                mem_max = device_host_physical_memory(false);
            }

            uint64_t mem_low = read_value_from_file("/sys/fs/cgroup/memory.low");

            auto stats = read_memory_stat();

            uint64_t slab_reclaimable = 0;
            uint64_t mmap_file        = 0;

            if (stats.find("slab_reclaimable") != stats.end()) {
                slab_reclaimable = stats["slab_reclaimable"];
            }
            if (stats.find("file") != stats.end()) {
                mmap_file = stats["file"];
            }

            uint64_t available_memory = mem_max - mem_current;
            if (mem_low > 0 && available_memory < mem_low) {
                available_memory = mem_low;
            }
            available_memory += slab_reclaimable * 0.5 + mmap_file * 0.5;
            
            return available_memory < mem_max ? available_memory : mem_max;
        } else {
            LOG_WRN("Using cgroup v1, the available memory could be error, will be addressed later\n");
            uint64_t mem_limit = read_value_from_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
            uint64_t mem_usage = read_value_from_file("/sys/fs/cgroup/memory/memory.usage_in_bytes");
            return mem_limit - mem_usage > 0 ? mem_limit - mem_usage : 0;
        }
    }
}

uint64_t device_physical_memory(bool available) {
    if (device_is_docker_container()) {
        return device_cgroup_physical_memory(available);
    } else {
        return device_host_physical_memory(available);
    }
}

static uint64_t device_host_swap_memory(bool available) {
    uint64_t swap_memory = 0;

#if defined(_WIN32) || defined(_WIN64)
    PERFORMANCE_INFORMATION performance_info;
    performance_info.cb = sizeof(performance_info);
    if (GetPerformanceInfo(&performance_info, sizeof(performance_info))) {
        if (available) {
            swap_memory = (performance_info.PageFileTotal - performance_info.PageFileUsage) * performance_info.PageSize;
        } else {
            swap_memory = performance_info.PageFileTotal * performance_info.PageSize;
        }
    }
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    uint64_t total_swap = 0;
    uint64_t free_swap = 0;

    if (meminfo.is_open()) {
        while (std::getline(meminfo, line)) {
            if (line.find("SwapTotal:") == 0) {
                std::istringstream iss(line);
                std::string key;
                uint64_t kb;
                iss >> key >> kb;
                total_swap = kb * 1024;
            } else if (line.find("SwapFree:") == 0) {
                std::istringstream iss(line);
                std::string key;
                uint64_t kb;
                iss >> key >> kb;
                free_swap = kb * 1024;
            }
        }
        meminfo.close();
    }

    if (available) {
        swap_memory = free_swap;
    } else {
        swap_memory = total_swap;
    }

#elif defined(__APPLE__) && defined(__MACH__)
    int mib[2] = {CTL_VM, VM_SWAPUSAGE};
    struct xsw_usage swap;
    size_t len = sizeof(swap);

    if (sysctl(mib, 2, &swap, &len, NULL, 0) == 0) {
        if (available) {
            swap_memory = swap.xsu_avail;
        } else {
            swap_memory = swap.xsu_total;
        }
    }
#endif

    return swap_memory;
}

static uint64_t device_cgroup_swap_memory(bool available) {
    if (available) return 0;

#if defined(__linux__)
    const char * file_path = nullptr;
    uint64_t swap_limit    = 0;

    std::ifstream cgroup_file("/proc/cgroups");
    bool is_cgroup_v2 = false;
    if (cgroup_file.is_open()) {
        std::string line;
        while (std::getline(cgroup_file, line)) {
            if (line.find("0") != std::string::npos) {
                is_cgroup_v2 = true;
                break;
            }
        }
        cgroup_file.close();
    }

    if (is_cgroup_v2) {
        file_path = "/sys/fs/cgroup/memory.swap.max"; 
    } else {
        file_path = "/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes"; 
    }

    std::ifstream mem_swap_file(file_path);
    if (mem_swap_file.is_open()) {
        std::string line;
        if (std::getline(mem_swap_file, line)) {
            try {
                swap_limit = std::stoull(line);
            } catch (const std::exception &e) {
                swap_limit = 0;
            }
        }
        mem_swap_file.close();
    }

    return swap_limit;
#else
    return 0; // Unsupported on non-Linux platforms
#endif
}

uint64_t device_swap_memory(bool available) {
    if (device_is_docker_container()) {
        return device_cgroup_swap_memory(available);
    } else {
        return device_host_swap_memory(available);
    }
}

static std::string get_default_device_path() {
#ifdef __linux__
    // find the first block device under /sys/block
    const std::string block_path = "/sys/block/";
    DIR * dir = opendir(block_path.c_str());
    if (!dir) {
        LOG_INF("Unable to open %s\n", block_path.c_str());
        return "";
    }
    struct dirent * entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] != '.') { // ignore hidden files/directories
            std::string device = entry->d_name;
            closedir(dir);
            return "/dev/" + device;
        }
    }
    closedir(dir);
    LOG_INF("No block devices found in %s\n", block_path.c_str());
    return "";
#elif __APPLE__
    // use the root device as a default
    return "/";
#elif _WIN32
    // use the default drive (usually C:)
    char volume_name[MAX_PATH];
    if (GetVolumeInformation("C:\\", volume_name, sizeof(volume_name), NULL, NULL, NULL, NULL, 0)) {
        return "C:\\";
    } else {
        LOG_INF("Failed to determine default volume\n");
        return "";
    }
#else
    LOG_INF("Unsupported platform\n");
    return "";
#endif
}

static size_t get_default_readahead_size() {
    const std::string device_path = get_default_device_path();

#ifdef __linux__
    std::string device = device_path.empty() ? get_default_device_path() : device_path;
    if (device.empty()) return 0;

    // read from sysfs
    std::string sysfs_path = "/sys/block/" + device.substr(device.find_last_of("/") + 1) + "/queue/read_ahead_kb";
    std::ifstream file(sysfs_path);
    if (file.is_open()) {
        size_t read_ahead_kb;
        file >> read_ahead_kb;
        file.close();
        return read_ahead_kb * 1024; // convert to bytes
    } else {
        return 0;
    }
#elif __APPLE__
    // use statfs to determine default block size
    struct statfs stats;
    std::string path = device_path.empty() ? "/" : device_path;
    if (statfs(path.c_str(), &stats) == 0) {
        return stats.f_iosize; // return in bytes
    } else {
        LOG_INF("statfs failed\n");
        return 0;
    }
#elif _WIN32
    // use GetDiskFreeSpace to get default cluster size
    std::string drive = device_path.empty() ? "C:\\" : device_path;
    DWORD sectorsPerCluster, bytesPerSector, numberOfFreeClusters, totalNumberOfClusters;
    if (GetDiskFreeSpace(drive.c_str(), &sectorsPerCluster, &bytesPerSector, &numberOfFreeClusters, &totalNumberOfClusters)) {
        return sectorsPerCluster * bytesPerSector; // return in bytes
    } else {
        LOG_INF("GetDiskFreeSpace failed\n");
        return 0;
    }
#else
    LOG_INF("Unsupported platform\n");
    return 0;
#endif
}

static std::vector<std::string> split(const std::string & str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

static bool path_exist_in_env(const std::string & path, const std::string & env_path) {
    auto paths = split(env_path, ':');
    return std::find(paths.begin(), paths.end(), path) != paths.end();
}

static bool path_exist_in_fs(const std::string & path) {
    struct stat info;
    return (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR));
}

static void check_env_path() {
    const char * cur_env_path   = std::getenv("PATH");
    std::string update_env_path = cur_env_path ? cur_env_path : "";
    std::vector<std::string> paths_to_check = {"/opt/homebrew/bin", "/usr/local/bin"};

    for (const auto & path : paths_to_check) {
        if (!path_exist_in_env(path, update_env_path) && path_exist_in_fs(path)) {
            if (!update_env_path.empty() && update_env_path.back() != ':') {
                update_env_path += ':';
            }
            update_env_path += path;
            LOG_INF("add missing path: %s, current env path: %s\n", path.c_str(), update_env_path.c_str());
        }
    }

    setenv("PATH", update_env_path.c_str(), 1);
}

static void external_fio_impl(float * read_bw, float * write_bw, bool op_rand, int n_threads) {
    pid_t pid = getpid(); // avoid conflict with other processes

    std::string test_file   = "fio_test_"   + std::to_string(pid);
    std::string output_file = "fio_output_" + std::to_string(pid) + ".log";
    std::string conf_file   = "config_"     + std::to_string(pid) + ".fio";

    const char * fio_conf_template = R"(
[global]
ioengine=%s
direct=1
time_based=1
runtime=1
size=1G
group_reporting=1
iodepth=1

[write-job]
rw=%s
bs=%s
filename=%s
numjobs=%d

[read-job]
startdelay=1.5
rw=%s
bs=%s
filename=%s
numjobs=%d
)";

    size_t page_size = get_page_size();
    if (page_size == 0) {
        LOG_INF("Unable to get system page size, use 4KB by default\n");
        page_size = 4 * 1024;
    }
    // format the page size as a readable string (e.g., "16k" or "4k")
    char page_size_str[8];
    if (page_size >= 1024) {
        snprintf(page_size_str, sizeof(page_size_str), "%zuk", page_size / 1024);
    } else {
        snprintf(page_size_str, sizeof(page_size_str), "%zu", page_size);
    }

    size_t readahead_size = get_default_readahead_size();
    if (readahead_size == 0) {
        LOG_INF("Unable to get system readahead size, use 128KB by default\n");
        readahead_size = 128 * 1024;
    }
    // format the readahead size as a readable string (e.g., "128k" or "1m")
    char readahead_str[8];
    if (readahead_size >= 1024 * 1024) {
        snprintf(readahead_str, sizeof(readahead_str), "%zuM", readahead_size / 1024 / 1024);
    } else if (readahead_size >= 1024) {
        snprintf(readahead_str, sizeof(readahead_str), "%zuk", readahead_size / 1024);
    } else {
        snprintf(readahead_str, sizeof(readahead_str), "%zu",  readahead_size);
    }

    const char * read_type  = op_rand ? "randread" : "read";
    const char * write_type = op_rand ? "randwrite" : "write";
    const char * block_size = op_rand ? page_size_str : readahead_str;
    const char * ioengine   = "posixaio";

    check_env_path(); // ensure the fio bin file can be found

    int num_try = 0;
    int ret;
    while (num_try < 2) {
        char fio_conf[1024];
        snprintf(fio_conf, sizeof(fio_conf), fio_conf_template, ioengine,
                 read_type,  block_size, test_file.c_str(), n_threads,
                 write_type, block_size, test_file.c_str(), n_threads);
        
        std::ofstream conf(conf_file.c_str());
        if (!conf) {
            LOG_INF("Error: Unable to create configuration file\n");
            return;
        }
        conf << fio_conf;
        conf.close();

        std::string command = "fio " + conf_file + " > " + output_file + " 2>&1";
        ret = std::system(command.c_str());

        num_try += 1;

        if (ret != 0) {
            LOG_WRN("Engine posixaio not loadable, retrying with sync engine\n");
            ioengine = "sync";
        } else {
            num_try = 2;
        }
    }

    if (ret != 0) {
        throw std::runtime_error("Engine posixaio and sync not loadable, fio test failed\n");
    }

    // parse fio output
    std::ifstream result(output_file.c_str());
    if (!result) {
        LOG_INF("Error: Failed to open fio output file\n");
        return;
    }
    *read_bw = 0.0f;
    *write_bw = 0.0f;
    
    std::string line;
    std::regex read_regex(R"(READ: bw=([0-9.]+)([a-zA-Z/]+))");
    std::regex write_regex(R"(WRITE: bw=([0-9.]+)([a-zA-Z/]+))");
    std::smatch match;

    while (std::getline(result, line)) {
        if (std::regex_search(line, match, read_regex)) {
            float value = std::stof(match[1]);
            std::string unit = match[2];
            if (unit == "MiB/s") {
                *read_bw = value * 1024.0f * 1024.0f / 1e9;  // convert MiB/s to GB/s
            } else if (unit == "MB/s") {
                *read_bw = value / 1000.0f;  // convert MB/s to GB/s
            }
        } else if (std::regex_search(line, match, write_regex)) {
            float value = std::stof(match[1]);
            std::string unit = match[2];
            if (unit == "MiB/s") {
                *write_bw = value * 1024.0f * 1024.0f / 1e9;  // convert MiB/s to GB/s
            } else if (unit == "MB/s") {
                *write_bw = value / 1000.0f;  // convert MB/s to GB/s
            }
        }
    }

    // clean up temporary files
    std::remove(test_file.c_str());
    std::remove(conf_file.c_str());
    std::remove(output_file.c_str());
}

void device_disk_rnd_bw(float * read_rnd_bw, float * write_rnd_bw, int n_threads) {
    external_fio_impl(read_rnd_bw, write_rnd_bw, true, n_threads);
}

void device_disk_seq_bw(float * read_seq_bw, float * write_seq_bw, int n_threads) {
    external_fio_impl(read_seq_bw, write_seq_bw, false, n_threads);
}

float device_memory_bw(int n_thread) {
    // simulate large model weights, set to 100 MiB
    size_t buffer_size = 100L * 1024 * 1024;
    std::vector<char> data(buffer_size);
    std::fill(data.begin(), data.end(), 1); // initialize data to avoid lazy loading

    std::vector<double> results(n_thread);

    // memory bandwidth test function
    auto memory_bw_test = [](char * data, size_t total_size, size_t block_size, double & result) {
        size_t n_iters = total_size / block_size; 
        volatile char temp = 0; // volatile to prevent compiler optimization
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < n_iters; i++) {
            // simulate block-wise sequential access
            size_t offset = i * block_size;
            for (size_t j = 0; j < block_size; j += 64) {
                temp += data[offset + j];
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        result = total_size / elapsed.count() / 1e9; // GB/s

        (void)temp;
    };

    std::vector<std::thread> thread_pool;
    for (int i = 0; i < n_thread; ++i) {
        thread_pool.emplace_back(
            memory_bw_test, 
            data.data(), 
            buffer_size / n_thread, 
            MEM_TEST_BLOCK_SIZE, 
            std::ref(results[i])
        );
    }

    for (auto & t : thread_pool) {
        t.join();
    }

    double bandwidth = std::accumulate(results.begin(), results.end(), 0.0);
    return static_cast<float>(bandwidth);
}

static float device_read_vram_bw(enum profiler_backend_type btype) {
    const int n_embd = 8192;
    std::vector<float> matrix_A(n_embd * n_embd, 1.0f);

    ggml_backend_t backend = NULL;
    switch (btype) {
        case PROFILER_BACKEND_TYPE_METAL:
#ifdef GGML_USE_METAL
            backend = ggml_backend_metal_init();
#endif
            break;
        case PROFILER_BACKEND_TYPE_CUDA:
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
#endif
            break;
        case PROFILER_BACKEND_TYPE_CPU:
            break;
    }

    if (!backend) {
        LOG_INF("%s: ggml backend init failed\n", __func__);
        return 0.0f;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
    tensor_a->op = GGML_OP_READ;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(tensor_a, matrix_A.data(), 0, ggml_nbytes(tensor_a));

    struct ggml_cgraph  * gf         = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph(ctx_cgraph);
        ggml_build_forward_expand(gf, tensor_a);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    const int64_t t_start = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    const int64_t t_end = ggml_time_us();

    double elapsed_s = ((double)t_end - (double)t_start) / 1e6;
    size_t total_bytes = n_embd * n_embd * sizeof(float);
    float bandwidth = (total_bytes / elapsed_s) / 1e9; // GB/s

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return bandwidth;
}

float device_metal_read_vram_bw() {
    float bw = 0.0f;

#ifdef GGML_USE_METAL
    bw = device_read_vram_bw(PROFILER_BACKEND_TYPE_METAL);
#endif

    return bw;
}

float device_cuda_read_vram_bw() {
    float bw = 0.0f;

#ifdef GGML_USE_CUDA
    bw = device_read_vram_bw(PROFILER_BACKEND_TYPE_CUDA);
#endif

    return bw;
}

// return ggml_cpy delay in kvcache in ms
static float device_mem_copy(struct llama_model * model, enum profiler_backend_type btype, int n_threads) {
    const int64_t n_embd_k_gqa  = llama_model_n_embd_k_gqa(model);
    const int64_t n_embd_v_gqa  = llama_model_n_embd_v_gqa(model);

    std::vector<float> src_mat_k(n_embd_k_gqa, 1.0f); 
    std::vector<float> src_mat_v(n_embd_v_gqa, 1.0f); 
    std::vector<float> dst_mat_k(n_embd_k_gqa, 0.0f);
    std::vector<float> dst_mat_v(n_embd_v_gqa, 0.0f);

    ggml_backend_t backend = NULL;
    switch (btype) {
        case PROFILER_BACKEND_TYPE_CPU:
            backend = ggml_backend_cpu_init();
            break;
        case PROFILER_BACKEND_TYPE_METAL:
#ifdef GGML_USE_METAL
            backend = ggml_backend_metal_init();
#endif
            break;
        case PROFILER_BACKEND_TYPE_CUDA:
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
#endif
            break;
    }

    if (!backend) {
        LOG_INF("%s: ggml backend init failed\n", __func__);
        return 0.0f;
    }

    size_t ctx_size = 0;
    ctx_size += 4 * ggml_tensor_overhead(); // tensors

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src_tensor_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd_k_gqa);
    struct ggml_tensor * src_tensor_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd_v_gqa);
    struct ggml_tensor * dst_tensor_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd_k_gqa);
    struct ggml_tensor * dst_tensor_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd_v_gqa);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    ggml_backend_tensor_set(src_tensor_k, src_mat_k.data(), 0, ggml_nbytes(src_tensor_k));
    ggml_backend_tensor_set(src_tensor_v, src_mat_v.data(), 0, ggml_nbytes(src_tensor_v));
    ggml_backend_tensor_set(dst_tensor_k, dst_mat_k.data(), 0, ggml_nbytes(dst_tensor_k));
    ggml_backend_tensor_set(dst_tensor_v, dst_mat_v.data(), 0, ggml_nbytes(dst_tensor_v));

    struct ggml_cgraph  * gf         = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph(ctx_cgraph);
        ggml_build_forward_expand(gf, ggml_cpy(ctx_cgraph, src_tensor_k, dst_tensor_k));
        ggml_build_forward_expand(gf, ggml_cpy(ctx_cgraph, src_tensor_v, dst_tensor_v));
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    // warm-up 
    ggml_backend_graph_compute(backend, gf);

    const int64_t t_start = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    const int64_t t_end = ggml_time_us();

    double elapsed_ms = ((double)t_end - (double)t_start) / 1e3; // ms

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return (float)elapsed_ms;
}

float device_cpu_mem_copy(struct llama_model * model, int n_threads) {
    return device_mem_copy(model, PROFILER_BACKEND_TYPE_CPU, n_threads);
}

float device_metal_mem_copy(struct llama_model * model) {
    float delay = 0.0f;

#ifdef GGML_USE_METAL
    delay = device_mem_copy(model, PROFILER_BACKEND_TYPE_METAL, 4);
#endif

    (void)model;
    return delay;
}

float device_cuda_mem_copy(struct llama_model * model) {
    float delay = 0.0f;

#ifdef GGML_USE_CUDA
    delay = device_mem_copy(model, PROFILER_BACKEND_TYPE_CUDA, 4);
#endif

    (void)model;
    return delay;
}

int device_has_metal(void) {
    return ggml_cpu_has_metal();
}

int device_has_cuda(void) {
    return ggml_cpu_has_cuda();
}

int device_has_vulkan(void) {
    return ggml_cpu_has_vulkan();
}

int device_has_kompute(void) {
    return ggml_cpu_has_kompute();
}

int device_has_gpublas(void) {
    return ggml_cpu_has_gpublas();
}

int device_has_blas(void) {
    return ggml_cpu_has_blas();
}

int device_has_sycl(void) {
    return ggml_cpu_has_sycl();
}

void device_get_props(struct llama_model * model, int device, struct ggml_backend_dev_props * props) {
    ggml_backend_buffer_type_t buft_type;
    if (device == -1) { // type cpu
        buft_type = ggml_backend_cpu_buffer_type();
    } else { // type gpu
        buft_type = llama_dev_buffer_type(model, device);
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft_type);
    ggml_backend_dev_get_props(dev, props);
}

static float device_compute_delay(struct device_info & dev_info, int n_layers, const struct llama_context_params cparams) {
    struct model_flops n_flops   = dev_info.model_flops;
    struct cpu_props cpu         = dev_info.cpu_props;
    int n_gpu_layers             = std::min(static_cast<int>(cparams.n_gpu_layers), n_layers);

    double gpu_latency_per_layer = 0.0f;
    double cpu_latency_per_layer = 0.0f;

#ifdef GGML_USE_CUDA
    struct gpu_props gpu = dev_info.gpu_props;

    gpu_latency_per_layer += (double)n_flops.layer_f32_f32    / ((double)gpu.cuda_flops_f32_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_f16_f32    / ((double)gpu.cuda_flops_f16_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q2k_f32    / ((double)gpu.cuda_flops_q2k_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q4k_f32    / ((double)gpu.cuda_flops_q4k_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q5k_f32    / ((double)gpu.cuda_flops_q5k_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q6k_f32    / ((double)gpu.cuda_flops_q6k_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_iq2xxs_f32 / ((double)gpu.cuda_flops_iq2xxs_f32 + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q50_f32    / ((double)gpu.cuda_flops_q50_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q80_f32    / ((double)gpu.cuda_flops_q80_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_iq1s_f32   / ((double)gpu.cuda_flops_iq1s_f32   + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_iq4nl_f32  / ((double)gpu.cuda_flops_iq4nl_f32  + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_iq1m_f32   / ((double)gpu.cuda_flops_iq1m_f32   + EPS) / 1e9;
#elif GGML_USE_METAL
    struct gpu_props gpu = dev_info.gpu_props;

    gpu_latency_per_layer += (double)n_flops.layer_f32_f32    / ((double)gpu.metal_flops_f32_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_f16_f32    / ((double)gpu.metal_flops_f16_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q2k_f32    / ((double)gpu.metal_flops_q2k_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q4k_f32    / ((double)gpu.metal_flops_q4k_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q5k_f32    / ((double)gpu.metal_flops_q5k_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q6k_f32    / ((double)gpu.metal_flops_q6k_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_iq2xxs_f32 / ((double)gpu.metal_flops_iq2xxs_f32 + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q50_f32    / ((double)gpu.metal_flops_q50_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q80_f32    / ((double)gpu.metal_flops_q80_f32    + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_iq1s_f32   / ((double)gpu.metal_flops_iq1s_f32   + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_iq4nl_f32  / ((double)gpu.metal_flops_iq4nl_f32  + EPS) / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_iq1m_f32   / ((double)gpu.metal_flops_iq1m_f32   + EPS) / 1e9;
#endif

    cpu_latency_per_layer += (double)n_flops.layer_f32_f32    / ((double)cpu.flops_f32_f32    + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_f16_f32    / ((double)cpu.flops_f16_f32    + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q2k_f32    / ((double)cpu.flops_q2k_f32    + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q4k_f32    / ((double)cpu.flops_q4k_f32    + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q5k_f32    / ((double)cpu.flops_q5k_f32    + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q6k_f32    / ((double)cpu.flops_q6k_f32    + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_iq2xxs_f32 / ((double)cpu.flops_iq2xxs_f32 + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q50_f32    / ((double)cpu.flops_q50_f32    + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q80_f32    / ((double)cpu.flops_q80_f32    + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_iq1s_f32   / ((double)cpu.flops_iq1s_f32   + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_iq4nl_f32  / ((double)cpu.flops_iq4nl_f32  + EPS) / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_iq1m_f32   / ((double)cpu.flops_iq1m_f32   + EPS) / 1e9;
    double total_latency = 0.0f;
    
#if defined(GGML_USE_METAL) || defined(GGML_USE_CUDA)
    total_latency += gpu_latency_per_layer * n_gpu_layers;
    total_latency += cpu_latency_per_layer * (n_layers - n_gpu_layers);
#else
    (void)n_gpu_layers;
    (void)gpu_latency_per_layer;
    total_latency += cpu_latency_per_layer * n_layers;
#endif

    total_latency += (double)n_flops.output_f32_f32    / ((double)cpu.flops_f32_f32    + EPS) / 1e9;
    total_latency += (double)n_flops.output_f16_f32    / ((double)cpu.flops_f16_f32    + EPS) / 1e9;
    total_latency += (double)n_flops.output_q2k_f32    / ((double)cpu.flops_q2k_f32    + EPS) / 1e9;
    total_latency += (double)n_flops.output_q4k_f32    / ((double)cpu.flops_q4k_f32    + EPS) / 1e9;
    total_latency += (double)n_flops.output_q5k_f32    / ((double)cpu.flops_q5k_f32    + EPS) / 1e9;
    total_latency += (double)n_flops.output_q6k_f32    / ((double)cpu.flops_q6k_f32    + EPS) / 1e9;
    total_latency += (double)n_flops.output_iq2xxs_f32 / ((double)cpu.flops_iq2xxs_f32 + EPS) / 1e9;
    total_latency += (double)n_flops.output_q50_f32    / ((double)cpu.flops_q50_f32    + EPS) / 1e9;
    total_latency += (double)n_flops.output_q80_f32    / ((double)cpu.flops_q80_f32    + EPS) / 1e9;
    total_latency += (double)n_flops.output_iq1s_f32   / ((double)cpu.flops_iq1s_f32   + EPS) / 1e9;
    total_latency += (double)n_flops.output_iq4nl_f32  / ((double)cpu.flops_iq4nl_f32  + EPS) / 1e9;
    total_latency += (double)n_flops.output_iq1m_f32   / ((double)cpu.flops_iq1m_f32   + EPS) / 1e9;

    total_latency *= 1000; // convert to ms

    total_latency += n_flops.inp_embd_ms;

    return static_cast<float>(total_latency);
}

// estimate the memory access delay, except for the input embedding because it has been considered in n_flops.inp_embd_ms
static float device_memory_access_delay(struct device_info & dev_info, struct llama_model * model, const struct llama_context_params cparams, int n_layers) {
    auto n_bytes     = dev_info.model_bytes;
    int n_gpu_layers = std::min(static_cast<int>(cparams.n_gpu_layers), n_layers);
    
    int64_t cpu_kv_size;
    int64_t gpu_kv_size;

#if defined(GGML_USE_METAL) || defined(GGML_USE_CUDA)
    llama_kv_size(&cpu_kv_size, &gpu_kv_size, model, cparams, true);

    int64_t vram_bytes = n_bytes.nb_layer * n_gpu_layers + gpu_kv_size;
    int64_t ram_bytes  = n_bytes.nb_layer * (n_layers - n_gpu_layers) + n_bytes.nb_output + cpu_kv_size;

#ifdef GGML_USE_CUDA
    double vram_access_delay = (double)(vram_bytes) / 1e6 / dev_info.gpu_props.cuda_read_vram_bw;
#elif GGML_USE_METAL
    double vram_access_delay = (double)(vram_bytes) / 1e6 / dev_info.gpu_props.metal_read_vram_bw;
#endif

    double ram_access_delay  = (double)(ram_bytes)  / 1e6 / dev_info.memory.cpu_read_ram_bw;
    return static_cast<float>(vram_access_delay + ram_access_delay); // ms

#else
    llama_kv_size(&cpu_kv_size, &gpu_kv_size, model, cparams, false);

    (void)n_gpu_layers;
    (void)gpu_kv_size;
    int64_t ram_bytes = n_bytes.nb_layer * n_layers + n_bytes.nb_output + cpu_kv_size;
    double ram_access_delay = (double)(ram_bytes) / 1e6 / dev_info.memory.cpu_read_ram_bw;
    return static_cast<float>(ram_access_delay); // ms
#endif
}

static uint64_t device_termux_swappable_memory() {
    if (access("/data/data/com.termux/files/usr/bin", F_OK) != 0) {
        LOG_ERR("Not in a Termux environment\n");
        return 0;
    }

    uint64_t total_swappable = 0;
    uint64_t active_anon     = 0; 
    uint64_t inactive_anon   = 0;

    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    if (meminfo.is_open()) {
        while (std::getline(meminfo, line)) {
            if (line.find("Active(anon):") == 0) {
                sscanf(line.c_str(), "Active(anon): %" SCNu64 " kB", &active_anon);
            } else if (line.find("Inactive(anon):") == 0) {
                sscanf(line.c_str(), "Inactive(anon): %" SCNu64 " kB", &inactive_anon);
            }
        }
        meminfo.close();
    }

    total_swappable = (active_anon + inactive_anon) * 1024;

    DIR * proc_dir = opendir("/proc");
    if (proc_dir) {
        struct dirent * entry;
        while ((entry = readdir(proc_dir)) != nullptr) {
            if (!isdigit(entry->d_name[0])) continue;

            std::string smaps_path = "/proc/" + std::string(entry->d_name) + "/smaps";
            std::ifstream smaps_file(smaps_path);
            if (!smaps_file.is_open()) {
                LOG_WRN("Failed to open smaps file: %s\n", smaps_path.c_str());
                continue;
            }

            uint64_t locked_pages = 0;
            while (std::getline(smaps_file, line)) {
                if (line.find("Locked:") == 0) {
                    uint64_t kb;
                    sscanf(line.c_str(), "Locked: %" SCNu64 " kB", &kb);
                    locked_pages += kb * 1024;
                }
            }
            smaps_file.close();

            // Subtract locked pages from swappable memory
            total_swappable -= locked_pages;
        }
        closedir(proc_dir);
    }

    return total_swappable;
}

static uint64_t device_macos_swappable_memory() {
#if defined(__APPLE__) && defined(__MACH__)
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stats;
    kern_return_t kr = host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stats, &count);
    if (kr != KERN_SUCCESS) {
        LOG_INF("Failed to get VM statistics\n");
        return 0;
    }
    return vm_stats.internal_page_count * get_page_size();

#else
    return 0;

#endif
}

uint64_t device_swappable_memory() {
#if defined(__APPLE__) && defined(__MACH__)
    return device_macos_swappable_memory();
#endif

    if (access("/data/data/com.termux/files/usr/bin", F_OK) == 0) {
        return device_termux_swappable_memory();
    }
    return 0;
}

static float device_disk_access_delay(struct device_info & dev_info, struct llama_model * model, const struct llama_context_params cparams) {
    auto n_bytes     = dev_info.model_bytes;
    int n_layers     = llama_model_n_layers(model);
    int n_gpu_layers = std::min(static_cast<int>(cparams.n_gpu_layers), n_layers);
    int n_vocab      = llama_n_vocab(model);

    int64_t cpu_total_bytes = 0;
    int64_t input_bytes = n_bytes.nb_input / n_vocab; // lookup table, retrieve only n_embd elements
    cpu_total_bytes += input_bytes;

#if defined(GGML_USE_METAL) || defined(GGML_USE_CUDA)
    cpu_total_bytes += n_bytes.nb_layer * (n_layers - n_gpu_layers);
#if defined(GGML_USE_METAL)
    int64_t gpu_total_bytes = n_bytes.nb_layer * n_gpu_layers;
#endif
#else
    (void)n_gpu_layers;
    cpu_total_bytes += n_bytes.nb_layer * n_layers;
#endif

    cpu_total_bytes += n_bytes.nb_output;

    int64_t cpu_kv_size;
    int64_t gpu_kv_size;
    int64_t cpu_compute_buf;
    int64_t gpu_compute_buf;

#if defined(GGML_USE_METAL) || defined(GGML_USE_CUDA)
    llama_kv_size(&cpu_kv_size, &gpu_kv_size, model, cparams, true);
    llama_model_compute_buf_size(&cpu_compute_buf, &gpu_compute_buf, model, cparams, true,  true, n_layers, n_gpu_layers);
#else
    llama_kv_size(&cpu_kv_size, &gpu_kv_size, model, cparams, false);
    llama_model_compute_buf_size(&cpu_compute_buf, &gpu_compute_buf, model, cparams, false, true, n_layers, n_gpu_layers);
#endif

    double cpu_kv_size_gib     = static_cast<double>(cpu_kv_size) / 1024.0 / 1024.0 / 1024.0;     // convert to GiB
    double gpu_kv_size_gib     = static_cast<double>(gpu_kv_size) / 1024.0 / 1024.0 / 1024.0;     // convert to GiB
    double cpu_compute_buf_gib = static_cast<double>(cpu_compute_buf) / 1024.0 / 1024.0 / 1024.0; // convert to GiB
    double gpu_compute_buf_gib = static_cast<double>(gpu_compute_buf) / 1024.0 / 1024.0 / 1024.0; // convert to GiB

#if defined(GGML_USE_METAL)
    if (n_gpu_layers > 0) {
        double total_bytes_gib       = static_cast<double>(cpu_total_bytes + gpu_total_bytes) / 1024.0 / 1024.0 / 1024.0;
        double total_kv_size_gib     = cpu_kv_size_gib + gpu_kv_size_gib;
        double total_compute_buf_gib = cpu_compute_buf_gib + gpu_compute_buf_gib;
        double total_mem_needed      = total_bytes_gib + total_kv_size_gib + total_compute_buf_gib;
        float  disk_read_bw          = dev_info.disk.read_rnd_bw;

        if (total_mem_needed < dev_info.memory.total_physical - 1) { // -1 is an empirical value reserved by system processes
            // each time one new row of lookup table will be loaded
            return static_cast<double>(input_bytes) / 1e6 / disk_read_bw; // convert to ms
        } else {
            // warn: OOM error may occur if -ngl is set large
            // inactive pages are swapped out or compressed to free memory for Metal
            // mmap pages are not locked so they will be released when memory is busy
            return total_bytes_gib * 1024.0 * 1024.0 * 1024.0 / 1e6 / disk_read_bw; // ms
        }
    }
#endif

    (void)gpu_kv_size_gib;
    (void)gpu_compute_buf_gib;

    float cpu_total_bytes_gib = (double)cpu_total_bytes / 1024.0 / 1024.0 / 1024.0; // convert to GiB
    float cpu_mem_avail       = dev_info.memory.available_physical; // GiB
    float total_mem_needed    = cpu_total_bytes_gib + cpu_kv_size_gib + cpu_compute_buf_gib;
    
    // non-linux os uses random read bandwidth
    float disk_read_bw = dev_info.disk.read_rnd_bw * 1e9 / 1024.0 / 1024.0 / 1024.0; // convert GB/s to GiB/s

    if (total_mem_needed > cpu_mem_avail) {

#if defined(__APPLE__) && defined(__MACH__)
        // if physical memory reaches busy, all mapped tensors should be re-loaded
        return cpu_total_bytes_gib / disk_read_bw * 1000;  // convert to ms
#else

#if defined(__linux__)
        if (getenv("TERMUX_VERSION") != NULL) {
            // Android will forcibly reserve some physical memory, usually 128 MiB
            dev_info.memory.available_physical -= 0.128;
            // termux on android: swap has higher priority than releasing mmap
            // non-app memory that can be swapped to disk
            float swapout_gib = std::min(
                std::max(0.0f, total_mem_needed - dev_info.memory.available_physical),
                std::min(dev_info.memory.used_can_swap, dev_info.memory.available_swap)
            );
            float mmapin_gib = total_mem_needed - (dev_info.memory.available_physical + swapout_gib);
            return mmapin_gib / disk_read_bw * 1000; // ms
        } else {
            // if this linux not in termux env, use sequantial read bandwidth
            // POSIX_FADV_SEQUENTIAL is set on linux
            disk_read_bw = dev_info.disk.read_seq_bw * 1e9 / 1024.0 / 1024.0 / 1024.0; 
        }
#endif

        // only part of the mapped tensors needs to be re-loaded
        float gbytes_to_load = cpu_total_bytes_gib - (cpu_mem_avail - cpu_kv_size_gib - cpu_compute_buf_gib);
        return gbytes_to_load / disk_read_bw * 1000;  // convert to ms
#endif

    } else {
        // if physical memory is enough, all mapped tensors can be stored in memory and will not be released
        return 0.0f;
    }
}

static float device_mem_copy_delay(struct device_info & dev_info, struct llama_model * model, const struct llama_context_params cparams) {
    int n_layers     = llama_model_n_layers(model);
    int n_gpu_layers = std::min(static_cast<int>(cparams.n_gpu_layers), n_layers);

    float layer_delay_cpu   = dev_info.memory.mem_cpy_delay;

#ifdef GGML_USE_METAL
    float layer_delay_metal = dev_info.gpu_props.metal_mem_cpy_delay;
    return layer_delay_metal * n_gpu_layers + layer_delay_cpu * (n_layers - n_gpu_layers);
#elif GGML_USE_CUDA
    float layer_delay_cuda  = dev_info.gpu_props.cuda_mem_cpy_delay;
    return layer_delay_cuda * n_gpu_layers + layer_delay_cpu * (n_layers - n_gpu_layers);
#else
    (void)n_gpu_layers;
    return layer_delay_cpu * n_layers;
#endif
}

void device_print_props(struct device_info * dev_info_set, int n, struct llama_model * model, const struct llama_context_params cparams) {
    LOG_INF("\n-------------------------------------------------------------------------------------------\n");
    LOG_INF("| Property                        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| Rank %-8d", i);
        GGML_ASSERT((int)dev_info_set[i].rank == i);
    }
    LOG_INF("\n-------------------------------------------------------------------------------------------\n");

    LOG_INF("| Device Name                     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].device_name);
    }
    LOG_INF("\n");

    LOG_INF("| Device OS                       ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].device_os);
    }
    LOG_INF("\n");

    LOG_INF("| CPU Name                        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].cpu_props.name);
    }
    LOG_INF("\n");

    LOG_INF("| CPU Description                 ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].cpu_props.description);
    }
    LOG_INF("\n");

    LOG_INF("| Number of CPU cores             ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10u   ", dev_info_set[i].cpu_props.cores);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (F32xF32, GFLOPS)     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_f32_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (F16xF32, GFLOPS)     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_f16_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q2K x F32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q2k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q4K x F32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q4k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q5K x F32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q5k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q6K x F32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q6k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (IQ2XXS x F32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_iq2xxs_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q50 x F32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q50_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q80 x F32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q80_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (IQ1S x F32, GFLOPS)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_iq1s_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (IQ4NL x F32, GFLOPS) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_iq4nl_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (IQ1M x F32, GFLOPS)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_iq1m_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Physical Mem Total (GiB)        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.total_physical);
    }
    LOG_INF("\n");

    LOG_INF("| Physical Mem Available (GiB)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.available_physical);
    }
    LOG_INF("\n");

    LOG_INF("| Used Mem Swappable (GiB)        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.used_can_swap);
    }
    LOG_INF("\n");

    LOG_INF("| Swap Mem Total (GiB)            ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.total_swap);
    }
    LOG_INF("\n");

    LOG_INF("| Swap Mem Available (GiB)        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.available_swap);
    }
    LOG_INF("\n");

    LOG_INF("| CPU RAM Read BW (GB/s)          ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.cpu_read_ram_bw);
    }
    LOG_INF("\n");

    LOG_INF("| CPU KVCache Copy Time (ms/l)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.mem_cpy_delay);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Read Seq Speed (GB/s)      ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk.read_seq_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Write Seq Speed (GB/s)     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk.write_seq_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Read Rnd Speed (GB/s)      ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk.read_rnd_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Write Rnd Speed (GB/s)     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk.write_rnd_bw);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Metal                       ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.metal);
    }
    LOG_INF("\n");

    LOG_INF("| GPU CUDA                        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.cuda);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Vulkan                      ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.vulkan);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Kompute                     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.kompute);
    }
    LOG_INF("\n");

    LOG_INF("| GPU BLAS                        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.gpublas);
    }
    LOG_INF("\n");

    LOG_INF("| BLAS                            ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.blas);
    }
    LOG_INF("\n");

    LOG_INF("| SYCL                            ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.sycl);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Name                        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].gpu_props.name);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Description                 ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].gpu_props.description);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Mem Free (GiB)              ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.memory_free);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Mem Total (GiB)             ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.memory_total);
    }
    LOG_INF("\n");

    LOG_INF("| Metal VRAM Read BW (GB/s)       ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.metal_read_vram_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Metal KVCache Copy Time(ms/l)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.metal_mem_cpy_delay);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (F32xF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_f32_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (F16xF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_f16_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q2KxF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q2k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q4KxF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q4k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q5KxF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q5k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q6KxF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q6k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (IQ2XXSxF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_iq2xxs_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q50xF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q50_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q80xF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q80_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (IQ1SxF32, GFLOPS)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_iq1s_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (IQ4NLxF32, GFLOPS) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_iq4nl_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (IQ1MxF32, GFLOPS)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_iq1m_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA VRAM Read BW (GB/s)        ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.cuda_read_vram_bw);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA KVCache Copy Time (ms/l)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.cuda_mem_cpy_delay);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (F32xF32, GFLOPS)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_f32_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (F16xF32, GFLOPS)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_f16_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q2KxF32, GFLOPS)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q2k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q4KxF32, GFLOPS)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q4k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q5KxF32, GFLOPS)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q5k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q6KxF32, GFLOPS)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q6k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (IQ2XXSxF32, GFLOPS) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_iq2xxs_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q50xF32, GFLOPS)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q50_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q80xF32, GFLOPS)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q80_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (IQ1SxF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_iq1s_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (IQ4NLxF32, GFLOPS)  ");  
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_iq4nl_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (IQ1MxF32, GFLOPS)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_iq1m_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Model flops (output F32xF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_f32_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output F16xF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_f16_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output Q2KxF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q2k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output Q4KxF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q4k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output Q5KxF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q5k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output Q6KxF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q6k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output IQ2XXSxF32) ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_iq2xxs_f32);
    LOG_INF("\n");
    
    LOG_INF("| Model flops (output Q50xF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q50_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output Q80xF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q80_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output IQ1SxF32)   ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_iq1s_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output IQ4NLxF32)  ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_iq4nl_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output IQ1MxF32)   ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_iq1m_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer F32xF32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_f32_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer F16xF32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_f16_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q2KxF32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q2k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q4KxF32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q4k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q5KxF32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q5k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q6KxF32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q6k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer IQ2XXSxF32)  ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_iq2xxs_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q50xF32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q50_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q80xF32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q80_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer IQ1SxF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_iq1s_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer IQ4NLxF32)   ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_iq4nl_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer IQ1MxF32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_iq1m_f32);
    LOG_INF("\n");

    LOG_INF("| Model params (input F32)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_f32);
    LOG_INF("\n");

    LOG_INF("| Model params (input F16)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_f16);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q2K)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q2k);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q4K)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q4k);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q5K)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q5k);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q6K)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q6k);
    LOG_INF("\n");

    LOG_INF("| Model params (input IQ2XXS)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_iq2xxs);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q50)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q50);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q80)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q80);
    LOG_INF("\n");

    LOG_INF("| Model params (input IQ1S)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_iq1s);
    LOG_INF("\n");

    LOG_INF("| Model params (input IQ4NL)      ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_iq4nl);
    LOG_INF("\n");

    LOG_INF("| Model params (input IQ1M)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_iq1m);
    LOG_INF("\n");

    LOG_INF("| Model params (layer F32)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_f32);
    LOG_INF("\n");

    LOG_INF("| Model params (layer F16)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_f16);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q2K)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q2k);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q4K)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q4k);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q5K)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q5k);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q6K)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q6k);
    LOG_INF("\n");

    LOG_INF("| Model params (layer IQ2XXS)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_iq2xxs);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q50)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q50);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q80)        ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q80);
    LOG_INF("\n");

    LOG_INF("| Model params (layer IQ1S)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_iq1s);
    LOG_INF("\n");

    LOG_INF("| Model params (layer IQ4NL)      ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_iq4nl);
    LOG_INF("\n");        
    
    LOG_INF("| Model params (layer IQ1M)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_iq1m);
    LOG_INF("\n");

    LOG_INF("| Model params (output F32)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_f32);
    LOG_INF("\n");

    LOG_INF("| Model params (output F16)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_f16);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q2K)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q2k);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q4K)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q4k);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q5K)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q5k);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q6K)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q6k);
    LOG_INF("\n");

    LOG_INF("| Model params (output IQ2XXS)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_iq2xxs);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q50)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q50);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q80)       ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q80);
    LOG_INF("\n");

    LOG_INF("| Model params (output IQ1S)      ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_iq1s);
    LOG_INF("\n");    

    LOG_INF("| Model params (output IQ4NL)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_iq4nl);
    LOG_INF("\n");  

    LOG_INF("| Model params (output IQ1M)      ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_iq1m);
    LOG_INF("\n");

    LOG_INF("| Model bytes  (input)            ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_bytes.nb_input);
    LOG_INF("\n");

    LOG_INF("| Model bytes  (layer)            ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_bytes.nb_layer);
    LOG_INF("\n");

    LOG_INF("| Model bytes  (output)           ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_bytes.nb_output);
    LOG_INF("\n");

    // float latency = 0.0f;
    // int n_layers  = llama_model_n_layers (model);
    // latency += device_compute_delay      (dev_info_set[0], n_layers, cparams);
    // latency += device_memory_access_delay(dev_info_set[0], model, cparams, n_layers);
    // latency += device_disk_access_delay  (dev_info_set[0], model, cparams); // if physical memory is not enough, some mapped data will be released and reloaded later
    // latency += device_mem_copy_delay     (dev_info_set[0], model, cparams); // memory copy delay in kvcache

    // LOG_INF("| Token latency (ms)           ");
    // LOG_INF("| %-10.2f   ", latency);
    // LOG_INF("\n");

    (void)model;
    (void)cparams;

    LOG_INF("-------------------------------------------------------------------------------------------\n\n");
}


size_t serialize(const struct device_info * dev_info, char ** buffer) {
    // calculate total size for serialized buffer
    size_t device_name_len     = strlen(dev_info->device_name) + 1;
    size_t device_os_len       = strlen(dev_info->device_os) + 1;
    size_t next_ip_len         = strlen(dev_info->next_ip) + 1;
    size_t cpu_name_len        = strlen(dev_info->cpu_props.name) + 1;
    size_t cpu_description_len = strlen(dev_info->cpu_props.description) + 1;
    size_t gpu_name_len        = strlen(dev_info->gpu_props.name) + 1;
    size_t gpu_description_len = strlen(dev_info->gpu_props.description) + 1;

    size_t total_size = sizeof(uint32_t)
                      + sizeof(size_t) * 7  // for lengths of strings
                      + device_name_len
                      + device_os_len
                      + next_ip_len
                      + cpu_name_len
                      + cpu_description_len
                      + gpu_name_len
                      + gpu_description_len
                      + sizeof(struct disk_props)
                      + sizeof(uint32_t)    // cpu_props.cores
                      + sizeof(float) * 12  // - cpu_props.flops_f32_f32,   cpu_props.flops_f16_f32,
                                            // - cpu_props.flops_q2k_f32,   cpu_props.flops_q4k_f32, cpu_props.flops_q5k_f32, cpu_props.flops_q6k_f32
                                            // - cpu_props.flops_iq2xxs_f32
                                            // - cpu_props.flops_q50_f32,   cpu_props.flops_q80_f32
                                            // - cpu_props.flops_iq1s_f32,  cpu_props.flops_iq4nl_f32
                                            // - cpu_props.flops_iq1m_f32
                      + sizeof(struct memory_info)
                      + sizeof(struct gpu_support)
                      + sizeof(float) * 30;     // GPU attributes
                                                // memory:
                                                // - memory_free,           memory_total
                                                // - metal_read_vram_bw,    cuda_read_vram_bw
                                                // Metal floating-point performance:
                                                // - metal_flops_f32_f32,   metal_flops_f16_f32
                                                // - metal_flops_q2k_f32,   metal_flops_q4k_f32, metal_flops_q5k_f32, metal_flops_q6k_f32
                                                // - metal_flops_iq2xxs_f32
                                                // - metal_flops_q50_f32,   metal_flops_q80_f32
                                                // - metal_flops_iq1s_f32,  metal_flops_iq4nl_f32
                                                // - metal_flops_iq1m_f32
                                                // CUDA floating-point performance:
                                                // - cuda_flops_f32_f32,    cuda_flops_f16_f32
                                                // - cuda_flops_q2k_f32,    cuda_flops_q4k_f32, cuda_flops_q5k_f32, cuda_flops_q6k_f32
                                                // - cuda_flops_iq2xxs_f32
                                                // - cuda_flops_q50_f32,    cuda_flops_q80_f32
                                                // - cuda_flops_iq1s_f32,   cuda_flops_iq4nl_f32
                                                // - cuda_flops_iq1m_f32
                                                // delay:
                                                // - metal_mem_cpy_delay,   cuda_mem_cpy_delay

    *buffer = (char *)malloc(total_size);
    char * ptr = *buffer;

    if (*buffer == NULL) {
        LOG_ERR("%s: failed to allocate %zu bytes for device info serialization\n", 
                __func__, total_size);
        return 0;
    }

    // rank
    memcpy(ptr, &dev_info->rank, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // copy string lengths and string data
    memcpy(ptr, &device_name_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->device_name, device_name_len);
    ptr += device_name_len;

    memcpy(ptr, &device_os_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->device_os, device_os_len);
    ptr += device_os_len;

    memcpy(ptr, &next_ip_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->next_ip, next_ip_len);
    ptr += next_ip_len;

    memcpy(ptr, &cpu_name_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->cpu_props.name, cpu_name_len);
    ptr += cpu_name_len;

    memcpy(ptr, &cpu_description_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->cpu_props.description, cpu_description_len);
    ptr += cpu_description_len;

    memcpy(ptr, &gpu_name_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->gpu_props.name, gpu_name_len);
    ptr += gpu_name_len;

    memcpy(ptr, &gpu_description_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->gpu_props.description, gpu_description_len);
    ptr += gpu_description_len;

    // copy the non-string members
    memcpy(ptr, &dev_info->disk, sizeof(struct disk_props));
    ptr += sizeof(struct disk_props);

    memcpy(ptr, &dev_info->cpu_props.cores, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    memcpy(ptr, &dev_info->cpu_props.flops_f32_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_f16_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q2k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q5k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q6k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_iq2xxs_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q50_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q80_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_iq1s_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_iq4nl_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_iq1m_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->memory, sizeof(struct memory_info));
    ptr += sizeof(struct memory_info);

    memcpy(ptr, &dev_info->gpu_support, sizeof(struct gpu_support));
    ptr += sizeof(struct gpu_support);

    memcpy(ptr, &dev_info->gpu_props.memory_free, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.memory_total, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_read_vram_bw, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_f32_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_f16_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q2k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q5k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q6k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_iq2xxs_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q50_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q80_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_iq1s_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_iq4nl_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_iq1m_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_mem_cpy_delay, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_read_vram_bw, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_f32_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_f16_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q2k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q5k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q6k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_iq2xxs_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q50_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q80_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_iq1s_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_iq4nl_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_iq1m_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_mem_cpy_delay, sizeof(float));

    // no need to synchronize model flops and model params
    return total_size;
}

void deserialize(const char * buffer, struct device_info * dev_info) {
    const char * ptr = buffer;

    // rank
    memcpy(&dev_info->rank, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // device_name
    size_t device_name_len;
    memcpy(&device_name_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->device_name = (char *)malloc(device_name_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->device_name)), ptr, device_name_len);
    ptr += device_name_len;

    // device_os
    size_t device_os_len;
    memcpy(&device_os_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->device_os = (char *)malloc(device_os_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->device_os)), ptr, device_os_len);
    ptr += device_os_len;

    // next ip
    size_t next_ip_len;
    memcpy(&next_ip_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->next_ip = (char *)malloc(next_ip_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->next_ip)), ptr, next_ip_len);
    ptr += next_ip_len;

    // cpu_props.name
    size_t cpu_name_len;
    memcpy(&cpu_name_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->cpu_props.name = (char *)malloc(cpu_name_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->cpu_props.name)), ptr, cpu_name_len);
    ptr += cpu_name_len;

    // cpu_props.description
    size_t cpu_description_len;
    memcpy(&cpu_description_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->cpu_props.description = (char *)malloc(cpu_description_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->cpu_props.description)), ptr, cpu_description_len);
    ptr += cpu_description_len;

    // gpu_props.name
    size_t gpu_name_len;
    memcpy(&gpu_name_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->gpu_props.name = (char *)malloc(gpu_name_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->gpu_props.name)), ptr, gpu_name_len);
    ptr += gpu_name_len;

    // gpu_props.description
    size_t gpu_description_len;
    memcpy(&gpu_description_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->gpu_props.description = (char *)malloc(gpu_description_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->gpu_props.description)), ptr, gpu_description_len);
    ptr += gpu_description_len;

    // other non-string members
    memcpy(&dev_info->disk, ptr, sizeof(struct disk_props));
    ptr += sizeof(struct disk_props);

    memcpy(&dev_info->cpu_props.cores, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    memcpy(&dev_info->cpu_props.flops_f32_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_f16_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q2k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q5k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q6k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_iq2xxs_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q50_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q80_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_iq1s_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_iq4nl_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_iq1m_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->memory, ptr, sizeof(struct memory_info));
    ptr += sizeof(struct memory_info);

    memcpy(&dev_info->gpu_support, ptr, sizeof(struct gpu_support));
    ptr += sizeof(struct gpu_support);

    memcpy(&dev_info->gpu_props.memory_free, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.memory_total, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_read_vram_bw, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_f32_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_f16_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q2k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q5k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q6k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_iq2xxs_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q50_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q80_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_iq1s_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_iq4nl_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_iq1m_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_mem_cpy_delay, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_read_vram_bw, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_f32_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_f16_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q2k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q5k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q6k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_iq2xxs_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q50_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q80_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_iq1s_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_iq4nl_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_iq1m_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_mem_cpy_delay, ptr, sizeof(float));

    // no need to synchronize model flops and model params
}
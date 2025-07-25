// Various helper functions and utilities

#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <sstream>

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

#define print_build_info() do {                                                                     \
    fprintf(stderr, "%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);      \
    fprintf(stderr, "%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);    \
} while(0)

#define DEFAULT_MODEL_PATH "models/7B/ggml-model-f16.gguf"

struct llama_lora_adapter_info {
    std::string path;
    float scale;
};

struct llama_lora_adapter_container : llama_lora_adapter_info {
    struct llama_lora_adapter * adapter;
};

using llama_tokens = std::vector<llama_token>;

// build info
extern int LLAMA_BUILD_NUMBER;
extern char const * LLAMA_COMMIT;
extern char const * LLAMA_COMPILER;
extern char const * LLAMA_BUILD_TARGET;

struct llama_control_vector_load_info;

//
// CPU utils
//

struct cpu_params {
    int      n_threads                   = -1;
    bool     cpumask[GGML_MAX_N_THREADS] = {false}; // CPU affinity mask.
    bool     mask_valid                  = false;   // Default: any CPU
    enum ggml_sched_priority  priority   = GGML_SCHED_PRIO_NORMAL;  // Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
    bool     strict_cpu                  = false;   // Use strict CPU placement
    uint32_t poll                        = 50;      // Polling (busywait) level (0 - no polling, 100 - mostly polling)
};

int32_t cpu_get_num_physical_cores();
int32_t cpu_get_num_math();

//
// Common params
//

enum llama_example {
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_MAIN,
    LLAMA_EXAMPLE_INFILL,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_LLAVA,
    LLAMA_EXAMPLE_LOOKUP,
    LLAMA_EXAMPLE_PARALLEL,

    LLAMA_EXAMPLE_COUNT,
};

enum gpt_sampler_type {
    GPT_SAMPLER_TYPE_NONE        = 0,
    GPT_SAMPLER_TYPE_TOP_K       = 1,
    GPT_SAMPLER_TYPE_TOP_P       = 2,
    GPT_SAMPLER_TYPE_MIN_P       = 3,
    GPT_SAMPLER_TYPE_TFS_Z       = 4,
    GPT_SAMPLER_TYPE_TYPICAL_P   = 5,
    GPT_SAMPLER_TYPE_TEMPERATURE = 6,
};

// dimensionality reduction methods, used by cvector-generator
enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

// sampler parameters
struct gpt_sampler_params {
    uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

    int32_t n_prev            = 64;    // number of previous tokens to remember
    int32_t n_probs           = 0;     // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep          = 0;     // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   min_p             = 0.05f; // 0.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typ_p             = 1.00f; // typical_p, 1.0 = disabled
    float   temp              = 0.80f; // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range    = 0.00f; // 0.0 = disabled
    float   dynatemp_exponent = 1.00f; // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n    = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat    = 1.00f; // 1.0 = disabled
    float   penalty_freq      = 0.00f; // 0.0 = disabled
    float   penalty_present   = 0.00f; // 0.0 = disabled
    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate
    bool    penalize_nl       = false; // consider newlines as a repeatable token
    bool    ignore_eos        = false;
    bool    no_perf           = false; // disable performance metrics

    std::vector<enum gpt_sampler_type> samplers = {
        GPT_SAMPLER_TYPE_TOP_K,
        GPT_SAMPLER_TYPE_TFS_Z,
        GPT_SAMPLER_TYPE_TYPICAL_P,
        GPT_SAMPLER_TYPE_TOP_P,
        GPT_SAMPLER_TYPE_MIN_P,
        GPT_SAMPLER_TYPE_TEMPERATURE
    };

    std::string grammar; // optional BNF-like grammar to constrain sampling

    std::vector<llama_logit_bias> logit_bias; // logit biases to apply

    // print the parameters into a string
    std::string print() const;
};

struct common_params_speculative {
    int32_t n_ctx        =     0; // draft context size
    int32_t n_max        =    16; // maximum number of tokens to draft during speculative decoding
    int32_t n_min        =     5; // minimum number of draft tokens to use for speculative decoding
    int32_t n_gpu_layers =    -1; // number of layers to store in VRAM for the draft model (-1 - use default)
    float   p_split      =  0.1f; // speculative decoding split probability
    float   p_min        =  0.9f; // minimum speculative decoding probability (greedy)

    struct cpu_params cpuparams;
    struct cpu_params cpuparams_batch;

    std::string model = ""; // draft model for speculative decoding                          // NOLINT
};

struct gpt_params {
    int32_t n_world               =     1; // number of devices to use
    int32_t rank                  =     0; // my rank for distributed inference
    uint32_t n_layer_window[32]   =   {0}; // layer window size on each node
    std::string master_ip         = "127.0.0.1"; // ip address of the master node
    std::string next_node_ip      = "127.0.0.1"; // ip address of my next node
    uint32_t data_port            =  9000;  // data port for distributed inference
    uint32_t signal_port          =  10000; // signal port for distributed inference
    bool    prefetch              = false; // prefetch layer weights
    bool    keep_out_in_metal     =  true; // whether to keep output weights in metal memory, true by default
    bool    keep_out_in_cuda      = false; // whether to run the output layer on CUDA, false by default
    bool    force                 = false; // force to start prefetching after computation
    float   master_priority       =  1.01; // priority to assign workload to the master (set 1.01 to use master first, and 0.99 to offload to other devices)
    int32_t gpu_mem               = 999.0; // gpu memory to use, in GiB
    int32_t n_cycles              =     0; // number of cycles to output one token
    int32_t n_predict             =    -1; // new tokens to predict
    int32_t n_ctx                 =     0; // context size
    int32_t n_batch               =  2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_ubatch              =   512; // physical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                =     0; // number of tokens to keep from initial prompt
    int32_t n_chunks              =    -1; // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel            =     1; // number of parallel sequences to decode
    int32_t n_sequences           =     1; // number of sequences to decode
    float   p_split               =  0.1f; // speculative decoding split probability
    int32_t n_gpu_layers          =     0; // number of layers to store in VRAM (0 - do not use by default)
    int32_t n_gpu_layers_draft    =    -1; // number of layers to store in VRAM for the draft model (-1 - use default)
    int32_t main_gpu              =     0; // the GPU that is used for scratch and small tensors
    float   tensor_split[128]     =   {0}; // how split tensors should be distributed across GPUs
    int32_t grp_attn_n            =     1; // group-attention factor
    int32_t grp_attn_w            =   512; // group-attention width
    int32_t n_print               =    -1; // print token count every n tokens (-1 = disabled)
    float   rope_freq_base        =  0.0f; // RoPE base frequency
    float   rope_freq_scale       =  0.0f; // RoPE frequency scaling factor
    float   yarn_ext_factor       = -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor      =  1.0f; // YaRN magnitude scaling factor
    float   yarn_beta_fast        = 32.0f; // YaRN low correction dim
    float   yarn_beta_slow        =  1.0f; // YaRN high correction dim
    int32_t yarn_orig_ctx         =     0; // YaRN original context length
    float   defrag_thold          = -1.0f; // KV cache defragmentation threshold

    struct cpu_params cpuparams;
    struct cpu_params cpuparams_batch;
    struct cpu_params draft_cpuparams;
    struct cpu_params draft_cpuparams_batch;

    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data                 = nullptr;

    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    enum llama_split_mode        split_mode        = LLAMA_SPLIT_MODE_LAYER; // how to split the model across GPUs
    enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    enum llama_pooling_type      pooling_type      = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling type for embeddings
    enum llama_attention_type    attention_type    = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // attention type for embeddings

    struct gpt_sampler_params sparams;
    struct common_params_speculative speculative;

    std::string model                = ""; // model path                                                    // NOLINT
    std::string model_alias          = "unknown"; // model alias                                            // NOLINT
    std::string model_url            = ""; // model url to download                                         // NOLINT
    std::string hf_token             = ""; // HF token                                                      // NOLINT
    std::string hf_repo              = ""; // HF repo                                                       // NOLINT
    std::string hf_file              = ""; // HF file                                                       // NOLINT
    std::string prompt               = "";                                                                  // NOLINT
    std::string prompt_file          = ""; // store the external prompt file name                           // NOLINT
    std::string path_prompt_cache    = ""; // path to file for saving/loading prompt eval state             // NOLINT
    std::string input_prefix         = ""; // string to prefix user inputs with                             // NOLINT
    std::string input_suffix         = ""; // string to suffix user inputs with                             // NOLINT
    std::string logdir               = ""; // directory in which to save YAML log files                     // NOLINT
    std::string lookup_cache_static  = ""; // path of static ngram cache file for lookup decoding           // NOLINT
    std::string lookup_cache_dynamic = ""; // path of dynamic ngram cache file for lookup decoding          // NOLINT
    std::string logits_file          = ""; // file for saving *all* logits                                  // NOLINT
    std::string rpc_servers          = ""; // comma separated list of RPC servers                           // NOLINT

    std::vector<std::string> in_files;   // all input files
    std::vector<std::string> antiprompt; // strings upon which more user input is prompted (a.k.a. reverse prompts)
    std::vector<llama_model_kv_override> kv_overrides;

    bool lora_init_without_apply = false; // only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply)
    std::vector<llama_lora_adapter_info> lora_adapters; // lora adapter path with user defined scale

    std::vector<llama_control_vector_load_info> control_vectors; // control vector with user defined scale

    int32_t verbosity                  = 0;
    int32_t control_vector_layer_start = -1; // layer range for control vector
    int32_t control_vector_layer_end   = -1; // layer range for control vector

    int32_t ppl_stride      = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int32_t ppl_output_type = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                     //                                       (which is more convenient to use for plotting)
                                     //
    bool   hellaswag        = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks  = 400;   // number of tasks to use when computing the HellaSwag score

    bool   winogrande       = false; // compute Winogrande score over random tasks from datafile supplied in prompt
    size_t winogrande_tasks = 0;     // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice  = false;  // compute TruthfulQA score over random tasks from datafile supplied in prompt
    size_t multiple_choice_tasks = 0; // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool kl_divergence     = false; // compute KL divergence
    bool usage             = false; // print usage
    bool use_color         = false; // use color to distinguish generations and inputs
    bool special           = false; // enable special token output
    bool interactive       = false; // interactive mode
    bool interactive_first = false; // wait for user input immediately
    bool conversation      = false; // conversation mode (does not print special tokens and suffix/prefix)
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool escape            = true;  // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = true;  // insert new sequences for decoding on-the-fly
    bool flash_attn        = false; // flash attention
    bool no_perf           = false; // disable performance metrics
    bool ctx_shift         = true;  // context shift on inifinite text generation

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool logits_all        = false; // return logits for all tokens in the batch
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool display_prompt    = true;  // print prompt before generation
    bool dump_kv_cache     = false; // dump the KV cache contents for debugging purposes
    bool no_kv_offload     = false; // disable KV offloading
    bool warmup            = true;  // warmup run
    bool check_tensors     = false; // validate tensor data

    std::string cache_type_k = "f16"; // KV cache data type for the K
    std::string cache_type_v = "f16"; // KV cache data type for the V

    // multimodal models (see examples/llava)
    std::string mmproj = "";        // path to multimodal projector                                         // NOLINT
    std::vector<std::string> image; // path to image file(s)

    // embedding
    bool embedding         = false; // get only sentence embedding
    int32_t embd_normalize = 2;     // normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
    std::string embd_out   = "";    // empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
    std::string embd_sep   = "\n";  // separator of embendings
    bool reranking         = false; // enable reranking support on server

    // server params
    int32_t port           = 8080;         // server listens on this network port
    int32_t timeout_read   = 600;          // http read timeout in seconds
    int32_t timeout_write  = timeout_read; // http write timeout in seconds
    int     n_threads_http = -1;           // number of threads to process HTTP requests (TODO: support threadpool)

    std::string hostname      = "127.0.0.1";
    std::string public_path   = "";                                                                         // NOLINT
    std::string chat_template = "";                                                                         // NOLINT
    std::string system_prompt = "";                                                                         // NOLINT
    bool enable_chat_template = true;

    std::vector<std::string> api_keys;

    std::string ssl_file_key  = "";                                                                         // NOLINT
    std::string ssl_file_cert = "";                                                                         // NOLINT

    bool endpoint_slots   = true;
    bool endpoint_metrics = false;

    bool log_json = false;

    std::string slot_save_path;

    float slot_prompt_similarity = 0.5f;

    // batched-bench params
    bool is_pp_shared = false;

    std::vector<int32_t> n_pp;
    std::vector<int32_t> n_tg;
    std::vector<int32_t> n_pl;

    // retrieval params
    std::vector<std::string> context_files; // context files to embed

    int32_t chunk_size = 64; // chunk size for context embedding

    std::string chunk_separator = "\n"; // chunk separator for context embedding

    // passkey params
    int32_t n_junk = 250; // number of times to repeat the junk text
    int32_t i_pos  = -1;  // position of the passkey in the junk text

    // imatrix params
    std::string out_file = "imatrix.dat"; // save the resulting imatrix to this file

    int32_t n_out_freq  = 10; // output the imatrix every n_out_freq iterations
    int32_t n_save_freq =  0; // save the imatrix every n_save_freq iterations
    int32_t i_chunk     =  0; // start processing from this chunk

    bool process_output = false; // collect data for the output tensor
    bool compute_ppl    = true;  // whether to compute perplexity

    // cvector-generator params
    int n_pca_batch = 100;
    int n_pca_iterations = 1000;
    dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    std::string cvector_outfile       = "control_vector.gguf";
    std::string cvector_positive_file = "examples/cvector-generator/positive.txt";
    std::string cvector_negative_file = "examples/cvector-generator/negative.txt";

    bool spm_infill = false; // suffix/prefix/middle pattern for infill

    std::string lora_outfile = "ggml-lora-merged-f16.gguf";

    // batched-bench params
    bool batched_bench_output_jsonl = false;
};

// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
void gpt_init();

std::string gpt_params_get_system_info(const gpt_params & params);

bool parse_cpu_range(const std::string& range, bool(&boolmask)[GGML_MAX_N_THREADS]);
bool parse_cpu_mask(const std::string& mask, bool(&boolmask)[GGML_MAX_N_THREADS]);
void postprocess_cpu_params(cpu_params& cpuparams, const cpu_params* role_model = nullptr);
bool set_process_priority(enum ggml_sched_priority prio);

//
// String utils
//

std::vector<std::string> string_split(std::string input, char separator);

std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();

void string_replace_all(std::string & s, const std::string & search, const std::string & replace);

template<class T>
static std::vector<T> string_split(const std::string & str, char delim) {
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides);
void string_process_escapes(std::string & input);

std::string string_from(bool value);
std::string string_from(const std::vector<int> & values);
std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens);
std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch);

//
// Filesystem utils
//

bool fs_validate_filename(const std::string & filename);
bool fs_create_directory_with_parents(const std::string & path);

std::string fs_get_cache_directory();
std::string fs_get_cache_file(const std::string & filename);

//
// Model utils
//

struct llama_init_result {
    struct llama_model   * model   = nullptr;
    struct llama_context * context = nullptr;
    std::vector<llama_lora_adapter_container> lora_adapters;
};

struct llama_init_result    llama_init_from_gpt_params(gpt_params & params);

struct llama_model_params     llama_model_params_from_gpt_params    (const gpt_params & params);
struct llama_context_params   llama_context_params_from_gpt_params  (const gpt_params & params);
struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const cpu_params & params);

struct llama_model * llama_load_model_from_url(const char * model_url, const char * path_model, const char * hf_token, const struct llama_model_params & params);
struct llama_model * llama_load_model_from_hf(const char * repo, const char * file, const char * path_model, const char * hf_token, const struct llama_model_params & params);

// clear LoRA adapters from context, then apply new list of adapters
void llama_lora_adapters_apply(struct llama_context * ctx, std::vector<llama_lora_adapter_container> & lora_adapters);

// Batch utils

void llama_batch_clear(struct llama_batch & batch);

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string llama_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token,
                       bool          special = true);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// optionally renders special/control tokens
std::string llama_detokenize(
                         llama_context * ctx,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

//
// Chat template utils
//

// same with llama_chat_message, but uses std::string
struct llama_chat_msg {
    std::string role;
    std::string content;
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool llama_chat_verify_template(const std::string & tmpl);

// CPP wrapper for llama_chat_apply_template
// If the built-in template is not supported, we default to chatml
// If the custom "tmpl" is not supported, we throw an error
std::string llama_chat_apply_template(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & chat,
        bool add_ass);

// Format single message, while taking into account the position of that message in chat history
std::string llama_chat_format_single(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & past_msg,
        const llama_chat_msg & new_msg,
        bool add_ass);

// Returns an example of formatted chat
std::string llama_chat_format_example(const struct llama_model * model,
        const std::string & tmpl);

//
// KV cache utils
//

// Dump the KV cache view with the number of sequences per cell.
void llama_kv_cache_dump_view(const llama_kv_cache_view & view, int row_size = 80);

// Dump the KV cache view showing individual sequences in each cell (long output).
void llama_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size = 40);

//
// Embedding utils
//

void llama_embd_normalize(const float * inp, float * out, int n, int embd_norm = 2);

float llama_embd_similarity_cos(const float * embd1, const float * embd2, int n);

//
// Control vector utils
//

struct llama_control_vector_data {
    int n_embd;

    // stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
    std::vector<float> data;
};

struct llama_control_vector_load_info {
    float strength;

    std::string fname;
};

// Load control vectors, scale each by strength, and add them together.
// On error, returns {-1, empty}
llama_control_vector_data llama_control_vector_load(const std::vector<llama_control_vector_load_info> & load_infos);

//
// Split utils
//

static const char * const LLM_KV_SPLIT_NO            = "split.no";
static const char * const LLM_KV_SPLIT_COUNT         = "split.count";
static const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count";

//
// YAML utils
//

void yaml_dump_vector_float    (FILE * stream, const char * prop_name, const std::vector<float> & data);
void yaml_dump_vector_int      (FILE * stream, const char * prop_name, const std::vector<int> & data);
void yaml_dump_string_multiline(FILE * stream, const char * prop_name, const char * data);

void yaml_dump_non_result_info(
    FILE * stream, const gpt_params & params, const llama_context * lctx,
    const std::string & timestamp, const std::vector<int> & prompt_tokens, const char * model_desc);

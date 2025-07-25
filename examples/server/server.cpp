#include "utils.hpp"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "speculative.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

// auto generated files (update with ./deps.sh)
#include "colorthemes.css.hpp"
#include "style.css.hpp"
#include "theme-beeninorder.css.hpp"
#include "theme-ketivah.css.hpp"
#include "theme-mangotango.css.hpp"
#include "theme-playground.css.hpp"
#include "theme-polarnight.css.hpp"
#include "theme-snowstorm.css.hpp"
#include "index.html.hpp"
#include "index-new.html.hpp"
#include "index.js.hpp"
#include "completion.js.hpp"
#include "system-prompts.js.hpp"
#include "prompt-formats.js.hpp"
#include "json-schema-to-grammar.mjs.hpp"
#include "loading.html.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cinttypes>
#include <deque>
#include <memory>
#include <mutex>
#include <signal.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

using json = nlohmann::ordered_json;

enum stop_type {
    STOP_TYPE_FULL,
    STOP_TYPE_PARTIAL,
};

// state diagram: https://github.com/ggerganov/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
};

enum server_task_cmpl_type {
    SERVER_TASK_CMPL_TYPE_NORMAL,
    SERVER_TASK_CMPL_TYPE_EMBEDDING,
    SERVER_TASK_CMPL_TYPE_RERANK,
    SERVER_TASK_CMPL_TYPE_INFILL,
};

struct server_task {
    int id        = -1; // to be filled by server_queue
    int id_target = -1; // used by SERVER_TASK_TYPE_CANCEL

    server_task_type type;
    json data;

    server_task_cmpl_type cmpl_type = SERVER_TASK_CMPL_TYPE_NORMAL;

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
        }
        return ids;
    }
};

struct server_task_result {
    int id = -1;

    json data;

    bool stop;
    bool error;
};

struct slot_params {
    bool stream       = true;
    bool cache_prompt = true; // remember the prompt to avoid reprocessing all prompt

    int32_t  n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t  n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t  n_predict = -1; // new tokens to predict

    std::vector<std::string> antiprompt;
    
    struct gpt_sampler_params        sampling;
    struct common_params_speculative speculative;

    json input_prefix;
    json input_suffix;
};

struct server_slot {
    int id;
    int id_task = -1;

    llama_batch batch_spec;

    llama_context * ctx_dft = nullptr;

    common_speculative * spec = nullptr;

    // the index relative to completion multi-task request
    size_t index = 0;

    struct slot_params params;

    slot_state state = SLOT_STATE_IDLE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1; // TODO: disambiguate from params.n_predict

    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;

    json prompt; // can be either a string, array of strings or array of token ids

    // when a task is submitted, we first tokenize the prompt and store it here
    std::vector<llama_token> prompt_tokens;

    std::string generated_text;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    server_task_cmpl_type cmpl_type = SERVER_TASK_CMPL_TYPE_NORMAL;

    bool has_next_token = true;
    bool truncated      = false;
    bool stopped_eos    = false;
    bool stopped_word   = false;
    bool stopped_limit  = false;

    bool oaicompat = false;

    std::string oaicompat_model;
    std::string stopping_word;

    // sampling
    json json_schema;

    struct gpt_sampler_params sparams;
    struct gpt_sampler * smpl = nullptr;

    llama_token sampled;

    int32_t ga_i = 0;   // group-attention state
    int32_t ga_n = 1;   // group-attention factor
    int32_t ga_w = 512; // group-attention width

    int32_t n_past_se = 0; // self-extend

    // stats
    size_t n_sent_text = 0; // number of sent text character
    size_t n_sent_token_probs = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation; // ms

    std::function<void(int)> callback_on_release;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens    = 0;
        generated_text     = "";
        truncated          = false;
        stopped_eos        = false;
        stopped_word       = false;
        stopped_limit      = false;
        stopping_word      = "";
        n_past             = 0;
        n_sent_text        = 0;
        n_sent_token_probs = 0;
        cmpl_type          = SERVER_TASK_CMPL_TYPE_NORMAL;
        ga_i               = 0;
        n_past_se          = 0;

        generated_token_probs.clear();
    }

    bool has_budget(const gpt_params &global_params) {
        if (params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1) {
            n_remaining = params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return ctx_dft && params.speculative.n_max > 0 && params.cache_prompt;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }
        generated_token_probs.push_back(token);
    }

    void release() {
        if (is_processing()) {
            SLT_INF(*this, "stop processing: n_past = %d, truncated = %d\n", n_past, truncated);

            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;
            state = SLOT_STATE_IDLE;
            callback_on_release(id);
        }
    }

    json get_formated_timings() const {
        return json {
            {"prompt_n",               n_prompt_tokens_processed},
            {"prompt_ms",              t_prompt_processing},
            {"prompt_per_token_ms",    t_prompt_processing / n_prompt_tokens_processed},
            {"prompt_per_second",      1e3 / t_prompt_processing * n_prompt_tokens_processed},

            {"predicted_n",            n_decoded},
            {"predicted_ms",           t_token_generation},
            {"predicted_per_token_ms", t_token_generation / n_decoded},
            {"predicted_per_second",   1e3 / t_token_generation * n_decoded},
        };
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, const stop_type type) {
        size_t stop_pos = std::string::npos;

        for (const std::string & word : params.antiprompt) {
            size_t pos;

            if (type == STOP_TYPE_FULL) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                pos = find_partial_stop_string(word, text);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (type == STOP_TYPE_FULL) {
                    stopped_word   = true;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "\n"
                "\rprompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "\r       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "\r      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);
    }
};

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

struct server_queue {
    int id = 0;
    bool running;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task&)> callback_new_task;
    std::function<void(void)>         callback_update_slots;

    // Add a new task to the end of the queue
    int post(server_task task, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (task.id == -1) {
            task.id = id++;
        }
        QUE_DBG("new task, id = %d, front = %d\n", task.id, front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
        condition_tasks.notify_one();
        return task.id;
    }

    // multi-task version of post()
    int post(std::vector<server_task> & tasks, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        for (auto & task : tasks) {
            if (task.id == -1) {
                task.id = id++;
            }
            QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
            if (front) {
                queue_tasks.push_front(std::move(task));
            } else {
                queue_tasks.push_back(std::move(task));
            }
        }
        condition_tasks.notify_one();
        return 0;
    }

    // Add a new task, but defer until one slot is available
    void defer(server_task task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        QUE_DBG("defer task, id = %d\n", task.id);
        queue_tasks_deferred.push_back(std::move(task));
        condition_tasks.notify_one();
    }

    // Get the next id for creating a new task
    int get_new_id() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        int new_id = id++;
        return new_id;
    }

    // Register function to process a new task
    void on_new_task(std::function<void(server_task &)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    void pop_deferred_task() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (!queue_tasks_deferred.empty()) {
            queue_tasks.emplace_back(std::move(queue_tasks_deferred.front()));
            queue_tasks_deferred.pop_front();
        }
        condition_tasks.notify_one();
    }

    // end the start_loop routine
    void terminate() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        running = false;
        condition_tasks.notify_all();
    }

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     */
    void start_loop() {
        running = true;

        while (true) {
            QUE_DBG("%s", "processing new tasks\n");

            while (true) {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    lock.unlock();
                    break;
                }
                server_task task = queue_tasks.front();
                queue_tasks.pop_front();
                lock.unlock();

                QUE_DBG("processing task, id = %d\n", task.id);
                callback_new_task(task);
            }

            // all tasks in the current loop is processed, slots data is now ready
            QUE_DBG("%s", "update slots\n");

            callback_update_slots();

            QUE_DBG("%s", "waiting for new tasks\n");
            {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    if (!running) {
                        QUE_DBG("%s", "terminate\n");
                        return;
                    }
                    condition_tasks.wait(lock, [&]{
                        return (!queue_tasks.empty() || !running);
                    });
                }
            }
        }
    }
};

struct server_response {
    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue
    std::vector<server_task_result> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task) {
        SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.insert(id_task);
    }

    void add_waiting_tasks(const std::vector<server_task> & tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (const auto & task : tasks) {
            SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", task.id, (int) waiting_task_ids.size());
            waiting_task_ids.insert(task.id);
        }
    }

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task) {
        SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.erase(id_task);
    }

    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (const auto & id_task : id_tasks) {
            SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
            waiting_task_ids.erase(id_task);
        }
    }

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result recv(const std::unordered_set<int> & id_tasks) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_results);
            condition_results.wait(lock, [&]{
                return !queue_results.empty();
            });

            for (int i = 0; i < (int) queue_results.size(); i++) {
                if (id_tasks.find(queue_results[i].id) != id_tasks.end()) {
                    server_task_result res = queue_results[i];
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }
        }

        // should never reach here
    }

    // single-task version of recv()
    server_task_result recv(int id_task) {
        std::unordered_set<int> id_tasks = {id_task};
        return recv(id_tasks);
    }

    // Send a new result to a waiting id_task
    void send(server_task_result & result) {
        SRV_DBG("sending result for task id = %d\n", result.id);

        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto & id_task : waiting_task_ids) {
            if (result.id == id_task) {
                SRV_DBG("task id = %d moved to result queue\n", result.id);

                queue_results.push_back(std::move(result));
                condition_results.notify_all();
                return;
            }
        }
    }
};

struct server_context {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::vector<llama_lora_adapter_container> loras;

    gpt_params params;

    llama_model * model_dft = nullptr;
    llama_context_params cparams_dft;

    llama_batch batch = {};

    bool clean_kv_cache = true;
    bool add_bos_token  = true;
    bool has_eos_token  = false;

    int32_t n_ctx; // total context for all clients / slots

    // system prompt
    bool system_need_update = false;

    std::string              system_prompt;
    std::vector<llama_token> system_tokens;

    // slots / clients
    std::vector<server_slot> slots;
    json default_generation_settings_for_props;

    server_queue    queue_tasks;
    server_response queue_results;

    server_metrics metrics;

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    ~server_context() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }

        if (model) {
            llama_free_model(model);
            model = nullptr;
        }

        if (model_dft) {
            llama_free_model(model_dft);
            model_dft = nullptr;
        }

        // Clear any sampling context
        for (server_slot & slot : slots) {
            if (slot.smpl != nullptr) {
                gpt_sampler_free(slot.smpl);
            }
            slot.smpl = nullptr;

            llama_free(slot.ctx_dft);
            slot.ctx_dft = nullptr;

            common_speculative_free(slot.spec);
            slot.spec = nullptr;

            llama_batch_free(slot.batch_spec);
        }

        llama_batch_free(batch);
    }

    bool load_model(const gpt_params & params_) {
        SRV_INF("loading model '%s'\n", params.model.c_str());

        params = params_;

        // dedicate one sequence to the system prompt
        params.n_parallel += 1;
        
        // load draft model first
        llama_init_result llama_init_dft;
        if (!params.speculative.model.empty()) {
            SRV_INF("loading draft model '%s'\n", params.speculative.model.c_str());

            auto params_dft = params;

            params_dft.model        = params.speculative.model;
            params_dft.n_ctx        = params.speculative.n_ctx;
            params_dft.n_gpu_layers = params.speculative.n_gpu_layers;
            params_dft.use_mlock    = true;
            params_dft.n_world      = 1;  // do not split the draft model across devicesAdd commentMore actions
            params_dft.rank         = 0;  // always load the draft model on the head device

            std::fill_n(params_dft.n_layer_window, params.n_world, 0);
            
            llama_init_dft = llama_init_from_gpt_params(params_dft);

            model_dft = llama_init_dft.model;

            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params.speculative.model.c_str());
                return false;
            }
            
            cparams_dft = llama_context_params_from_gpt_params(params);
            cparams_dft.n_batch = llama_n_ctx(llama_init_dft.context);
            cparams_dft.n_world = 1;
            cparams_dft.rank    = 0;
            std::fill_n(cparams_dft.n_layer_window, 32, 0);
            cparams_dft.n_layer_window[0] = llama_n_layer(model_dft);
            cparams_dft.n_gpu_layers      = params.speculative.n_gpu_layers;
        }

        llama_init_result llama_init = llama_init_from_gpt_params(params);

        model = llama_init.model;
        ctx   = llama_init.context;
        loras = llama_init.lora_adapters;

        params.n_parallel -= 1; // but be sneaky about it

        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params.model.c_str());
            return false;
        }

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_add_bos_token(model);
        has_eos_token = !llama_add_eos_token(model);
        
        if (!params.speculative.model.empty()){
            if (!common_speculative_are_compatible(ctx, llama_init_dft.context)) {
                SRV_ERR("the draft model '%s' is not compatible with the target model '%s'\n", params.speculative.model.c_str(), params.model.c_str());

                llama_free      (llama_init_dft.context);
                llama_free_model(llama_init_dft.model);

                model_dft = nullptr;

                return false;
            }

            // the context is not needed - we will create one for each slot
            llama_free(llama_init_dft.context);
        }

        return true;
    }

    bool validate_model_chat_template() const {
        llama_chat_message chat[] = {{"user", "test"}};

        const int res = llama_chat_apply_template(model, nullptr, chat, 1, true, nullptr, 0);

        return res > 0;
    }

    void init() {
        const int32_t n_ctx_slot = n_ctx / params.n_parallel;

        SRV_INF("initializing slots, n_slots = %d\n", params.n_parallel);

        for (int i = 0; i < params.n_parallel; i++) {
            server_slot slot;

            slot.id = i;
            slot.n_ctx = n_ctx_slot;
            slot.n_predict = params.n_predict;
            
            if (model_dft) {
                slot.batch_spec = llama_batch_init(params.speculative.n_max + 1, 0, 1);

                slot.ctx_dft = llama_new_context_with_model(model_dft, cparams_dft);
                
                if (llama_context_setup_backend(model_dft, cparams_dft, slot.ctx_dft) == nullptr) {
                    SRV_ERR("%s: failed to setup context with model '%s'\n", __func__, params.speculative.model.c_str());
                    llama_free(slot.ctx_dft);
                    llama_free_model(model_dft);
                    return;
                }
                
                if (slot.ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create draft context\n");
                    return;
                }

                slot.spec = common_speculative_init(slot.ctx_dft);
                if (slot.spec == nullptr) {
                    SRV_ERR("%s", "failed to create speculator\n");
                    return;
                }
            }

            SLT_INF(slot, "new slot n_ctx_slot = %d\n", slot.n_ctx);

            const int ga_n = params.grp_attn_n;
            const int ga_w = params.grp_attn_w;

            if (ga_n != 1) {
                GGML_ASSERT(ga_n > 0                    && "ga_n must be positive");                       // NOLINT
                GGML_ASSERT(ga_w % ga_n == 0            && "ga_w must be a multiple of ga_n");             // NOLINT
                //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of ga_w");    // NOLINT
                //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * ga_n"); // NOLINT

                SLT_INF(slot, "slot self-extend: ga_n = %d, ga_w = %d\n", ga_n, ga_w);
            }

            slot.ga_i = 0;
            slot.ga_n = ga_n;
            slot.ga_w = ga_w;

            slot.sparams = params.sparams;

            slot.callback_on_release = [this](int) {
                queue_tasks.pop_deferred_task();
            };

            slot.reset();

            slots.push_back(slot);
        }

        default_generation_settings_for_props = get_formated_generation(slots.front());
        default_generation_settings_for_props["seed"] = -1;

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);

            // only a single seq_id per token is needed
            batch = llama_batch_init(std::max(n_batch, params.n_parallel), 0, 1);
        }

        metrics.init();
    }

    std::vector<llama_token> tokenize(const json & json_prompt, bool add_special) const {
        // TODO: currently, we tokenize using special tokens by default
        //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
        //       but it's better compared to completely ignoring ChatML and other chat templates
        const bool TMP_FORCE_SPECIAL = true;

        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (json_prompt.is_array()) {
            bool first = true;
            for (const auto & p : json_prompt) {
                if (p.is_string()) {
                    auto s = p.template get<std::string>();

                    std::vector<llama_token> p;
                    if (first) {
                        p = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
                        first = false;
                    } else {
                        p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                    }

                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                } else {
                    if (first) {
                        first = false;
                    }

                    prompt_tokens.push_back(p.template get<llama_token>());
                }
            }
        } else {
            auto s = json_prompt.template get<std::string>();
            prompt_tokens = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
        }

        return prompt_tokens;
    }

    server_slot * get_slot_by_id(int id) {
        for (server_slot & slot : slots) {
            if (slot.id == id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const std::string & prompt) {
        server_slot * ret = nullptr;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f && !prompt.empty()) {
            int max_lcp_len = 0;
            float similarity = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // skip the slot if it does not contains prompt
                if (!slot.prompt.is_string()) {
                    continue;
                }

                // current slot's prompt
                std::string slot_prompt = slot.prompt.get<std::string>();

                // length of the current slot's prompt
                int slot_prompt_len = slot_prompt.size();

                // length of the Longest Common Prefix between the current slot's prompt and the input prompt
                int lcp_len = common_part(slot_prompt, prompt);

                // fraction of the common substring length compared to the current slot's prompt length
                similarity = static_cast<float>(lcp_len) / slot_prompt_len;

                // select the current slot if the criteria match
                if (lcp_len > max_lcp_len && similarity > slot_prompt_similarity) {
                    max_lcp_len = lcp_len;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_DBG(*ret, "selected slot by lcp similarity, max_lcp_len = %d, similarity = %f\n", max_lcp_len, similarity);
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = ggml_time_us();
            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (slot.t_last_used < t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_DBG(*ret, "selected slot by lru, t_last = %" PRId64 "\n", t_last);
            }
        }

        return ret;
    }

    bool launch_slot_with_task(server_slot & slot, const server_task & task) {
        slot_params default_params;
        // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
        auto default_sparams = params.sparams;
        default_params.speculative = params.speculative;
        
        const auto & data = task.data;

        if (data.count("__oaicompat") != 0) {
            slot.oaicompat = true;
            slot.oaicompat_model = json_value(data, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
        } else {
            slot.oaicompat = false;
            slot.oaicompat_model = "";
        }

        slot.params.stream             = json_value(data, "stream",            false);
        slot.params.cache_prompt       = json_value(data, "cache_prompt",      true);
        slot.params.n_predict          = json_value(data, "n_predict",         json_value(data, "max_tokens", default_params.n_predict));
        slot.sparams.top_k             = json_value(data, "top_k",             default_sparams.top_k);
        slot.sparams.top_p             = json_value(data, "top_p",             default_sparams.top_p);
        slot.sparams.min_p             = json_value(data, "min_p",             default_sparams.min_p);
        slot.sparams.tfs_z             = json_value(data, "tfs_z",             default_sparams.tfs_z);
        slot.sparams.typ_p             = json_value(data, "typical_p",         default_sparams.typ_p);
        slot.sparams.temp              = json_value(data, "temperature",       default_sparams.temp);
        slot.sparams.dynatemp_range    = json_value(data, "dynatemp_range",    default_sparams.dynatemp_range);
        slot.sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", default_sparams.dynatemp_exponent);
        slot.sparams.penalty_last_n    = json_value(data, "repeat_last_n",     default_sparams.penalty_last_n);
        slot.sparams.penalty_repeat    = json_value(data, "repeat_penalty",    default_sparams.penalty_repeat);
        slot.sparams.penalty_freq      = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
        slot.sparams.penalty_present   = json_value(data, "presence_penalty",  default_sparams.penalty_present);
        slot.sparams.mirostat          = json_value(data, "mirostat",          default_sparams.mirostat);
        slot.sparams.mirostat_tau      = json_value(data, "mirostat_tau",      default_sparams.mirostat_tau);
        slot.sparams.mirostat_eta      = json_value(data, "mirostat_eta",      default_sparams.mirostat_eta);
        slot.sparams.penalize_nl       = json_value(data, "penalize_nl",       default_sparams.penalize_nl);
        slot.params.n_keep             = json_value(data, "n_keep",            params.n_keep);
        slot.params.n_discard          = json_value(data, "n_discard",         default_params.n_discard);
        slot.sparams.seed              = json_value(data, "seed",              default_sparams.seed);
        slot.sparams.n_probs           = json_value(data, "n_probs",           default_sparams.n_probs);
        slot.sparams.min_keep          = json_value(data, "min_keep",          default_sparams.min_keep);
        
        slot.params.speculative.n_min = json_value(data, "speculative.n_min", default_params.speculative.n_min);
        slot.params.speculative.n_max = json_value(data, "speculative.n_max", default_params.speculative.n_max);
        slot.params.speculative.p_min = json_value(data, "speculative.p_min", default_params.speculative.p_min);

        slot.params.speculative.n_min = std::min(slot.params.speculative.n_max, slot.params.speculative.n_min);

        // process "json_schema" and "grammar"
        if (data.contains("json_schema") && !data.at("json_schema").is_null() && data.contains("grammar") && !data.at("grammar").is_null()) {
            send_error(task, "Either \"json_schema\" or \"grammar\" can be specified, but not both", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        if (data.contains("json_schema") && !data.contains("grammar")) {
            try {
                auto schema                = json_value(data, "json_schema", json::object());
                slot.sparams.grammar       = json_schema_to_grammar(schema);
            } catch (const std::exception & e) {
                send_error(task, std::string("\"json_schema\": ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        } else {
            slot.sparams.grammar       = json_value(data, "grammar",           default_sparams.grammar);
        }

        if (slot.params.cache_prompt && slot.ga_n != 1) {
            slot.params.cache_prompt = false;
            SLT_WRN(slot, "%s", "group-attention is not supported with prompt caching. disabling cache\n");
        }

        if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict) {
            // Might be better to reject the request with a 400 ?
            slot.params.n_predict = slot.n_predict;
            SLT_WRN(slot, "n_predict = %d exceeds server configuration, setting to %d", slot.n_predict, slot.n_predict);
        }

        // infill
        slot.params.input_prefix = json_value(data, "input_prefix", default_params.input_prefix);
        slot.params.input_suffix = json_value(data, "input_suffix", default_params.input_suffix);

        // get prompt
        if (task.cmpl_type != SERVER_TASK_CMPL_TYPE_INFILL) {
            const auto & prompt = data.find("prompt");
            if (prompt == data.end()) {
                send_error(task, "\"prompt\" must be provided", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            if ((prompt->is_string()) ||
                (prompt->is_array() &&  prompt->size() == 1 && prompt->at(0).is_string()) ||
                (prompt->is_array() && !prompt->empty()     && prompt->at(0).is_number_integer())) {
                slot.prompt = *prompt;
            } else if (prompt->is_array() && prompt->size() == 1 && prompt->at(0).is_array()) {
                slot.prompt = prompt->at(0);
            } else if (prompt->is_array() && prompt->size() > 1) {
                // array of strings
                for (const auto & el : *prompt) {
                    if (!el.is_string()) {
                        send_error(task, "\"prompt\" must be a string, an array of strings or an array of integers", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
                }
                slot.prompt = *prompt;
            } else {
                send_error(task, "\"prompt\" must be a string, an array of strings or an array of integers", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }

        {
            slot.sparams.logit_bias.clear();

            if (json_value(data, "ignore_eos", false) && has_eos_token) {
                slot.sparams.logit_bias.push_back({llama_token_eos(model), -INFINITY});
            }

            const auto & logit_bias = data.find("logit_bias");
            if (logit_bias != data.end() && logit_bias->is_array()) {
                const int n_vocab = llama_n_vocab(model);
                for (const auto & el : *logit_bias) {
                    // TODO: we may want to throw errors here, in case "el" is incorrect
                    if (el.is_array() && el.size() == 2) {
                        float bias;
                        if (el[1].is_number()) {
                            bias = el[1].get<float>();
                        } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                            bias = -INFINITY;
                        } else {
                            continue;
                        }

                        if (el[0].is_number_integer()) {
                            llama_token tok = el[0].get<llama_token>();
                            if (tok >= 0 && tok < n_vocab) {
                                slot.sparams.logit_bias.push_back({tok, bias});
                            }
                        } else if (el[0].is_string()) {
                            auto toks = llama_tokenize(model, el[0].get<std::string>(), false);
                            for (auto tok : toks) {
                                slot.sparams.logit_bias.push_back({tok, bias});
                            }
                        }
                    }
                }
            }
        }

        {
            slot.params.antiprompt.clear();

            const auto & stop = data.find("stop");
            if (stop != data.end() && stop->is_array()) {
                for (const auto & word : *stop) {
                    if (!word.empty()) {
                        slot.params.antiprompt.push_back(word);
                    }
                }
            }
        }

        {
            const auto & samplers = data.find("samplers");
            if (samplers != data.end() && samplers->is_array()) {
                std::vector<std::string> sampler_names;
                for (const auto & name : *samplers) {
                    if (name.is_string()) {
                        sampler_names.emplace_back(name);
                    }
                }
                slot.sparams.samplers = gpt_sampler_types_from_names(sampler_names, false);
            } else {
                slot.sparams.samplers = default_sparams.samplers;
            }
        }

        {
            if (slot.smpl != nullptr) {
                gpt_sampler_free(slot.smpl);
            }

            slot.smpl = gpt_sampler_init(model, slot.sparams);
            if (slot.smpl == nullptr) {
                // for now, the only error that may happen here is invalid grammar
                send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }
        
        if (slot.ctx_dft) {
            llama_batch_free(slot.batch_spec);

            slot.batch_spec = llama_batch_init(slot.params.speculative.n_max + 1, 0, 1);
        }

        slot.state = SLOT_STATE_PROCESSING_PROMPT;
        slot.prompt_tokens.clear();

        SLT_INF(slot, "%s", "processing task\n");

        return true;
    }

    void kv_cache_clear() {
        SRV_DBG("%s", "clearing all KV cache\n");
        llama_kv_cache_clear(ctx);
        llama_send_kv_cache_clear(ctx);
        clean_kv_cache = false;
    }

    void system_prompt_update() {
        SRV_DBG("updating system prompt: '%s'\n", system_prompt.c_str());

        kv_cache_clear();
        system_tokens.clear();

        if (!system_prompt.empty()) {
            system_tokens = ::llama_tokenize(ctx, system_prompt, true);

            const int32_t n_batch = llama_n_batch(ctx);
            const int32_t n_tokens_prompt = system_tokens.size();

            for (int32_t i = 0; i < n_tokens_prompt; i += n_batch) {
                const int32_t n_tokens = std::min(n_batch, n_tokens_prompt - i);

                llama_batch_clear(batch);

                for (int32_t j = 0; j < n_tokens; ++j) {
                    llama_batch_add(batch, system_tokens[i + j], i + j, { 0 }, false);
                }

                if (llama_decode(ctx, batch, true) != 0) {
                    SRV_ERR("%s", "llama_decode() failed\n");
                    return;
                }
            }

            // assign the system KV cache to all parallel sequences
            for (int32_t i = 1; i <= params.n_parallel; ++i) {
                llama_kv_cache_seq_cp     (ctx, 0, i,     -1, -1);
                llama_send_kv_cache_seq_cp(ctx, 0, i - 1, -1, -1);
            }
        }

        system_need_update = false;
    }

    bool system_prompt_set(const std::string & sys_prompt) {
        SRV_DBG("system prompt set: '%s'\n", system_prompt.c_str());

        system_prompt = sys_prompt;

        // release all slots
        for (server_slot & slot : slots) {
            slot.release();
        }

        system_need_update = true;
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = llama_token_to_piece(ctx, result.tok, params.special);
        slot.sampled = result.tok;

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = false;
        for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i) {
            unsigned char c = slot.generated_text[slot.generated_text.size() - i];
            if ((c & 0xC0) == 0x80) {
                // continuation byte: 10xxxxxx
                continue;
            }
            if ((c & 0xE0) == 0xC0) {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            } else if ((c & 0xF0) == 0xE0) {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            } else if ((c & 0xF8) == 0xF0) {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool is_stop_full = false;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), STOP_TYPE_FULL);
            if (stop_pos != std::string::npos) {
                is_stop_full = true;
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else {
                is_stop_full = false;
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), STOP_TYPE_PARTIAL);
            }

            // check if there is any token to predict
            if (stop_pos == std::string::npos || (!slot.has_next_token && !is_stop_full && stop_pos > 0)) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            }

            slot.add_token(result);
            if (slot.params.stream) {
                send_partial_response(slot, result);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params)) {
            slot.stopped_limit  = true;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.params.n_predict);
        }

        // we stop when it reaches the context limit, otherwise it may run forever
        if (slot.n_decoded >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stopped_limit  = true;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, n_decoded = %d, n_ctx = %d\n", slot.n_decoded, slot.n_ctx);
        }

        if (llama_token_is_eog(model, result.tok)) {
            slot.stopped_eos    = true;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        const auto n_ctx_train = llama_n_ctx_train(model);

        if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.ga_n == 1 && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
            slot.truncated      = true;
            slot.stopped_limit  = true;
            slot.has_next_token = false; // stop prediction

            SLT_WRN(slot,
                    "n_predict (%d) is not set and self-context extend is disabled. "
                    "Limiting generated tokens to n_ctx_train (%d) to avoid EOS-less generation infinite loop\n",
                    slot.params.n_predict, n_ctx_train);
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: '%s'\n", slot.n_decoded, slot.n_remaining, token_str.c_str());

        return slot.has_next_token; // continue
    }

    json get_formated_generation(const server_slot & slot) const {
        std::vector<std::string> samplers;
        samplers.reserve(slot.sparams.samplers.size());
        for (const auto & sampler : slot.sparams.samplers) {
            samplers.emplace_back(gpt_sampler_type_to_str(sampler));
        }

        return json {
            {"n_ctx",                     slot.n_ctx},
            {"n_predict",                 slot.n_predict},     // Server configured n_predict
            {"model",                     params.model_alias},
            {"seed",                      slot.sparams.seed},
            {"seed_cur",                  slot.smpl ? gpt_sampler_get_seed(slot.smpl) : 0},
            {"temperature",               slot.sparams.temp},
            {"dynatemp_range",            slot.sparams.dynatemp_range},
            {"dynatemp_exponent",         slot.sparams.dynatemp_exponent},
            {"top_k",                     slot.sparams.top_k},
            {"top_p",                     slot.sparams.top_p},
            {"min_p",                     slot.sparams.min_p},
            {"tfs_z",                     slot.sparams.tfs_z},
            {"typical_p",                 slot.sparams.typ_p},
            {"repeat_last_n",             slot.sparams.penalty_last_n},
            {"repeat_penalty",            slot.sparams.penalty_repeat},
            {"presence_penalty",          slot.sparams.penalty_present},
            {"frequency_penalty",         slot.sparams.penalty_freq},
            {"mirostat",                  slot.sparams.mirostat},
            {"mirostat_tau",              slot.sparams.mirostat_tau},
            {"mirostat_eta",              slot.sparams.mirostat_eta},
            {"penalize_nl",               slot.sparams.penalize_nl},
            {"stop",                      slot.params.antiprompt},
            {"max_tokens",                slot.params.n_predict}, // User configured n_predict
            {"n_keep",                    slot.params.n_keep},
            {"n_discard",                 slot.params.n_discard},
            {"ignore_eos",                slot.sparams.ignore_eos},
            {"stream",                    slot.params.stream},
          //{"logit_bias",                slot.sparams.logit_bias},
            {"n_probs",                   slot.sparams.n_probs},
            {"min_keep",                  slot.sparams.min_keep},
            {"grammar",                   slot.sparams.grammar},
            {"samplers",                  samplers},
        };
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.id_task, error, type);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        server_task_result res;
        res.id       = id_task;
        res.stop     = false;
        res.error    = true;
        res.data     = format_error_response(error, type);

        queue_results.send(res);
    }

    void send_partial_response(server_slot & slot, completion_token_output tkn) {
        server_task_result res;
        res.id       = slot.id_task;
        res.error    = false;
        res.stop     = false;
        res.data     = json {
            {"content",    tkn.text_to_send},
            {"stop",       false},
            {"id_slot",    slot.id},
            {"multimodal", false},
            {"index",      slot.index},
        };

        if (slot.sparams.n_probs > 0) {
            const std::vector<llama_token> to_send_toks = llama_tokenize(ctx, tkn.text_to_send, false);
            const size_t probs_pos      = std::min(slot.n_sent_token_probs,                       slot.generated_token_probs.size());
            const size_t probs_stop_pos = std::min(slot.n_sent_token_probs + to_send_toks.size(), slot.generated_token_probs.size());

            std::vector<completion_token_output> probs_output;
            if (probs_pos < probs_stop_pos) {
                probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin() + probs_pos,
                        slot.generated_token_probs.begin() + probs_stop_pos);
            }
            slot.n_sent_token_probs = probs_stop_pos;

            res.data["completion_probabilities"] = probs_vector_to_json(ctx, probs_output);
        }

        if (slot.oaicompat) {
            res.data["oaicompat_token_ctr"] = slot.n_decoded;
            res.data["model"] = slot.oaicompat_model;
        }

        queue_results.send(res);
    }

    void send_final_response(const server_slot & slot) {
        server_task_result res;
        res.id       = slot.id_task;
        res.error    = false;
        res.stop     = true;
        res.data     = json {
            {"content",             !slot.params.stream ? slot.generated_text : ""},
            {"id_slot",             slot.id},
            {"stop",                true},
            {"model",               params.model_alias},
            {"tokens_predicted",    slot.n_decoded},
            {"tokens_evaluated",    slot.n_prompt_tokens},
            {"generation_settings", get_formated_generation(slot)},
            {"prompt",              slot.prompt},
            {"truncated",           slot.truncated},
            {"stopped_eos",         slot.stopped_eos},
            {"stopped_word",        slot.stopped_word},
            {"stopped_limit",       slot.stopped_limit},
            {"stopping_word",       slot.stopping_word},
            {"tokens_cached",       slot.n_past},
            {"timings",             slot.get_formated_timings()},
            {"index",               slot.index},
        };

        if (slot.sparams.n_probs > 0) {
            std::vector<completion_token_output> probs;
            if (!slot.params.stream && slot.stopped_word) {
                const std::vector<llama_token> stop_word_toks = llama_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                probs = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                probs = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }

            res.data["completion_probabilities"] = probs_vector_to_json(ctx, probs);
        }

        if (slot.oaicompat) {
            res.data["oaicompat_token_ctr"] = slot.n_decoded;
            res.data["model"] = slot.oaicompat_model;
        }

        queue_results.send(res);
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        server_task_result res;
        res.id       = slot.id_task;
        res.error    = false;
        res.stop     = true;

        const int n_embd = llama_n_embd(model);

        std::vector<float> embd_res(n_embd, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id + 1) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res.data = json {
                    {"embedding", std::vector<float>(n_embd, 0.0f)},
                    {"index",     slot.index},
                };

                continue;
            }

            llama_embd_normalize(embd, embd_res.data(), n_embd);

            res.data = json {
                {"embedding", embd_res},
                {"index",     slot.index},
            };
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(res);
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        server_task_result res;
        res.id       = slot.id_task;
        res.error    = false;
        res.stop     = true;

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id + 1) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res.data = json {
                    {"index", slot.index},
                    {"score", -1e6},
                };

                continue;
            }

            res.data = json {
                {"index", slot.index},
                {"score", embd[0]},
            };
        }

        SLT_DBG(slot, "sending rerank result, res = '%s'\n", res.data.dump().c_str());

        queue_results.send(res);
    }

    //
    // Functions to create new task(s) and receive result(s)
    //

    std::vector<server_task> create_tasks_cmpl(json data, server_task_cmpl_type cmpl_type) {
        std::vector<server_task> tasks;
        auto create_task = [&](json & task_data, bool replace_prompt, json prompt) {
            server_task task;
            task.id        = queue_tasks.get_new_id();
            task.cmpl_type = cmpl_type;
            task.type      = SERVER_TASK_TYPE_COMPLETION;
            if (replace_prompt) {
                task.data  = task_data;
                task.data["prompt"] = std::move(prompt);
            } else {
                task.data  = std::move(task_data);
            }
            tasks.push_back(std::move(task));
        };

        static constexpr const char * error_msg = "\"prompt\" must be a string, an array of token ids or an array of prompts";
        if (!data.contains("prompt")) {
            throw std::runtime_error(error_msg);
        }

        json prompt = data.at("prompt");

        // if the prompt is a singleton (i.e. a string or a list of tokens), we only need to create single task
        if (prompt.is_string() || json_is_array_of_numbers(prompt)) {
            data["index"] = 0;
            create_task(data, false, nullptr);
        }
        // otherwise, it's a multiple-prompt task, we break it into smaller tasks
        else if (prompt.is_array()) {
            std::vector<json> prompts = prompt;
            if (cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK) {
                // prompts[0] is the question
                // the rest are the answers/documents
                SRV_DBG("creating rerank tasks, n_prompts = %d\n", (int) prompts.size() - 1);
                for (size_t i = 1; i < prompts.size(); i++) {
                    json qd;
                    qd.push_back(prompts[0]);
                    qd.push_back(prompts[i]);
                    data["index"] = i - 1;
                    create_task(data, true, qd);
                }
            } else {
                SRV_DBG("creating multi-prompt tasks, n_prompts = %d\n", (int) prompts.size());
                for (size_t i = 0; i < prompts.size(); i++) {
                    const auto & e = prompts[i];
                    if (e.is_string() || json_is_array_of_numbers(e)) {
                        data["index"] = i;
                        create_task(data, true, e);
                    } else {
                        throw std::runtime_error(error_msg);
                    }
                }
            }
        }
        // invalid case
        else {
            throw std::runtime_error(error_msg);
        }

        return tasks;
    }

    void cancel_tasks(const std::unordered_set<int> & id_tasks) {
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            SRV_WRN("cancel task, id_task = %d\n", id_task);
            
            // create a cancel task for id_task
            server_task task;
            task.type      = SERVER_TASK_TYPE_CANCEL;
            task.id_target = id_task;
            cancel_tasks.push_back(task);

            // notify the results queue that the task is cancelled
            server_task_result cancel_res;
            cancel_res.id    = id_task;
            cancel_res.stop  = true;      
            cancel_res.error = false;     
            cancel_res.data  = {{"cancelled", true}};
            queue_results.send(cancel_res);

            // remove the task from the waiting queue
            queue_results.remove_waiting_task_id(id_task);
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(cancel_tasks, true);
    }

    // receive the results from task(s) created by create_tasks_cmpl
    void receive_cmpl_results(
            const std::unordered_set<int> & id_tasks,
            const std::function<void(std::vector<server_task_result>&)> & result_handler,
            const std::function<void(json)> & error_handler) {
        // TODO: currently, there is no way to detect the client has cancelled the request
        std::vector<server_task_result> results(id_tasks.size());
        for (size_t i = 0; i < id_tasks.size(); i++) {
            server_task_result result = queue_results.recv(id_tasks);

            if (result.error) {
                error_handler(result.data);
                cancel_tasks(id_tasks);
                return;
            }

            const size_t idx = result.data["index"];
            GGML_ASSERT(idx < results.size() && "index out of range");

            results[idx] = result;
        }
        result_handler(results);
    }

    // receive the results from task(s) created by create_tasks_cmpl, in stream mode
    void receive_cmpl_results_stream(
            const std::unordered_set<int> & id_tasks, const
            std::function<bool(server_task_result&)> & result_handler, const
            std::function<void(json)> & error_handler) {
        size_t n_finished = 0;
        while (true) {
            server_task_result result = queue_results.recv(id_tasks);
            if (!result_handler(result)) {
                cancel_tasks(id_tasks);
                break;
            }

            if (result.error) {
                error_handler(result.data);
                cancel_tasks(id_tasks);
                break;
            }

            if (result.stop) {
                if (++n_finished == id_tasks.size()) {
                    break;
                }
            }
        }
    }

    //
    // Functions to process the task
    //

    void process_single_task(const server_task & task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
                {
                    const int id_slot = json_value(task.data, "id_slot", -1);

                    server_slot * slot;

                    if (id_slot != -1) {
                        slot = get_slot_by_id(id_slot);
                    } else {
                        std::string prompt;
                        if (task.data.contains("prompt") && task.data.at("prompt").is_string()) {
                            prompt = json_value(task.data, "prompt", std::string());
                        }

                        slot = get_available_slot(prompt);
                    }

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }

                    if (task.data.contains("system_prompt")) {
                        std::string sys_prompt = json_value(task.data, "system_prompt", std::string());
                        system_prompt_set(sys_prompt);

                        for (server_slot & slot : slots) {
                            slot.n_past    = 0;
                            slot.n_past_se = 0;
                        }
                    }

                    slot->reset();

                    slot->id_task   = task.id;
                    slot->cmpl_type = task.cmpl_type;
                    slot->index     = json_value(task.data, "index", 0);

                    if (!launch_slot_with_task(*slot, task)) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", task.id);
                        break;
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.id_task == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = get_formated_generation(slot);
                        slot_data["id"]         = slot.id;
                        slot_data["id_task"]    = slot.id_task;
                        slot_data["state"]      = slot.state;
                        slot_data["prompt"]     = slot.prompt;
                        slot_data["next_token"] = {
                            {"has_next_token", slot.has_next_token},
                            {"n_remain",       slot.n_remaining},
                            {"n_decoded",      slot.n_decoded},
                            {"stopped_eos",    slot.stopped_eos},
                            {"stopped_word",   slot.stopped_word},
                            {"stopped_limit",  slot.stopped_limit},
                            {"stopping_word",  slot.stopping_word},
                        };

                        if (slot_data["state"] == SLOT_STATE_IDLE) {
                            n_idle_slots++;
                        } else {
                            n_processing_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    server_task_result res;
                    res.id       = task.id;
                    res.stop     = true;
                    res.error    = false;
                    res.data     = {
                        { "idle",                            n_idle_slots       },
                        { "processing",                      n_processing_slots },
                        { "deferred",                        queue_tasks.queue_tasks_deferred.size() },
                        { "t_start",                         metrics.t_start},

                        { "n_prompt_tokens_processed_total", metrics.n_prompt_tokens_processed_total},
                        { "t_tokens_generation_total",       metrics.t_tokens_generation_total},
                        { "n_tokens_predicted_total",        metrics.n_tokens_predicted_total},
                        { "t_prompt_processing_total",       metrics.t_prompt_processing_total},

                        { "n_prompt_tokens_processed",       metrics.n_prompt_tokens_processed},
                        { "t_prompt_processing",             metrics.t_prompt_processing},
                        { "n_tokens_predicted",              metrics.n_tokens_predicted},
                        { "t_tokens_generation",             metrics.t_tokens_generation},

                        { "n_decode_total",                  metrics.n_decode_total},
                        { "n_busy_slots_total",              metrics.n_busy_slots_total},

                        { "kv_cache_tokens_count",           llama_get_kv_cache_token_count(ctx)},
                        { "kv_cache_used_cells",             llama_get_kv_cache_used_cells(ctx)},

                        { "slots",                           slots_data },
                    };

                    if (json_value(task.data, "reset_bucket", false)) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(res);
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }

                    const size_t token_count = slot->cache_tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.data.at("filename");
                    std::string filepath = task.data.at("filepath");

                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id + 1, slot->cache_tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    server_task_result result;
                    result.id = task.id;
                    result.stop = true;
                    result.error = false;
                    result.data = json {
                        { "id_slot",   id_slot },
                        { "filename",  filename },
                        { "n_saved",   token_count }, // tokens saved
                        { "n_written", nwrite },      // bytes written
                        { "timings", {
                            { "save_ms", t_save_ms }
                        } }
                    };
                    queue_results.send(result);
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.data.at("filename");
                    std::string filepath = task.data.at("filepath");

                    slot->cache_tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id + 1, slot->cache_tokens.data(), slot->cache_tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->cache_tokens.resize(0);
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    slot->cache_tokens.resize(token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    server_task_result result;
                    result.id = task.id;
                    result.stop = true;
                    result.error = false;
                    result.data = json {
                        { "id_slot",    id_slot },
                        { "filename",   filename },
                        { "n_restored", token_count }, // tokens restored
                        { "n_read",     nread },       // bytes read
                        { "timings", {
                            { "restore_ms", t_restore_ms }
                        } }
                    };
                    queue_results.send(result);
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->cache_tokens.size();
                    llama_kv_cache_seq_rm(ctx, slot->id + 1, -1, -1);
                    slot->cache_tokens.clear();

                    server_task_result result;
                    result.id = task.id;
                    result.stop = true;
                    result.error = false;
                    result.data = json {
                        { "id_slot",  id_slot },
                        { "n_erased", n_erased }
                    };
                    queue_results.send(result);
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    llama_lora_adapters_apply(ctx, loras);
                    server_task_result result;
                    result.id = task.id;
                    result.stop = true;
                    result.error = false;
                    result.data = json{{ "success", true }};
                    queue_results.send(result);
                } break;
        }
    }

    void update_slots() {
        if (system_need_update) {
            system_prompt_update();
        }

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");
                if (system_prompt.empty() && clean_kv_cache) {
                    kv_cache_clear();
                }

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task;
            task.type      = SERVER_TASK_TYPE_NEXT_RESPONSE;
            task.id_target = -1;

            queue_tasks.post(task);
        }

        // apply context-shift if needed
        for (server_slot & slot : slots) {
            if (slot.ga_n == 1) {
                if (slot.is_processing() && (int) system_tokens.size() + slot.n_past >= slot.n_ctx - 1) {
                    if (!params.ctx_shift) {
                        // this check is redundant (for good)
                        // we should never get here, because generation should already stopped in process_token()
                        slot.release();
                        send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                        continue;
                    }

                    // Shift context
                    const int n_keep    = slot.params.n_keep + add_bos_token;
                    const int n_left    = (int) system_tokens.size() + slot.n_past - n_keep;
                    const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

                    SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                    llama_kv_cache_seq_rm      (ctx, slot.id + 1, n_keep            , n_keep + n_discard);
                    llama_kv_cache_seq_add     (ctx, slot.id + 1, n_keep + n_discard, system_tokens.size() + slot.n_past, -n_discard);

                    llama_send_kv_cache_seq_rm (ctx, slot.id    , n_keep            , n_keep + n_discard);
                    llama_send_kv_cache_seq_add(ctx, slot.id    , n_keep + n_discard, system_tokens.size() + slot.n_past, -n_discard);

                    if (slot.params.cache_prompt) {
                        for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size(); i++) {
                            slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                        }

                        slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);
                    }

                    slot.n_past -= n_discard;

                    slot.truncated = true;
                }
            }
        }

        // start populating the batch for this iteration
        llama_batch_clear(batch);

        // frist, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            slot.i_batch = batch.n_tokens;

            const int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

            // TODO: we always have to take into account the "system_tokens"
            //       this is not great and needs to be improved somehow
            llama_batch_add(batch, slot.sampled, system_tokens.size() + slot_npast, { slot.id + 1 }, true);

            slot.n_past += 1;

            if (slot.params.cache_prompt) {
                slot.cache_tokens.push_back(slot.sampled);
            }

            SLT_DBG(slot, "slot decode token, n_ctx = %d, n_past = %d, n_system_tokens = %d, n_cache_tokens = %d, truncated = %d\n",
                    slot.n_ctx, slot.n_past, (int) system_tokens.size(), (int) slot.cache_tokens.size(), slot.truncated);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        // track if this is an embedding or non-embedding batch
        // if we've added sampled tokens above, we are in non-embedding mode
        // -1: none, 0: non-embedding, 1: embedding
        // TODO: make enum
        int32_t batch_type = batch.n_tokens > 0 ? 0 : -1;

        // next, batch any pending prompts without exceeding n_batch
        if (params.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                    auto & prompt_tokens = slot.prompt_tokens;

                    // we haven't tokenized the prompt yet - do it now:
                    if (prompt_tokens.empty()) {
                        SLT_INF(slot, "tokenizing prompt, len = %d\n", (int) slot.prompt.size());

                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_INFILL) {
                            const bool add_bos = llama_add_bos_token(model);
                            bool suff_rm_leading_spc = true;
                            if (params.input_suffix.find_first_of(' ') == 0 && params.input_suffix.size() > 1) {
                                params.input_suffix.erase(0, 1);
                                suff_rm_leading_spc = false;
                            }

                            auto prefix_tokens = tokenize(slot.params.input_prefix, false);
                            auto suffix_tokens = tokenize(slot.params.input_suffix, false);

                            const int space_token = 29871; // TODO: this should not be hardcoded
                            if (suff_rm_leading_spc && !suffix_tokens.empty() && suffix_tokens[0] == space_token) {
                                suffix_tokens.erase(suffix_tokens.begin());
                            }

                            prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(model));
                            suffix_tokens.insert(suffix_tokens.begin(), llama_token_suffix(model));

                            auto embd_inp = params.spm_infill ? suffix_tokens : prefix_tokens;
                            auto embd_end = params.spm_infill ? prefix_tokens : suffix_tokens;
                            if (add_bos) {
                                embd_inp.insert(embd_inp.begin(), llama_token_bos(model));
                            }
                            embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());

                            const llama_token middle_token = llama_token_middle(model);
                            if (middle_token >= 0) {
                                embd_inp.push_back(middle_token);
                            }

                            prompt_tokens = embd_inp;
                        } else if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK) {
                            // require slot.prompt to be array of 2 strings
                            if (!slot.prompt.is_array() || slot.prompt.size() != 2) {
                                SLT_ERR(slot, "%s", "invalid prompt for rerank task\n");
                                slot.release();
                                send_error(slot, "invalid prompt for rerank task", ERROR_TYPE_INVALID_REQUEST);
                                continue;
                            }

                            // prompt: [BOS]query[EOS][SEP]doc[EOS]
                            prompt_tokens.clear();
                            prompt_tokens.push_back(llama_token_bos(model));
                            {
                                const auto part = tokenize(slot.prompt[0], false);
                                prompt_tokens.insert(prompt_tokens.end(), part.begin(), part.end());
                            }
                            prompt_tokens.push_back(llama_token_eos(model));
                            prompt_tokens.push_back(llama_token_sep(model));
                            {
                                const auto part = tokenize(slot.prompt[1], false);
                                prompt_tokens.insert(prompt_tokens.end(), part.begin(), part.end());
                            }
                            prompt_tokens.push_back(llama_token_eos(model));
                        } else {
                            prompt_tokens = tokenize(slot.prompt, system_prompt.empty()); // add BOS if there isn't system prompt
                        }

                        slot.n_past = 0;
                        slot.n_prompt_tokens = prompt_tokens.size();

                        SLT_INF(slot, "prompt tokenized, n_ctx_slot = %d, n_keep = %d, n_prompt_tokens = %d\n", slot.n_ctx, slot.params.n_keep, slot.n_prompt_tokens);

                        // empty prompt passed -> release the slot and send empty response
                        if (prompt_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.release();
                            slot.print_timings();
                            send_final_response(slot);
                            continue;
                        }

                        if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_EMBEDDING || slot.cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK) {
                            // this prompt is too large to process - discard it
                            if (slot.n_prompt_tokens > n_ubatch) {
                                slot.release();
                                send_error(slot, "input is too large to process. increase the physical batch size", ERROR_TYPE_SERVER);
                                continue;
                            }
                        } else {
                            if (!params.ctx_shift) {
                                // if context shift is disabled, we make sure prompt size is smaller than KV size
                                if ((int)system_tokens.size() + slot.n_prompt_tokens >= slot.n_ctx) {
                                    slot.release();
                                    send_error(slot, "the request exceeds the available context size. try increasing the context size or enable context shift", ERROR_TYPE_INVALID_REQUEST);
                                    continue;
                                }
                            }
                            if (slot.params.n_keep < 0) {
                                slot.params.n_keep = (int)system_tokens.size() + slot.n_prompt_tokens + 3; // +3 for <think> tag
                            }
                            slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                            // if input prompt is too big, truncate it (if group attention self-extend is disabled)
                            if (slot.ga_n == 1 && slot.n_prompt_tokens >= slot.n_ctx) {
                                const int n_left = slot.n_ctx - slot.params.n_keep;
                                const int n_block_size = n_left / 2;
                                const int erased_blocks = (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                                std::vector<llama_token> new_tokens(
                                        prompt_tokens.begin(),
                                        prompt_tokens.begin() + slot.params.n_keep);

                                new_tokens.insert(
                                        new_tokens.end(),
                                        prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                                        prompt_tokens.end());

                                prompt_tokens = std::move(new_tokens);

                                slot.truncated = true;
                                slot.n_prompt_tokens = prompt_tokens.size();

                                SLT_WRN(slot, "input truncated, n_ctx = %d, n_keep = %d, n_left = %d, n_prompt_tokens = %d\n", slot.n_ctx, slot.params.n_keep, n_left, slot.n_prompt_tokens);

                                GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                            }

                            gpt_sampler_reset(slot.smpl);

                            if (!slot.params.cache_prompt) {
                                slot.n_past_se = 0;
                                slot.ga_i      = 0;
                            } else {
                                GGML_ASSERT(slot.ga_n == 1);

                                // reuse any previously computed tokens that are common with the new prompt
                                slot.n_past = common_part(slot.cache_tokens, prompt_tokens);

                                // push the prompt into the sampling context (do not apply grammar)
                                for (int i = 0; i < slot.n_past; ++i) {
                                    gpt_sampler_accept(slot.smpl, slot.cache_tokens[i], false);
                                }
                            }
                        }

                        if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
                            // we have to evaluate at least 1 token to generate logits.
                            SLT_WRN(slot, "need to evaluate at least 1 token to generate logits, n_past = %d, n_prompt_tokens = %d\n", slot.n_past, slot.n_prompt_tokens);

                            slot.n_past--;
                            if (slot.ga_i > 0) {
                                slot.n_past_se--;
                            }
                        }

                        slot.n_prompt_tokens_processed = 0;
                    }

                    // non-causal tasks require to fit the entire prompt in the physical batch
                    if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_EMBEDDING || slot.cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.n_prompt_tokens > n_batch) {
                            continue;
                        }
                    }

                    // check that we are in the right batch_type, if not defer the slot
                    const bool slot_type =
                        slot.cmpl_type == SERVER_TASK_CMPL_TYPE_EMBEDDING ||
                        slot.cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK     ? 1 : 0;

                    if (batch_type == -1) {
                        batch_type = slot_type;
                    } else if (batch_type != slot_type) {
                        continue;
                    }

                    // keep only the common part
                    int p0 = (int) system_tokens.size() + slot.n_past;
                    if (!llama_kv_cache_seq_rm(ctx, slot.id + 1, p0, -1)) {
                        // could not partially delete (likely using a non-Transformer model)
                        llama_kv_cache_seq_rm     (ctx, slot.id + 1, -1, -1);
                        llama_send_kv_cache_seq_rm(ctx, slot.id    , -1, -1);

                        p0 = (int) system_tokens.size();
                        if (p0 != 0) {
                            // copy over the system prompt when there is one
                            llama_kv_cache_seq_cp     (ctx, 0, slot.id + 1, -1, -1);
                            llama_send_kv_cache_seq_cp(ctx, 0, slot.id    , -1, -1);
                        }

                        // there is no common part left (except for the system prompt)
                        slot.n_past = 0;
                        slot.n_past_se = 0;
                        slot.ga_i = 0;
                        // TODO: is the system prompt ever in the sampling context?
                        gpt_sampler_reset(slot.smpl);
                    } else {
                        llama_send_kv_cache_seq_rm(ctx, slot.id, p0, -1);
                    }

                    // remove the non-common part from the cache
                    slot.cache_tokens.resize(slot.n_past);

                    SLT_INF(slot, "kv cache rm [%d, end)\n", p0);

                    int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

                    int32_t ga_i = slot.ga_i;
                    int32_t ga_n = slot.ga_n;
                    int32_t ga_w = slot.ga_w;

                    // add prompt tokens for processing in the current batch
                    // TODO: the self-extend stuff here is a mess - simplify and/or abstract it somehow
                    for (; slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch; ++slot.n_past) {
                        if (slot.ga_n != 1) {
                            while (slot_npast >= ga_i + ga_w) {
                                const int bd = (ga_w/ga_n)*(ga_n - 1);
                                slot_npast -= bd;
                                ga_i += ga_w/ga_n;
                            }
                        }

                        llama_batch_add(batch, prompt_tokens[slot.n_past], system_tokens.size() + slot_npast, { slot.id + 1 }, false);

                        if (slot.params.cache_prompt) {
                            slot.cache_tokens.push_back(prompt_tokens[slot.n_past]);
                        }

                        slot.n_prompt_tokens_processed++;
                        slot_npast++;
                    }

                    SLT_INF(slot, "prompt processing progress, n_past = %d, n_tokens = %d, progress = %f\n", slot.n_past, batch.n_tokens, (float) slot.n_prompt_tokens_processed / slot.n_prompt_tokens);

                    // entire prompt has been processed
                    if (slot.n_past == slot.n_prompt_tokens) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        SLT_INF(slot, "prompt done, n_past = %d, n_tokens = %d\n", slot.n_past, batch.n_tokens);
                    }
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");
            return;
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        // make sure we're in the right embedding mode
        llama_set_embeddings(ctx, batch_type == 1);

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            for (auto & slot : slots) {
                if (slot.ga_n != 1) {
                    // context extension via Self-Extend
                    // TODO: simplify and/or abstract this
                    while (slot.n_past_se >= slot.ga_i + slot.ga_w) {
                        const int ib = (slot.ga_n * slot.ga_i) / slot.ga_w;
                        const int bd = (slot.ga_w / slot.ga_n) * (slot.ga_n - 1);
                        const int dd = (slot.ga_w / slot.ga_n) - ib * bd - slot.ga_w;

                        SLT_DBG(slot, "shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i, slot.n_past_se, ib * bd, slot.ga_i + ib * bd, slot.n_past_se + ib * bd);
                        SLT_DBG(slot, "div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n, (slot.ga_i + ib * bd) / slot.ga_n, (slot.ga_i + ib * bd + slot.ga_w) / slot.ga_n);
                        SLT_DBG(slot, "shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd, slot.ga_i + ib * bd + slot.ga_w + dd, slot.n_past_se + ib * bd + dd);

                        llama_kv_cache_seq_add     (ctx, slot.id + 1, slot.ga_i, slot.n_past_se, ib * bd);
                        llama_send_kv_cache_seq_add(ctx, slot.id    , slot.ga_i, slot.n_past_se, ib * bd);

                        llama_kv_cache_seq_div     (ctx, slot.id + 1, slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n);
                        llama_send_kv_cache_seq_div(ctx, slot.id    , slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n);

                        llama_kv_cache_seq_add     (ctx, slot.id + 1, slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd);
                        llama_send_kv_cache_seq_add(ctx, slot.id    , slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd);

                        slot.n_past_se -= bd;

                        slot.ga_i += slot.ga_w / slot.ga_n;

                        SLT_DBG(slot, "\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", slot.n_past_se + bd, slot.n_past_se, slot.ga_i);
                    }

                    slot.n_past_se += n_tokens;
                }
            }

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view, true);
            metrics.on_decoded(slots);

            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    SRV_ERR("failed to decode the batch: KV cache is full - try increasing it via the context size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);
                    for (auto & slot : slots) {
                        slot.release();
                        send_error(slot, "Input prompt is too big compared to KV size. Please try increasing KV size.");
                    }
                    break; // break loop of n_batch
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size - try increasing it via the context size or enable defragmentation, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            for (auto & slot : slots) {
                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                llama_token id;

                {
                    completion_token_output result;

                    id = gpt_sampler_sample(slot.smpl, ctx, slot.i_batch - i);

                    slot.i_batch = -1;

                    gpt_sampler_accept(slot.smpl, id, true);

                    slot.n_decoded += 1;
                    if (slot.n_decoded == 1) {
                        slot.t_start_generation = ggml_time_us();
                        slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                        metrics.on_prompt_eval(slot);
                    }

                    result.tok = id;

                    const auto * cur_p = gpt_sampler_get_candidates(slot.smpl);

                    for (size_t i = 0; i < (size_t) slot.params.sampling.n_probs; ++i) {
                        result.probs.push_back({
                            cur_p->data[i].id,
                                i >= cur_p->size ? 0.0f : cur_p->data[i].p,
                        });
                    }

                    if (!process_token(result, slot)) {
                        // release slot because of stop condition
                        slot.release();
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        continue;
                    }
                }

                // check if the slot supports speculative decoding
                if (!slot.can_speculate()) {
                    continue;
                }

                struct common_speculative_params params_spec;
                params_spec.n_draft   = slot.params.speculative.n_max;
                params_spec.n_reuse   = llama_n_ctx(slot.ctx_dft) - slot.params.speculative.n_max;
                params_spec.p_min     = slot.params.speculative.p_min;

                llama_tokens draft = common_speculative_gen_draft(slot.spec, params_spec, slot.cache_tokens, id);

                // ignore small drafts
                if (slot.params.speculative.n_min > (int) draft.size()) {
                    continue;
                }

                // construct the speculation batch
                llama_batch_clear(slot.batch_spec);
                llama_batch_add  (slot.batch_spec, id, slot.n_past, { slot.id + 1 }, true);

                for (size_t i = 0; i < draft.size(); ++i) {
                    llama_batch_add(slot.batch_spec, draft[i], slot.n_past + 1 + i, { slot.id + 1 }, true);
                }

                llama_decode(ctx, slot.batch_spec, true);

                // the accepted tokens from the speculation
                const auto ids = gpt_sampler_sample_and_accept_n(slot.smpl, ctx, draft);

                slot.n_past    += ids.size();
                slot.n_decoded += ids.size();

                slot.cache_tokens.push_back(id);
                slot.cache_tokens.insert(slot.cache_tokens.end(), ids.begin(), ids.end() - 1);

                llama_kv_cache_seq_rm     (ctx, slot.id + 1, slot.n_past, -1);
                llama_send_kv_cache_seq_rm(ctx, slot.id    , slot.n_past, -1);

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok = ids[i];

                    if (!process_token(result, slot)) {
                        // release slot because of stop condition
                        slot.release();
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        break;
                    }
                }

                SRV_DBG("accepted %d/%d draft tokens\n", (int) ids.size() - 1, (int) draft.size());
                
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    json model_meta() const {
        return json {
            {"vocab_type",  llama_vocab_type    (model)},
            {"n_vocab",     llama_n_vocab       (model)},
            {"n_ctx_train", llama_n_ctx_train   (model)},
            {"n_embd",      llama_n_embd        (model)},
            {"n_params",    llama_model_n_params(model)},
            {"size",        llama_model_size    (model)},
        };
    }
};

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health" || req.path == "/v1/completions") {
        return;
    }

    LOG_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);

    LOG_DBG("request:  %s\n", req.body.c_str());
    LOG_DBG("response: %s\n", res.body.c_str());
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

int main(int argc, char ** argv) {
    // own arguments required by this example
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    gpt_init();

    // enabling this will output extra debug information in the HTTP responses from the server
    // see format_final_response_oaicompat()
    const bool verbose = params.verbosity > 9;

    // struct that contains llama context and inference
    server_context ctx_server;

    if (!params.system_prompt.empty()) {
        ctx_server.system_prompt_set(params.system_prompt);
    }

    if (params.model_alias == "unknown") {
        params.model_alias = params.model;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", gpt_params_get_system_info(params).c_str());
    LOG_INF("\n");

    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INF("Running with SSL: key = %s, cert = %s\n", params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
        svr.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INF("Running without SSL\n");
        svr.reset(new httplib::Server());
    }
#else
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_ERR("Server is built without SSL support\n");
        return 1;
    }
    svr.reset(new httplib::Server());
#endif

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr->set_default_headers({{"Server", "llama.cpp"}});

    // CORS preflight
    svr->Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        // Access-Control-Allow-Origin is already set by middleware
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods",     "POST");
        res.set_header("Access-Control-Allow-Headers",     "*");
        return res.set_content("", "text/html"); // blank response, no data
    });

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, const json & error_data) {
        json final_response {{"error", error_data}};
        res.set_content(final_response.dump(-1, ' ', false, json::error_handler_t::replace), MIMETYPE_JSON);
        res.status = json_value(error_data, "code", 500);
    };

    auto res_ok = [](httplib::Response & res, const json & data) {
        res.set_content(data.dump(-1, ' ', false, json::error_handler_t::replace), MIMETYPE_JSON);
        res.status = 200;
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        LOG_WRN("got exception: %s\n", formatted_error.dump().c_str());
        res_error(res, formatted_error);
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params.hostname;
    log_data["port"]     = std::to_string(params.port);

    if (params.api_keys.size() == 1) {
        auto key = params.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    } else if (params.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(params.api_keys.size()) + " keys loaded";
    }

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

    //
    // Middlewares
    //

    auto middleware_validate_api_key = [&params, &res_error](const httplib::Request & req, httplib::Response & res) {
        // TODO: should we apply API key to all endpoints, including "/health" and "/models"?
        static const std::unordered_set<std::string> protected_endpoints = {
            "/props",
            "/completion",
            "/completions",
            "/v1/completions",
            "/chat/completions",
            "/v1/chat/completions",
            "/infill",
            "/tokenize",
            "/detokenize",
            "/embedding",
            "/embeddings",
            "/v1/embeddings",
        };

        // If API key is not set, skip validation
        if (params.api_keys.empty()) {
            return true;
        }

        // If path is not in protected_endpoints list, skip validation
        if (protected_endpoints.find(req.path) == protected_endpoints.end()) {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");

        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
                return true; // API key is valid
            }
        }

        // API key is invalid or not provided
        res_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

        LOG_WRN("Unauthorized: Invalid API Key\n");

        return false;
    };

    auto middleware_server_state = [&res_error, &state](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
                res.status = 503;
            } else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key, &middleware_server_state](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    //
    // Route handlers (or controllers)
    //

    const auto handle_health = [&](const httplib::Request &, httplib::Response & res) {
        // error and loading states are handled by middleware
        json health = {{"status", "ok"}};
        res_ok(res, health);
    };

    const auto handle_cancel_tasks = [&](const httplib::Request & req, httplib::Response & res) {
        json request_data = json::parse(req.body);
        if (!request_data.contains("task_id") || !request_data["task_id"].is_number_integer()) {
            res.status = 400;
            res_error(res, format_error_response(
                "Invalid request: 'task_id' field is required and must be integer",
                ERROR_TYPE_INVALID_REQUEST
            ));
            return;
        }
        int task_id = request_data["task_id"].get<int>();
        ctx_server.cancel_tasks({task_id});
        json reply = {
            {"task_id", task_id},
            {"status", "cancelled"}
        };
        res_ok(res, reply);
    };

    const auto handle_slots = [&](const httplib::Request & req, httplib::Response & res) {
        if (!params.endpoint_slots) {
            res_error(res, format_error_response("This server does not support slots endpoint. Start it without `--no-slots`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task;
        task.id = ctx_server.queue_tasks.get_new_id();
        task.type = SERVER_TASK_TYPE_METRICS;

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(task, true); // high-priority task

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        // optionally return "fail_on_no_slot" error
        const int n_idle_slots = result.data.at("idle");
        if (req.has_param("fail_on_no_slot")) {
            if (n_idle_slots == 0) {
                res_error(res, format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return;
            }
        }

        res_ok(res, result.data.at("slots"));
    };

    const auto handle_metrics = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_metrics) {
            res_error(res, format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task;
        task.id = ctx_server.queue_tasks.get_new_id();
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;
        task.data.push_back({{"reset_bucket", true}});

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(task, true); // high-priority task

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        json data = result.data;

        const uint64_t n_prompt_tokens_processed = data.at("n_prompt_tokens_processed");
        const uint64_t t_prompt_processing       = data.at("t_prompt_processing");

        const uint64_t n_tokens_predicted  = data.at("n_tokens_predicted");
        const uint64_t t_tokens_generation = data.at("t_tokens_generation");

        const uint64_t n_decode_total     = data.at("n_decode_total");
        const uint64_t n_busy_slots_total = data.at("n_busy_slots_total");

        const int32_t kv_cache_used_cells = data.at("kv_cache_used_cells");

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) data.at("n_prompt_tokens_processed_total")}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) data.at("t_prompt_processing_total") / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) data.at("n_tokens_predicted_total")}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) data.at("t_tokens_generation_total") / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  n_decode_total}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) n_busy_slots_total / (float) n_decode_total}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  n_prompt_tokens_processed ? 1.e3 / t_prompt_processing * n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  n_tokens_predicted ? 1.e3 / t_tokens_generation * n_tokens_predicted : 0.}
            },{
                    {"name",  "kv_cache_usage_ratio"},
                    {"help",  "KV-cache usage. 1 means 100 percent usage."},
                    {"value",  1. * kv_cache_used_cells / params.n_ctx}
            },{
                    {"name",  "kv_cache_tokens"},
                    {"help",  "KV-cache tokens."},
                    {"value",  (uint64_t) data.at("kv_cache_tokens_count")}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of request processing."},
                    {"value",  (uint64_t) data.at("processing")}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of request deferred."},
                    {"value",  (uint64_t) data.at("deferred")}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        const int64_t t_start = data.at("t_start");
        res.set_header("Process-Start-Time-Unix", std::to_string(t_start));

        res.set_content(prometheus.str(), "text/plain; version=0.0.4");
        res.status = 200; // HTTP OK
    };

    const auto handle_slots_save = [&ctx_server, &res_error, &res_ok, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_SAVE;
        task.data = {
            { "id_slot", id_slot },
            { "filename", filename },
            { "filepath", filepath },
        };

        const int id_task = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res_ok(res, result.data);
        }
    };

    const auto handle_slots_restore = [&ctx_server, &res_error, &res_ok, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_RESTORE;
        task.data = {
            { "id_slot", id_slot },
            { "filename", filename },
            { "filepath", filepath },
        };

        const int id_task = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res_ok(res, result.data);
        }
    };

    const auto handle_slots_erase = [&ctx_server, &res_error, &res_ok](const httplib::Request & /* req */, httplib::Response & res, int id_slot) {
        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_ERASE;
        task.data = {
            { "id_slot", id_slot },
        };

        const int id_task = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res_ok(res, result.data);
        }
    };

    const auto handle_slots_action = [&params, &res_error, &handle_slots_save, &handle_slots_restore, &handle_slots_erase](const httplib::Request & req, httplib::Response & res) {
        if (params.slot_save_path.empty()) {
            res_error(res, format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        std::string id_slot_str = req.path_params.at("id_slot");
        int id_slot;

        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res_error(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::string action = req.get_param_value("action");

        if (action == "save") {
            handle_slots_save(req, res, id_slot);
        } else if (action == "restore") {
            handle_slots_restore(req, res, id_slot);
        } else if (action == "erase") {
            handle_slots_erase(req, res, id_slot);
        } else {
            res_error(res, format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        }
    };

    const auto handle_props = [&ctx_server, &res_ok](const httplib::Request &, httplib::Response & res) {
        std::string template_key = "tokenizer.chat_template", curr_tmpl;
        int32_t tlen = llama_model_meta_val_str(ctx_server.model, template_key.c_str(), nullptr, 0);
        if (tlen > 0) {
            std::vector<char> curr_tmpl_buf(tlen + 1, 0);
            if (llama_model_meta_val_str(ctx_server.model, template_key.c_str(), curr_tmpl_buf.data(), curr_tmpl_buf.size()) == tlen) {
                curr_tmpl = std::string(curr_tmpl_buf.data(), tlen);
            }
        }
        json data = {
            { "system_prompt",               ctx_server.system_prompt.c_str() },
            { "default_generation_settings", ctx_server.default_generation_settings_for_props },
            { "total_slots",                 ctx_server.params.n_parallel },
            { "chat_template",               curr_tmpl.c_str() },
        };

        res_ok(res, data);
    };

    const auto handle_completions_generic = [&ctx_server, &res_error, &res_ok](server_task_cmpl_type cmpl_type, json & data, httplib::Response & res) {
        if (ctx_server.params.embedding || ctx_server.params.reranking) {
            res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings` or `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        std::vector<server_task> tasks = ctx_server.create_tasks_cmpl(data, cmpl_type);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
            ctx_server.receive_cmpl_results(task_ids, [&](std::vector<server_task_result> & results) {
                if (results.size() == 1) {
                    // single result
                    res_ok(res, results[0].data);
                } else {
                    // multiple results (multitask)
                    json arr = json::array();
                    for (const auto & res : results) {
                        arr.push_back(res.data);
                    }
                    res_ok(res, arr);
                }
            }, [&](const json & error_data) {
                res_error(res, error_data);
            });

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        } else {
            const auto chunked_content_provider = [task_ids, &ctx_server](size_t, httplib::DataSink & sink) {
                ctx_server.receive_cmpl_results_stream(task_ids, [&](const server_task_result & result) -> bool {
                    return server_sent_event(sink, "data", result.data);
                }, [&](const json & error_data) {
                    server_sent_event(sink, "error", error_data);
                });
                sink.done();
                return false;
            };

            auto on_complete = [task_ids, &ctx_server] (bool) {
                ctx_server.queue_results.remove_waiting_task_ids(task_ids);
            };

            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        }
    };

    const auto handle_completions = [&handle_completions_generic](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        return handle_completions_generic(SERVER_TASK_CMPL_TYPE_NORMAL, data, res);
    };

    const auto handle_infill = [&handle_completions_generic](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        return handle_completions_generic(SERVER_TASK_CMPL_TYPE_INFILL, data, res);
    };

    // TODO: maybe merge this function with "handle_completions_generic"
    const auto handle_chat_completions = [&ctx_server, &params, &res_error, &res_ok, verbose](const httplib::Request & req, httplib::Response & res) {
        if (ctx_server.params.embedding || ctx_server.params.reranking) {
            res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings` or `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        json data = oaicompat_completion_params_parse(ctx_server.model, json::parse(req.body), params.chat_template);

        std::vector<server_task> tasks = ctx_server.create_tasks_cmpl(data, SERVER_TASK_CMPL_TYPE_NORMAL);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);
        const auto completion_id = gen_chatcmplid();

        if (!stream) {
            ctx_server.receive_cmpl_results(task_ids, [&](const std::vector<server_task_result> & results) {
                // multitask is never support in chat completion, there is only one result
                json result_oai = format_final_response_oaicompat(data, results[0].data, completion_id, /*.streaming =*/ false, verbose);
                res_ok(res, result_oai);
            }, [&](const json & error_data) {
                res_error(res, error_data);
            });

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        } else {
            const auto chunked_content_provider = [task_ids, &ctx_server, completion_id](size_t, httplib::DataSink & sink) {
                ctx_server.receive_cmpl_results_stream(task_ids, [&](const server_task_result & result) -> bool {
                    std::vector<json> result_array = format_partial_response_oaicompat(result.data, completion_id);
                    for (auto & event_data : result_array) {
                        if (event_data.empty()) {
                            continue; // skip the stop token
                        }
                        if (!server_sent_event(sink, "data", event_data)) {
                            return false; // connection is closed
                        }
                    }
                    return true; // ok
                }, [&](const json & error_data) {
                    server_sent_event(sink, "error", error_data);
                });
                static const std::string ev_done = "data: [DONE]\n\n";
                sink.write(ev_done.data(), ev_done.size());
                sink.done();
                return true;
            };

            auto on_complete = [task_ids, &ctx_server] (bool) {
                ctx_server.queue_results.remove_waiting_task_ids(task_ids);
            };

            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        }
    };

    const auto handle_models = [&params, &ctx_server](const httplib::Request &, httplib::Response & res) {
        json models = {
            {"object", "list"},
            {"data", {
                {
                    {"id",       params.model_alias},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     ctx_server.model_meta()}
                },
             }}
        };

        res.set_content(models.dump(), MIMETYPE_JSON);
    };

    const auto handle_tokenize = [&ctx_server, &res_ok](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool with_pieces = json_value(body, "with_pieces", false);
            std::vector<llama_token> tokens = ctx_server.tokenize(body.at("content"), add_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = llama_token_to_piece(ctx_server.ctx, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        const json data = format_tokenizer_response(tokens_response);
        res_ok(res, data);
    };

    const auto handle_detokenize = [&ctx_server, &res_ok](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const std::vector<llama_token> tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        res_ok(res, data);
    };

    const auto handle_embeddings = [&ctx_server, &res_error, &res_ok](const httplib::Request & req, httplib::Response & res) {
        // TODO: somehow clean up this checks in the future
        if (!ctx_server.params.embedding || ctx_server.params.reranking) {
            res_error(res, format_error_response("This server does not support embeddings. Start it with `--embeddings` and without `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }
        const json body = json::parse(req.body);
        bool is_openai = false;

        // an input prompt can be a string or a list of tokens (integer)
        json prompt;
        if (body.count("input") != 0) {
            is_openai = true;
            prompt = body.at("input");
        } else if (body.count("content") != 0) {
            // with "content", we only support single prompt
            prompt = std::vector<std::string>{body.at("content")};
        } else {
            res_error(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        // create and queue the task
        json responses = json::array();
        bool error = false;
        {
            std::vector<server_task> tasks = ctx_server.create_tasks_cmpl({{"prompt", prompt}}, SERVER_TASK_CMPL_TYPE_EMBEDDING);
            ctx_server.queue_results.add_waiting_tasks(tasks);
            ctx_server.queue_tasks.post(tasks);

            // get the result
            std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

            ctx_server.receive_cmpl_results(task_ids, [&](std::vector<server_task_result> & results) {
                for (const auto & res : results) {
                    responses.push_back(res.data);
                }
            }, [&](const json & error_data) {
                res_error(res, error_data);
                error = true;
            });

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        }

        if (error) {
            return;
        }

        // write JSON response
        json root = is_openai
            ? format_embeddings_response_oaicompat(body, responses)
            : responses[0];
        res_ok(res, root);
    };

    const auto handle_rerank = [&ctx_server, &res_error, &res_ok](const httplib::Request & req, httplib::Response & res) {
        if (!ctx_server.params.reranking) {
            res_error(res, format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }
        const json body = json::parse(req.body);

        // TODO: implement
        //int top_n = 1;
        //if (body.count("top_n") != 1) {
        //    top_n = body.at("top_n");
        //} else {
        //    res_error(res, format_error_response("\"top_n\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        //    return;
        //}

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res_error(res, format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        } else {
            res_error(res, format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::vector<std::string> documents = json_value(body, "documents", std::vector<std::string>());
        if (documents.empty()) {
            res_error(res, format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        // construct prompt object: array of ["query", "doc0", "doc1", ...]
        json prompt;
        prompt.push_back(query);
        for (const auto & doc : documents) {
            prompt.push_back(doc);
        }

        LOG_DBG("rerank prompt: %s\n", prompt.dump().c_str());

        // create and queue the task
        json responses = json::array();
        bool error = false;
        {
            std::vector<server_task> tasks = ctx_server.create_tasks_cmpl({{"prompt", prompt}}, SERVER_TASK_CMPL_TYPE_RERANK);
            ctx_server.queue_results.add_waiting_tasks(tasks);
            ctx_server.queue_tasks.post(tasks);

            // get the result
            std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

            ctx_server.receive_cmpl_results(task_ids, [&](std::vector<server_task_result> & results) {
                for (const auto & res : results) {
                    responses.push_back(res.data);
                }
            }, [&](const json & error_data) {
                res_error(res, error_data);
                error = true;
            });
        }

        if (error) {
            return;
        }

        // write JSON response
        json root = format_response_rerank(body, responses);
        res_ok(res, root);
    };

    const auto handle_lora_adapters_list = [&](const httplib::Request &, httplib::Response & res) {
        json result = json::array();
        for (size_t i = 0; i < ctx_server.loras.size(); ++i) {
            auto & lora = ctx_server.loras[i];
            result.push_back({
                {"id", i},
                {"path", lora.path},
                {"scale", lora.scale},
            });
        }
        res_ok(res, result);
        res.status = 200; // HTTP OK
    };

    const auto handle_lora_adapters_apply = [&](const httplib::Request & req, httplib::Response & res) {
        const std::vector<json> body = json::parse(req.body);
        int max_idx = ctx_server.loras.size();

        // clear existing value
        for (auto & lora : ctx_server.loras) {
            lora.scale = 0.0f;
        }

        // set value
        for (auto entry : body) {
            int id      = entry.at("id");
            float scale = entry.at("scale");
            if (0 <= id && id < max_idx) {
                ctx_server.loras[id].scale = scale;
            } else {
                throw std::runtime_error("invalid adapter id");
            }
        }

        server_task task;
        task.type = SERVER_TASK_TYPE_SET_LORA;
        const int id_task = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        res_ok(res, result.data);
        res.status = 200; // HTTP OK
    };

    auto handle_static_file = [](unsigned char * content, size_t len, const char * mime_type) {
        return [content, len, mime_type](const httplib::Request &, httplib::Response & res) {
            res.set_content(reinterpret_cast<const char*>(content), len, mime_type);
            return false;
        };
    };

    //
    // Router
    //

    // register static assets routes
    if (!params.public_path.empty()) {
        // Set the base directory for serving static files
        svr->set_base_dir(params.public_path);
    }

    // using embedded static files
    svr->Get("/",                           handle_static_file(index_html, index_html_len, "text/html; charset=utf-8"));
    svr->Get("/index.js",                   handle_static_file(index_js, index_js_len, "text/javascript; charset=utf-8"));
    svr->Get("/completion.js",              handle_static_file(completion_js, completion_js_len, "text/javascript; charset=utf-8"));
    svr->Get("/json-schema-to-grammar.mjs", handle_static_file(json_schema_to_grammar_mjs, json_schema_to_grammar_mjs_len, "text/javascript; charset=utf-8"));

    // add new-ui files
    svr->Get("/colorthemes.css",       handle_static_file(colorthemes_css, colorthemes_css_len, "text/css; charset=utf-8"));
    svr->Get("/style.css",             handle_static_file(style_css, style_css_len, "text/css; charset=utf-8"));
    svr->Get("/theme-beeninorder.css", handle_static_file(theme_beeninorder_css, theme_beeninorder_css_len, "text/css; charset=utf-8"));
    svr->Get("/theme-ketivah.css",     handle_static_file(theme_ketivah_css, theme_ketivah_css_len, "text/css; charset=utf-8"));
    svr->Get("/theme-mangotango.css",  handle_static_file(theme_mangotango_css, theme_mangotango_css_len, "text/css; charset=utf-8"));
    svr->Get("/theme-playground.css",  handle_static_file(theme_playground_css, theme_playground_css_len, "text/css; charset=utf-8"));
    svr->Get("/theme-polarnight.css",  handle_static_file(theme_polarnight_css, theme_polarnight_css_len, "text/css; charset=utf-8"));
    svr->Get("/theme-snowstorm.css",   handle_static_file(theme_snowstorm_css, theme_snowstorm_css_len, "text/css; charset=utf-8"));
    svr->Get("/index-new.html",        handle_static_file(index_new_html, index_new_html_len, "text/html; charset=utf-8"));
    svr->Get("/system-prompts.js",     handle_static_file(system_prompts_js, system_prompts_js_len, "text/javascript; charset=utf-8"));
    svr->Get("/prompt-formats.js",     handle_static_file(prompt_formats_js, prompt_formats_js_len, "text/javascript; charset=utf-8"));

    // register API routes
    svr->Get ("/health",              handle_health);
    svr->Get ("/metrics",             handle_metrics);
    svr->Get ("/props",               handle_props);
    svr->Get ("/v1/models",           handle_models);
    svr->Post("/completion",          handle_completions); // legacy
    svr->Post("/completions",         handle_completions);
    svr->Post("/v1/completions",      handle_completions);
    svr->Post("/chat/completions",    handle_chat_completions);
    svr->Post("/v1/chat/completions", handle_chat_completions);
    svr->Post("/infill",              handle_infill);
    svr->Post("/embedding",           handle_embeddings); // legacy
    svr->Post("/embeddings",          handle_embeddings);
    svr->Post("/v1/embeddings",       handle_embeddings);
    svr->Post("/rerank",              handle_rerank);
    svr->Post("/reranking",           handle_rerank);
    svr->Post("/v1/rerank",           handle_rerank);
    svr->Post("/v1/reranking",        handle_rerank);
    svr->Post("/tokenize",            handle_tokenize);
    svr->Post("/detokenize",          handle_detokenize);
    // LoRA adapters hotswap
    svr->Get ("/lora-adapters",       handle_lora_adapters_list);
    svr->Post("/lora-adapters",       handle_lora_adapters_apply);
    // Save & load slots
    svr->Get ("/slots",               handle_slots);
    svr->Post("/slots/:id_slot",      handle_slots_action);
    // Stop tasks
    svr->Post("/v1/cancel",           handle_cancel_tasks);

    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(params.n_threads_http);
    svr->new_task_queue = [&params] { return new httplib::ThreadPool(params.n_threads_http); };

    // clean up function, to be called before exit
    auto clean_up = [&svr]() {
        svr->stop();
        llama_backend_free();
    };

    // bind HTTP listen port, run the HTTP server in a thread
    if (!svr->bind_to_port(params.hostname, params.port)) {
        LOG_ERR("%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n", __func__, params.hostname.c_str(), params.port);
        clean_up();
        return 1;
    }
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    LOG_INF("%s: HTTP server is listening, hostname: %s, port: %d, http threads: %d\n", __func__, params.hostname.c_str(), params.port, params.n_threads_http);

    // load the model
    LOG_INF("%s: loading model\n", __func__);

    if (!ctx_server.load_model(params)) {
        char * stop_signal = nullptr;
        llama_free_sockets(ctx_server.ctx, &stop_signal);
        clean_up();
        t.join();
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return 1;
    }

    ctx_server.init();
    state.store(SERVER_STATE_READY);

    LOG_INF("%s: model loaded\n", __func__);

    // if a custom chat template is not supplied, we will use the one that comes with the model (if any)
    if (params.chat_template.empty()) {
        if (!ctx_server.validate_model_chat_template()) {
            LOG_WRN("%s: The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses\n", __func__);
            params.chat_template = "chatml";
        }
    }

    // print sample chat example to make it clear which template is used
    // LOG_INF("%s: chat template, built_in: %d, chat_example: '%s'\n", __func__, params.chat_template.empty(), llama_chat_format_example(ctx_server.model, params.chat_template).c_str());

    ctx_server.queue_tasks.on_new_task(std::bind(
                &server_context::process_single_task, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_update_slots(std::bind(
                &server_context::update_slots, &ctx_server));

    shutdown_handler = [&](int) {
        ctx_server.queue_tasks.terminate();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    LOG_INF("%s: server is listening on %s:%d - starting the main loop\n", __func__, params.hostname.c_str(), params.port);

    ctx_server.queue_tasks.start_loop();

    char * stop_signal = nullptr;
    llama_free_sockets(ctx_server.ctx, &stop_signal);

    clean_up();
    t.join();

    return 0;
}

#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "llama.h"

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    // load the model and apply lora adapter, if any
    auto mparams = llama_model_params_from_gpt_params(params);
    struct llama_context_params cparams = llama_context_params_from_gpt_params(params);

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
        return -1;
    }

    llama_model_loader * ml = llama_model_load(params.model.c_str(), model, &mparams);

    device_info dev_info;
    llama_profile_device(&dev_info, model, ml, params.gpu_mem, params.n_predict, params.n_ctx, params.cpuparams.n_threads, params.flash_attn);
    device_print_props(&dev_info, 1, model, cparams);

    llama_free_model(model);
    return 0;
}
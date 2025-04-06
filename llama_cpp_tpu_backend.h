#ifndef LLAMA_CPP_TPU_BACKEND_H
#define LLAMA_CPP_TPU_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Register the TPU backend with llama.cpp
 * This function should be called before any other llama.cpp functions
 */
void ggml_backend_tpu_register(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_CPP_TPU_BACKEND_H

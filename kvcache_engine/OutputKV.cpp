#include "OutputKV.h"

extern "C++" {
const std::vector<ggml_tensor *> *get_key_vector_cpp(const llama_context *ctx);
}
void outputKV(llama_context *ctx) {
  uint32_t num_token = get_kv_self_used(ctx);
  const std::vector<ggml_tensor *> *k_l = get_key_vector_cpp(ctx);
}

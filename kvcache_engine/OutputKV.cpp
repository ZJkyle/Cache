#include "OutputKV.h"
#include <cstdint>
#include <fstream>

extern "C++" {
const std::vector<ggml_tensor *> *get_key_vector_cpp(const llama_context *ctx);
const std::vector<ggml_tensor *> *
get_value_vector_cpp(const llama_context *ctx);
}
void outputKV(llama_context *ctx) {
  uint32_t num_token = get_kv_self_used(ctx);
  uint32_t n_head_kv = get_n_head_kv(ctx);
  uint32_t n_layer = get_n_layer(ctx);
  uint32_t n_embd_head = get_n_embd_head(ctx);
  uint32_t n_embd_gqa = n_head_kv * n_embd_head;
  const std::vector<ggml_tensor *> *k_l = get_key_vector_cpp(ctx);
  const std::vector<ggml_tensor *> *v_l = get_value_vector_cpp(ctx);

  std::ofstream outFile_k("my_prompts/output_kv/wiki_key_float.csv");
  std::ofstream outFile_v("my_prompts/output_kv/wiki_value_float.csv");

  outFile_k << "layer,channel,sequence,value\n";
  outFile_v << "layer,channel,sequence,value\n";

  for (uint32_t l = 0; l < n_layer; ++l) {
    const ggml_tensor *layer_tensor_k = (*k_l)[l];
    const ggml_tensor *layer_tensor_v = (*v_l)[l];
    uint16_t *d_k = static_cast<uint16_t *>(layer_tensor_k->data);
    uint16_t *d_v = static_cast<uint16_t *>(layer_tensor_v->data);

    for (uint32_t t = 0; t < num_token; ++t) {
      for (uint32_t e = 0; e < n_embd_gqa; ++e) {
        uint32_t idx = n_embd_gqa * t + e;
        float key = d_k[idx];
        float value = d_v[idx];

        outFile_k << l << "," << e << "," << t << ", " << key << "\n";
        outFile_v << l << "," << e << "," << t << ", " << value << "\n";
      }
    }
  }

  outFile_k.close();
  outFile_v.close();
}

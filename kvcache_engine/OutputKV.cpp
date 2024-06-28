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

  std::ofstream outFile("my_prompts/output_kv/tmp_k.csv");
  if (!outFile.is_open()) {
    std::cerr << "Unable to open file for writing\n";
    return;
  }

  for (uint32_t l = 0; l < n_layer; ++l) {
    const ggml_tensor *layer_tensor = (*k_l)[l];
    uint16_t *data = static_cast<uint16_t *>(layer_tensor->data);
    for (uint32_t t = 0; t < num_token; ++t) {
      for (uint32_t e = 0; e < n_embd_gqa; ++e) {
        uint32_t idx = n_embd_gqa * t + e;
        uint16_t value = data[idx];
        outFile << e << "," << num_token * l + t << "," << value << "\n";
      }
    }
  }
  outFile.close();
}
// #include "OutputKV.h"
// #include <cmath>
// #include <cstdint>
// #include <fstream>
// #include <iostream>
// #include <vector>
//
// extern "C++" {
// const std::vector<ggml_tensor *> *get_key_vector_cpp(const llama_context
// *ctx); const std::vector<ggml_tensor *> * get_value_vector_cpp(const
// llama_context *ctx);
// }
//
// // Function to convert uint16_t (FP16) to float (FP32)
// float fp16_to_fp32(uint16_t value) {
//   // Extract sign, exponent, and fraction from the half-precision value
//   uint16_t sign = (value >> 15) & 0x0001;
//   uint16_t exponent = (value >> 10) & 0x001F;
//   uint16_t fraction = value & 0x03FF;
//
//   // Handle special cases
//   if (exponent == 0) {
//     if (fraction == 0) {
//       // Zero
//       return sign ? -0.0f : 0.0f;
//     } else {
//       // Subnormal numbers
//       return std::ldexp(static_cast<float>(fraction) / 1024.0f, -14) *
//              (sign ? -1.0f : 1.0f);
//     }
//   } else if (exponent == 31) {
//     if (fraction == 0) {
//       // Infinity
//       return sign ? -INFINITY : INFINITY;
//     } else {
//       // NaN
//       return NAN;
//     }
//   }
//
//   // Normalized numbers
//   float result =
//       std::ldexp(1.0f + static_cast<float>(fraction) / 1024.0f, exponent -
//       15);
//   return sign ? -result : result;
// }
//
// void outputKV(llama_context *ctx) {
//   uint32_t num_token = get_kv_self_used(ctx);
//   uint32_t n_head_kv = get_n_head_kv(ctx);
//   uint32_t n_layer = get_n_layer(ctx);
//   uint32_t n_embd_head = get_n_embd_head(ctx);
//   uint32_t n_embd_gqa = n_head_kv * n_embd_head;
//   const std::vector<ggml_tensor *> *k_l = get_key_vector_cpp(ctx);
//   const std::vector<ggml_tensor *> *v_l = get_value_vector_cpp(ctx);
//
//   std::ofstream outFile("my_prompts/output_kv/tmp_k_float.csv");
//   if (!outFile.is_open()) {
//     std::cerr << "Unable to open file for writing\n";
//     return;
//   }
//
//   for (uint32_t l = 0; l < n_layer; ++l) {
//     const ggml_tensor *layer_tensor = (*k_l)[l];
//     uint16_t *data = static_cast<uint16_t *>(layer_tensor->data);
//     for (uint32_t t = 0; t < num_token; ++t) {
//       for (uint32_t e = 0; e < n_embd_gqa; ++e) {
//         uint32_t idx = n_embd_gqa * t + e;
//         uint16_t value = data[idx];
//         float fp32_value = fp16_to_fp32(value);
//
//         outFile << e << "," << num_token * l + t << "," << fp32_value <<
//         "\n";
//       }
//     }
//   }
//   outFile.close();
// }

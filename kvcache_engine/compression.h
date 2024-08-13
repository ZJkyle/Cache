#ifndef COMPRESSION_H
#define COMPRESSION_H

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"
#include <stddef.h>
#ifdef __cplusplus
#include <fcntl.h>
#include <map>
#include <stdint.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" {
#endif
struct Node;
struct HuffmanResult;
void k_quant_time_c(double time);
void k_matmul_time_c(double time);
void v_quant_time_c(double time);
void v_matmul_time_c(double time);
uint8_t *key_decoding_c(uint8_t *data, const uint8_t *code, int64_t token_id,
                        int64_t quant_group_id, int64_t layer_id);
uint8_t *value_decoding_c(uint8_t *data, const uint8_t *code,
                          int64_t quant_block_id, int64_t channel_id,
                          int64_t layer_id);
uint8_t *store_fetch_addr_key_c(int quant_group_id, int layer_id);
block_q4_roy *store_fetch_block_addr_key_c(int quant_group_id, int layer_id);
float *store_fetch_addr_value_c(int channel_id, int layer_id);
uint8_t *mulmat_fetch_addr_key_c(int64_t token_id, int64_t quant_group_id,
                                 int64_t layer_id);
block_q4_roy *mulmat_fetch_block_addr_key_c(int64_t token_id,
                                            int64_t quant_group_id,
                                            int layer_id);
float *mulmat_fetch_addr_value_c(int64_t channel_id, int64_t layer_id);
void update_token_len_key_c(int quant_group_id, int layer_id);
void update_token_len_value_c(int channel_id, int layer_id);
bool is_encoded_c(int64_t token_id, int64_t quant_group_id, int64_t layer_id);
uint32_t fetch_total_token_cnt_c(void);
uint8_t fetch_value_token_len_c(int64_t channel_id, int64_t layer_id);
block_q4_v_roy *fetch_value_block_addr_c(int64_t channel_id, int64_t layer_id);
bool enable_encoding_c(void);
#ifdef __cplusplus
}

struct Node {
  uint8_t data;
  unsigned freq;
  Node *left, *right;

  Node(uint8_t data, unsigned freq)
      : data(data), freq(freq), left(nullptr), right(nullptr) {}
};

struct HuffmanResult {
  uint8_t symbols[16];
  uint8_t codelengths[16];
};

// Functions
void key_encode(const std::map<uint8_t, std::string> &codes, block_q4_roy *addr,
                uint32_t table_idx);
void value_encode(uint8_t *data, const std::map<uint8_t, std::string> &codes,
                  block_q4_v_roy *addr, uint32_t table_idx);
Node *buildHuffmanTree(const uint8_t *data, size_t size);
Node *buildHuffmanTree(block_q4_roy *block);
std::map<uint8_t, std::string>
generateCanonicalCodes(Node *root, struct HuffmanResult &table);
uint8_t *decodeHuffman(uint8_t *data, const uint8_t *encodedData,
                       uint32_t quant_block_size,
                       const std::map<std::string, uint8_t> &huffmanCodes);
std::map<std::string, uint8_t> reconstructHuffmanCodes(uint8_t *symbols,
                                                       uint8_t *codeLengths);
void key_entrypoint_encode(uint32_t abs_token_id, int quant_group_id,
                           int layer_id);
void value_entrypoint_encode(int channel_id, int layer_id);
uint8_t *key_entrypoint_decode(uint8_t *data, const uint8_t *code,
                               int64_t abs_token_id, int64_t quant_group_id,
                               int64_t layer_id);
uint8_t *value_entrypoint_decode(uint8_t *data, const uint8_t *code,
                                 int64_t quant_block_id, int64_t channel_id,
                                 int64_t layer_id);
void v_quant(int channel_id, int layer_id);
void dump_bits();
void init_kv_cache();
void clear_kv_cache();
//
bool ensureFileSize_com(int fd, size_t size);
void *mapFileToMemory_com(const std::string &filename, size_t size, int &fd);
void unmapFileFromMemory_com(void *addr, size_t size);
void init_parameters(uint32_t n_size, uint32_t p_size, uint32_t k_en_size,
                     uint32_t v_en_size, bool enable_encoding,
                     bool enable_cache_mmap);
void cleanup_buffers();
template <typename T> void cleanup_1d_array(T *&array);
template <typename T> void init_1d_array(T *&array, size_t ne);
template <typename T> void cleanup_2d_array(T **&array, size_t rows);
template <typename T> void init_2d_array(T **&array, size_t rows, size_t cols);
#endif // __cplusplus
#endif // COMPRESSION_H

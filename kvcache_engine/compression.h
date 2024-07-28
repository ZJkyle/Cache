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
uint8_t *decoding_c(const uint8_t *code, int64_t token_id, int64_t head_id,
                    int64_t layer_id);
uint8_t *encode_fetch_addr_key_c(int head_id, int layer_id);
ggml_fp16_t *encode_fetch_addr_value_c(int channel_id, int layer_id);
uint8_t *decode_fetch_addr_key_c(int64_t token_id, int64_t head_id,
                                 int64_t layer_id);
ggml_fp16_t *decode_fetch_addr_value_c(int64_t channel_id, int64_t layer_id);
void store_key_code_addr_c(uint8_t *addr, int head_id, int layer_id);
void update_token_len_key_c(int head_id, int layer_id);
void update_token_len_value_c(int channel_id, int layer_id);
bool is_encoded_c(int64_t token_id, int64_t head_id, int64_t layer_id);
uint8_t fetch_value_token_len_c(int64_t channel_id, int64_t layer_id);
block_q4_v_roy *fetch_value_block_addr_c(int64_t channel_id, int64_t layer_id);
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
std::map<uint8_t, unsigned> generateFrequencyTable(const uint8_t *data,
                                                   size_t size);
void prepareDecodingInfo(const std::map<uint8_t, std::string> &canonicalCodes,
                         struct HuffmanResult &table);
void encode(uint8_t *data, size_t size,
            const std::map<uint8_t, std::string> &codes, uint8_t **addr,
            uint64_t table_idx);
Node *buildHuffmanTree(const std::map<uint8_t, unsigned> &freqs);
std::map<uint8_t, std::string> generateCanonicalCodes(Node *root);
uint8_t *decodeHuffman(const uint8_t *encodedData,
                       const std::map<std::string, uint8_t> &huffmanCodes);
std::map<std::string, uint8_t> reconstructHuffmanCodes(uint8_t *symbols,
                                                       uint8_t *codeLengths);
void key_entrypoint_encode(uint64_t abs_token_id, int head_id, int layer_id);
void value_entrypoint_encode(int channel_id, int layer_id);
uint8_t *entrypoint_decode(const uint8_t *code, int64_t abs_token_id,
                           int64_t head_it, int64_t layer_id);
void v_quant(int channel_id, int layer_id);
void dump_bits();
void init_value_cache();
void clear_value_cache();
//
bool ensureFileSize_com(int fd, size_t size);
void *mapFileToMemory_com(const std::string &filename, size_t size, int &fd);
void unmapFileFromMemory_com(void *addr, size_t size);
#endif // __cplusplus
#endif // COMPRESSION_H

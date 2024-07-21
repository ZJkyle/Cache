#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <stddef.h>

#ifdef __cplusplus
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

extern "C" {
#endif
struct Node;
struct HuffmanResult;
uint8_t *decoding_c(const uint8_t *code, int64_t token_id, int64_t head_id,
                    int64_t layer_id);
uint8_t *encode_fetch_addr_c(int head_id, int layer_id);
uint8_t *decode_fetch_addr_c(int64_t token_id, int64_t head_id,
                             int64_t layer_id);
void store_code_addr_c(uint8_t *addr, int head_id, int layer_id);
void update_token_len_c(int head_id, int layer_id);
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
            const std::map<uint8_t, std::string> &codes, uint8_t **addr);
Node *buildHuffmanTree(const std::map<uint8_t, unsigned> &freqs);
std::map<uint8_t, std::string> generateCanonicalCodes(Node *root);
uint8_t *decodeHuffman(const uint8_t *encodedData,
                       const std::map<std::string, uint8_t> &huffmanCodes);
std::map<std::string, uint8_t> reconstructHuffmanCodes(uint8_t *symbols,
                                                       uint8_t *codeLengths);
void entrypoint_encode(uint64_t abs_token_id, int head_id, int layer_id);
uint8_t *entrypoint_decode(const uint8_t *code, int64_t token_id,
                           int64_t head_it, int64_t layer_id);
#endif // __cplusplus
#endif // COMPRESSION_H

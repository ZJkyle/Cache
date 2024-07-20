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
uint8_t *decoding_c(const void *key);
uint8_t *fetch_addr_c(int head_id, int token_id, int layer_id);
void store_code_addr_c(uint8_t *addr, int head_id, int token_id, int layer_id);
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
  std::vector<uint8_t> encodeddata;
  std::vector<uint8_t> symbols;
  std::vector<uint8_t> codelengths;
};

// Functions
std::map<uint8_t, unsigned> generateFrequencyTable(const uint8_t *data,
                                                   size_t size);

HuffmanResult
prepareDecodingInfo(const std::map<uint8_t, std::string> &canonicalCodes);
void encode(uint8_t *data, size_t size,
            const std::map<uint8_t, std::string> &codes, uint8_t **addr);
Node *buildHuffmanTree(const std::map<uint8_t, unsigned> &freqs);
std::map<uint8_t, std::string> generateCanonicalCodes(Node *root);
std::vector<uint8_t> decode(const std::vector<uint8_t> &encoded, Node *root,
                            int total_bits);
uint8_t *decodeHuffman(const std::vector<uint8_t> &encodedData,
                       const std::map<std::string, uint8_t> &huffmanCodes);
std::map<std::string, uint8_t>
reconstructHuffmanCodes(const std::vector<uint8_t> &symbols,
                        const std::vector<uint8_t> &codeLengths);

void entrypoint_encode(uint8_t *data, size_t size, const void *key);
uint8_t *entrypoint_decode(const void *key);
#endif // __cplusplus
#endif // COMPRESSION_H

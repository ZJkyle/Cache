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
void encoding_c(uint8_t *data, size_t size, void *addr);
uint8_t *decoding_c(void *addr);

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
std::vector<uint8_t> encode(uint8_t *data, size_t size,
                            const std::map<uint8_t, std::string> &codes);
Node *buildHuffmanTree(const std::map<uint8_t, unsigned> &freqs);
std::map<uint8_t, std::string> generateCanonicalCodes(Node *root);
std::vector<uint8_t> decode(const std::vector<uint8_t> &encoded, Node *root,
                            int total_bits);
uint8_t *decodeHuffman(const std::vector<uint8_t> &encodedData,
                       const std::map<std::string, uint8_t> &huffmanCodes);
std::map<std::string, uint8_t>
reconstructHuffmanCodes(const std::vector<uint8_t> &symbols,
                        const std::vector<uint8_t> &codeLengths);

void entrypoint_encode(uint8_t *data, size_t size, void *addr);
uint8_t *entrypoint_decode(void *addr);
#endif // __cplusplus
#endif // COMPRESSION_H

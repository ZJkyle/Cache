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

#ifdef __cplusplus
}

struct Node {
  uint8_t data;
  unsigned freq;
  Node *left, *right;

  Node(uint8_t data, unsigned freq)
      : data(data), freq(freq), left(nullptr), right(nullptr) {}
};

// Functions
std::map<uint8_t, unsigned>
generateFrequencyTable(const std::vector<uint8_t> &data);
Node *buildHuffmanTree(const std::map<uint8_t, unsigned> &freqs);
std::map<uint8_t, std::string> generateCanonicalCodes(Node *root);
std::vector<uint8_t> encode(const std::vector<uint8_t> &data,
                            const std::map<uint8_t, std::string> &codes);
std::vector<uint8_t> decode(const std::vector<uint8_t> &encoded, Node *root,
                            int total_bits);

#endif // __cplusplus
#endif // COMPRESSION_H

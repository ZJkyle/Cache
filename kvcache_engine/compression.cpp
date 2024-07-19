#include <cstdint>
#ifdef __cplusplus
#include "compression.h"
#include <algorithm>
#include <bitset>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#include <unordered_map>
std::unordered_map<const void *, HuffmanResult> database;
std::mutex mapMutex;
uint32_t less_cnt = 0;
uint32_t larger_cnt = 0;
uint32_t less_original = 0;
uint32_t larger_original = 0;
std::map<uint8_t, unsigned> generateFrequencyTable(const uint8_t *data,
                                                   size_t size) {
  std::map<uint8_t, unsigned> freqs;
  for (size_t i = 0; i < size; i++) {
    freqs[data[i]]++;
  }
  return freqs;
}

Node *buildHuffmanTree(const std::map<uint8_t, unsigned> &freqs) {
  std::priority_queue<Node *, std::vector<Node *>,
                      std::function<bool(Node *, Node *)>>
      pq([](Node *l, Node *r) { return l->freq > r->freq; });
  for (auto it = freqs.begin(); it != freqs.end(); ++it) {
    pq.push(new Node(it->first, it->second));
  }
  while (pq.size() > 1) {
    Node *left = pq.top();
    pq.pop();
    Node *right = pq.top();
    pq.pop();
    Node *parent = new Node(255, left->freq + right->freq);
    parent->left = left;
    parent->right = right;
    pq.push(parent);
  }
  return pq.top();
}
std::map<uint8_t, std::string> generateCanonicalCodes(Node *root) {
  std::map<uint8_t, unsigned> codeLengths;
  std::function<void(Node *, std::string)> traverse = [&](Node *node,
                                                          std::string code) {
    if (!node) {
      return;
    }
    if (node->data != 255) {
      codeLengths[node->data] = code.length();
    }
    traverse(node->left, code + "0");
    traverse(node->right, code + "1");
  };
  traverse(root, "");

  std::map<int, std::vector<uint8_t>> sortedByLength;
  for (auto it = codeLengths.begin(); it != codeLengths.end(); ++it) {
    sortedByLength[it->second].push_back(it->first);
  }

  std::map<uint8_t, std::string> canonicalCodes;
  int code = 0;
  int previousLength = 0; // To track the previous group's code length

  for (auto &lengthGroup : sortedByLength) {
    sort(lengthGroup.second.begin(), lengthGroup.second.end());

    if (previousLength == 0) {
      code = 0; // Start at 0 for the smallest length
    } else {
      code <<= (lengthGroup.first -
                previousLength); // Adjust the 'code' to match the new length
    }

    for (auto &ch : lengthGroup.second) {
      canonicalCodes[ch] = std::bitset<32>(code).to_string().substr(
          32 - lengthGroup.first, lengthGroup.first);
      code++;
    }

    previousLength =
        lengthGroup.first; // Update previousLength after processing this group
  }
  return canonicalCodes;
}

std::vector<uint8_t> encode(uint8_t *data, size_t size,
                            const std::map<uint8_t, std::string> &codes) {
  std::string bitstring;
  for (size_t i = 0; i < size; i++) {
    bitstring += codes.at(data[i]);
  }

  std::vector<uint8_t> encoded;
  uint8_t current_byte = 0;
  int bit_count = 0;

  for (auto bit = bitstring.begin(); bit != bitstring.end(); ++bit) {
    current_byte = (current_byte << 1) | (*bit - '0');
    bit_count++;
    if (bit_count == 8) {
      encoded.push_back(current_byte);
      current_byte = 0;
      bit_count = 0;
    }
  }
  if (bit_count > 0) {
    current_byte <<= (8 - bit_count);
    encoded.push_back(current_byte);
  }
  return encoded;
}

HuffmanResult
prepareDecodingInfo(const std::map<uint8_t, std::string> &canonicalCodes) {
  HuffmanResult info;
  for (const auto &pair : canonicalCodes) {
    info.symbols.push_back(pair.first);
    info.codelengths.push_back(pair.second.length());
  }

  // Sort symbols by their code lengths (and lexicographically within the same
  // length)
  std::vector<size_t> indices(info.symbols.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
    return info.codelengths[i] < info.codelengths[j] ||
           (info.codelengths[i] == info.codelengths[j] &&
            info.symbols[i] < info.symbols[j]);
  });

  std::vector<uint8_t> sortedSymbols(info.symbols.size());
  std::vector<uint8_t> sortedCodeLengths(info.codelengths.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    sortedSymbols[i] = info.symbols[indices[i]];
    sortedCodeLengths[i] = info.codelengths[indices[i]];
  }
  info.symbols = std::move(sortedSymbols);
  info.codelengths = std::move(sortedCodeLengths);

  return info;
}

// Reconstruct the canonical Huffman codes from the symbol and code lengths
std::map<std::string, uint8_t>
reconstructHuffmanCodes(const std::vector<uint8_t> &symbols,
                        const std::vector<uint8_t> &codeLengths) {
  std::map<std::string, uint8_t> huffmanCodes;
  int code = 0;
  int previousLength = 0;

  for (size_t i = 0; i < symbols.size(); ++i) {
    if (previousLength != 0 && codeLengths[i] > previousLength) {
      code <<= (codeLengths[i] - previousLength);
    }

    std::string bitString = std::bitset<32>(code).to_string().substr(
        32 - codeLengths[i], codeLengths[i]);
    huffmanCodes[bitString] = symbols[i];
    code++;
    previousLength = codeLengths[i];
  }

  return huffmanCodes;
}

uint8_t *decodeHuffman(const std::vector<uint8_t> &encodedData,
                       const std::map<std::string, uint8_t> &huffmanCodes) {
  std::string currentCode;
  std::vector<uint8_t> tempDecoded; // Temporarily store decoded data

  for (uint8_t byte : encodedData) {
    for (int i = 7; i >= 0; --i) {
      currentCode.push_back(((byte >> i) & 1) ? '1' : '0');
      if (huffmanCodes.count(currentCode)) {
        tempDecoded.push_back(huffmanCodes.at(currentCode));
        currentCode.clear();
      }
    }
  }
  size_t decodedSize = tempDecoded.size();
  uint8_t *decodedData =
      new uint8_t[decodedSize]; // Allocate the array dynamically

  // Copy from vector to allocated array
  std::copy(tempDecoded.begin(), tempDecoded.end(), decodedData);

  return decodedData;
}

void entrypoint_encode(uint8_t *data, size_t size, const void *key) {
  auto freq = generateFrequencyTable(data, size);
  Node *root = buildHuffmanTree(freq);
  auto codes = generateCanonicalCodes(root);
  auto encoded = encode(data, size, codes);
  HuffmanResult result = prepareDecodingInfo(codes);
  result.encodeddata = encoded;
  // auto it = database.find(key);
  // if (it != database.end()) {
  //   // Handle the case where the key is not found
  //   std::cerr << "Error: Double Key\n";
  //   return;
  // }
  std::lock_guard<std::mutex> guard(mapMutex);
  database[key] = result;
  uint8_t s = result.codelengths.size() + result.encodeddata.size() +
              result.symbols.size();
  if (s < size / 2) {
    less_cnt += s;
    less_original += size / 2;
  } else {
    larger_cnt += s;
    larger_original += size / 2;
  }
}
uint8_t *entrypoint_decode(const void *key) {
  // Check if the key exists in the database
  auto it = database.find(key);
  if (it == database.end()) {
    // Handle the case where the key is not found
    std::cerr << "Error: Key " << key << " not found in database\n";
    return nullptr;
  }
  auto data = it->second;
  auto huffmanCodes = reconstructHuffmanCodes(data.symbols, data.codelengths);
  auto originalData = decodeHuffman(data.encodeddata, huffmanCodes);
  return originalData;
}

extern "C" {
void encoding_c(uint8_t *data, size_t size, const void *key) {
  entrypoint_encode(data, size, key);
}
uint8_t *decoding_c(const void *key) { return entrypoint_decode(key); }
#endif
#ifdef __cplusplus
}
#endif

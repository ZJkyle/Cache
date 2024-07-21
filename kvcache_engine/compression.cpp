#include <cstdint>
#include <cstdlib>
#ifdef __cplusplus
#include "compression.h"
#include <algorithm>
#include <bitset>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <unordered_map>

std::mutex mtx;
const uint8_t block_size = 128;
const uint8_t layers = 32;
const uint8_t heads = 8;
const uint8_t tokens = 32;
const uint32_t tmp_quantized_data_size = block_size * layers * heads * tokens;
const uint32_t backup_addr_size = layers * heads * tokens;

// 1 layer / 1 head is assigned to a thread
uint8_t cur_tokens[layers][heads] = {{0}};
uint8_t *code_addr[backup_addr_size] = {0};
uint8_t tmp_quantized_data[tmp_quantized_data_size] = {0};
// uint8_t *coded_flag[backup_addr_size] = {0};

std::unordered_map<const uint8_t *, struct HuffmanResult> huffmantable;

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

void encode(uint8_t *data, size_t size,
            const std::map<uint8_t, std::string> &codes, uint8_t **addr) {
  for (int t = 0; t < tokens; t++) {
    std::string bitstring;
    for (size_t i = 0; i < size; i++) {
      bitstring += codes.at(data[t * size + i]);
    }

    uint8_t *encoded = *(addr + t);
    uint8_t current_byte = 0;
    int bit_count = 0;
    int idx_cnt = 0;

    for (auto bit = bitstring.begin(); bit != bitstring.end(); ++bit) {
      current_byte = (current_byte << 1) | (*bit - '0');
      bit_count++;
      if (bit_count == 8) {
        encoded[idx_cnt++] = current_byte;
        current_byte = 0;
        bit_count = 0;
      }
    }
    if (bit_count > 0) {
      current_byte <<= (8 - bit_count);
      encoded[idx_cnt++] = current_byte;
    }
    if (idx_cnt > 200) {
      abort();
    }
  }
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

uint8_t *decodeHuffman(const uint8_t *encodedData,
                       const std::map<std::string, uint8_t> &huffmanCodes) {
  std::string currentCode;
  uint8_t *decodedData = (uint8_t *)malloc(block_size * sizeof(uint8_t));
  uint8_t data_cnt = 0;
  uint8_t outer_idx = 0;

  while (data_cnt < block_size) {
    uint8_t byte = encodedData[outer_idx++];
    for (int i = 7; i >= 0 && data_cnt < block_size; --i) {
      currentCode.push_back(((byte >> i) & 1) ? '1' : '0');
      if (huffmanCodes.count(currentCode)) {
        decodedData[data_cnt++] = huffmanCodes.at(currentCode);
        currentCode.clear();
      }
    }
  }
  return decodedData;
}

void entrypoint_encode(int head_id, int layer_id) {
  uint32_t q_idx = layer_id * (block_size * heads * tokens) +
                   head_id * (tokens * block_size);
  uint32_t code_idx = layer_id * (heads * tokens) + head_id * tokens;
  uint8_t *data = tmp_quantized_data + q_idx;

  auto freq = generateFrequencyTable(data, block_size * tokens);
  Node *root = buildHuffmanTree(freq);
  auto codes = generateCanonicalCodes(root);

  uint8_t **b_addr = code_addr + code_idx;

  encode(data, block_size, codes, b_addr);
  HuffmanResult result = prepareDecodingInfo(codes);
  std::lock_guard<std::mutex> lock(mtx);
  for (int i = 0; i < tokens; i++) {
    huffmantable[*(b_addr + i)] = result;
  }
}
uint8_t *entrypoint_decode(const uint8_t *code) {
  // Check if the key exists in the database
  auto it = huffmantable.find(code);
  if (it == huffmantable.end()) {
    std::cout << "key not found" << std::endl;
    abort();
  }
  auto data = it->second;
  auto huffmanCodes = reconstructHuffmanCodes(data.symbols, data.codelengths);
  auto originalData = decodeHuffman(code, huffmanCodes);
  return originalData;
}

extern "C" {
uint8_t *decoding_c(const uint8_t *code) { return entrypoint_decode(code); }
uint8_t *encode_fetch_addr_c(int head_id, int layer_id) {
  unsigned int abs_token_id = cur_tokens[layer_id][head_id];
  unsigned int index = layer_id * (heads * block_size * tokens) +
                       head_id * (block_size * tokens) +
                       abs_token_id * block_size;

  return tmp_quantized_data + index;
}
uint8_t *decode_fetch_addr_c(int64_t token_id, int64_t head_id,
                             int64_t layer_id) {
  unsigned int index = layer_id * (heads * block_size * tokens) +
                       head_id * (block_size * tokens) + token_id * block_size;

  return tmp_quantized_data + index;
}

void store_code_addr_c(uint8_t *addr, int head_id, int layer_id) {
  unsigned int abs_token_id = cur_tokens[layer_id][head_id];
  unsigned int index =
      layer_id * (heads * tokens) + head_id * tokens + abs_token_id;
  code_addr[index] = addr;
}

void update_token_len_c(int head_id, int layer_id) {
  cur_tokens[layer_id][head_id] += 1;
  if (cur_tokens[layer_id][head_id] == tokens) {
    entrypoint_encode(head_id, layer_id);
    cur_tokens[layer_id][head_id] = 0;
  }
}

#endif
#ifdef __cplusplus
}
#endif

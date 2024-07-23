#include <cstddef>
#include <cstdint>
#include <cstdlib>
#ifdef __cplusplus
#include "compression.h"
#include <algorithm>
#include <bitset>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>

const uint8_t block_size = 128;
const uint8_t layers = 32;
const uint8_t heads = 8;
const uint8_t token_group_size = 32;
const uint32_t kv_size = 512;
const uint32_t token_groups = kv_size / token_group_size;
const uint32_t tmp_quantized_data_size =
    block_size * layers * heads * token_group_size;
const uint32_t backup_addr_size = layers * heads * token_group_size;
const uint32_t huffmantable_size = layers * heads * token_groups;

// 1 layer / 1 head is assigned to a thread
uint8_t cur_tokens[layers][heads] = {{0}};
uint64_t total_tokens[layers][heads] = {{0}};
uint8_t *code_addr[backup_addr_size] = {0};
uint8_t tmp_quantized_data[tmp_quantized_data_size] = {0};
struct HuffmanResult huffmantable[huffmantable_size];
uint64_t bits_cnt[huffmantable_size] = {0};

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
            const std::map<uint8_t, std::string> &codes, uint8_t **addr,
            uint64_t table_idx) {
  for (int t = 0; t < token_group_size; t++) {
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
      // too long;
      abort();
    }
    bits_cnt[table_idx] += idx_cnt;
  }
}

void prepareDecodingInfo(const std::map<uint8_t, std::string> &canonicalCodes,
                         HuffmanResult &table) {
  // Temporary local variable to process data
  HuffmanResult info;

  // Initialize all elements to a safe value, assuming 255 is safe
  // std::fill(std::begin(info.symbols), std::end(info.symbols), 255);
  // std::fill(std::begin(info.codelengths), std::end(info.codelengths), 255);

  size_t len = 0;
  for (const auto &pair : canonicalCodes) {
    info.symbols[len] = pair.first;
    info.codelengths[len] = static_cast<uint8_t>(pair.second.length());
    ++len;
  }

  // Sort symbols by their code lengths (and lexicographically within the same
  // length)
  std::vector<size_t> indices(len); // Use actual number of elements
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
    return info.codelengths[i] < info.codelengths[j] ||
           (info.codelengths[i] == info.codelengths[j] &&
            info.symbols[i] < info.symbols[j]);
  });

  for (size_t i = 0; i < 16; ++i) {
    if (i < len) {
      table.symbols[i] = info.symbols[indices[i]];
      table.codelengths[i] = info.codelengths[indices[i]];
    } else {
      table.symbols[i] = 255;
      table.codelengths[i] = 255;
    }
  }
}
// Reconstruct the canonical Huffman codes from the symbol and code lengths
std::map<std::string, uint8_t> reconstructHuffmanCodes(uint8_t *symbols,
                                                       uint8_t *codeLengths) {
  std::map<std::string, uint8_t> huffmanCodes;
  int code = 0;
  int previousLength = 0;
  size_t len = 16;
  while (symbols[len - 1] == 255) {
    len--;
  }

  for (size_t i = 0; i < len; ++i) {
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

void entrypoint_encode(uint64_t abs_token_id, int head_id, int layer_id) {
  uint32_t q_idx = layer_id * (block_size * heads * token_group_size) +
                   head_id * (token_group_size * block_size);
  uint32_t code_idx =
      layer_id * (heads * token_group_size) + head_id * token_group_size;
  uint64_t table_idx = layer_id * (heads * token_groups) +
                       head_id * token_groups + abs_token_id / token_group_size;
  uint8_t *data = tmp_quantized_data + q_idx;

  auto freq = generateFrequencyTable(data, block_size * token_group_size);
  Node *root = buildHuffmanTree(freq);
  auto codes = generateCanonicalCodes(root);

  uint8_t **b_addr = code_addr + code_idx;
  encode(data, block_size, codes, b_addr, table_idx);
  prepareDecodingInfo(codes, huffmantable[table_idx]);
}
uint8_t *entrypoint_decode(const uint8_t *code, int64_t abs_token_id,
                           int64_t head_id, int64_t layer_id) {
  int64_t table_idx = layer_id * (heads * token_groups) +
                      head_id * token_groups + abs_token_id / token_group_size;
  auto huffmanCodes = reconstructHuffmanCodes(
      huffmantable[table_idx].symbols, huffmantable[table_idx].codelengths);
  auto originalData = decodeHuffman(code, huffmanCodes);
  return originalData;
}

void dump_bits() {

  std::ofstream outFile("kvcache_engine/dump_bits.csv");

  uint64_t total_t = total_tokens[0][0] / token_group_size;

  for (uint8_t l = 0; l < layers; l++) {
    for (uint8_t h = 0; h < heads; h++) {
      for (uint64_t g = 0; g < total_t; g++) {
        uint32_t index = l * (heads * token_groups) + h * token_groups + g;
        outFile << bits_cnt[index];
        if (g != total_t - 1) {
          outFile << ",";
        }
      }
      outFile << "\n";
    }
  }
}
extern "C" {
uint8_t *decoding_c(const uint8_t *code, int64_t token_id, int64_t head_id,
                    int64_t layer_id) {
  return entrypoint_decode(code, token_id, head_id, layer_id);
}
uint8_t *encode_fetch_addr_c(int head_id, int layer_id) {
  unsigned int abs_token_id = cur_tokens[layer_id][head_id];
  unsigned int index = layer_id * (heads * block_size * token_group_size) +
                       head_id * (block_size * token_group_size) +
                       abs_token_id * block_size;

  return tmp_quantized_data + index;
}
uint8_t *decode_fetch_addr_c(int64_t token_id, int64_t head_id,
                             int64_t layer_id) {
  unsigned int index = layer_id * (heads * block_size * token_group_size) +
                       head_id * (block_size * token_group_size) +
                       (token_id % token_group_size) * block_size;

  return tmp_quantized_data + index;
}

void store_code_addr_c(uint8_t *addr, int head_id, int layer_id) {
  unsigned int abs_token_id = cur_tokens[layer_id][head_id];
  unsigned int index = layer_id * (heads * token_group_size) +
                       head_id * token_group_size + abs_token_id;
  code_addr[index] = addr;
}

void update_token_len_c(int head_id, int layer_id) {
  cur_tokens[layer_id][head_id] += 1;
  total_tokens[layer_id][head_id] += 1;
  if (cur_tokens[layer_id][head_id] == token_group_size) {
    entrypoint_encode(total_tokens[layer_id][head_id] - 1, head_id, layer_id);
    cur_tokens[layer_id][head_id] = 0;
  }
}

bool is_encoded_c(int64_t token_id, int64_t head_id, int64_t layer_id) {
  int64_t index = layer_id * (heads * token_groups) + head_id * token_groups +
                  token_id / token_group_size;

  return !(huffmantable[index].symbols[0] == huffmantable[index].symbols[1]);
}

#endif
#ifdef __cplusplus
}
#endif

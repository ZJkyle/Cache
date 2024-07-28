#include "ggml.h"
#include <immintrin.h>
#include <xmmintrin.h>
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "compression.h"
#include "ggml-impl.h"

#ifdef __cplusplus
#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>

/////////////
// common
/////////////
const uint8_t layers = 32;
const uint32_t kv_size = 512;
void *value_cache[32];
/////////////
// Key
/////////////
const uint8_t k_quant_block_size = 128;
// encoding along sequence
const uint8_t k_encode_group_size = 32;
const uint8_t k_heads = 8;
const uint32_t k_encode_groups = kv_size / k_encode_group_size;
const uint32_t k_buffer_size =
    k_quant_block_size * layers * k_heads * k_encode_group_size;
const uint32_t k_code_addr_size = layers * k_heads * k_encode_group_size;
const uint32_t k_huffmantable_size = layers * k_heads * k_encode_groups;
//
uint8_t k_token_cnt[layers][k_heads] = {{0}};
uint8_t *k_code_addr[k_code_addr_size] = {0};
uint8_t k_buffer[k_buffer_size] = {0};
struct HuffmanResult k_huffmantable[k_huffmantable_size];
uint64_t k_bits_cnt[k_huffmantable_size] = {0};

/////////////
// Value
/////////////
const uint8_t v_quant_block_size = 32;
const uint8_t v_encode_group_size = 32;
const uint32_t v_quant_blocks = kv_size / v_quant_block_size;
const uint32_t v_channels = 1024;
const uint32_t v_buffer_size = layers * v_channels;
//
uint8_t v_token_cnt[layers][v_channels] = {{0}};
ggml_fp16_t v_buffer[v_buffer_size] = {0};
uint8_t v_quant_tmp[v_channels][v_quant_block_size];
uint32_t v_compressed_cnt[layers][v_channels] = {{0}};
/////////////
// common
/////////////
uint64_t total_tokens[layers][k_heads] = {{0}};
//
//
//

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
  for (int t = 0; t < k_encode_group_size; t++) {
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
    k_bits_cnt[table_idx] += idx_cnt;
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
  uint8_t *decodedData =
      (uint8_t *)malloc(k_quant_block_size * sizeof(uint8_t));
  uint8_t data_cnt = 0;
  uint8_t outer_idx = 0;

  while (data_cnt < k_quant_block_size) {
    uint8_t byte = encodedData[outer_idx++];
    for (int i = 7; i >= 0 && data_cnt < k_quant_block_size; --i) {
      currentCode.push_back(((byte >> i) & 1) ? '1' : '0');
      if (huffmanCodes.count(currentCode)) {
        decodedData[data_cnt++] = huffmanCodes.at(currentCode);
        currentCode.clear();
      }
    }
  }
  return decodedData;
}

void key_entrypoint_encode(uint64_t abs_token_id, int head_id, int layer_id) {
  uint32_t q_idx =
      layer_id * (k_quant_block_size * k_heads * k_encode_group_size) +
      head_id * (k_encode_group_size * k_quant_block_size);
  uint32_t code_idx = layer_id * (k_heads * k_encode_group_size) +
                      head_id * k_encode_group_size;
  uint64_t table_idx = layer_id * (k_heads * k_encode_groups) +
                       head_id * k_encode_groups +
                       abs_token_id / k_encode_group_size;
  uint8_t *data = k_buffer + q_idx;

  auto freq =
      generateFrequencyTable(data, k_quant_block_size * k_encode_group_size);
  Node *root = buildHuffmanTree(freq);
  auto codes = generateCanonicalCodes(root);

  uint8_t **b_addr = k_code_addr + code_idx;
  encode(data, k_quant_block_size, codes, b_addr, table_idx);
  prepareDecodingInfo(codes, k_huffmantable[table_idx]);
}

void value_entrypoint_encode(int channel_id, int layer_id) {

  v_quant(channel_id, layer_id);
  // uint32_t q_idx =
  //     layer_id * (k_quant_block_size * k_heads * k_encode_group_size) +
  //     head_id * (k_encode_group_size * k_quant_block_size);
  // uint32_t code_idx = layer_id * (k_heads * k_encode_group_size) +
  //                     head_id * k_encode_group_size;
  // uint64_t table_idx = layer_id * (k_heads * k_encode_groups) +
  //                      head_id * k_encode_groups +
  //                      abs_token_id / k_encode_group_size;
  // uint8_t *data = k_buffer + q_idx;
  //
  // auto freq =
  //     generateFrequencyTable(data, k_quant_block_size *
  //     k_encode_group_size);
  // Node *root = buildHuffmanTree(freq);
  // auto codes = generateCanonicalCodes(root);
  //
  // uint8_t **b_addr = k_code_addr + code_idx;
  // encode(data, k_quant_block_size, codes, b_addr, table_idx);
  // prepareDecodingInfo(codes, k_huffmantable[table_idx]);
}

uint8_t *entrypoint_decode(const uint8_t *code, int64_t abs_token_id,
                           int64_t head_id, int64_t layer_id) {
  int64_t table_idx = layer_id * (k_heads * k_encode_groups) +
                      head_id * k_encode_groups +
                      abs_token_id / k_encode_group_size;
  auto huffmanCodes = reconstructHuffmanCodes(
      k_huffmantable[table_idx].symbols, k_huffmantable[table_idx].codelengths);
  auto originalData = decodeHuffman(code, huffmanCodes);
  return originalData;
}

void v_quant(int channel_id, int layer_id) {
  // per channel
  for (int c = 0; c < v_encode_group_size; c++) {
    int s_channel_id = channel_id - c;
    uint32_t buffer_index = layer_id * (v_channels * v_quant_block_size) +
                            s_channel_id * v_quant_block_size;
    ggml_fp16_t *buffer_s_addr = v_buffer + buffer_index;
    // quantize tokens within the channel
    //
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int t = 0; t < v_quant_block_size; t++) {
      const float v = GGML_FP16_TO_FP32(buffer_s_addr[t]);
      if (v < min) {
        min = v;
      }
      if (v > max) {
        max = v;
      }
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    uint32_t s_block_addr_index = s_channel_id * v_quant_blocks +
                                  v_compressed_cnt[layer_id][s_channel_id];
    block_q4_v_roy *block_addr =
        (block_q4_v_roy *)value_cache[layer_id] + s_block_addr_index;
    (*block_addr).d = GGML_FP32_TO_FP16(d);
    (*block_addr).m = GGML_FP32_TO_FP16(min);

    uint8_t *quant_tmp_addr = v_quant_tmp[s_channel_id];

    for (int t = 0; t < v_quant_block_size; t++) {
      const float x0 = (GGML_FP16_TO_FP32(buffer_s_addr[t]) - min) * id;
      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
      quant_tmp_addr[t] = xi0;
      (*block_addr).qs[t] = xi0;
    }
    v_compressed_cnt[layer_id][s_channel_id] += 1;
  }
}

void dump_bits() {

  std::ofstream outFile("my_prompts/dump_bits.csv");

  uint64_t total_t = total_tokens[0][0] / k_encode_group_size;

  for (uint8_t l = 0; l < layers; l++) {
    for (uint8_t h = 0; h < k_heads; h++) {
      for (uint64_t g = 0; g < total_t; g++) {
        uint32_t index =
            l * (k_heads * k_encode_groups) + h * k_encode_groups + g;
        outFile << k_bits_cnt[index];
        if (g != total_t - 1) {
          outFile << ",";
        }
      }
      outFile << "\n";
    }
  }
}

void init_value_cache() {
  std::string mmap_dir = "kvcache_engine/mmap_data/";
  for (int i = 0; i < 32; i++) {
    std::string filename_v = mmap_dir + "layer_" + std::to_string(i) + "_v.dat";
    size_t file_size_v = v_quant_blocks * 1024 * sizeof(block_q4_v_roy);
    int fd_v;
    void *mapped_v = mapFileToMemory_com(filename_v, file_size_v, fd_v);
    if (mapped_v == MAP_FAILED) {
      abort();
    }
    // value cache
    block_q4_v_roy *v_data = static_cast<block_q4_v_roy *>(mapped_v);
    memset(v_data, 0, file_size_v);
    value_cache[i] = v_data;
    close(fd_v);
  }
}
void clear_value_cache() {
  size_t file_size_v = v_quant_blocks * 1024 * sizeof(block_q4_v_roy);
  for (size_t i = 0; i < 32; i++) {
    unmapFileFromMemory_com(value_cache[i], file_size_v);
  }
}

bool ensureFileSize_com(int fd, size_t size) {
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    return false;
  }

  if (static_cast<long unsigned int>(sb.st_size) < size) {
    if (ftruncate(fd, size) == -1) {
      perror("ftruncate");
      return false;
    }
  }
  return true;
}

void *mapFileToMemory_com(const std::string &filename, size_t size, int &fd) {
  fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd == -1) {
    perror("open");
    return MAP_FAILED;
  }

  if (!ensureFileSize_com(fd, size)) {
    close(fd);
    return MAP_FAILED;
  }

  void *mapped = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    perror("mmap");
    close(fd);
  }
  return mapped;
}

void unmapFileFromMemory_com(void *addr, size_t size) {
  if (munmap(addr, size) == -1) {
    perror("munmap");
    fprintf(stderr, "Failed to unmap memory at address %p with size %zu\n",
            addr, size);
  }
}
extern "C" {
uint8_t *decoding_c(const uint8_t *code, int64_t token_id, int64_t head_id,
                    int64_t layer_id) {
  return entrypoint_decode(code, token_id, head_id, layer_id);
}
uint8_t *encode_fetch_addr_key_c(int head_id, int layer_id) {
  unsigned int abs_token_id = k_token_cnt[layer_id][head_id];
  unsigned int index =
      layer_id * (k_heads * k_quant_block_size * k_encode_group_size) +
      head_id * (k_quant_block_size * k_encode_group_size) +
      abs_token_id * k_quant_block_size;

  return k_buffer + index;
}
ggml_fp16_t *encode_fetch_addr_value_c(int channel_id, int layer_id) {
  unsigned int abs_token_id = v_token_cnt[layer_id][channel_id];
  unsigned int index = layer_id * (v_channels * v_quant_block_size) +
                       channel_id * v_quant_block_size + abs_token_id;

  return v_buffer + index;
}
uint8_t *decode_fetch_addr_key_c(int64_t token_id, int64_t head_id,
                                 int64_t layer_id) {
  unsigned int index =
      layer_id * (k_heads * k_quant_block_size * k_encode_group_size) +
      head_id * (k_quant_block_size * k_encode_group_size) +
      (token_id % k_encode_group_size) * k_quant_block_size;

  return k_buffer + index;
}

ggml_fp16_t *decode_fetch_addr_value_c(int64_t channel_id, int64_t layer_id) {
  unsigned int index = layer_id * (v_channels * v_quant_block_size) +
                       channel_id * v_quant_block_size;
  return v_buffer + index;
}

void store_key_code_addr_c(uint8_t *addr, int head_id, int layer_id) {
  unsigned int abs_token_id = k_token_cnt[layer_id][head_id];
  unsigned int index = layer_id * (k_heads * k_encode_group_size) +
                       head_id * k_encode_group_size + abs_token_id;
  k_code_addr[index] = addr;
}

void update_token_len_key_c(int head_id, int layer_id) {
  k_token_cnt[layer_id][head_id] += 1;
  total_tokens[layer_id][head_id] += 1;
  if (k_token_cnt[layer_id][head_id] == k_encode_group_size) {
    key_entrypoint_encode(total_tokens[layer_id][head_id] - 1, head_id,
                          layer_id);
    k_token_cnt[layer_id][head_id] = 0;
  }
}

void update_token_len_value_c(int channel_id, int layer_id) {
  v_token_cnt[layer_id][channel_id] += 1;
  if ((channel_id % v_encode_group_size) == (v_encode_group_size - 1) &&
      v_token_cnt[layer_id][channel_id] == v_quant_block_size) {

    value_entrypoint_encode(channel_id, layer_id);

    for (int i = channel_id; i > channel_id - v_encode_group_size; i--) {
      v_token_cnt[layer_id][channel_id] = 0;
    }
  }
}

bool is_encoded_c(int64_t token_id, int64_t head_id, int64_t layer_id) {
  int64_t index = layer_id * (k_heads * k_encode_groups) +
                  head_id * k_encode_groups + token_id / k_encode_group_size;

  return !(k_huffmantable[index].symbols[0] ==
           k_huffmantable[index].symbols[1]);
}

uint8_t fetch_value_token_len_c(int64_t channel_id, int64_t layer_id) {
  return v_token_cnt[layer_id][channel_id];
}

block_q4_v_roy *fetch_value_block_addr_c(int64_t channel_id, int64_t layer_id) {
  uint32_t index = channel_id * v_quant_blocks;
  return (block_q4_v_roy *)value_cache[layer_id] + index;
}

#endif
#ifdef __cplusplus
}
#endif

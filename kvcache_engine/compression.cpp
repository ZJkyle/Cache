#include "ggml.h"
#include <sys/mman.h>
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
const uint32_t channels = 1024;
const uint8_t layers = 32;
block_q4_roy *key_cache[layers];
block_q4_v_roy *value_cache[layers];
// *parameters*
uint32_t kv_size;
uint32_t prompt_size;
bool use_encode;
bool use_cache_mmap;
/////////////

/////////////
// Key
const uint32_t k_quant_block_size = QK4_ROY;
uint32_t k_quant_blocks = channels / k_quant_block_size;
// *parameters*
uint32_t k_encode_group_size;
// *affected by parameters*
uint32_t k_encode_groups;
uint32_t k_buffer_size;
uint32_t k_huffmantable_size;
// *arrays*
uint8_t **k_token_cnt;
uint32_t **k_total_token_cnt;
uint8_t *k_buffer;
uint32_t **k_encoded_cnt;
uint32_t *k_bits_cnt;
HuffmanResult *k_huffmantable;
/////////////

/////////////
// Value
const uint32_t v_quant_block_size = QK4_V_ROY;
const uint32_t v_buffer_size = layers * channels * v_quant_block_size;
// *parameters*
uint32_t v_encode_group_size;
uint32_t v_encode_groups;
// *affected by parameters*
uint32_t v_quant_blocks;
uint32_t v_huffmantable_size;
// *arrays*
uint8_t **v_token_cnt;
float *v_buffer;
uint8_t **v_quant_tmp;
uint32_t *v_quanted_cnt;
uint32_t **v_total_quanted_cnt;
uint32_t *v_bits_cnt;
HuffmanResult *v_huffmantable;
/////////////

Node *buildHuffmanTree(const uint8_t *data, size_t size) {

  std::map<uint8_t, unsigned> freqs;
  for (size_t i = 0; i < size; i++) {
    freqs[data[i]]++;
  }

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
std::map<uint8_t, std::string> generateCanonicalCodes(Node *root,
                                                      HuffmanResult &table) {
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
  size_t len = 0;

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
      table.symbols[len] = ch;
      table.codelengths[len] = static_cast<uint8_t>(lengthGroup.first);
      code++;
      len++;
    }

    previousLength =
        lengthGroup.first; // Update previousLength after processing this group
  }

  for (size_t i = len; i < 16; ++i) {
    table.symbols[i] = 255;
    table.codelengths[i] = 255;
  }
  return canonicalCodes;
}

void key_encode(uint8_t *data, const std::map<uint8_t, std::string> &codes,
                block_q4_roy *addr, uint32_t table_idx) {
  for (uint32_t t = 0; t < k_encode_group_size; t++) {
    std::string bitstring;
    for (size_t c = 0; c < k_quant_block_size; c++) {
      bitstring += codes.at(data[t * k_quant_block_size + c]);
    }

    block_q4_roy *encoded = addr + t * k_quant_blocks;
    uint8_t current_byte = 0;
    int bit_count = 0;
    int idx_cnt = 0;

    for (auto bit = bitstring.begin(); bit != bitstring.end(); ++bit) {
      current_byte = (current_byte << 1) | (*bit - '0');
      bit_count++;
      if (bit_count == 8) {
        encoded->code[idx_cnt++] = current_byte;
        current_byte = 0;
        bit_count = 0;
      }
    }
    if (bit_count > 0) {
      current_byte <<= (8 - bit_count);
      encoded->code[idx_cnt++] = current_byte;
    }
    if (idx_cnt > (QK4_ROY / 2 * 3)) {
      // too long;
      abort();
    }
    k_bits_cnt[table_idx] += idx_cnt;
  }
}

void value_encode(uint8_t *data, const std::map<uint8_t, std::string> &codes,
                  block_q4_v_roy *addr, uint32_t table_idx) {
  for (uint32_t c = 0; c < v_encode_group_size; c++) {
    std::string bitstring;
    for (size_t t = 0; t < v_quant_block_size; t++) {
      bitstring += codes.at(data[c * v_quant_block_size + t]);
    }

    block_q4_v_roy *encoded = addr + c * v_quant_blocks;
    uint8_t current_byte = 0;
    int bit_count = 0;
    int idx_cnt = 0;

    for (auto bit = bitstring.begin(); bit != bitstring.end(); ++bit) {
      current_byte = (current_byte << 1) | (*bit - '0');
      bit_count++;
      if (bit_count == 8) {
        encoded->code[idx_cnt++] = current_byte;
        current_byte = 0;
        bit_count = 0;
      }
    }
    if (bit_count > 0) {
      current_byte <<= (8 - bit_count);
      encoded->code[idx_cnt++] = current_byte;
    }
    if (idx_cnt > (QK4_V_ROY / 2 * 3)) {
      // too long;
      abort();
    }
    v_bits_cnt[table_idx] += idx_cnt;
  }
}

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

uint8_t *decodeHuffman(uint8_t *data, const uint8_t *encodedData,
                       uint32_t quant_block_size,
                       const std::map<std::string, uint8_t> &huffmanCodes) {
  std::string currentCode;
  uint8_t data_cnt = 0;
  uint8_t outer_idx = 0;

  while (data_cnt < quant_block_size) {
    uint8_t byte = encodedData[outer_idx++];
    for (int i = 7; i >= 0 && data_cnt < quant_block_size; --i) {
      currentCode.push_back(((byte >> i) & 1) ? '1' : '0');
      if (huffmanCodes.count(currentCode)) {
        data[data_cnt++] = huffmanCodes.at(currentCode);
        currentCode.clear();
      }
    }
  }
  return data;
}

void key_entrypoint_encode(uint32_t abs_token_id, int quant_group_id,
                           int layer_id) {
  uint32_t q_idx =
      layer_id * (k_quant_block_size * k_quant_blocks * k_encode_group_size) +
      quant_group_id * (k_encode_group_size * k_quant_block_size);
  uint32_t s_token_id = abs_token_id - k_encode_group_size;
  uint32_t block_idx = s_token_id * k_quant_blocks + quant_group_id;
  block_q4_roy *b_addr = key_cache[layer_id] + block_idx;
  uint32_t table_idx = layer_id * (k_quant_blocks * k_encode_groups) +
                       quant_group_id * k_encode_groups +
                       s_token_id / k_encode_group_size;
  uint8_t *data = k_buffer + q_idx;

  Node *root = buildHuffmanTree(data, k_quant_block_size * k_encode_group_size);
  auto codes = generateCanonicalCodes(root, k_huffmantable[table_idx]);
  key_encode(data, codes, b_addr, table_idx);

  k_encoded_cnt[layer_id][quant_group_id] += 1;
}

void value_entrypoint_encode(int channel_id, int layer_id) {
  int s_channel_id = channel_id - (v_encode_group_size - 1);
  // process multiple groups of encoding within same channel group
  for (uint32_t g = 0; g < v_quanted_cnt[s_channel_id]; g++) {
    if (use_encode) {
      uint32_t s_code_idx =
          s_channel_id * v_quant_blocks +
          v_total_quanted_cnt[layer_id][s_channel_id / v_encode_group_size];
      block_q4_v_roy *b_addr = value_cache[layer_id] + s_code_idx;
      uint32_t table_idx =
          layer_id * (v_encode_groups * v_quant_blocks) +
          v_total_quanted_cnt[layer_id][s_channel_id / v_encode_group_size] *
              v_encode_groups +
          s_channel_id / v_encode_group_size;
      uint8_t *data = v_quant_tmp[g] + s_channel_id * v_quant_block_size;

      Node *root =
          buildHuffmanTree(data, v_quant_block_size * v_encode_group_size);
      auto codes = generateCanonicalCodes(root, v_huffmantable[table_idx]);
      value_encode(data, codes, b_addr, table_idx);
    }
    v_total_quanted_cnt[layer_id][s_channel_id / v_encode_group_size] += 1;
  }

  for (int c = s_channel_id; c <= channel_id; c++) {
    v_quanted_cnt[c] = 0;
  }
}

uint8_t *key_entrypoint_decode(uint8_t *data, const uint8_t *code,
                               int64_t abs_token_id, int64_t quant_group_id,
                               int64_t layer_id) {
  int64_t table_idx = layer_id * (k_quant_blocks * k_encode_groups) +
                      quant_group_id * k_encode_groups +
                      abs_token_id / k_encode_group_size;
  auto huffmanCodes = reconstructHuffmanCodes(
      k_huffmantable[table_idx].symbols, k_huffmantable[table_idx].codelengths);
  auto originalData =
      decodeHuffman(data, code, k_quant_block_size, huffmanCodes);
  return originalData;
}

uint8_t *value_entrypoint_decode(uint8_t *data, const uint8_t *code,
                                 int64_t quant_block_id, int64_t channel_id,
                                 int64_t layer_id) {
  int64_t table_idx = layer_id * (v_encode_groups * v_quant_blocks) +
                      quant_block_id * v_encode_groups +
                      channel_id / v_encode_group_size;
  auto huffmanCodes = reconstructHuffmanCodes(
      v_huffmantable[table_idx].symbols, v_huffmantable[table_idx].codelengths);
  auto originalData =
      decodeHuffman(data, code, v_quant_block_size, huffmanCodes);
  return originalData;
}

void v_quant(int channel_id, int layer_id) {
  uint32_t buffer_index = layer_id * (channels * v_quant_block_size) +
                          channel_id * v_quant_block_size;
  float *buffer_s_addr = v_buffer + buffer_index;

  float min = FLT_MAX;
  float max = -FLT_MAX;

  for (uint32_t t = 0; t < v_quant_block_size; t++) {
    const float v = buffer_s_addr[t];
    if (v < min) {
      min = v;
    }
    if (v > max) {
      max = v;
    }
  }

  const float d = (max - min) / ((1 << 4) - 1);
  const float id = d ? 1.0f / d : 0.0f;

  uint32_t s_block_addr_index =
      channel_id * v_quant_blocks +
      v_total_quanted_cnt[layer_id][channel_id / v_encode_group_size] +
      v_quanted_cnt[channel_id];
  block_q4_v_roy *block_addr = value_cache[layer_id] + s_block_addr_index;

  (*block_addr).d = GGML_FP32_TO_FP16(d);
  (*block_addr).m = GGML_FP32_TO_FP16(min);
  uint8_t *quant_addr;
  if (use_encode) {
    uint32_t q_tmp_row = v_quanted_cnt[channel_id];
    uint32_t q_tmp_col = channel_id * v_quant_block_size;
    quant_addr = &(v_quant_tmp[q_tmp_row][q_tmp_col]);
  } else {
    quant_addr = (*block_addr).code;
  }

  for (uint32_t t = 0; t < v_quant_block_size; t++) {
    const float x0 = (buffer_s_addr[t] - min) * id;
    const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
    quant_addr[t] = xi0;
  }

  v_quanted_cnt[channel_id] += 1;
}

void dump_bits() {

  std::ofstream outFile_k("my_prompts/k_compression_rate.csv");
  std::ofstream outFile_v("my_prompts/v_compression_rate.csv");

  uint32_t k_encodes = k_encoded_cnt[0][0];
  uint32_t v_quants = v_total_quanted_cnt[0][0];

  outFile_k << "layer, channel_group, token_group, compression_rate" << "\n";
  outFile_v << "layer, channel_group, token_group, compression_rate" << "\n";

  for (uint32_t l = 0; l < layers; l++) {
    for (uint32_t k_c = 0; k_c < k_quant_blocks; k_c++) {
      for (uint32_t k_t = 0; k_t < k_encodes; k_t++) {
        uint32_t index = l * (k_encode_groups * k_quant_blocks) +
                         k_c * k_encode_groups + k_t;
        outFile_k << l << ", " << k_c << ", " << k_t << ", ";
        outFile_k << ((float)k_bits_cnt[index] /
                      (k_quant_block_size * k_encode_group_size / 2));
        outFile_k << "\n";
      }
    }
    for (uint32_t v_t = 0; v_t < v_quants; v_t++) {
      for (uint32_t v_c = 0; v_c < v_encode_groups; v_c++) {
        uint32_t index = l * (v_encode_groups * v_quant_blocks) +
                         v_t * v_encode_groups + v_c;
        outFile_v << l << ", " << v_c << ", " << v_t << ", ";
        outFile_v << ((float)v_bits_cnt[index] /
                      (v_quant_block_size * v_encode_group_size / 2));
        outFile_v << "\n";
      }
    }
  }
  outFile_k.close();
  outFile_v.close();
}

template <typename T> void init_2d_array(T **&array, size_t rows, size_t cols) {
  array = new T *[rows];
  for (size_t i = 0; i < rows; ++i) {
    array[i] = new T[cols];
    memset(array[i], 0, cols * sizeof(T));
  }
}
// template
template void init_2d_array<uint8_t>(uint8_t **&array, size_t rows,
                                     size_t cols);
template void init_2d_array<uint32_t>(uint32_t **&array, size_t rows,
                                      size_t cols);

template <typename T> void cleanup_2d_array(T **&array, size_t rows) {
  for (size_t i = 0; i < rows; ++i) {
    delete[] array[i];
  }
  delete[] array;
  array = nullptr;
}
template void cleanup_2d_array<uint8_t>(uint8_t **&array, size_t rows);
template void cleanup_2d_array<uint32_t>(uint32_t **&array, size_t rows);

template <typename T> void init_1d_array(T *&array, size_t ne) {
  array = new T[ne];
  memset(array, 0, ne * sizeof(T));
}
// template
template void init_1d_array<uint8_t>(uint8_t *&array, size_t ne);
template void init_1d_array<HuffmanResult>(HuffmanResult *&array, size_t ne);
template void init_1d_array<float>(float *&array, size_t ne);
template void init_1d_array<uint32_t>(uint32_t *&array, size_t ne);

template <typename T> void cleanup_1d_array(T *&array) {
  delete[] array;
  array = nullptr;
}
// template
template void cleanup_1d_array<uint8_t>(uint8_t *&array);
template void cleanup_1d_array<HuffmanResult>(HuffmanResult *&array);
template void cleanup_1d_array<float>(float *&array);
template void cleanup_1d_array<uint32_t>(uint32_t *&array);

void init_parameters(uint32_t n_size, uint32_t p_size, uint32_t k_en_size,
                     uint32_t v_en_size, bool enable_encoding,
                     bool enable_cache_mmap) {
  // parameters
  kv_size = n_size;
  prompt_size = p_size;
  k_encode_group_size = k_en_size;
  v_encode_group_size = v_en_size;
  use_encode = enable_encoding;
  use_cache_mmap = enable_cache_mmap;
  // init key variables
  k_encode_groups = kv_size / k_encode_group_size;
  k_buffer_size = layers * channels * k_encode_group_size;
  k_huffmantable_size = layers * k_quant_blocks * k_encode_groups;
  // init key arrays
  init_2d_array<uint8_t>(k_token_cnt, layers, k_quant_blocks);
  init_2d_array<uint32_t>(k_total_token_cnt, layers, k_quant_blocks);
  init_1d_array<uint8_t>(k_buffer, k_buffer_size);
  init_1d_array<uint32_t>(k_bits_cnt, k_huffmantable_size);
  init_2d_array<uint32_t>(k_encoded_cnt, layers, k_quant_blocks);
  init_1d_array<HuffmanResult>(k_huffmantable, k_huffmantable_size);
  // init value variables
  v_quant_blocks = kv_size / v_quant_block_size;
  v_encode_groups = channels / v_encode_group_size;
  v_huffmantable_size = layers * v_encode_groups * v_quant_blocks;
  // init value arrays
  init_2d_array<uint8_t>(v_token_cnt, layers, channels);
  init_1d_array<float>(v_buffer, v_buffer_size);
  init_2d_array<uint8_t>(v_quant_tmp, prompt_size / v_quant_block_size,
                         v_quant_block_size * channels);
  init_1d_array<uint32_t>(v_quanted_cnt, channels);
  init_2d_array<uint32_t>(v_total_quanted_cnt, layers, v_encode_groups);
  init_1d_array<uint32_t>(v_bits_cnt, v_huffmantable_size);
  init_1d_array<HuffmanResult>(v_huffmantable, v_huffmantable_size);
  // init key/value cache
  init_kv_cache();
}

void cleanup_buffers() {
  // key buffers
  cleanup_2d_array<uint8_t>(k_token_cnt, layers);
  cleanup_2d_array<uint32_t>(k_total_token_cnt, layers);
  cleanup_1d_array<uint8_t>(k_buffer);
  cleanup_2d_array<uint32_t>(k_encoded_cnt, layers);
  cleanup_1d_array<uint32_t>(k_bits_cnt);
  cleanup_1d_array<HuffmanResult>(k_huffmantable);
  // value buffers
  cleanup_2d_array<uint8_t>(v_token_cnt, layers);
  cleanup_1d_array<float>(v_buffer);
  cleanup_2d_array<uint8_t>(v_quant_tmp, prompt_size / v_quant_block_size);
  cleanup_1d_array<uint32_t>(v_quanted_cnt);
  cleanup_2d_array<uint32_t>(v_total_quanted_cnt, layers);
  cleanup_1d_array<uint32_t>(v_bits_cnt);
  cleanup_1d_array<HuffmanResult>(v_huffmantable);
  // value cache
  clear_kv_cache();
}

void init_kv_cache() {
  std::string mmap_dir = "kvcache_engine/mmap_data/";
  for (int i = 0; i < layers; i++) {
    if (use_cache_mmap) {
      std::string filename_k =
          mmap_dir + "layer_" + std::to_string(i) + "_k.dat";
      std::string filename_v =
          mmap_dir + "layer_" + std::to_string(i) + "_v.dat";
      size_t file_size_k = kv_size * k_quant_blocks * sizeof(block_q4_roy);
      size_t file_size_v = channels * v_quant_blocks * sizeof(block_q4_v_roy);
      int fd_k;
      int fd_v;
      void *mapped_k = mapFileToMemory_com(filename_k, file_size_k, fd_k);
      void *mapped_v = mapFileToMemory_com(filename_v, file_size_v, fd_v);
      if (mapped_k == MAP_FAILED || mapped_v == MAP_FAILED) {
        abort();
      }
      // key cache
      block_q4_roy *k_data = static_cast<block_q4_roy *>(mapped_k);
      memset(k_data, 0, file_size_k);
      key_cache[i] = k_data;
      close(fd_k);
      // value cache
      block_q4_v_roy *v_data = static_cast<block_q4_v_roy *>(mapped_v);
      memset(v_data, 0, file_size_v);
      value_cache[i] = v_data;
      close(fd_v);
    } else {
      key_cache[i] = new block_q4_roy[kv_size * k_quant_blocks];
      value_cache[i] = new block_q4_v_roy[channels * v_quant_blocks];
    }
  }
}

void clear_kv_cache() {
  size_t file_size_v = channels * v_quant_blocks * sizeof(block_q4_v_roy);
  size_t file_size_k = kv_size * k_quant_blocks * sizeof(block_q4_roy);
  for (size_t i = 0; i < layers; i++) {
    if (use_cache_mmap) {
      unmapFileFromMemory_com(key_cache[i], file_size_k);
      unmapFileFromMemory_com(value_cache[i], file_size_v);
    } else {
      delete[] key_cache[i];
      delete[] value_cache[i];
    }
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
uint8_t *key_decoding_c(uint8_t *data, const uint8_t *code, int64_t token_id,
                        int64_t quant_group_id, int64_t layer_id) {
  return key_entrypoint_decode(data, code, token_id, quant_group_id, layer_id);
}
uint8_t *value_decoding_c(uint8_t *data, const uint8_t *code,
                          int64_t quant_block_id, int64_t channel_id,
                          int64_t layer_id) {
  return value_entrypoint_decode(data, code, quant_block_id, channel_id,
                                 layer_id);
}

uint8_t *store_fetch_addr_key_c(int quant_group_id, int layer_id) {
  unsigned int abs_token_id = k_token_cnt[layer_id][quant_group_id];
  unsigned int index =
      layer_id * (k_quant_blocks * k_quant_block_size * k_encode_group_size) +
      quant_group_id * (k_quant_block_size * k_encode_group_size) +
      abs_token_id * k_quant_block_size;

  return k_buffer + index;
}

float *store_fetch_addr_value_c(int channel_id, int layer_id) {
  unsigned int abs_token_id = v_token_cnt[layer_id][channel_id];
  unsigned int index = layer_id * (channels * v_quant_block_size) +
                       channel_id * v_quant_block_size + abs_token_id;

  return v_buffer + index;
}

uint8_t *mulmat_fetch_addr_key_c(int64_t token_id, int64_t quant_group_id,
                                 int64_t layer_id) {
  unsigned int index =
      layer_id * (k_quant_blocks * k_quant_block_size * k_encode_group_size) +
      quant_group_id * (k_quant_block_size * k_encode_group_size) +
      (token_id % k_encode_group_size) * k_quant_block_size;

  return k_buffer + index;
}

float *mulmat_fetch_addr_value_c(int64_t channel_id, int64_t layer_id) {
  unsigned int index = layer_id * (channels * v_quant_block_size) +
                       channel_id * v_quant_block_size;
  return v_buffer + index;
}

void update_token_len_key_c(int quant_group_id, int layer_id) {
  k_token_cnt[layer_id][quant_group_id] += 1;
  k_total_token_cnt[layer_id][quant_group_id] += 1;
  if (use_encode &&
      k_token_cnt[layer_id][quant_group_id] == k_encode_group_size) {
    key_entrypoint_encode(k_total_token_cnt[layer_id][quant_group_id],
                          quant_group_id, layer_id);
    k_token_cnt[layer_id][quant_group_id] = 0;
  }
}

void update_token_len_value_c(int channel_id, int layer_id) {
  v_token_cnt[layer_id][channel_id] += 1;
  if (v_token_cnt[layer_id][channel_id] == v_quant_block_size) {
    v_quant(channel_id, layer_id);
    v_token_cnt[layer_id][channel_id] = 0;
    if ((channel_id % v_encode_group_size) == (v_encode_group_size - 1) &&
        (v_quanted_cnt[channel_id] ==
         v_quanted_cnt[channel_id - (v_encode_group_size - 1)])) {
      value_entrypoint_encode(channel_id, layer_id);
    }
  }
}

bool is_encoded_c(int64_t token_id, int64_t quant_group_id, int64_t layer_id) {
  int64_t index = layer_id * (k_quant_blocks * k_encode_groups) +
                  quant_group_id * k_encode_groups +
                  token_id / k_encode_group_size;

  return !(k_huffmantable[index].symbols[0] ==
           k_huffmantable[index].symbols[1]);
}

uint32_t fetch_total_token_cnt_c(void) { return k_total_token_cnt[0][0]; }
uint8_t fetch_value_token_len_c(int64_t channel_id, int64_t layer_id) {
  return v_token_cnt[layer_id][channel_id];
}

block_q4_v_roy *fetch_value_block_addr_c(int64_t channel_id, int64_t layer_id) {
  uint32_t index = channel_id * v_quant_blocks;
  return value_cache[layer_id] + index;
}

block_q4_roy *store_fetch_block_addr_key_c(int quant_group_id, int layer_id) {
  uint32_t index =
      k_total_token_cnt[layer_id][quant_group_id] * k_quant_blocks +
      quant_group_id;
  return key_cache[layer_id] + index;
}
block_q4_roy *mulmat_fetch_block_addr_key_c(int64_t token_id,
                                            int64_t quant_group_id,
                                            int layer_id) {
  uint32_t index = token_id * k_quant_blocks + quant_group_id;
  return key_cache[layer_id] + index;
}
bool enable_encoding_c(void) { return use_encode; }
#endif
#ifdef __cplusplus
}
#endif

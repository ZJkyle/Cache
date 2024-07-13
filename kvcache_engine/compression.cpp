#ifdef __cplusplus
#include "compression.h"
#include <algorithm>
#include <bitset>
#include <functional>
#include <queue>

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
    if (!node)
      return;
    if (node->data != 255)
      codeLengths[node->data] = code.length();
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
  for (auto lengthGroup = sortedByLength.begin();
       lengthGroup != sortedByLength.end(); ++lengthGroup) {
    sort(lengthGroup->second.begin(), lengthGroup->second.end());
    for (auto ch = lengthGroup->second.begin(); ch != lengthGroup->second.end();
         ++ch) {
      canonicalCodes[*ch] = std::bitset<32>(code++).to_string().substr(
          32 - lengthGroup->first, lengthGroup->first);
    }
    code <<= 1;
  }

  return canonicalCodes;
}

std::vector<uint8_t> encode(const std::vector<uint8_t> &data,
                            const std::map<uint8_t, std::string> &codes) {
  std::string bitstring;
  for (auto it = data.begin(); it != data.end(); ++it) {
    bitstring += codes.at(*it);
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

std::vector<uint8_t> decode(const std::vector<uint8_t> &encoded, Node *root,
                            int total_bits) {
  std::vector<uint8_t> decoded;
  Node *current = root;
  int bits_processed = 0;

  for (auto byte = encoded.begin(); byte != encoded.end(); ++byte) {
    for (int i = 7; i >= 0 && bits_processed < total_bits; --i) {
      bool bit = *byte & (1 << i);
      current = bit ? current->right : current->left;
      if (current->left == nullptr && current->right == nullptr) {
        decoded.push_back(current->data);
        current = root;
      }
      bits_processed++;
    }
  }
  return decoded;
}
extern "C" {

#endif
#ifdef __cplusplus
}
#endif

#include "OutputKV.h"
#include <cstdint>
#include <fstream>

extern "C++" {
const std::vector<ggml_tensor *> *get_key_vector_cpp(const llama_context *ctx);
const std::vector<ggml_tensor *> *
get_value_vector_cpp(const llama_context *ctx);
}
void outputKV(llama_context *ctx) {
  uint32_t num_token = get_kv_self_used(ctx);
  uint32_t n_head_kv = get_n_head_kv(ctx);
  uint32_t n_layer = get_n_layer(ctx);
  uint32_t n_embd_head = get_n_embd_head(ctx);
  uint32_t n_embd_gqa = n_head_kv * n_embd_head;
  const std::vector<ggml_tensor *> *k_l = get_key_vector_cpp(ctx);
  const std::vector<ggml_tensor *> *v_l = get_value_vector_cpp(ctx);

  std::ofstream outFile("my_prompts/output_kv/rope_key_float.csv");

  for (uint32_t l = 0; l < n_layer; ++l) {
    const ggml_tensor *layer_tensor = (*k_l)[l];
    float *data = static_cast<float *>(layer_tensor->data);

    for (uint32_t t = 0; t < num_token; ++t) {
      for (uint32_t e = 0; e < n_embd_gqa; ++e) {
        uint32_t idx = n_embd_gqa * t + e;
        float value = data[idx];

        outFile << l << "," << e << "," << t << ", " << value << "\n";
      }
    }
  }
  outFile.close();
}
void testmmap() {
  const char *filename = "kvcache_engine/kvcache_mmap/data.dat";
  MyStruct array[3] = {{1, 1.1f}, {2, 2.2f}, {3, 3.3f}};
  int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0666);
  if (fd == -1) {
    perror("open");
    exit(1);
  }

  size_t filesize = 3 * sizeof(MyStruct);
  // 設置文件大小
  if (ftruncate(fd, filesize) == -1) {
    perror("ftruncate");
    close(fd);
    exit(1);
  }

  // 將數組寫入文件
  if (write(fd, array, filesize) == -1) {
    perror("write");
    close(fd);
    exit(1);
  }

  close(fd);
  fd = open(filename, O_RDWR);
  if (fd == -1) {
    perror("open");
    exit(1);
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    close(fd);
    exit(1);
  }

  filesize = sb.st_size;
  size_t num_elements = filesize / sizeof(MyStruct);

  MyStruct *mapped = static_cast<MyStruct *>(
      mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (mapped == MAP_FAILED) {
    perror("mmap");
    close(fd);
    exit(1);
  }

  // 打印數組內容
  std::cout << "Array contents before modification:\n";
  for (size_t i = 0; i < num_elements; ++i) {
    std::cout << "ID: " << mapped[i].id << ", Value: " << mapped[i].value
              << "\n";
  }

  // 修改數組中的元素
  if (num_elements > 0) {
    mapped[0].id = 42;       // 修改第一個元素的id
    mapped[0].value = 3.14f; // 修改第一個元素的value
  }

  std::cout << "Array contents after modification:\n";
  for (size_t i = 0; i < num_elements; ++i) {
    std::cout << "ID: " << mapped[i].id << ", Value: " << mapped[i].value
              << "\n";
  }

  // 解除內存映射
  if (munmap(mapped, filesize) == -1) {
    perror("munmap");
  }

  close(fd);
}

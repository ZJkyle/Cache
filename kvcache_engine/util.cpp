#include "util.h"

bool ensureFileSize(int fd, size_t size) {
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    return false;
  }

  if (sb.st_size < size) {
    if (ftruncate(fd, size) == -1) {
      perror("ftruncate");
      return false;
    }
  }
  return true;
}

void *mapFileToMemory(const std::string &filename, size_t size, int &fd) {
  fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd == -1) {
    perror("open");
    return MAP_FAILED;
  }

  if (!ensureFileSize(fd, size)) {
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

void unmapFileFromMemory(void *addr, size_t size) {
  if (munmap(addr, size) == -1) {
    perror("munmap");
    fprintf(stderr, "Failed to unmap memory at address %p with size %zu\n",
            addr, size);
  }
}

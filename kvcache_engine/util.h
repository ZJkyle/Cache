#ifndef UTIL_H
#define UTIL_H
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

bool ensureFileSize(int fd, size_t size);
void *mapFileToMemory(const std::string &filename, size_t size, int &fd);
void unmapFileFromMemory(void *addr, size_t size);
#endif // UTIL_H

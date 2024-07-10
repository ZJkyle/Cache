#ifndef OUTPUTKV_H
#define OUTPUTKV_H
#include "llama.h"
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

void outputKV(llama_context *ctx);
void testmmap();
struct MyStruct {
  int id;
  float value;
};
#endif // OUTPUTKV_H

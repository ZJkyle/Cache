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
#endif // OUTPUTKV_H

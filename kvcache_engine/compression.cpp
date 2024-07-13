extern "C" {
#include "compression.h"
}
#include <iostream>
#include <vector>

extern "C" void displayNumbers(const int *numbers, size_t size) {
  std::vector<int> nums(numbers, numbers + size);
  for (int num : nums) {
    std::cout << num << " ";
  }
  std::cout << std::endl;
}

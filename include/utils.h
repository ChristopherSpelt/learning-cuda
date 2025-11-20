#pragma once

#include <fstream>

void write_binary(const char *filename, uint32_t *data, uint32_t size) {
  std::ofstream file(filename, std::ios::binary);
  file.write(reinterpret_cast<char *>(data), size * size * sizeof(uint32_t));
  file.close();
}

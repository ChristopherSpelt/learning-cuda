#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>


inline void write_binary(std::string_view filename, uint32_t *data, uint32_t size) {
  std::ofstream file(std::string(filename), std::ios::binary);
  file.write(reinterpret_cast<char *>(data), size * size * sizeof(uint32_t));
  file.close();
}

#pragma once

#include <cstdint>
#include <fstream>
#include <filesystem>
#include <span>


inline void write_binary(const std::filesystem::path& path, std::span<const std::uint32_t> data) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(data.data()), data.size_bytes());
  file.close();
}

#include <string>
#include <unistd.h>
#include <filesystem>
#include "tempfile.h"

TempFile::TempFile(const std::string &data) {
  _path = std::filesystem::temp_directory_path().string()
    + std::filesystem::path::preferred_separator + "pocvXXXXXX";

  int fd = mkstemp(_path.data());
  write(fd, data.data(), data.size());
  close(fd);
}

TempFile::~TempFile() {
  std::remove(_path.c_str());
}

const char *TempFile::get_path() {
  return _path.c_str();
}

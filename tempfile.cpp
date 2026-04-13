#include <string>
#include <unistd.h>
#include <filesystem>
#include <phpcpp.h>
#include "tempfile.h"

TempFile::TempFile(const std::string &data) {
  _path = std::filesystem::temp_directory_path().string()
    + std::filesystem::path::preferred_separator + "pocvXXXXXX";

  int fd = mkstemp(_path.data());
  if (fd == -1) {
    std::string error = strerror(errno);
    throw Php::Exception(
        "Failed to create temporary file " + _path + ": " + error);
  }

  if (write(fd, data.data(), data.size()) == -1) {
    close(fd);
    std::string error = strerror(errno);
    throw Php::Exception(
        "Failed to write to temporary file: " + error);
  }

  close(fd);
}

TempFile::~TempFile() {
  std::remove(_path.c_str());
}

const char *TempFile::get_path() {
  return _path.c_str();
}

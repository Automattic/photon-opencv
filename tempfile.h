class TempFile {
private:
  std::string _path;

public:
  TempFile(const std::string &data);
  ~TempFile();

  const char *get_path();
};

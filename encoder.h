class Encoder {
protected:
  std::string _last_error;

public:
  Encoder() {};
  virtual bool add_frame(const Frame &frame) = 0;
  virtual bool finalize() = 0;

  virtual bool requires_original_palette() {
    return false;
  }

  std::string get_last_error() {
    return std::move(_last_error);
  }
};

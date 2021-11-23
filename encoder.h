class Encoder {
public:
  Encoder() {};
  virtual bool add_frame(const Frame &frame) = 0;
  virtual bool finalize() = 0;
  virtual bool requires_original_palette() {
    return false;
  }
};

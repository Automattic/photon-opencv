class Decoder {
public:
  Decoder() {};
  virtual ~Decoder() {};
  virtual bool loaded() = 0;
  virtual void reset() = 0;
  virtual bool get_next_frame(Frame &dst) = 0;

  virtual bool get_icc_profile(std::vector<uint8_t> &dst) {
    (void) dst;
    return false;
  }

  virtual bool provides_optimized_frames() {
    return false;
  }

  virtual bool provides_animation() {
    return false;
  }
};

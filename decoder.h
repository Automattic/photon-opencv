class Decoder {
 public:
  Decoder() {};
  virtual bool loaded() = 0;
  virtual void reset() = 0;
  virtual bool get_next_frame(cv::Mat &dst, int &delay) = 0;

  bool get_icc_profile(std::vector<uint8_t> &dst) {
    (void) dst;
    return false;
  }
};

class Libheif_Decoder : public Decoder {
protected:
  const std::string *_data;
  cv::Mat _frame;
  bool _ok;
  std::vector<uint8_t> _icc_profile;

public:
  Libheif_Decoder(const std::string *data);
  bool loaded();
  void reset();
  bool get_next_frame(Frame &dst);
  bool get_icc_profile(std::vector<uint8_t> &dst);
};

class Libheif_Decoder : public Decoder {
protected:
  const std::string *_data;
  cv::Mat _frame;
  bool _ok;
  std::vector<uint8_t> _icc_profile;
  int _thread_count;
  
public:
  Libheif_Decoder(const std::string *data, int thread_count = 1);
  bool loaded();
  void reset();
  bool get_next_frame(Frame &dst);
  bool get_icc_profile(std::vector<uint8_t> &dst);
};

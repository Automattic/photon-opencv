class OpenCV_Decoder : public Decoder {
protected:
  const std::string *_data;
  cv::Mat _frame;
  bool _ok;

public:
  OpenCV_Decoder(const std::string *data);
  bool loaded();
  void reset();
  bool get_next_frame(Frame &dst);
};

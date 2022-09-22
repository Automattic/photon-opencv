class Msfgif_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;
  MsfGifState _gif_state;
  int _delay_error;
  bool _initialized;
  cv::Mat _last_frame;

  bool _init_state(const Frame &frame);
  void _composite(cv::Mat &dst, const cv::Mat &src);

public:
  Msfgif_Encoder(const std::string &format,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output);
  bool add_frame(const Frame &frame);
  bool finalize();
};

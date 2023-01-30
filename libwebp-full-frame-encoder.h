class LibWebP_Full_Frame_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;
  std::unique_ptr<WebPAnimEncoder, decltype(&WebPAnimEncoderDelete)> _encoder;
  WebPConfig _config;
  int _timestamp;

  bool _init_encoder(const Frame &frame);

public:
  LibWebP_Full_Frame_Encoder(const std::string &,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output);
  bool add_frame(const Frame &frame);
  bool finalize();
  bool supports_multiple_frames();
};

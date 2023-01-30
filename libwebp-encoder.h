class LibWebP_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;
  std::unique_ptr<WebPMux, decltype(&WebPMuxDelete)> _mux;
  int _delay_error;
  struct WebPMuxFrameInfo _next;
  std::vector<
      std::unique_ptr<WebPMemoryWriter, void (*) (WebPMemoryWriter *)>>
      _encoded_frames;
  cv::Mat _state;
  WebPConfig _config;
  int _inserted_frames;

  static void _delete_writer(WebPMemoryWriter *writer);
  bool _init_mux(const Frame &frame);
  bool _maybe_insert_frame(bool finalizing);

public:
  LibWebP_Encoder(const std::string &format,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output);
  bool add_frame(const Frame &frame);
  bool finalize();
  bool supports_multiple_frames();
  bool supports_optimized_frames();
};

class Giflib_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;
  std::unique_ptr<GifFileType, void (*) (GifFileType *)> _gif;
  int _delay_error;
  bool _initialized;
  GraphicsControlBlock _next_gcb;
  cv::Mat _next;
  std::shared_ptr<Gif_Palette> _next_palette;
  int _next_x;
  int _next_y;
  bool _has_global_palette;

  bool _init_state(const Frame &frame);
  bool _insert_gcb();
  bool _maybe_insert_frame();
  cv::Mat _apply_palette(const Frame &frame);

public:
  Giflib_Encoder(const std::string &format,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output);
  bool add_frame(const Frame &frame);
  bool finalize();
  bool requires_original_palette();
};

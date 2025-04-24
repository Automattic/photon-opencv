class LibWebP_Decoder : public Decoder {
protected:
  const std::string *_data;
  std::unique_ptr<WebPAnimDecoder, decltype(&WebPAnimDecoderDelete)> _decoder;
  WebPAnimInfo _anim_info;
  int _last_ts;

public:
  LibWebP_Decoder(const std::string *data);
  bool loaded();
  void reset();
  bool get_next_frame(Frame &dst);
  bool provides_animation();
  std::string default_format();
  bool default_format_is_accurate();
};

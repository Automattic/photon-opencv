class Giflib_Decoder : public Decoder {
protected:
  const std::string *_data;
  std::unique_ptr<GifFileType, void (*) (GifFileType *)> _gif;
  std::pair<int, const std::string *> _offset_and_data;
  std::shared_ptr<Gif_Palette> _global_palette;
  int _loops;
  bool _can_read_loops;
  bool _may_dispose_to_previous;

  bool _has_previous_disposal();

public:
  Giflib_Decoder(const std::string *data);
  bool loaded();
  void reset();
  bool get_next_frame(Frame &dst);
  bool provides_optimized_frames();
  bool provides_animation();
  std::string default_format();
  bool default_format_is_accurate();
};

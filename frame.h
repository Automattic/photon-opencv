struct Frame {
  cv::Mat img;
  int delay;
  int x;
  int y;
  int canvas_width;
  int canvas_height;
  bool empty;
  int loops;

  enum disposal_type {
    DISPOSAL_UNDEFINED,
    DISPOSAL_NONE,
    DISPOSAL_BACKGROUND,
    DISPOSAL_PREVIOUS,
  };
  disposal_type disposal;

  enum blending_type {
    BLENDING_UNDEFINED,
    BLENDING_BLEND,
    BLENDING_NO_BLEND,
  };
  blending_type blending;

  std::shared_ptr<Gif_Palette> gif_frame_palette;
  std::shared_ptr<Gif_Palette> gif_global_palette;
  int gif_transparent_index;

  bool may_dispose_to_previous;

  Frame();
  Frame(const Frame &src);
  void reset();
};

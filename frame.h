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

  Frame() :
    empty(true),
    loops(0),
    disposal(DISPOSAL_UNDEFINED),
    blending(BLENDING_UNDEFINED),
    gif_transparent_index(-1),
    may_dispose_to_previous(false) {
  }

  Frame(const Frame &src) :
    img(src.img),
    delay(src.delay),
    x(src.x),
    y(src.y),
    canvas_width(src.canvas_width),
    canvas_height(src.canvas_height),
    empty(src.empty),
    loops(src.loops),
    disposal(src.disposal),
    blending(src.blending),
    gif_frame_palette(src.gif_frame_palette),
    gif_global_palette(src.gif_global_palette),
    gif_transparent_index(src.gif_transparent_index),
    may_dispose_to_previous(src.may_dispose_to_previous) {
  }
};

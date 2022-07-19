class LibWebP_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;
  std::unique_ptr<WebPMux, decltype(&WebPMuxDelete)> _mux;
  int _delay_error;
  struct WebPMuxFrameInfo _next;
  std::vector<WebPData> _encoded_frames;
  bool _lossless;

  bool _init_mux(const Frame &frame) {
    _mux.reset(WebPMuxNew());
    if (!_mux.get()) {
      return false;
    }

    if (WEBP_MUX_OK != WebPMuxSetCanvasSize(
          _mux.get(), frame.canvas_width, frame.canvas_height)) {
      return false;
    }

    struct WebPMuxAnimParams params;
    params.loop_count = frame.loops;
    params.bgcolor = 0;
    if (WEBP_MUX_OK != WebPMuxSetAnimationParams(_mux.get(), &params)) {
      return false;
    }

    auto lossless_option = _options->find("webp:lossless");
    if (lossless_option != _options->end()
        && "true" == lossless_option->second) {
      _lossless = true;
    }

    return true;
  }

  bool _maybe_insert_frame() {
    if (_encoded_frames.empty() && !_delay_error) {
      return true;
    }

    // Need to insert a delay, but there is no frame. Create a dummy one
    if (_encoded_frames.empty()) {
      _next.x_offset = 0;
      _next.y_offset = 0;
      _next.id = WEBP_CHUNK_ANMF;
      _next.dispose_method = WEBP_MUX_DISPOSE_NONE;
      _next.blend_method = WEBP_MUX_NO_BLEND;


      uint8_t *encoded_data;
      uint32_t pixel = 0;
      int size = WebPEncodeLosslessBGRA((uint8_t *) &pixel,
          1,
          1,
          1,
          &encoded_data);

      if (!size) {
        return 0;
      }

      WebPData encoded_frame;
      encoded_frame.bytes = encoded_data;
      encoded_frame.size = size;
      _encoded_frames.push_back(encoded_frame);
    }

    _next.duration = _delay_error;
    _next.bitstream = _encoded_frames.back();
    _delay_error = 0;

    if (WEBP_MUX_OK != WebPMuxPushFrame(_mux.get(), &_next, 0)) {
      return false;
    }

    return true;
  }

public:
  LibWebP_Encoder(const std::string &format,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output) :
      _mux(nullptr, &WebPMuxDelete) {
    _options = options;
    _quality = quality;
    _format = format;
    _output = output;

    _output->clear();
    _delay_error = 0;
    _lossless = false;
  }

  ~LibWebP_Encoder() {
    for (auto frame : _encoded_frames) {
      WebPDataClear(&frame);
    }
  }

  bool add_frame(const Frame &frame) {
    if ("webp" != _format) {
      return false;
    }

    if (!_mux.get() && !_init_mux(frame)) {
      return false;
    }

    if (!frame.empty && frame.img.empty()) {
      _delay_error += frame.delay;
      return true;
    }

    if (!_maybe_insert_frame()) {
      return false;
    }

    cv::Mat img;
    switch (frame.img.channels()) {
      case 1:
        cv::cvtColor(frame.img, img, cv::COLOR_GRAY2BGRA);
        break;

      case 2:
        {
          std::vector<cv::Mat> ga_channels, bgra_channels;
          cv::split(frame.img, ga_channels);
          cv::cvtColor(ga_channels[0], img, cv::COLOR_GRAY2BGRA);
          cv::split(img, bgra_channels);
          bgra_channels[3] = ga_channels[1];
          cv::merge(bgra_channels, img);
        }
        break;

      case 3:
        cv::cvtColor(frame.img, img, cv::COLOR_BGR2BGRA);
        break;

      default:
        img = frame.img;
        break;
    }

    // WebP only supports even X and Y
    if (frame.x & 1 || frame.y & 1) {
      cv::Mat bordered(img.rows + (frame.y & 1),
          img.cols + (frame.x & 1),
          CV_8UC4,
          cv::Vec4b(0, 0, 0, 0));

      img.copyTo(bordered(
          cv::Rect(frame.x & 1, frame.y & 1, img.cols, img.rows)));
      img = bordered;
    }

    _next.x_offset = frame.x & ~1;
    _next.y_offset = frame.y & ~1;
    _next.id = WEBP_CHUNK_ANMF;
    // TODO: dispose previous
    _next.dispose_method = Frame::DISPOSAL_BACKGROUND == frame.disposal?
      WEBP_MUX_DISPOSE_BACKGROUND : WEBP_MUX_DISPOSE_NONE;
    _next.blend_method = Frame::BLENDING_BLEND == frame.blending?
      WEBP_MUX_BLEND : WEBP_MUX_NO_BLEND;

    size_t size;
    uint8_t *encoded_data;
    if (_lossless) {
      size = WebPEncodeLosslessBGRA(img.data,
          img.cols,
          img.rows,
          img.step,
          &encoded_data);
    }
    else {
      size = WebPEncodeBGRA(img.data,
          img.cols,
          img.rows,
          img.step,
          _quality,
          &encoded_data);
    }

    if (!size) {
      return false;
    }

    WebPData encoded_frame;
    encoded_frame.bytes = encoded_data;
    encoded_frame.size = size;
    _encoded_frames.push_back(encoded_frame);

    _delay_error += frame.delay;

    return true;
  }

  bool finalize() {
    WebPData data;
    WebPDataInit(&data);

    if (!_mux.get()) {
      return false;
    }

    if (!_maybe_insert_frame()) {
      return false;
    }

    if (WEBP_MUX_OK != WebPMuxAssemble(_mux.get(), &data)) {
      return false;
    }

    _output->resize(data.size);
    memcpy(_output->data(), data.bytes, data.size);
    WebPDataClear(&data);

    return true;
  }
};

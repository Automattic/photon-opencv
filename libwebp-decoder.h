class LibWebP_Decoder : public Decoder {
 private:
  const std::string *_data;
  std::unique_ptr<WebPAnimDecoder, decltype(&WebPAnimDecoderDelete)> _decoder;
  WebPAnimInfo _anim_info;
  int _last_ts;
  
 public:
  LibWebP_Decoder(const std::string *data) :
    _decoder(nullptr, &WebPAnimDecoderDelete) {

    _data = data;
    reset();
  }

  bool loaded() {
    return _decoder.get();
  }

  void reset() {
    WebPAnimDecoderOptions options;
    WebPAnimDecoderOptionsInit(&options);
    options.color_mode = MODE_BGRA;
    options.use_threads = true;

    WebPData webp_data;
    WebPDataInit(&webp_data);
    webp_data.size = _data->size();
    webp_data.bytes = (const uint8_t *) _data->data();

    _decoder.reset(WebPAnimDecoderNew(&webp_data, &options));
    if (!_decoder.get()) {
      return;
    }

    if (!WebPAnimDecoderGetInfo(_decoder.get(), &_anim_info)) {
      _decoder.reset(nullptr);
      return;
    }

    _last_ts = 0;
  }

  bool get_next_frame(cv::Mat &dst, int &delay) {
    uint8_t *buffer;
    int ts;
    if (!WebPAnimDecoderHasMoreFrames(_decoder.get())
        || !WebPAnimDecoderGetNext(_decoder.get(), &buffer, &ts)) {
      return false;
    }

    delay = ts - _last_ts;
    _last_ts = ts;

    dst = cv::Mat(_anim_info.canvas_height,
        _anim_info.canvas_width,
        CV_8UC4);
    memcpy(dst.data,
        buffer,
        _anim_info.canvas_width * _anim_info.canvas_height * 4);

    return true;
  }
};

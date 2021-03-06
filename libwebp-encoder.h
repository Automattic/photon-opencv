class LibWebP_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;
  std::unique_ptr<WebPAnimEncoder, decltype(&WebPAnimEncoderDelete)> _encoder;
  WebPConfig _config;
  int _timestamp;

  bool _init_encoder(const Frame &frame) {
    WebPAnimEncoderOptions webp_options;
    WebPAnimEncoderOptionsInit(&webp_options);

    bool lossless = false;
    auto lossless_option = _options->find("webp:lossless");
    if (lossless_option != _options->end()
        && "true" == lossless_option->second) {
      lossless = true;
    }

    bool minimize_requested = false;
    auto minimize_option = _options->find("webp:minimize_size");
    if (minimize_option != _options->end()
        && "true" == minimize_option->second) {
      minimize_requested = true;
    }
    webp_options.minimize_size = minimize_requested;

    // Lower is faster, higher is slower, but better (range: [0-6])
    // Default to auto select, unless option is set explicitly and correctly
    int method = minimize_requested? 4 : 1;
    auto encoding_effort_option = _options->find("webp:encoding_effort");
    if (encoding_effort_option != _options->end()) {
      try {
        int encoding_effort = stoi(encoding_effort_option->second);
        if (encoding_effort >= 0 && encoding_effort <= 6) {
          method = encoding_effort;
        }
      }
      catch (const std::invalid_argument &e) {
      }
    }

    _encoder.reset(WebPAnimEncoderNew(frame.img.cols,
          frame.img.rows,
          &webp_options));
    if (!_encoder.get()) {
      return false;
    }

    WebPConfigInit(&_config);
    _config.lossless = lossless;
    // If lossless, quality indicates the effort put into compression
    _config.quality = _quality;
    _config.method = method;

    return true;
  }

public:
  LibWebP_Encoder(const std::string &format,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output) :
      _encoder(nullptr, &WebPAnimEncoderDelete) {
    _options = options;
    _quality = quality;
    _format = format;
    _output = output;

    _output->clear();
    _timestamp = 0;
  }


  bool add_frame(const Frame &frame) {
    if ("webp" != _format) {
      return false;
    }

    if (!_encoder.get() && !_init_encoder(frame)) {
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

    WebPPicture picture;
    WebPPictureInit(&picture);
    picture.use_argb = 1;
    picture.argb = (uint32_t *) img.data;
    picture.argb_stride = img.step / 4;
    picture.width = img.cols;
    picture.height = img.rows;

    if (!WebPAnimEncoderAdd(_encoder.get(), &picture, _timestamp, &_config)) {
      return false;
    }

    _timestamp += frame.delay;

    return true;
  }

  bool finalize() {
    WebPData wdata;
    WebPDataInit(&wdata);

    if (!_encoder.get() || !WebPAnimEncoderAssemble(_encoder.get(), &wdata)) {
      WebPDataClear(&wdata);
      return false;
    }

    _output->resize(wdata.size);
    memcpy(_output->data(), wdata.bytes, wdata.size);
    WebPDataClear(&wdata);

    return true;
  }
};

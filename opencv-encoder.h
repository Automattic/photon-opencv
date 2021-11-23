class OpenCV_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;

public:
  OpenCV_Encoder(const std::string &format,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output) {
    _options = options;
    _quality = quality;
    _format = format;
    _output = output;

    _output->clear();
  }

  bool add_frame(const Frame &frame) {
    if (_output->size()) {
      return false;
    }

    std::vector<int> img_parameters;
    if ("jpeg" == _format) {
      img_parameters.push_back(cv::IMWRITE_JPEG_QUALITY);
      img_parameters.push_back(_quality);
    }
    else if ("png" == _format) {
      /* GMagick uses a single scalar for storing two values:
        _compressioN_quality = compression_level*10 + filter_type */
      img_parameters.push_back(cv::IMWRITE_PNG_COMPRESSION);
      img_parameters.push_back(_quality/10);
      /* OpenCV does not support setting the filters like GMagick,
         instead we get to pick the strategy, so we ignore it
         https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html */
    }
    else if ("webp" == _format) {
      auto lossless_option = _options->find("webp:lossless");

      img_parameters.push_back(cv::IMWRITE_WEBP_QUALITY);
      if (lossless_option != _options->end()
          && "true" == lossless_option->second) {
        img_parameters.push_back(101);
      }
      else {
        img_parameters.push_back(_quality);
      }
    }

    bool encoded = false;
    try {
      encoded = cv::imencode("." + _format,
          frame.img,
          *_output,
          img_parameters);
    }
    catch (cv::Exception &e) {
      return false;
    }

    return encoded;
  }

  bool finalize() {
    return _output->size();
  }
};

#include <opencv2/opencv.hpp>
#include <gif_lib.h>
#include <webp/encode.h>
#include <webp/mux.h>

#include "gif-palette.h"
#include "frame.h"
#include "encoder.h"
#include "libwebp-full-frame-encoder.h"

bool LibWebP_Full_Frame_Encoder::_init_encoder(const Frame &frame) {
  WebPAnimEncoderOptions webp_options;
  WebPAnimEncoderOptionsInit(&webp_options);

  bool lossless = false;
  auto lossless_option = _options->find("webp:lossless");
  if (lossless_option != _options->end()
      && "true" == lossless_option->second) {
    lossless = true;
  }

  // Lower is faster, higher is slower, but better (range: [0-6])
  int method = 1;
  auto encoding_effort_option = _options->find("webp:method");
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

  int lossless_effort = 35;
  auto lossless_effort_option = _options->find("webp:lossless_effort");
  if (lossless_effort_option != _options->end()) {
    try {
      lossless_effort = stoi(lossless_effort_option->second);
    }
    catch (const std::invalid_argument &e) {
      lossless_effort = -1;
    }
    if (lossless_effort < 0 || lossless_effort > 100) {
      _last_error = "Invalid value for lossless effort option: "
        "Expected int between 0 and 100";
      return false;
    }
  }

  _encoder.reset(WebPAnimEncoderNew(frame.img.cols,
        frame.img.rows,
        &webp_options));
  if (!_encoder.get()) {
    _last_error = "Failed to instantiate encoder";
    return false;
  }

  WebPConfigInit(&_config);
  _config.lossless = lossless;
  if (lossless) {
    // Quality indicates the effort put into compression, maximize speed
    _config.quality = lossless_effort;
  }
  else {
    _config.quality = _quality;
    _config.alpha_quality = _quality;
  }
  _config.thread_level = 1;
  _config.method = method;

  return true;
}

LibWebP_Full_Frame_Encoder::LibWebP_Full_Frame_Encoder(
    const std::string &format,
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

bool LibWebP_Full_Frame_Encoder::add_frame(const Frame &frame) {
  if ("webp" != _format) {
    _last_error = "Expected webp format, got " + _format;
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
    _last_error = std::string("Failed to feed frame into encoder: ") +
      WebPAnimEncoderGetError(_encoder.get());
    return false;
  }

  _timestamp += frame.delay;

  return true;
}

bool LibWebP_Full_Frame_Encoder::finalize() {
  WebPData wdata;
  WebPDataInit(&wdata);

  if (!_encoder.get() || !WebPAnimEncoderAssemble(_encoder.get(), &wdata)) {
    WebPDataClear(&wdata);
    _last_error = std::string("Failed to assemble: ") +
      WebPAnimEncoderGetError(_encoder.get());
    return false;
  }

  _output->resize(wdata.size);
  memcpy(_output->data(), wdata.bytes, wdata.size);
  WebPDataClear(&wdata);

  return true;
}

bool LibWebP_Full_Frame_Encoder::supports_multiple_frames() {
  return true;
}

bool LibWebP_Full_Frame_Encoder::supports_optimized_frames() {
  return true;
}

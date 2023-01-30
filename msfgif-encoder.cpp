#include <opencv2/opencv.hpp>
#include <gif_lib.h>
#define MSF_USE_ALPHA
#define MSF_GIF_BGR
#define MSF_GIF_IMPL
#include <msf_gif.h>

#include "gif-palette.h"
#include "frame.h"
#include "encoder.h"
#include "msfgif-encoder.h"

bool Msfgif_Encoder::_init_state(const Frame &frame) {
  if (!msf_gif_begin(&_gif_state, frame.canvas_width, frame.canvas_height)) {
    _last_error = "Failed to initialize encoding state";
    return false;
  }

  _initialized = true;
  _last_frame = cv::Mat::zeros(frame.canvas_height,
      frame.canvas_width,
      CV_8UC4);
  return true;
}

void Msfgif_Encoder::_composite(cv::Mat &dst, const cv::Mat &src) {
  cv::Vec4b *dst_line = (cv::Vec4b *) dst.data;
  cv::Vec4b *src_line = (cv::Vec4b *) src.data;
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      cv::Vec4b &d = dst_line[j];
      cv::Vec4b &s = src_line[j];
      double ad = d[3] / 255.0;
      double as = s[3] / 255.0;
      double ar = (s[3] + d[3]*(1-as)) / 255.0;
      dst_line[j] = cv::Vec4b(
          round((s[0]*as + d[0]*ad*(1-as))/ar),
          round((s[1]*as + d[1]*ad*(1-as))/ar),
          round((s[2]*as + d[2]*ad*(1-as))/ar),
          round(ar * 255.0));
    }
    dst_line += dst.step / sizeof(cv::Vec4b);
    src_line += src.step / sizeof(cv::Vec4b);
  }
}

Msfgif_Encoder::Msfgif_Encoder(const std::string &format,
    int quality,
    const std::map<std::string, std::string> *options,
    std::vector<uint8_t> *output) {
  _options = options;
  _quality = quality;
  _format = format;
  _output = output;

  _output->clear();
  _delay_error = 0;
  _initialized = false;
}


bool Msfgif_Encoder::add_frame(const Frame &frame) {
  if ("gif" != _format) {
    _last_error = "Expected gif format, got " + _format;
    return false;
  }

  if (!_initialized && !_init_state(frame)) {
    return false;
  }


  cv::Mat img;

  if (!frame.img.empty()) {
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
  }

  cv::Mat bg;
  _last_frame.copyTo(bg);

  if (!img.empty()) {
    cv::Mat dst = bg(cv::Rect(frame.x, frame.y, img.cols, img.rows));
    if (Frame::BLENDING_BLEND == frame.blending) {
      _composite(dst, img);
    }
    else {
      img.copyTo(dst);
    }
  }
  img = bg;

  // Convert milliseconds to centiseconds
  _delay_error += frame.delay % 10;
  int delay = frame.delay / 10;
  if (_delay_error >= 10) {
    delay += 1;
    _delay_error -= 10;
  }
  if (!msf_gif_frame(&_gif_state,
        img.data,
        delay,
        16,
        img.step)) {
    _last_error = "Failed to encode frame";
    return false;
  }

  if (Frame::DISPOSAL_BACKGROUND == frame.disposal) {
    img(cv::Rect(frame.x, frame.y, frame.img.cols, frame.img.rows)) =
      cv::Vec4b(0, 0, 0, 0);
  }
  if (Frame::DISPOSAL_PREVIOUS != frame.disposal) {
    _last_frame = img;
  }

  return true;
}

bool Msfgif_Encoder::finalize() {
  if (!_initialized) {
    _last_error = "Not initialized";
    return false;
  }

  MsfGifResult result = msf_gif_end(&_gif_state);
  bool success = result.data != nullptr;
  if (success) {
    _output->resize(result.dataSize);
    memcpy(_output->data(), result.data, result.dataSize);
  }
  else {
    _last_error = "Failed to end encoding";
  }

  msf_gif_free(result);
  return success;
}

bool Msfgif_Encoder::supports_multiple_frames() {
  return true;
}

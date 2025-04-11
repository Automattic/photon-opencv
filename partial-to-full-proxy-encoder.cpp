#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <gif_lib.h>

#include "gif-palette.h"
#include "frame.h"
#include "encoder.h"
#include "partial-to-full-proxy-encoder.h"

Partial_To_Full_Proxy_Encoder::Partial_To_Full_Proxy_Encoder(
    std::shared_ptr<Encoder> internal_encoder) :
  _internal_encoder(internal_encoder) {
}
void Partial_To_Full_Proxy_Encoder::_initialize_full_frame(
    const Frame &frame) {
  _full_frame.img = cv::Mat(frame.canvas_height,
      frame.canvas_width,
      CV_8UC4,
      cv::Vec4b(0, 0, 0, 0));
  _full_frame.x = 0;
  _full_frame.y = 0;
  _full_frame.canvas_width = frame.canvas_width;
  _full_frame.canvas_height = frame.canvas_height;
  _full_frame.empty = false;
  _full_frame.delay = 0;
}

bool Partial_To_Full_Proxy_Encoder::add_frame(const Frame &frame) {
  if (!_full_frame.empty && !_internal_encoder->add_frame(_full_frame)) {
    _last_error = "Failed to insert previous frame: ";
    _last_error += _internal_encoder->get_last_error();
    return false;
  }

  if (_full_frame.empty) {
    _initialize_full_frame(frame);
  }

  // Valid empty frames might come from crops or aggressive resizing
  if (frame.empty || frame.img.empty()) {
    _full_frame.delay += frame.delay;
    return true;
  }

  if (4 != frame.img.channels()) {
    _last_error = "Unsupported number of channels";
    return false;
  }

  cv::Rect previous_roi(
      _previous.x,
      _previous.y,
      _previous.img.cols,
      _previous.img.rows
  );
  switch (_previous.disposal) {
    case Frame::DISPOSAL_BACKGROUND:
      _full_frame.img(previous_roi).setTo(0);
      break;

    case Frame::DISPOSAL_PREVIOUS:
      _previous.img.copyTo(_full_frame.img(previous_roi));
      break;

    default:
      break;
  }

  int overlap_x = std::min(_full_frame.canvas_width, std::max(0, frame.x));
  int overlap_width = std::min(_full_frame.canvas_width,
      std::max(0, frame.x + frame.img.cols)) - overlap_x;
  int overlap_y = std::min(_full_frame.canvas_height, std::max(0, frame.y));
  int overlap_height = std::min(_full_frame.canvas_height,
      std::max(0, frame.y + frame.img.rows)) - overlap_y;

  if (Frame::DISPOSAL_PREVIOUS == frame.disposal) {
    cv::Rect roi(
        overlap_x,
        overlap_y,
        overlap_width,
        overlap_height
    );
    _previous.img = _full_frame.img(roi).clone();
  }
  else if (!_previous.img.empty()) {
    _previous.img = cv::Mat();
  }
  _previous.x = overlap_x;
  _previous.y = overlap_y;
  _previous.img.cols = overlap_width;
  _previous.img.rows = overlap_height;
  _previous.disposal = frame.disposal;

  cv::Mat overlapping_img = frame.img(cv::Rect(
        overlap_y - frame.y,
        overlap_x - frame.x,
        overlap_width,
        overlap_height));
  for (int i = 0; i < overlapping_img.rows; i++) {
    cv::Vec4b *src_line = (cv::Vec4b *) overlapping_img.data
      + i * overlapping_img.step / sizeof(cv::Vec4b);
    cv::Vec4b *dst_line = (cv::Vec4b *) _full_frame.img.data
      + (overlap_y + i) * _full_frame.img.step / sizeof(cv::Vec4b) + overlap_x;
    for (int j = 0; j < overlapping_img.cols; j++) {
      if (255 == src_line[j][3]) {
        dst_line[j] = src_line[j];
      }
      else if (src_line[j][3]) {
        _last_error = "Unsupported partial transparency";
        return false;
      }
    }
  }

  _full_frame.delay = frame.delay;
  return true;
}

bool Partial_To_Full_Proxy_Encoder::finalize() {
  if (_full_frame.empty) {
    _last_error = "Can't finalize image with no frames";
    return false;
  }

  if (!_internal_encoder->add_frame(_full_frame)) {
    _last_error = "Failed to insert last frame: ";
    _last_error += _internal_encoder->get_last_error();
    return false;
  }

  if (!_internal_encoder->finalize()) {
    _last_error = "Failed to finalize: ";
    _last_error += _internal_encoder->get_last_error();
    return false;
  }

  return true;
}

bool Partial_To_Full_Proxy_Encoder::supports_optimized_frames() {
  return true;
}

bool Partial_To_Full_Proxy_Encoder::supports_multiple_frames() {
  return _internal_encoder->supports_multiple_frames();
}

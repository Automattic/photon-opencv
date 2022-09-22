#include <opencv2/opencv.hpp>
#include <gif_lib.h>
#include <webp/demux.h>

#include "gif-palette.h"
#include "frame.h"
#include "decoder.h"
#include "libwebp-decoder.h"

LibWebP_Decoder::LibWebP_Decoder(const std::string *data) :
  _decoder(nullptr, &WebPAnimDecoderDelete) {

  _data = data;
  reset();
}

bool LibWebP_Decoder::loaded() {
  return _decoder.get();
}

void LibWebP_Decoder::reset() {
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

bool LibWebP_Decoder::get_next_frame(Frame &dst) {
  uint8_t *buffer;
  int ts;

  dst.reset();

  if (!WebPAnimDecoderHasMoreFrames(_decoder.get())
      || !WebPAnimDecoderGetNext(_decoder.get(), &buffer, &ts)) {
    return false;
  }

  dst.delay = ts - _last_ts;
  _last_ts = ts;

  dst.img = cv::Mat(_anim_info.canvas_height,
      _anim_info.canvas_width,
      CV_8UC4);
  memcpy(dst.img.data,
      buffer,
      _anim_info.canvas_width * _anim_info.canvas_height * 4);
  dst.x = 0;
  dst.y = 0;
  dst.canvas_width = dst.img.cols;
  dst.canvas_height = dst.img.rows;
  dst.empty = false;

  return true;
}

bool LibWebP_Decoder::provides_animation() {
  return true;
}

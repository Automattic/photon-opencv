#include <opencv2/opencv.hpp>
#include <gif_lib.h>
#include <libheif/heif.h>

#include "gif-palette.h"
#include "frame.h"
#include "encoder.h"
#include "giflib-encoder.h"

bool Giflib_Encoder::_init_state(const Frame &frame) {
  int error = GIF_OK;
  GifFileType *raw_gif = EGifOpen(_output,
      [] (GifFileType *gif, const GifByteType *buffer, int size) {
        std::vector<uint8_t> *output =
          (std::vector<uint8_t> *) gif->UserData;
        output->insert(output->end(), buffer, buffer + size);

        return size;
      },
      &error);

  if (GIF_OK != error) {
    _last_error = "Failed to open encoder";
    return false;
  }

  _gif.reset(raw_gif);

  EGifSetGifVersion(_gif.get(), true);

  if (GIF_OK != EGifPutScreenDesc(_gif.get(),
      frame.canvas_width,
      frame.canvas_height,
      6,
      0,
      frame.gif_global_palette?
        frame.gif_global_palette->get_color_map() : nullptr)) {
    _last_error = "Failed to put screen descriptor";
    return false;
  }

  // Loops default to 1
  if (frame.loops != 1) {
    if (GIF_OK != EGifPutExtensionLeader(_gif.get(),
          APPLICATION_EXT_FUNC_CODE)) {
      _last_error = "Failed to put looping extension leader";
      return false;
    }
    if (GIF_OK != EGifPutExtensionBlock(_gif.get(), 11, "NETSCAPE2.0")) {
      _last_error = "Failed to put looping extension block";
      return false;
    }
    int loops = frame.loops >= (1 << 16)? 0 : frame.loops;
    uint8_t app_data[3] = {1, (uint8_t) loops, (uint8_t) (loops >> 8)};
    if (GIF_OK != EGifPutExtensionBlock(_gif.get(), 3, app_data)) {
      _last_error = "Failed to put looping extension block data";
      return false;
    }
    if (GIF_OK != EGifPutExtensionTrailer(_gif.get())) {
      _last_error = "Failed to put looping extension trailer";
      return false;
    }
  }

  _initialized = true;
  _has_global_palette = frame.gif_global_palette.get();
  return true;
}

bool Giflib_Encoder::_insert_gcb() {
  uint8_t extension[8];
  size_t len = EGifGCBToExtension(&_next_gcb, extension);

  return GIF_OK == EGifPutExtension(_gif.get(),
        GRAPHICS_EXT_FUNC_CODE,
        len,
        extension);
}

bool Giflib_Encoder::_maybe_insert_frame(bool finalizing) {
  if (_next.empty() && !_delay_error && (_inserted_frames || !finalizing)) {
    return true;
  }

  // Need to insert a delay, but there is no frame. Create a dummy one
  if (_next.empty()) {
    _next_gcb.TransparentColor = 0;
    _next_gcb.UserInputFlag = false;
    _next_gcb.DisposalMode = DISPOSE_DO_NOT;
    _next = cv::Mat(1, 1, CV_8UC1, (uint8_t) 0);
    _next_x = 0;
    _next_y = 0;

    if (_has_global_palette) {
      _next_palette.reset();
    }
    else {
      GifColorType colors[2];
      memset(colors, 0, sizeof(colors));

      auto raw_palette = GifMakeMapObject(2, colors);
      if (!raw_palette) {
        _last_error = "Failed to make raw palette map object";
        return false;
      }

      _next_palette.reset(new Gif_Palette(raw_palette));
    }
  }

  _next_gcb.DelayTime = _delay_error/10;
  _delay_error %= 10;
  _insert_gcb();

  if (GIF_OK != EGifPutImageDesc(_gif.get(),
        _next_x,
        _next_y,
        _next.cols,
        _next.rows,
        false,
        _next_palette? _next_palette->get_color_map() : nullptr)) {
    _last_error = "Failed to put image descriptor";
    return false;
  }

  uint8_t *line = _next.data;
  for (int i = 0; i < _next.rows; i++) {
    if (GIF_OK != EGifPutLine(_gif.get(), line, _next.cols)) {
      _last_error = "Failed to put line" + i;
      return false;
    }
    line += _next.step / sizeof(uint8_t);
  }

  _next = cv::Mat();
  _inserted_frames++;

  return true;
}

cv::Mat Giflib_Encoder::_apply_palette(const Frame &frame) {
  Gif_Palette *palette = frame.gif_frame_palette?
    frame.gif_frame_palette.get() : frame.gif_global_palette.get();

  if (!palette) {
    return cv::Mat();
  }

  cv::Mat src;
  switch (frame.img.channels()) {
    case 1:
      cv::cvtColor(frame.img, src, cv::COLOR_GRAY2BGRA);
      break;

    case 2:
      {
        std::vector<cv::Mat> ga_channels, bgra_channels;
        cv::split(frame.img, ga_channels);
        cv::cvtColor(ga_channels[0], src, cv::COLOR_GRAY2BGRA);
        cv::split(src, bgra_channels);
        bgra_channels[3] = ga_channels[1];
        cv::merge(bgra_channels, src);
      }
      break;

    case 3:
      cv::cvtColor(frame.img, src, cv::COLOR_BGR2BGRA);
      break;

    default:
      src = frame.img;
      break;
  }

  cv::Mat dst(src.rows, src.cols, CV_8UC1);
  uint8_t *dst_line = dst.data;
  cv::Vec4b *src_line = (cv::Vec4b *) src.data;
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      if (src_line[j][3] < 128 && frame.gif_transparent_index >= 0) {
        dst_line[j] = frame.gif_transparent_index;
        continue;
      }

      // Should throw exception if color is not in palette
      dst_line[j] = palette->get_index(src_line[j][0],
          src_line[j][1],
          src_line[j][2],
          frame.gif_transparent_index);
    }
    dst_line += dst.step / sizeof(uint8_t);
    src_line += src.step / sizeof(cv::Vec4b);
  }

  return dst;
}

Giflib_Encoder::Giflib_Encoder(const std::string &format,
    int quality,
    const std::map<std::string, std::string> *options,
    std::vector<uint8_t> *output) :
  _gif(nullptr, [] (GifFileType *gif) { EGifCloseFile(gif, nullptr); }) {

  _options = options;
  _quality = quality;
  _format = format;
  _output = output;

  _output->clear();
  _delay_error = 0;
  _initialized = false;
  _has_global_palette = false;
  _inserted_frames = 0;
}


bool Giflib_Encoder::add_frame(const Frame &frame) {
  if ("gif" != _format) {
    _last_error = "Expected gif format, got " + _format;
    return false;
  }

  if (!_initialized && !_init_state(frame)) {
    return false;
  }

  if (!frame.empty && frame.img.empty()) {
    _delay_error += frame.delay;
    return true;
  }

  if (!_maybe_insert_frame(false)) {
    return false;
  }

  _next = _apply_palette(frame);
  _delay_error += frame.delay;

  switch (frame.disposal) {
    case Frame::DISPOSAL_BACKGROUND:
      _next_gcb.DisposalMode = DISPOSE_BACKGROUND;
      break;

    case Frame::DISPOSAL_PREVIOUS:
      _next_gcb.DisposalMode = DISPOSE_PREVIOUS;
      break;

    default:
      _next_gcb.DisposalMode = DISPOSE_DO_NOT;
      break;
  }
  _next_gcb.TransparentColor = frame.gif_transparent_index >= 0?
    frame.gif_transparent_index : NO_TRANSPARENT_COLOR;
  _next_gcb.UserInputFlag = false;

  _next_palette = frame.gif_frame_palette;
  _next_x = frame.x;
  _next_y = frame.y;

  return true;
}

bool Giflib_Encoder::finalize() {
  if (!_initialized) {
    _last_error = "Tried to finalize uninitialized image";
    return false;
  }

  if (!_maybe_insert_frame(true)) {
    return false;
  }

  // Release so we can run close manually and capture possible errors
  if (GIF_OK != EGifCloseFile(_gif.release(), nullptr)) {
    _last_error = "Failed to close file";
    return false;
  }

  return true;
}

bool Giflib_Encoder::requires_original_palette() {
  return true;
}

bool Giflib_Encoder::supports_multiple_frames() {
  return true;
}

bool Giflib_Encoder::supports_optimized_frames() {
  return true;
}

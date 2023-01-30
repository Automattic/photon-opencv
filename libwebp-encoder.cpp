#include <opencv2/opencv.hpp>
#include <gif_lib.h>
#include <webp/encode.h>
#include <webp/mux.h>

#include "gif-palette.h"
#include "frame.h"
#include "encoder.h"
#include "libwebp-encoder.h"

void LibWebP_Encoder::_delete_writer(WebPMemoryWriter *writer) {
  WebPMemoryWriterClear(writer);
  delete writer;
}

bool LibWebP_Encoder::_init_mux(const Frame &frame) {
  _mux.reset(WebPMuxNew());
  if (!_mux.get()) {
    _last_error = "Failed to initialize WebPMux";
    return false;
  }

  if (WEBP_MUX_OK != WebPMuxSetCanvasSize(
        _mux.get(), frame.canvas_width, frame.canvas_height)) {
    _last_error = "Failed to set canvas size";
    return false;
  }

  struct WebPMuxAnimParams params;
  params.loop_count = frame.loops;
  params.bgcolor = 0;
  if (WEBP_MUX_OK != WebPMuxSetAnimationParams(_mux.get(), &params)) {
    _last_error = "Failed to set animation parameters";
    return false;
  }

  bool lossless = false;
  auto lossless_option = _options->find("webp:lossless");
  if (lossless_option != _options->end()
      && "true" == lossless_option->second) {
    lossless = true;
  }

  // Lower is faster, higher is slower, but better (range: [0-6])
  int method = 1;
  auto method_option = _options->find("webp:method");
  if (method_option != _options->end()) {
    try {
      method = stoi(method_option->second);
    }
    catch (const std::invalid_argument &e) {
      method = -1;
    }
    if (method < 0 || method > 6) {
      _last_error =
        "Invalid value for method option: Expected int between 0 and 6";
      return false;
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

  if (frame.may_dispose_to_previous) {
    _state = cv::Mat::zeros(frame.canvas_height,
        frame.canvas_width,
        CV_8UC4);
  }

  return true;
}

bool LibWebP_Encoder::_maybe_insert_frame(bool finalizing) {
  bool has_uninserted = (int) _encoded_frames.size() > _inserted_frames;
  if (!has_uninserted && !_delay_error && (_inserted_frames || !finalizing)) {
    return true;
  }

  // Need to insert a delay, but there is no frame. Create a dummy one
  if (!has_uninserted) {
    _next.x_offset = 0;
    _next.y_offset = 0;
    _next.id = WEBP_CHUNK_ANMF;
    _next.dispose_method = WEBP_MUX_DISPOSE_NONE;
    _next.blend_method = WEBP_MUX_BLEND;

    uint32_t pixel = 0;
    WebPPicture picture;
    WebPPictureInit(&picture);
    picture.use_argb = 1;
    picture.argb = &pixel;
    picture.argb_stride = 1;
    picture.width = 1;
    picture.height = 1;

    _encoded_frames.emplace_back(new WebPMemoryWriter, _delete_writer);
    WebPMemoryWriterInit(_encoded_frames.back().get());
    picture.writer = WebPMemoryWrite;
    picture.custom_ptr = _encoded_frames.back().get();

    bool encode_ok = WebPEncode(&_config, &picture);
    // WebPEncode may allocate new buffers that need to be freed
    WebPPictureFree(&picture);
    if (!encode_ok) {
      _last_error = "Failed to encode dummy frame";
      return false;
    }
  }

  _next.duration = _delay_error;
  _next.bitstream.bytes = _encoded_frames.back()->mem;
  _next.bitstream.size = _encoded_frames.back()->size;
  _delay_error = 0;

  if (WEBP_MUX_OK != WebPMuxPushFrame(_mux.get(), &_next, 0)) {
    _last_error = "Failed to push frame";
    return false;
  }

  _inserted_frames++;

  return true;
}

LibWebP_Encoder::LibWebP_Encoder(const std::string &format,
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
  _inserted_frames = 0;
}

bool LibWebP_Encoder::add_frame(const Frame &frame) {
  if ("webp" != _format) {
    _last_error = "Expected webp format, got " + _format;
    return false;
  }

  if (!_mux.get() && !_init_mux(frame)) {
    return false;
  }

  if (!frame.empty
      && (frame.img.empty()
        || (1 == frame.img.cols && frame.x & 1 )
        || (1 == frame.img.rows && frame.y & 1 ))) {
    _delay_error += frame.delay;
    return true;
  }

  if (!_maybe_insert_frame(false)) {
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
  // It is possible to extend the frame to include one col/row to the
  // top/left. However, this becomes non-trivial with the dispose to
  // background strategy, or with any non-blending frames. It is also possible
  // to the generate proper cleanup by keeping track of the state, enlarging
  // the frame, and affecting the next frame to include the disposal pixels
  // values for the current one. However, this increases the final file size,
  // as each frame grows, as well as it introduces artifacts when lossy
  // compression is used, as colors may bleed out into the bigger draw area
  // and not get properly covered by the next frame.
  // Therefore, we go with the simple approach of dropping the top and left
  // column and row
  img = img(cv::Rect(frame.x & 1,
        frame.y & 1,
        img.cols - (frame.x & 1),
        img.rows - (frame.y & 1)));
  _next.x_offset = (frame.x + 1) & ~1;
  _next.y_offset = (frame.y + 1) & ~1;
  _next.id = WEBP_CHUNK_ANMF;
  _next.dispose_method = Frame::DISPOSAL_BACKGROUND == frame.disposal?
    WEBP_MUX_DISPOSE_BACKGROUND : WEBP_MUX_DISPOSE_NONE;
  _next.blend_method = Frame::BLENDING_BLEND == frame.blending?
    WEBP_MUX_BLEND : WEBP_MUX_NO_BLEND;

  WebPPicture picture;
  WebPPictureInit(&picture);
  picture.use_argb = 1;
  picture.argb = (uint32_t *) img.data;
  picture.argb_stride = img.step / 4;
  picture.width = img.cols;
  picture.height = img.rows;

  _encoded_frames.emplace_back(new WebPMemoryWriter, _delete_writer);
  WebPMemoryWriterInit(_encoded_frames.back().get());
  picture.writer = WebPMemoryWrite;
  picture.custom_ptr = _encoded_frames.back().get();

  bool encode_ok = WebPEncode(&_config, &picture);
  // WebPEncode may allocate new buffers that need to be freed
  // The argb buffer isn't freed twice as the library keeps track of what
  // was allocated by it or by the user
  WebPPictureFree(&picture);
  if (!encode_ok) {
    _last_error = "Failed to encode image data";
    return false;
  }

  _delay_error += frame.delay;

  // Update the state to match what is expected after the disposal
  if (frame.may_dispose_to_previous) {
    switch (frame.disposal) {
      case Frame::DISPOSAL_PREVIOUS:
        break;

      case Frame::DISPOSAL_BACKGROUND:
        cv::rectangle(_state,
            cv::Rect(_next.x_offset,
              _next.y_offset,
              img.cols,
              img.rows),
            cv::Vec4b(0, 0, 0, 0),
            -1);
        break;

      case Frame::DISPOSAL_NONE:
      default:
        cv::Vec4b *src_line = (cv::Vec4b *) img.data;
        cv::Vec4b *dst_line =
          (cv::Vec4b *) (_state.data + _next.y_offset * _state.step)
          + _next.x_offset;
        for (int i = 0; i < img.rows; i++) {
          for (int j = 0; j < img.cols; j++) {
            if (Frame::BLENDING_BLEND == frame.blending
                && src_line[j][3] < 128) {
              break;
            }
            dst_line[j] = src_line[j];
          }
          src_line += img.step / sizeof(cv::Vec4b);
          dst_line += _state.step / sizeof(cv::Vec4b);
        }
        break;
    }
  }

  // Handle dispose to previous by inserting a cleanup frame with a duration
  // of 0. This introduces a small delay in practice, but it saves us from
  // the complex solution of enlarging the subsequent frames to include the
  // cleanup for this one. Doing so would increase the final file size, as
  // well as possibly introduce artifacts when using lossy compression, as
  // colors may bleed out into the enlargened draw area
  if (Frame::DISPOSAL_PREVIOUS == frame.disposal) {
    if (!frame.may_dispose_to_previous) {
      _last_error = "Unexpected dispose to previous";
      return false;
    }

    Frame cleanup_frame(frame);
    cleanup_frame.delay = 0;
    cleanup_frame.img = cv::Mat(_state,
        cv::Rect(_next.x_offset,
          _next.y_offset,
          img.cols,
          img.rows));
    cleanup_frame.disposal = Frame::DISPOSAL_NONE;
    cleanup_frame.blending = Frame::BLENDING_NO_BLEND;

    if (!add_frame(cleanup_frame)) {
      return false;
    }
  }

  return true;
}

bool LibWebP_Encoder::finalize() {
  WebPData data;
  WebPDataInit(&data);

  if (!_mux.get()) {
    _last_error = "Tried to finalize uninitilized image";
    return false;
  }

  if (!_maybe_insert_frame(true)) {
    return false;
  }

  if (WEBP_MUX_OK != WebPMuxAssemble(_mux.get(), &data)) {
    _last_error = "Failed to assemble";
    return false;
  }

  _output->resize(data.size);
  memcpy(_output->data(), data.bytes, data.size);
  WebPDataClear(&data);

  return true;
}

bool LibWebP_Encoder::supports_multiple_frames() {
  return true;
}

bool LibWebP_Encoder::supports_optimized_frames() {
  return true;
}

#include <opencv2/opencv.hpp>
#include <gif_lib.h>

#include "gif-palette.h"
#include "frame.h"

Frame::Frame() {
    reset();
}

Frame::Frame(const Frame &src) :
  img(src.img),
  delay(src.delay),
  x(src.x),
  y(src.y),
  canvas_width(src.canvas_width),
  canvas_height(src.canvas_height),
  empty(src.empty),
  loops(src.loops),
  disposal(src.disposal),
  blending(src.blending),
  gif_frame_palette(src.gif_frame_palette),
  gif_global_palette(src.gif_global_palette),
  gif_transparent_index(src.gif_transparent_index) {
}

void Frame::reset() {
  img = cv::Mat();
  empty = true;
  loops = 0;
  disposal = DISPOSAL_UNDEFINED;
  blending = BLENDING_UNDEFINED;
  gif_transparent_index = -1;
  gif_frame_palette.reset();
  gif_global_palette.reset();
  x = 0;
  y = 0;
  canvas_width = 0;
  canvas_height = 0;
  delay = 0;
}

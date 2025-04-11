#include <phpcpp.h>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include <exiv2/webpimage.hpp>
#include <lcms2.h>
#include <libheif/heif.h>
#include <gif_lib.h>
#include <msf_gif.h>
#include <webp/demux.h>
#include <webp/encode.h>
#include <webp/mux.h>
#include "gif-palette.h"
#include "frame.h"
#include "tempfile.h"
#include "srgb.icc.h"
#include "decoder.h"
#include "encoder.h"
#include "opencv-decoder.h"
#include "opencv-encoder.h"
#include "giflib-decoder.h"
#include "giflib-encoder.h"
#include "msfgif-encoder.h"
#include "libwebp-decoder.h"
#include "libwebp-full-frame-encoder.h"
#include "libheif-decoder.h"
#include "libheif-encoder.h"
#include "partial-to-full-proxy-encoder.h"

#define _checkimageloaded() { \
  if (_raw_image_data.empty()) { \
    throw Php::Exception("Cannot process empty object"); \
  } \
}

class Photon_OpenCV : public Php::Base {
protected:
  Frame _frame;
  std::string _last_error;
  std::string _format;
  int _type;
  int _compression_quality;
  std::vector<uint8_t> _icc_profile;
  std::string _raw_image_data;
  int _expected_width;
  int _expected_height;
  int _header_channels;
  bool _force_reencode;
  Exiv2::Value::UniquePtr _original_orientation;
  std::map<std::string, std::string> _image_options;
  std::vector<std::function<void()>> _operations;
  std::unique_ptr<Decoder> _decoder;
  bool _preserve_palette;
  bool _first_encode;

  const int WEBP_DEFAULT_QUALITY = 75;
  const int AVIF_DEFAULT_QUALITY = 75;
  const int JPEG_DEFAULT_QUALITY = 75;
  const int PNG_DEFAULT_QUALITY = 21;

  static cmsHPROFILE _srgb_profile;

  void _enforce8u() {
    if (CV_8U != _frame.img.depth()) {
      /* Proper conversion is mostly guess work, but it's fairly rare and
         these are reasonable assumptions */
      double alpha, beta;
      switch (_frame.img.depth()) {
        case CV_16U:
          alpha = 1./256;
          beta = 0;
          break;

        case CV_16S:
          alpha = 1./256;
          beta = 128;
          break;

        case CV_8S:
          alpha = 1;
          beta = 128;
          break;

        default:
          alpha = 1;
          beta = 0;
          break;
      }

      _frame.img.convertTo(_frame.img, CV_8U, alpha, beta);
    }
  }

  bool _converttosrgb() {
    if (_icc_profile.empty()) {
      return true;
    }

    cmsHPROFILE embedded_profile = cmsOpenProfileFromMem(_icc_profile.data(),
        _icc_profile.size());
    if (!embedded_profile) {
      _last_error = "Failed to decode embedded profile";
      _icc_profile.clear();
      return false;
    }

    int storage_format;
    int num_intensity_channels;
    switch (_frame.img.channels()) {
      case 1:
      case 2:
        storage_format = TYPE_GRAY_8;
        num_intensity_channels = 1;
        break;

      case 3:
      case 4:
        storage_format = TYPE_BGR_8;
        num_intensity_channels = 3;
        break;

      default:
        cmsCloseProfile(embedded_profile);
        _last_error = "Invalid number of channels";
        return false;
    }

    cmsHTRANSFORM transform = cmsCreateTransform(
      embedded_profile, storage_format,
      _srgb_profile, TYPE_BGR_8,
      INTENT_PERCEPTUAL, cmsFLAGS_BLACKPOINTCOMPENSATION
    );

    if (!transform) {
      cmsCloseProfile(embedded_profile);
      _icc_profile.clear();
      _last_error = "Failed to create transform to sRGB";
      return false;
    }

    cmsCloseProfile(embedded_profile);

    /* Alpha optimizations using `reshape()` require continuous data.
       If necessary, this can be optimized out for images without alpha */
    if (!_frame.img.isContinuous()) {
      _frame.img = _frame.img.clone();
    }

    int output_type = _imagehasalpha()? CV_8UC4 : CV_8UC3;
    cv::Mat transformed_img = cv::Mat(_frame.img.rows,
        _frame.img.cols,
        output_type);

    /* The sRGB profile can't handle the alpha channel. We make sure it's
       skipped when applying the profile */
    cv::Mat no_alpha_img = _frame.img.
      reshape(1, _frame.img.rows*_frame.img.cols).
      colRange(0, num_intensity_channels);
    cv::Mat no_alpha_transformed_img = transformed_img.
      reshape(1, transformed_img.rows*transformed_img.cols).
      colRange(0, 3);

    cmsDoTransformLineStride(
        transform,
        no_alpha_img.data, no_alpha_transformed_img.data,
        no_alpha_img.cols/num_intensity_channels, no_alpha_img.rows,
        no_alpha_img.step, no_alpha_transformed_img.step,
        0, 0
    );
    cmsDeleteTransform(transform);

    if (_imagehasalpha()) {
      /* Copy the original alpha information */
      cv::Mat alpha_only_img = _frame.img.reshape(1,
          _frame.img.rows*_frame.img.cols).
        colRange(num_intensity_channels, num_intensity_channels+1);
      cv::Mat alpha_only_transformed_img = transformed_img.
        reshape(1, transformed_img.rows*transformed_img.cols).
        colRange(3, 4);
      alpha_only_img.copyTo(alpha_only_transformed_img);
    }

    _frame.img = transformed_img;

    return true;
  }

  static int _gmagickfilter2opencvinter(int filter, int default_filter) {
    /* Disable custom filters for now. Photon uses lanczos to circumvent an
       issue that does not exist here. See:
       https://code.trac.wordpress.org/ticket/62
       https://sourceforge.net/p/graphicsmagick/bugs/381/ */
    return cv::INTER_AREA;

    int opencv_filter = default_filter;

    if (filter == FILTER_LANCZOS) {
      opencv_filter = cv::INTER_LANCZOS4;
    }
    else if (filter == FILTER_POINT) {
      opencv_filter = cv::INTER_NEAREST;
    }
    else if (filter == FILTER_BOX) {
      opencv_filter = cv::INTER_AREA;
    }
    else if (filter == FILTER_TRIANGLE) {
      opencv_filter = cv::INTER_LINEAR;
    }
    else if (filter == FILTER_CUBIC) {
      opencv_filter = cv::INTER_CUBIC;
    }

    return opencv_filter;
  }

  static void _initialize() {
    /* Load default sRGB profile */
    _srgb_profile = cmsOpenProfileFromMem(srgb_icc, sizeof(srgb_icc)-1);
    if (!_srgb_profile) {
      throw std::runtime_error("Failed to decode builtin sRGB profile");
    }
  }

  bool _imagehasalpha() {
    /* Even number of channels means it's either GA or BGRA */
    int num_channels = _frame.img.empty()?
      _header_channels : _frame.img.channels();
    return !(num_channels & 1);
  }

  /* Assumes alpha is the last channel */
  void _associatealpha() {
    /* Continuity required for `reshape()` */
    if (!_frame.img.isContinuous()) {
      _frame.img = _frame.img.clone();
    }

    int alpha_channel = _frame.img.channels()-1;
    cv::Mat alpha = _frame.img.reshape(1, _frame.img.rows*_frame.img.cols).
      colRange(alpha_channel, alpha_channel+1);

    for (int i = 0; i < alpha_channel; i++) {
      cv::Mat color = _frame.img.reshape(
          1, _frame.img.rows*_frame.img.cols).colRange(i, i+1);
      cv::multiply(color, alpha, color, 1./255);
    }
  }

  /* Assumes alpha is the last channel */
  void _dissociatealpha() {
    /* Continuity required for `reshape()` */
    if (!_frame.img.isContinuous()) {
      _frame.img = _frame.img.clone();
    }

    int alpha_channel = _frame.img.channels()-1;
    cv::Mat alpha = _frame.img.reshape(1, _frame.img.rows*_frame.img.cols).
      colRange(alpha_channel, alpha_channel+1);

    for (int i = 0; i < alpha_channel; i++) {
      cv::Mat color = _frame.img.reshape(
          1, _frame.img.rows*_frame.img.cols).colRange(i, i+1);
      cv::divide(color, alpha, color, 255.);
    }
  }

  void _transparencysaferesize(int width, int height, int filter) {
    if (_frame.img.empty()) {
      return;
    }

    double width_mul = (double) width / _frame.canvas_width;
    double height_mul = (double) height / _frame.canvas_height;

    // Changing this logic may introduce inconsistencies in animations
    int fx = _frame.x * width_mul;
    int fy = _frame.y * height_mul;
    int fw = ceil((_frame.x + _frame.img.cols) * width_mul) - fx;
    int fh = ceil((_frame.y + _frame.img.rows) * height_mul) - fy;
    // Preserve bottom right edge
    if (_frame.x + _frame.img.cols == _frame.canvas_width) {
      fw = width - fx;
    }
    if (_frame.y + _frame.img.rows == _frame.canvas_height) {
      fh = height - fy;
    }

    bool consistent_sampling_required =
      (_decoder.get() && _decoder->provides_animation()) || _preserve_palette;
    if (consistent_sampling_required) {
      // Ensure border data doesn't turn into garbage
      if (fw && (int) ((fx + 0.5) / width_mul) < _frame.x) {
        fx++;
        fw--;
      }
      if (fw &&
          (int) ((fx + fw - 0.5) / width_mul) >= _frame.x + _frame.img.cols) {
        fw--;
      }
      if (fh && (int) ((fy + 0.5) / height_mul) < _frame.y) {
        fy++;
        fh--;
      }
      if (fh &&
          (int) ((fy + fh - 0.5) / height_mul) >= _frame.y + _frame.img.rows) {
        fh--;
      }
    }

    if (!fw || !fh) {
      _frame.x = 0;
      _frame.y = 0;
      _frame.img = cv::Mat();
      return;
    }

    if (!consistent_sampling_required) {
      if (_imagehasalpha()) {
        _associatealpha();
      }
      cv::resize(_frame.img, _frame.img, cv::Size(fw, fh), 0, 0, filter);
      if (_imagehasalpha()) {
        _dissociatealpha();
      }
    }
    else {
      cv::Mat dst(fh, fw, _frame.img.type());

      // x-axis offset cache
      std::vector<int> x_off(fw);
      for (int i = 0; i < fw; i++) {
        x_off[i] = (fx + i + 0.5) / width_mul - _frame.x;
      }

      int channels = dst.channels();
      uint8_t *dst_line = dst.data;
      for (int i = 0; i < fh; i++) {
        int y = (fy + i + 0.5) / height_mul - _frame.y;
        void *src_line = _frame.img.data + y * _frame.img.step;
        for (int j = 0; j < fw; j++) {
          if (1 == channels) {
            ((uint8_t *) dst_line)[j] = ((uint8_t *) src_line)[x_off[j]];
          }
          else if (2 == channels) {
            ((uint16_t *) dst_line)[j] = ((uint16_t *) src_line)[x_off[j]];
          }
          else if (3 == channels) {
            ((cv::Vec3b *) dst_line)[j] = ((cv::Vec3b *) src_line)[x_off[j]];
          }
          else if (4 == channels) {
            ((uint32_t *) dst_line)[j] = ((uint32_t *) src_line)[x_off[j]];
          }
        }
        dst_line += dst.step;
      }

      _frame.img = dst;
    }

    _frame.canvas_width = width;
    _frame.canvas_height = height;
    _frame.x = fx;
    _frame.y = fy;
  }

  bool _setupdecoder(bool silent=true) {
    _decoder.reset(new OpenCV_Decoder(&_raw_image_data));

    if (!_decoder->loaded()) {
      _decoder.reset(new Giflib_Decoder(&_raw_image_data));
    }
    if (!_decoder->loaded()) {
      _decoder.reset(new LibWebP_Decoder(&_raw_image_data));
    }
    if (!_decoder->loaded()) {
      _decoder.reset(new Libheif_Decoder(&_raw_image_data));
    }

    if (!_decoder->loaded()) {
      _decoder.reset();

      std::string message = "Unable to decode image";

      if (!silent) {
        throw Php::Exception(message);
      }

      _last_error = message;
      return false;
    }

    // This may be reworked once exiv2 supports all relevant formats
    _decoder->get_icc_profile(_icc_profile);

    return true;
  }

  bool _loadnextframe(bool silent=true) {
    if (!_decoder->get_next_frame(_frame)) {
      if (!silent) {
        throw Php::Exception("Unable to load next frame");
      }

      return false;
    }

    _enforce8u();
    return true;
  }

  bool _loadimagefromrawdata() {
    /* Clear image in case object is being reused */
    _frame.img = cv::Mat();
    _decoder.reset(nullptr);
    _icc_profile.clear();
    _image_options.clear();
    _compression_quality = -1;
    _force_reencode = false;
    _preserve_palette = false;
    _first_encode = true;

    Exiv2::Image::UniquePtr exiv_img;
    bool exiv2_ok = true;
    try {
      exiv_img = Exiv2::ImageFactory::open(
        (Exiv2::byte *) _raw_image_data.data(), _raw_image_data.size());
      exiv_img->readMetadata();
    }
    catch (Exiv2::Error &error) {
      // Failing to open is not critical, but it requires us to
      // try to decode the image to get the proper information
      exiv2_ok = false;
    }

    if (exiv2_ok && (!exiv_img->pixelWidth() || !exiv_img->pixelHeight())) {
      // Stop exiv2 use if giberrish data was read
      exiv2_ok = false;
    }

    if (exiv2_ok &&
        exiv_img->imageType() == Exiv2::ImageType::bmff &&
        _raw_image_data.compare(8, 4, "avif")) {
      // Only let static avif through
      exiv2_ok = false;
    }

    Exiv2::ImageType image_type = exiv2_ok?
      exiv_img->imageType() : Exiv2::ImageType::none;
    switch (image_type) {
      case Exiv2::ImageType::png:
        _format = "png";
        _header_channels = _getchannelsfromrawpng();
        _image_options["png:lossless"] = "true";
        break;

      case Exiv2::ImageType::jpeg:
        _format = "jpeg";
        _header_channels = _getchannelsfromrawjpg();
        break;

      case Exiv2::ImageType::webp:
        _format = "webp";
        _header_channels = _getchannelsfromrawwebp();
        if (_israwwebplossless()) {
          _image_options["webp:lossless"] = "true";
        }
        break;

      case Exiv2::ImageType::gif:
        _format = "gif";
        _header_channels = _getchannelsfromrawgif();
        _image_options["gif:lossless"] = "true";
        break;

      case Exiv2::ImageType::bmff:
        // Other formats were filtered out early
        _format = "avif";
        _header_channels = _getchannelsfromrawavif();
        break;

      default:
        // Unexpected format, often an exiv2 limitation. Decode to find out
        _setupdecoder(false);
        _loadnextframe(false);
        _header_channels = _frame.img.channels();
        _force_reencode = !_decoder->default_format_is_accurate();
        _format = _decoder->default_format();
        break;
    }

    _original_orientation.release();
    if (exiv2_ok) {
      auto &exif = exiv_img->exifData();
      Exiv2::ExifKey orientation_key("Exif.Image.Orientation");
      auto orientation_pos = exif.findKey(orientation_key);

      if (orientation_pos != exif.end()) {
        _original_orientation = orientation_pos->getValue();
      }
    }

    _expected_width = _frame.empty?
      exiv_img->pixelWidth() : _frame.canvas_width;
    _expected_height = _frame.empty?
      exiv_img->pixelHeight() : _frame.canvas_height;

    if (exiv2_ok && exiv_img->iccProfileDefined()) {
      const Exiv2::DataBuf profile = exiv_img->iccProfile();
      _icc_profile.resize(profile.size());
      std::memcpy(_icc_profile.data(), profile.c_data(), profile.size());
    }

    /* Palettes are automatically converted to RGB on decode */
    switch (_header_channels) {
      case 1:
        _type = IMGTYPE_GRAYSCALE;
        break;

      case 2:
        _type = IMGTYPE_GRAYSCALEMATTE;
        break;

      case 3:
        _type = IMGTYPE_TRUECOLOR;
        break;

      case 4:
        _type = IMGTYPE_TRUECOLORMATTE;
        break;

      default:
        _raw_image_data.clear();
        _last_error = "Invalid number of channels";
        return false;
    }

    return true;
  }

  bool _decodehexcolor(std::string color, cv::Vec3b &decoded) {
      if ((color.size() != 4 && color.size() != 7) || color[0] != '#') {
        return false;
      }

      std::vector<int> values(color.size() - 1);
      for (size_t i = 1; i < color.size(); i++) {
        char c = std::tolower(color[i]);
        int v;
        if (c >= '0' && c <= '9') {
            v = c - '0';
        }
        else if (c >= 'a' && c <= 'f') {
            v = c - 'a' + 10;
        }
        else {
            return false;
        }

        values[i-1] = v;
      }

      // BGR
      if (values.size() == 3) {
        decoded[2] = (values[0]) << 4 | values[0];
        decoded[1] = (values[1]) << 4 | values[1];
        decoded[0] = (values[2]) << 4 | values[2];
      }
      else if (values.size() == 6) {
        decoded[2] = (values[0]) << 4 | values[1];
        decoded[1] = (values[2]) << 4 | values[3];
        decoded[0] = (values[4]) << 4 | values[5];
      }

      return true;
  }

  cv::Vec4b _bgrtoloadedimagetype(cv::Vec3b bgr_color) {
    cv::Vec4b color;

    // Assumes 1 byte per channel
    if (_frame.img.channels() <= 2) {
      color[0] = round(
          bgr_color[0]*.114 + bgr_color[1]*.587 + bgr_color[2]*.299);
      color[1] = 255;
    }
    else {
      color[0] = bgr_color[0];
      color[1] = bgr_color[1];
      color[2] = bgr_color[2];
      color[3] = 255;
    }

    return color;
  }

  void _forceaspectratio(int &width, int &height) {
    double img_ratio = (double) _expected_width / (double) _expected_height;
    double new_ratio = (double) width / (double) height;

    if (new_ratio > img_ratio) {
      // Wider
      width = std::max(1, (int) round(height * img_ratio));
    }
    else {
      // Taller
      height = std::max(1, (int) round(width / img_ratio));
    }
  }

  int _getchannelsfromrawjpg() {
    const uint8_t *data = (uint8_t *) _raw_image_data.data();

    // Assumes JPG was already validated, minimal confidence checks
    if (_raw_image_data.size() < 2
        || data[0] != 0xff
        || data[1] != 0xd8) {
      // Unexpected first bytes, default to 3, the most common case
      return 3;
    }

    // Look for a SOF segment (0xffc0 - 0xffcf)
    size_t o = 2;
    while (o + 3 < _raw_image_data.size()
        && (data[o] != 0xff || (data[o+1] & 0xf0) != 0xc0)) {
      o += 2 + ((data[o+2] << 8) | data[o+3]);
    }

    if (o + 9 >= _raw_image_data.size()) {
      // Somehow malformed
      return 3;
    }

    // No alpha. CMYK will result in RGB image when decoded
    return data[o+9] == 1? 1 : 3;
  }

  int _getchannelsfromrawgif() {
    // Determining the number of channels requires decoding and rendering all
    // graphics. Possible values are 3 and 4, and we default to 4
    return 4;
  }

  int _getchannelsfromrawpng() {
    const uint8_t expected_first_bytes[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a,
        0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52};
    const uint8_t *data = (uint8_t *) _raw_image_data.data();
    const uint8_t *end = data + _raw_image_data.size();

    // Assumes PNG was already validated, minimal confidence checks
    if (_raw_image_data.size() < 32
        || memcmp(expected_first_bytes, data, sizeof(expected_first_bytes))) {
      // Unexpected default to 3, the most common case
      return 3;
    }

    // Palettes will be automatically converted to RGB
    uint8_t color_type = data[sizeof(expected_first_bytes)+9];
    bool rgb = color_type & 2;
    if (color_type & 4) {
      return rgb? 4 : 2;
    }

    for (const uint8_t *chunk = data + 8; chunk + 8 <= end; ) {
      if (!strncmp("tRNS", (char *) chunk+4, 4)) {
        return rgb? 4 : 2;
      }
      uint32_t chunk_size = (chunk[0] << 24)
        | (chunk[1] << 16)
        | (chunk[2] << 8)
        | chunk[3];
      chunk += chunk_size+12;
    }

    return rgb? 3 : 1;
  }

  int _getchannelsfromrawavif() {
    std::unique_ptr<heif_context, decltype(&heif_context_free)> context(
      heif_context_alloc(), &heif_context_free);

    heif_error error;

    error = heif_context_read_from_memory_without_copy(context.get(),
      (void *) _raw_image_data.data(),
      _raw_image_data.size(),
      nullptr);
    if (error.code) {
      // Shouldn't happen, default to the most common value
      return 3;
    }

    std::unique_ptr<heif_image_handle,
      decltype(&heif_image_handle_release)>
      handle(nullptr, &heif_image_handle_release);
    heif_image_handle *raw_handle = nullptr;
    error = heif_context_get_primary_image_handle(context.get(),
      &raw_handle);
    handle.reset(raw_handle);
    if (error.code) {
      // Shouldn't happen, default to the most common value
      return 3;
    }

    // Assumes no grayscale images for simplicity
    return heif_image_handle_has_alpha_channel(handle.get())? 4 : 3;
  }


  int _getchannelsfromrawwebp() {
    if (_raw_image_data.size() < 32) {
      // Unexpected, default to 3, the most common case
      return 3;
    }

    std::string type(_raw_image_data.substr(8, 8));
    if ("WEBPVP8 " == type) {
      // Simple webps only support YUV420. OpenCV always decodes to BGR
      return 3;
    }

    int alpha_bit = 0;
    if ("WEBPVP8L" == type) {
      alpha_bit = _raw_image_data[24] & (1<<4);
    }
    else if ("WEBPVP8X" == type) {
      alpha_bit = _raw_image_data[20] & (1<<4);
    }

    return alpha_bit? 4 : 3;
  }

  int _israwwebplossless() {
    uint8_t *end = (uint8_t *) _raw_image_data.data() + _raw_image_data.size();
    for (uint8_t *chunk = (uint8_t *) _raw_image_data.data() + 12;
        chunk <= end - 8;) {
      if (!strncmp("VP8L", (char *) chunk, 4)) {
        return true;
      }
      uint32_t chunk_size = chunk[4]
        | (chunk[5]<<8)
        | (chunk[6]<<16)
        | (chunk[7]<<24);
      chunk += chunk_size + 8;
    }

    return false;
  }

  bool _encodeimage(std::vector<uint8_t> &output_buffer) {
    int quality = _compression_quality;
    if (-1 == quality) {
      if ("jpeg" == _format) {
        quality = JPEG_DEFAULT_QUALITY;
      }
      else if ("png" == _format) {
        quality = PNG_DEFAULT_QUALITY;
      }
      else if ("webp" == _format) {
        quality = WEBP_DEFAULT_QUALITY;
      }
      else if ("avif" == _format) {
        quality = AVIF_DEFAULT_QUALITY;
      }
    }

    if (!_first_encode && _decoder.get()) {
      // Rewind and load the first frame if it's not the first call
      _decoder->reset();
      _loadnextframe();
    }
    _first_encode = false;

    if ((!_decoder.get() && !_setupdecoder())
        || (_frame.empty && !_loadnextframe())) {
      // Compatibility: silently replace image with original if we are unable
      // to decode this late in the process
      _last_error.clear();
      output_buffer.assign(_raw_image_data.begin(), _raw_image_data.end());

      return true;
    }

    std::unique_ptr<Encoder> encoder;
    if ("avif" == _format) {
      encoder.reset(new Libheif_Encoder(
            _format,
            quality,
            &_image_options,
            &output_buffer));
    }
    else if ("webp" == _format && _decoder->provides_animation()) {
      encoder.reset(new LibWebP_Full_Frame_Encoder(
            _format,
            quality,
            &_image_options,
            &output_buffer));
      if (_decoder->provides_optimized_frames()) {
        std::shared_ptr<Encoder> internal_encoder(encoder.release());
        encoder.reset(new Partial_To_Full_Proxy_Encoder(internal_encoder));
      }
    }
    else if ("gif" == _format && _decoder->provides_animation()) {
      if (_decoder->provides_optimized_frames()) {
        encoder.reset(new Giflib_Encoder(
              _format,
              quality,
              &_image_options,
              &output_buffer));
      }
      else {
        encoder.reset(new Msfgif_Encoder(
              _format,
              quality,
              &_image_options,
              &output_buffer));
      }
    }
    else {
      encoder.reset(new OpenCV_Encoder(
            _format,
            quality,
            &_image_options,
            &output_buffer));
    }

    _preserve_palette = encoder->requires_original_palette();
    do {
      // Color profile gets silently stripped, apply it first
      _converttosrgb();

      for (auto &operation : _operations) {
        operation();
      }

      if (_decoder->provides_optimized_frames()
          && !encoder->supports_optimized_frames()) {
        _expandtocanvas();
      }

      if (!encoder->add_frame(_frame)) {
        _last_error = encoder->get_last_error();
        return false;
      }

      if (!encoder->supports_multiple_frames()) {
        break;
      }
    } while (_loadnextframe());

    _frame.img = cv::Mat();
    _decoder.reset(nullptr);

    if (!encoder->finalize()) {
      _last_error = encoder->get_last_error();
      return false;
    }

    /* Manually reinsert orientation exif data if it has meaning */
    int exif_orientation = _original_orientation.get()?
      _original_orientation.get()->toUint32() : 0;
    if (exif_orientation > 1 && exif_orientation <= 8) {
      Exiv2::Image::UniquePtr exiv_img;
      try {
        exiv_img = Exiv2::ImageFactory::open(output_buffer.data(),
            output_buffer.size());
        exiv_img->readMetadata();
      }
      catch (Exiv2::Error &error) {
        _last_error = error.what();
        return false;
      }

      auto &exif = exiv_img->exifData();
      Exiv2::ExifKey orientation_key("Exif.Image.Orientation");
      exif.add(orientation_key, _original_orientation.get());
      try {
        exiv_img->writeMetadata();
        output_buffer.resize(exiv_img->io().size());
        exiv_img->io().read(output_buffer.data(), output_buffer.size());
      }
      catch (Exiv2::Error &error) {
        _last_error = error.what();
        return false;
      }
    }

    return true;
  }

  void _rotate(int rotation) {
    if (_frame.img.empty()) {
      return;
    }

    if (cv::ROTATE_90_CLOCKWISE == rotation) {
      int nx = _frame.canvas_height - _frame.img.rows - _frame.y;
      int ny = _frame.x;

      _frame.x = nx;
      _frame.y = ny;

      std::swap(_frame.canvas_width, _frame.canvas_height);
    }
    else if (cv::ROTATE_90_COUNTERCLOCKWISE == rotation) {
      int nx = _frame.y;
      int ny = _frame.canvas_width - _frame.img.cols - _frame.x;

      _frame.x = nx;
      _frame.y = ny;

      std::swap(_frame.canvas_width, _frame.canvas_height);
    }
    else if (cv::ROTATE_180 == rotation) {
      _frame.x = _frame.canvas_width - _frame.img.cols - _frame.x;
      _frame.y = _frame.canvas_height - _frame.img.rows - _frame.y;
    }

    cv::rotate(_frame.img, _frame.img, rotation);
  }

  void _flip(int flip_code) {
    if (_frame.img.empty()) {
      return;
    }

    _frame.x = _frame.canvas_width - _frame.x - _frame.img.cols;
    _frame.y = _frame.canvas_height - _frame.y - _frame.img.rows;
    cv::flip(_frame.img, _frame.img, flip_code);
  }

  void _crop(int x, int y, int width, int height) {
    if (_frame.img.empty()) {
      return;
    }

    int fx = std::max(0, std::min(width, _frame.x - x));
    int fy = std::max(0, std::min(height, _frame.y - y));
    int fx2 = std::max(0, std::min(width, _frame.x + _frame.img.cols - x));
    int fy2 = std::max(0, std::min(height, _frame.y + _frame.img.rows - y));

    if (fx == fx2 || fy == fy2) {
      _frame.img = cv::Mat();
    }
    else {
      _frame.img = _frame.img(cv::Rect(
            std::max(0, x - _frame.x),
            std::max(0, y - _frame.y),
            fx2 - fx,
            fy2 - fy));
    }

    _frame.x = fx;
    _frame.y = fy;

    _frame.canvas_width = width;
    _frame.canvas_height = height;
  }

  void _border(int width, int height, cv::Vec3b color) {
    if (_frame.img.empty()) {
      return;
    }

    if (_preserve_palette) {
      throw Php::Exception("Unable to insert border and preserve palette");
    }

    cv::Mat dst(_frame.img.rows + height*2,
      _frame.img.cols + width*2,
      _frame.img.type(),
      _bgrtoloadedimagetype(color));

    _frame.img.copyTo(
        dst(cv::Rect(width, height, _frame.img.cols, _frame.img.rows)));
    _frame.img = dst;

    _frame.x -= width;
    _frame.y -= height;

    if (_frame.x < 0) {
      _frame.canvas_width -= _frame.x;
      _frame.x = 0;
    }
    if (_frame.y < 0) {
      _frame.canvas_height -= _frame.y;
      _frame.y = 0;
    }
    if (_frame.x + _frame.img.cols > _frame.canvas_width) {
      _frame.canvas_width = _frame.x + _frame.img.cols;
    }
    if (_frame.y + _frame.img.rows > _frame.canvas_height) {
      _frame.canvas_height = _frame.y + _frame.img.rows;
    }
  }

  void _expandtocanvas() {
    if (_frame.img.empty()) {
      return;
    }

    if (!_frame.x
        && !_frame.y
        && _frame.img.cols == _frame.canvas_width
        && _frame.img.rows == _frame.canvas_height) {
      return;
    }

    const cv::Scalar bg_color_from_channels[] = {
      cv::Scalar(0, 0, 0, 0),
      cv::Scalar(255, 0, 0, 0),
      cv::Scalar(255, 0, 0, 0),
      cv::Scalar(255, 255, 255, 0),
      cv::Scalar(255, 255, 255, 0),
    };

    cv::Mat full_img(_frame.canvas_width,
        _frame.canvas_height,
        _frame.img.type(),
        bg_color_from_channels[_frame.img.channels()]);

    cv::Rect canvas_intersection = cv::Rect(0, 0, full_img.cols, full_img.rows)
      & cv::Rect(_frame.x, _frame.y, _frame.img.cols, _frame.img.rows);
    cv::Rect frame_intersection =
      cv::Rect(-_frame.x, -_frame.y, full_img.cols, full_img.rows)
      & cv::Rect(0, 0, _frame.img.cols, _frame.img.rows);

    if (canvas_intersection.width > 0 && canvas_intersection.height > 0) {
      _frame.img(frame_intersection).copyTo(full_img(canvas_intersection));
    }

    _frame.x = 0;
    _frame.y = 0;
    _frame.img = full_img;
  }

  bool _requiresreencoding() {
    return _force_reencode || _operations.size();
  }

public:
  /* Gmagick constant replacements */
  static const int CHANNEL_OPACITY = 7;
  static const int COLORSPACE_RGB = 1;
  static const int FILTER_LANCZOS = 13;
  static const int FILTER_CUBIC = 10;
  static const int FILTER_TRIANGLE = 3;
  static const int FILTER_POINT = 1;
  static const int FILTER_BOX = 2;
  static const int IMGTYPE_COLORSEPARATIONMATTE = 9;
  static const int IMGTYPE_GRAYSCALE = 2;
  static const int IMGTYPE_GRAYSCALEMATTE = 3;
  static const int IMGTYPE_PALETTE = 4;
  static const int IMGTYPE_PALETTEMATTE = 5;
  static const int IMGTYPE_TRUECOLOR = 6;
  static const int IMGTYPE_TRUECOLORMATTE = 7;

  // These match the exif specs
  static const int ORIENTATION_UNDEFINED = 0;
  static const int ORIENTATION_TOPLEFT = 1;
  static const int ORIENTATION_TOPRIGHT = 2;
  static const int ORIENTATION_BOTTOMRIGHT = 3;
  static const int ORIENTATION_BOTTOMLEFT = 4;
  static const int ORIENTATION_LEFTTOP = 5;
  static const int ORIENTATION_RIGHTTOP = 6;
  static const int ORIENTATION_RIGHTBOTTOM = 7;
  static const int ORIENTATION_LEFTBOTTOM = 8;

  Photon_OpenCV() {
    cv::setNumThreads(Php::ini_get("photon.opencv_threads"));

    /* Static local intilization is thread safe */
    static std::once_flag initialized;
    std::call_once(initialized, _initialize);
  }

  void readimageblob(Php::Parameters &params) {
    _raw_image_data = params[0].stringValue();
    if (_raw_image_data.empty()) {
      throw Php::Exception("Zero size image string passed");
    }

    if (!_loadimagefromrawdata()) {
      throw Php::Exception("Unable to read image blob: " + _last_error);
    }
  }

  void readimage(Php::Parameters &params) {
    std::fstream input(params[0].stringValue(),
        std::ios::in | std::ios::binary);

    if (input.is_open()) {
      input.seekg(0, std::ios::end);
      _raw_image_data.resize(input.tellg());
      input.seekg(0, std::ios::beg);
      input.read((char *) _raw_image_data.data(), _raw_image_data.size());
    }
    else {
      _raw_image_data.resize(0);
    }

    if (!_loadimagefromrawdata()) {
      throw Php::Exception("Unable to read image: " + _last_error);
    }
  }

  void writeimage(Php::Parameters &params) {
    _checkimageloaded();

    std::string output_path = params[0].stringValue();

    if (output_path.empty()) {
      throw Php::Exception("Unable to write the image. "
          "Empty filename string provided");
    }

    // No ops, we can return the original image
    if (!_requiresreencoding()) {
      std::ofstream output(output_path,
          std::ios::out | std::ios::binary);
      output << _raw_image_data;
      if (output.fail()) {
        throw Php::Exception("Unable to write the image to disk");
      }
      return;
    }

    std::vector<uint8_t> output_buffer;
    if (!_encodeimage(output_buffer)) {
      throw Php::Exception("Unable to encode image: " + _last_error);
    }
    else {
      std::ofstream output(output_path,
          std::ios::out | std::ios::binary);
      output.write((char *) output_buffer.data(), output_buffer.size());
      if (output.fail()) {
        throw Php::Exception("Unable to write the encoded image to disk");
      }
    }
  }

  Php::Value getimageblob() {
    _checkimageloaded();

    // No ops, we can return the original image
    if (!_requiresreencoding()) {
      return _raw_image_data;
    }

    std::vector<uint8_t> output_buffer;
    if (!_encodeimage(output_buffer)) {
      throw Php::Exception("Unable to encode image: " + _last_error);
    }

    return std::string((char *) output_buffer.data(), output_buffer.size());
  }

  Php::Value getlasterror() {
    return _last_error;
  }

  Php::Value getimagewidth() {
    _checkimageloaded();
    return _expected_width;
  }

  Php::Value getimageheight() {
    _checkimageloaded();
    return _expected_height;
  }

  Php::Value getimageformat() {
    _checkimageloaded();
    return _format;
  }

  void setimageformat(Php::Parameters &params) {
    std::string new_format = params[0].stringValue();
    std::transform(new_format.begin(), new_format.end(),
      new_format.begin(), ::tolower);

    if (new_format == _format) {
      return;
    }

    _force_reencode = true;
    _format = new_format;
  }

  void setimageoption(Php::Parameters &params) {
    std::string format = params[0].stringValue();
    std::transform(format.begin(), format.end(), format.begin(), ::tolower);

    std::string key = params[1].stringValue();
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);

    std::string value = params[2].stringValue();
    std::string real_key = format + ":" + key;

    if ("lossless" == key) {
      auto option = _image_options.find(real_key);
      // Lossless defaults to false
      if ((option == _image_options.end() && "true" == value)
          || (option != _image_options.end()
            && value != option->second)) {
        _force_reencode = true;
      }
    }

    _image_options[real_key] = value;
  }

  void setcompressionquality(Php::Parameters &params) {
    _compression_quality = params[0];
    _compression_quality = std::min(std::max(0, _compression_quality), 100);
  }

  Php::Value getcompressionquality() {
    _checkimageloaded();
    return _compression_quality;
  }

  void autoorientimage(Php::Parameters &params) {
    _checkimageloaded();

    int orientation = params[0];

    // Orientation follows exif specs, valid values are in [1,8], 0 is undef
    if (orientation < 0 || orientation > 8) {
      throw Php::Exception("Invalid orientation requested");
    }

    if (ORIENTATION_UNDEFINED == orientation) {
      orientation = _original_orientation.get()?
        _original_orientation.get()->toUint32() : ORIENTATION_TOPLEFT;
    }

    int rotation;
    switch (orientation) {
      case ORIENTATION_BOTTOMRIGHT:
      case ORIENTATION_BOTTOMLEFT:
        rotation = cv::ROTATE_180;
        break;

      case ORIENTATION_LEFTTOP:
      case ORIENTATION_RIGHTTOP:
        rotation = cv::ROTATE_90_CLOCKWISE;
        break;

      case ORIENTATION_RIGHTBOTTOM:
      case ORIENTATION_LEFTBOTTOM:
        rotation = cv::ROTATE_90_COUNTERCLOCKWISE;
        break;

      case ORIENTATION_TOPRIGHT:
      case ORIENTATION_TOPLEFT:
      default:
        rotation = -1;
        break;
    }

    if (-1 != rotation) {
      _operations.push_back(std::bind(&Photon_OpenCV::_rotate,
            this,
            rotation));
      if (cv::ROTATE_180 != rotation) {
        std::swap(_expected_width, _expected_height);
      }
    }
    if (ORIENTATION_TOPRIGHT == orientation
        || ORIENTATION_BOTTOMLEFT == orientation
        || ORIENTATION_LEFTTOP == orientation
        || ORIENTATION_RIGHTBOTTOM == orientation) {
      _operations.push_back(std::bind(&Photon_OpenCV::_flip, this, 1));
    }

    // Exif not reset intentionally, GraphicsMagick doesn't support it for Jpeg
  }

  Php::Value getimageorientation() {
    _checkimageloaded();

    // Numerical values in exif spec match library defines
    int orientation = _original_orientation.get()?
      _original_orientation.get()->toUint32() : 0;

    return orientation > 0 && orientation <= 8? orientation : 0;
  }

  void setimageprofile(Php::Parameters &params) {
    std::string name = params[0].stringValue();
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);


    if ("exif" == name) {
      if (!params[1].isNull()) {
        throw Php::Exception("Exif replacement unimplemented, only removal");
      }
      // Don't force reencoding if it results in no visible change
      int orientation = _original_orientation.get()?
        _original_orientation.get()->toUint32() : 0;
      // All valid values except TOPLEFT
      if (orientation > 1 && orientation <= 8) {
        _force_reencode = true;
      }
      _original_orientation.release();
    }
    else if ("icc" == name) {
      if (params[1].isNull()) {
        if (_icc_profile.size()) {
          _force_reencode = true;
          _icc_profile.clear();
        }
      }
      else {
        _force_reencode = true;
        std::string new_icc = params[1].stringValue();
        _icc_profile.resize(new_icc.size());
        memcpy(_icc_profile.data(), new_icc.data(), _icc_profile.size());
      }
    }
    else {
      throw Php::Exception("Tried to modify unsupported profile");
    }
  }

  Php::Value getimagetype() {
    _checkimageloaded();
    return _type;
  }

  void setimagetype(Php::Parameters &params) {
    // Unimplemented
    (void) params;
  }

  void resizeimage(Php::Parameters &params) {
    _checkimageloaded();

    /* Blur is ignored */
    int width = std::max(1, (int) params[0]);
    int height = std::max(1, (int) params[1]);
    int filter = params.size() > 2? (int) params[2] : -1;
    bool fit = params.size() > 3? (int) params[3] : false;
    /* AREA is fast, looks excellent when downsampling, good when upscaling */
    const int default_filter = cv::INTER_AREA;

    if (fit) {
      _forceaspectratio(width, height);
    }

    /* Explicitly skip if it's a noop. */
    if (width == _expected_width && height == _expected_height) {
      return;
    }

    _operations.push_back(std::bind(&Photon_OpenCV::_transparencysaferesize,
          this,
          width,
          height,
          _gmagickfilter2opencvinter(filter, default_filter)));

    _expected_width = width;
    _expected_height = height;
  }

  /* Documentation is lacking, but scaleimage is resizeimage with
    filter=Gmagick::FILTER_BOX and blur=1.0 */
  void scaleimage(Php::Parameters &params) {
    _checkimageloaded();

    int width = std::max(1, (int) params[0]);
    int height = std::max(1, (int) params[1]);
    bool fit = params.size() > 2? (int) params[2] : false;

    if (fit) {
      _forceaspectratio(width, height);
    }

    /* Explicitly skip if it's a noop. */
    if (width == _expected_width && height == _expected_height) {
      return;
    }

    _operations.push_back(std::bind(&Photon_OpenCV::_transparencysaferesize,
          this,
          width,
          height,
          cv::INTER_AREA));

    _expected_width = width;
    _expected_height = height;
  }

  void cropimage(Php::Parameters &params) {
    _checkimageloaded();

    int x = params[2];
    int y = params[3];
    int x2 = x + (int) params[0];
    int y2 = y + (int) params[1];
    if (x > x2) {
      std::swap(x, x2);
    }
    if (y > y2) {
      std::swap(y, y2);
    }

    x = std::max(0, std::min(x, _expected_width));
    y = std::max(0, std::min(y, _expected_height));
    x2 = std::max(0, std::min(x2, _expected_width));
    y2 = std::max(0, std::min(y2, _expected_height));

    // Prevent image from being loaded if it's a noop
    if (!x && !y && x2 == _expected_width && y2 == _expected_height) {
      return;
    }

    _operations.push_back(std::bind(&Photon_OpenCV::_crop,
          this,
          x,
          y,
          x2-x,
          y2-y));

    _expected_width = x2-x;
    _expected_height = y2-y;
  }

  void rotateimage(Php::Parameters &params) {
    _checkimageloaded();

    int degrees = params[1];
    int rotation_constant = -1;
    switch (degrees) {
      case 0:
        return;

      case 90:
        rotation_constant = cv::ROTATE_90_CLOCKWISE;
        break;

      case 180:
        rotation_constant = cv::ROTATE_180;
        break;

      case 270:
        rotation_constant = cv::ROTATE_90_COUNTERCLOCKWISE;
        break;

      default:
        throw Php::Exception("Unsupported rotation angle");
    }

    _operations.push_back(std::bind(&Photon_OpenCV::_rotate,
          this,
          rotation_constant));

    if (cv::ROTATE_180 != rotation_constant) {
      std::swap(_expected_width, _expected_height);
    }
  }

  Php::Value getimagechanneldepth(Php::Parameters &params) {
    _checkimageloaded();

    int channel = params[0];

    /* Gmagick's channels don't map directly to OpenCV's, we convert everything
     * to 8 bits. Photon is only interested in the opacity. Assume all other
     * channels are valid */
    if (channel == CHANNEL_OPACITY) {
      return _imagehasalpha()? 8 : 0;
    }

    return 8;
  }

  void borderimage(Php::Parameters &params) {
    _checkimageloaded();

    std::string hex_color = params[0].stringValue();
    int width = params[1];
    int height = params[2];
    cv::Vec3b bgr_color;

    if (!_decodehexcolor(hex_color, bgr_color)) {
      throw Php::Exception("Unrecognized color string");
    }

    _operations.push_back(std::bind(&Photon_OpenCV::_border,
          this,
          width,
          height,
          bgr_color));

    _expected_width += width*2;
    _expected_height += height*2;
  }

  Php::Value blobrequiresreencoding() {
    return _requiresreencoding();
  }
};
cmsHPROFILE Photon_OpenCV::_srgb_profile = nullptr;

extern "C" {
  PHPCPP_EXPORT void *get_module() {
    static Php::Extension extension("photon-opencv", "0.2.34");

    Php::Class<Photon_OpenCV> photon_opencv("Photon_OpenCV");

    photon_opencv.constant("CHANNEL_OPACITY", Photon_OpenCV::CHANNEL_OPACITY);
    photon_opencv.constant("COLORSPACE_RGB", Photon_OpenCV::COLORSPACE_RGB);
    photon_opencv.constant("FILTER_LANCZOS", Photon_OpenCV::FILTER_LANCZOS);
    photon_opencv.constant("FILTER_CUBIC", Photon_OpenCV::FILTER_CUBIC);
    photon_opencv.constant("FILTER_TRIANGLE", Photon_OpenCV::FILTER_TRIANGLE);
    photon_opencv.constant("FILTER_POINT", Photon_OpenCV::FILTER_POINT);
    photon_opencv.constant("FILTER_BOX", Photon_OpenCV::FILTER_BOX);
    photon_opencv.constant("IMGTYPE_COLORSEPARATIONMATTE",
        Photon_OpenCV::IMGTYPE_COLORSEPARATIONMATTE);
    photon_opencv.constant("IMGTYPE_GRAYSCALE",
        Photon_OpenCV::IMGTYPE_GRAYSCALE);
    photon_opencv.constant("IMGTYPE_GRAYSCALEMATTE",
        Photon_OpenCV::IMGTYPE_GRAYSCALEMATTE);
    photon_opencv.constant("IMGTYPE_PALETTE",
        Photon_OpenCV::IMGTYPE_PALETTE);
    photon_opencv.constant("IMGTYPE_PALETTEMATTE",
        Photon_OpenCV::IMGTYPE_PALETTEMATTE);
    photon_opencv.constant("IMGTYPE_TRUECOLOR",
        Photon_OpenCV::IMGTYPE_TRUECOLOR);
    photon_opencv.constant("IMGTYPE_TRUECOLORMATTE",
        Photon_OpenCV::IMGTYPE_TRUECOLORMATTE);
    photon_opencv.constant("ORIENTATION_UNDEFINED",
        Photon_OpenCV::ORIENTATION_UNDEFINED);
    photon_opencv.constant("ORIENTATION_TOPLEFT",
        Photon_OpenCV::ORIENTATION_TOPLEFT);
    photon_opencv.constant("ORIENTATION_TOPRIGHT",
        Photon_OpenCV::ORIENTATION_TOPRIGHT);
    photon_opencv.constant("ORIENTATION_BOTTOMRIGHT",
        Photon_OpenCV::ORIENTATION_BOTTOMRIGHT);
    photon_opencv.constant("ORIENTATION_BOTTOMLEFT",
        Photon_OpenCV::ORIENTATION_BOTTOMLEFT);
    photon_opencv.constant("ORIENTATION_LEFTTOP",
        Photon_OpenCV::ORIENTATION_LEFTTOP);
    photon_opencv.constant("ORIENTATION_RIGHTTOP",
        Photon_OpenCV::ORIENTATION_RIGHTTOP);
    photon_opencv.constant("ORIENTATION_RIGHTBOTTOM",
        Photon_OpenCV::ORIENTATION_RIGHTBOTTOM);
    photon_opencv.constant("ORIENTATION_LEFTBOTTOM",
        Photon_OpenCV::ORIENTATION_LEFTBOTTOM);

    photon_opencv.method<&Photon_OpenCV::readimageblob>("readimageblob", {
      Php::ByRef("raw_image_data", Php::Type::String),
    });
    photon_opencv.method<&Photon_OpenCV::readimage>("readimage", {
      Php::ByVal("filepath", Php::Type::String),
    });
    photon_opencv.method<&Photon_OpenCV::writeimage>("writeimage", {
      Php::ByVal("output", Php::Type::String),
    });
    photon_opencv.method<&Photon_OpenCV::getimageblob>("getimageblob");

    // Not in Gmagick
    photon_opencv.method<&Photon_OpenCV::getlasterror>("getlasterror");

    photon_opencv.method<&Photon_OpenCV::getimagewidth>("getimagewidth");
    photon_opencv.method<&Photon_OpenCV::getimageheight>("getimageheight");
    photon_opencv.method<&Photon_OpenCV::getimagechanneldepth>(
      "getimagechanneldepth", {
      Php::ByVal("channel", Php::Type::Numeric),
    });

    photon_opencv.method<&Photon_OpenCV::getimageformat>("getimageformat");
    photon_opencv.method<&Photon_OpenCV::setimageformat>("setimageformat", {
      Php::ByVal("format", Php::Type::String),
    });

    photon_opencv.method<&Photon_OpenCV::setcompressionquality>(
      "setcompressionquality", {
      Php::ByVal("format", Php::Type::String),
    });
    // Not in Gmagick
    photon_opencv.method<&Photon_OpenCV::getcompressionquality>(
      "getcompressionquality");

    // Not in Gmagick, but in GraphicsMagick
    photon_opencv.method<&Photon_OpenCV::autoorientimage>("autoorientimage", {
      Php::ByVal("current_orientation", Php::Type::Numeric),
    });
    photon_opencv.method<&Photon_OpenCV::getimageorientation>(
        "getimageorientation");

    photon_opencv.method<&Photon_OpenCV::setimageprofile>(
      "setimageprofile", {
      Php::ByVal("name", Php::Type::String),
      Php::ByVal("profile", Php::Type::Null),
    });

    photon_opencv.method<&Photon_OpenCV::getimagetype>("getimagetype");
    photon_opencv.method<&Photon_OpenCV::setimagetype>("setimagetype", {
      Php::ByVal("type", Php::Type::Numeric)
    });

    photon_opencv.method<&Photon_OpenCV::resizeimage>("resizeimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
      Php::ByVal("filter", Php::Type::Numeric, false),
      Php::ByVal("fit", Php::Type::Bool, false),
    });
    photon_opencv.method<&Photon_OpenCV::scaleimage>("scaleimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
      Php::ByVal("fit", Php::Type::Bool, false),
    });

    photon_opencv.method<&Photon_OpenCV::rotateimage>("rotateimage", {
      Php::ByVal("background", Php::Type::String),
      Php::ByVal("degrees", Php::Type::Numeric),
    });

    photon_opencv.method<&Photon_OpenCV::cropimage>("cropimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
      Php::ByVal("x", Php::Type::Numeric),
      Php::ByVal("y", Php::Type::Numeric),
    });

    photon_opencv.method<&Photon_OpenCV::borderimage>("borderimage", {
      Php::ByVal("color", Php::Type::String),
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
    });

    photon_opencv.method<&Photon_OpenCV::setimageoption>("setimageoption", {
      Php::ByVal("format", Php::Type::String),
      Php::ByVal("key", Php::Type::String),
      Php::ByVal("value", Php::Type::String),
    });

    // Not in Gmagick
    photon_opencv.method<&Photon_OpenCV::blobrequiresreencoding>(
        "blobrequiresreencoding");

    extension.add(std::move(photon_opencv));

    // Default to 2 if not set
    extension.add(Php::Ini("photon.opencv_threads", 2));

    return extension;
  }
}

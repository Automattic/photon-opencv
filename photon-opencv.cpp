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
#include "tempfile.h"
#include "srgb.icc.h"

#define _checkimageloaded() { \
  if (_raw_image_data.empty()) { \
    throw Php::Exception("Can not process empty object"); \
  } \
}

class Photon_OpenCV : public Php::Base {
protected:
  cv::Mat _img;
  std::string _last_error;
  std::string _format;
  int _type;
  int _compression_quality;
  std::vector<uint8_t> _icc_profile;
  std::string _raw_image_data;
  int _header_width;
  int _header_height;
  int _header_channels;
  Exiv2::Value::AutoPtr _original_orientation;
  std::map<std::string, std::string> _image_options;

  const int WEBP_DEFAULT_QUALITY = 75;
  const int AVIF_DEFAULT_QUALITY = 75;
  const int JPEG_DEFAULT_QUALITY = 75;
  const int PNG_DEFAULT_QUALITY = 21;

  static cmsHPROFILE _srgb_profile;

  static const heif_encoder_descriptor *_aom_descriptor;

  void _enforce8u() {
    if (CV_8U != _img.depth()) {
      /* Proper convertion is mostly guess work, but it's fairly rare and
         these are reasonable assumptions */
      double alpha, beta;
      switch (_img.depth()) {
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
      _img.convertTo(_img, CV_8U, alpha, beta);
    }
  }

  bool _converttosrgb() {
    if (_icc_profile.empty()) {
      return true;
    }

    cmsHPROFILE embedded_profile = cmsOpenProfileFromMem(
        _icc_profile.data(), _icc_profile.size());
    if (!embedded_profile) {
      _last_error = "Failed to decode embedded profile";
      _icc_profile.clear();
      return false;
    }

    int storage_format;
    int num_intensity_channels;
    switch (_img.channels()) {
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
      INTENT_PERCEPTUAL, 0
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
    if (!_img.isContinuous()) {
      _img = _img.clone();
    }

    int output_type = _imagehasalpha()? CV_8UC4 : CV_8UC3;
    cv::Mat transformed_img = cv::Mat(_img.rows, _img.cols, output_type);

    /* The sRGB profile can't handle the alpha channel. We make sure it's
       skipped when applying the profile */
    cv::Mat no_alpha_img = _img.
      reshape(1, _img.rows*_img.cols).
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
      cv::Mat alpha_only_img = _img.reshape(1, _img.rows*_img.cols).
        colRange(num_intensity_channels, num_intensity_channels+1);
      cv::Mat alpha_only_transformed_img = transformed_img.
        reshape(1, transformed_img.rows*transformed_img.cols).
        colRange(3, 4);
      alpha_only_img.copyTo(alpha_only_transformed_img);
    }

    _img = transformed_img;
    _icc_profile.clear();

    return true;
  }

  static int _gmagickfilter2opencvinter(int filter, int default_filter) {
    /* Disable custom filters for now. Photon uses lanczos to circunvent an
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

    std::unique_ptr<heif_context, decltype(&heif_context_free)> context(
      heif_context_alloc(), &heif_context_free);

    if (!heif_context_get_encoder_descriptors(context.get(),
        heif_compression_AV1,
        "aom",
        &_aom_descriptor,
        1)) {
      throw std::runtime_error("AOM encoder for AVIF images not available");
    }
  }

  bool _imagehasalpha() {
    /* Even number of channels means it's either GA or BGRA */
    int num_channels = _img.empty()? _header_channels : _img.channels();
    return !(num_channels & 1);
  }

  /* Assumes alpha is the last channel */
  void _associatealpha() {
    /* Continuity required for `reshape()` */
    if (!_img.isContinuous()) {
      _img = _img.clone();
    }

    int alpha_channel = _img.channels()-1;
    cv::Mat alpha = _img.reshape(1, _img.rows*_img.cols).
      colRange(alpha_channel, alpha_channel+1);

    for (int i = 0; i < alpha_channel; i++) {
      cv::Mat color = _img.reshape(1, _img.rows*_img.cols).colRange(i, i+1);
      cv::multiply(color, alpha, color, 1./255);
    }
  }

  /* Assumes alpha is the last channel */
  void _dissociatealpha() {
    /* Continuity required for `reshape()` */
    if (!_img.isContinuous()) {
      _img = _img.clone();
    }

    int alpha_channel = _img.channels()-1;
    cv::Mat alpha = _img.reshape(1, _img.rows*_img.cols).
      colRange(alpha_channel, alpha_channel+1);

    for (int i = 0; i < alpha_channel; i++) {
      cv::Mat color = _img.reshape(1, _img.rows*_img.cols).colRange(i, i+1);
      cv::divide(color, alpha, color, 255.);
    }
  }

  void _transparencysaferesize(int width, int height, int filter) {
    if (_imagehasalpha()) {
      _associatealpha();
    }
    cv::resize(_img, _img, cv::Size(width, height), 0, 0, filter);
    if (_imagehasalpha()) {
      _dissociatealpha();
    }
  }

  bool _maybedecodeimage(bool silent=true) {
    _checkimageloaded();

    if (!_img.empty()) {
      return true;
    }

    /*
     * Ideally, we would decode directly from memory. However, opencv is more
     * strict when imdecode is used. This results in jpegs that are missing
     * bytes and pngs that have broken exif information to only be parsed by
     * imread. In order to support more files, we write the files to the
     * filesystem so that imread can be used.
    */
    TempFile temp_image_file(_raw_image_data);
    _img = cv::imread(temp_image_file.get_path(), cv::IMREAD_UNCHANGED);

    if (_img.empty()) {
      // Not supported by OpenCV, try manually with libheif

      std::unique_ptr<heif_context, decltype(&heif_context_free)> context(
        heif_context_alloc(), &heif_context_free);

      heif_error error;

      error = heif_context_read_from_memory_without_copy(context.get(),
        (void *) _raw_image_data.data(),
        _raw_image_data.size(),
        nullptr);
      if (error.code) {
        std::string message = std::string("Failed to read heif context: ")
          + error.message;

        if (!silent) {
          throw Php::Exception(message);
        }

        _last_error = message;
        return false;
      }

      std::unique_ptr<heif_image_handle,
        decltype(&heif_image_handle_release)>
        handle(nullptr, &heif_image_handle_release);
      heif_image_handle *raw_handle = nullptr;
      error = heif_context_get_primary_image_handle(context.get(),
        &raw_handle);
      handle.reset(raw_handle);
      if (error.code) {
        std::string message = std::string(
            "Failed to get primary image handle: ") + error.message;

        if (!silent) {
          throw Php::Exception(message);
        }

        _last_error = message;
        return false;
      }

      bool has_alpha = heif_image_handle_has_alpha_channel(handle.get());

      std::unique_ptr<heif_image, decltype(&heif_image_release)>
        h_image(nullptr, &heif_image_release);
      heif_image *raw_h_image = nullptr;
      error = heif_decode_image(handle.get(),
        &raw_h_image,
        heif_colorspace_RGB,
        has_alpha? heif_chroma_interleaved_RGBA : heif_chroma_interleaved_RGB,
        nullptr);
      h_image.reset(raw_h_image);
      if (error.code) {
        std::string message = std::string("Failed to decode image: ")
          + error.message;

        if (!silent) {
          throw Php::Exception(message);
        }

        _last_error = message;
        return false;
      }

      int stride;
      uint8_t *data = heif_image_get_plane(h_image.get(),
        heif_channel_interleaved,
        &stride);
      cv::Mat rgb(heif_image_handle_get_height(handle.get()),
        heif_image_handle_get_width(handle.get()),
        has_alpha? CV_8UC4 : CV_8UC3,
        data,
        stride);
      cv::cvtColor(rgb,
          _img,
          has_alpha? cv::COLOR_RGBA2BGRA : cv::COLOR_RGB2BGR);

      // This redundand ICC profile extraction code can be removed when
      // exiv2 0.27.4 is released, as it should support the new formats.
      // For now, it will fail gracefully later in the loading process
      size_t profile_size = heif_image_get_raw_color_profile_size(
        h_image.get());
      if (profile_size) {
        _icc_profile.resize(profile_size);
        error = heif_image_handle_get_raw_color_profile(handle.get(),
          _icc_profile.data());
        if (error.code) {
          std::string message = std::string(
              "Failed to extract color profile: ") + error.message;

          if (!silent) {
            throw Php::Exception(message);
          }

          _last_error = message;
          return false;
        }
      }
    }

    _enforce8u();
    return true;
  }

  bool _loadimagefromrawdata() {
    /* Clear image in case object is being reused */
    _img = cv::Mat();
    _icc_profile.clear();
    _image_options.clear();
    _compression_quality = -1;

    Exiv2::Image::AutoPtr exiv_img;
    try {
      exiv_img = Exiv2::ImageFactory::open(
        (Exiv2::byte *) _raw_image_data.data(), _raw_image_data.size());
    }
    catch (Exiv2::Error &error) {
      _last_error = "Exiv2 failed to open image: ";
      _last_error += error.what();
      _raw_image_data.clear();
      return false;
    }

    try {
      exiv_img->readMetadata();
    }
    catch(Exiv2::Error &error) {
      // Failing to read the metadata is not critical, but it requires us to
      // try to decode the image to get the proper information early
      _maybedecodeimage(false);
    }

    switch (exiv_img->imageType()) {
      case Exiv2::ImageType::png:
        _format = "png";
        _header_channels = _getchannelsfromrawpng();
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

      default:
        /* Default to jpeg, but disable lazy loading */
        _format = "jpeg";
        _maybedecodeimage(false);
        _header_channels = -1;
        break;
    }

    auto &exif = exiv_img->exifData();
    Exiv2::ExifKey orientation_key("Exif.Image.Orientation");
    auto orientation_pos = exif.findKey(orientation_key);

    _original_orientation.release();
    if (orientation_pos != exif.end()) {
      _original_orientation = orientation_pos->getValue();
    }

    _header_width = exiv_img->pixelWidth();
    _header_height = exiv_img->pixelHeight();

    if (exiv_img->iccProfileDefined()) {
      const Exiv2::DataBuf *profile = exiv_img->iccProfile();
      _icc_profile.resize(profile->size_);
      std::memcpy(_icc_profile.data(), profile->pData_, profile->size_);
    }

    /* Palettes are automatically converted to RGB on decode */
    switch (_img.empty()? _header_channels : _img.channels()) {
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
    if (_img.channels() <= 2) {
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
    int img_width = _img.empty()? _header_width : _img.cols;
    int img_height = _img.empty()? _header_height : _img.rows ;

    double img_ratio = (double) img_width / (double) img_height;
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
    /* OpenCV strips the profile, we need to apply it first */
    _converttosrgb();

    bool lossless_requested = false;
    auto lossless_option = _image_options.find(_format + ":lossless");
    if (lossless_option != _image_options.end()
        && "true" == lossless_option->second) {
      lossless_requested = true;
    }

    if ("avif" == _format) {
      // Explicit rotation handling and early return, as exiv2 doesn't support
      // the format and libheif provides no way of inseting orientation
      // metadata. Remove rotation code and refactor when exiv2 0.27.4 is out
      cv::Mat original_img;

      long orientation = _original_orientation.get()?
        _original_orientation.get()->toLong() : 1;
      if (orientation < 1 || orientation > 8) {
        orientation = 1;
      }

      if (orientation != 1) {
        int rotation = -1;
        switch (orientation) {
          case 3:
          case 4:
            rotation = cv::ROTATE_180;
            break;

          case 5:
          case 6:
            rotation = cv::ROTATE_90_CLOCKWISE;
            break;

          case 7:
          case 8:
            rotation = cv::ROTATE_90_COUNTERCLOCKWISE;
            break;
        }

        _img.copyTo(original_img);
        if (rotation != -1) {
          cv::rotate(_img, _img, rotation);
        }
        if (2 == orientation
            || 4 == orientation
            || 5 == orientation
            || 7 == orientation) {
          cv::flip(_img, _img, 1);
        }
      }

      bool status = _encodeheifimage(heif_compression_AV1,
          output_buffer,
          lossless_requested);

      if (!original_img.empty()) {
        _img = original_img;
      }

      return status;
    }

    std::vector<int> img_parameters;
    if ("jpeg" == _format) {
      int quality = -1 == _compression_quality?
        JPEG_DEFAULT_QUALITY : _compression_quality;
      img_parameters.push_back(cv::IMWRITE_JPEG_QUALITY);
      img_parameters.push_back(quality);
    }
    else if ("png" == _format) {
      int quality = -1 == _compression_quality?
        PNG_DEFAULT_QUALITY : _compression_quality;
      /* GMagick uses a single scalar for storing two values:
        _compressioN_quality = compression_level*10 + filter_type */
      img_parameters.push_back(cv::IMWRITE_PNG_COMPRESSION);
      img_parameters.push_back(quality/10);
      /* OpenCV does not support setting the filters like GMagick,
         instead we get to pick the strategy, so we ignore it
         https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html */
    }
    else if ("webp" == _format) {
      img_parameters.push_back(cv::IMWRITE_WEBP_QUALITY);
      if (lossless_requested) {
        img_parameters.push_back(101);
      }
      else {
        int quality = -1 == _compression_quality?
          WEBP_DEFAULT_QUALITY : _compression_quality;
        img_parameters.push_back(quality);
      }
    }

    std::string failure_details;
    bool encoded = false;
    try {
      encoded = cv::imencode("." + _format,
          _img,
          output_buffer,
          img_parameters);
    }
    catch (cv::Exception &e) {
      failure_details = e.what();
    }

    if (!encoded) {
      _last_error = "Failed to encode image with opencv";
      if (failure_details.size()) {
        _last_error += ": " + failure_details;
      }
      return false;
    }

    /* Manually reinsert orientation exif data if it has meaning */
    long exif_orientation = _original_orientation.get()?
      _original_orientation.get()->toLong() : 0;
    if (exif_orientation > 1 && exif_orientation <= 8) {
      Exiv2::Image::AutoPtr exiv_img;
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

  bool _encodeheifimage(heif_compression_format heif_format,
      std::vector<uint8_t> &output_buffer,
      bool lossless) {

    heif_error error;

    std::unique_ptr<heif_context, decltype(&heif_context_free)> context(
      heif_context_alloc(), &heif_context_free);

    std::unique_ptr<heif_encoder, decltype(&heif_encoder_release)> encoder(
      nullptr,
      &heif_encoder_release);
    heif_encoder *raw_encoder;

    // Force pick AOM for AVIF images, as it supports lossless encoding
    if (heif_compression_AV1 == heif_format) {
      error = heif_context_get_encoder(context.get(),
          _aom_descriptor,
          &raw_encoder);
    }
    else {
      error = heif_context_get_encoder_for_format(context.get(),
          heif_format,
          &raw_encoder);
    }
    encoder.reset(raw_encoder);
    if (error.code != heif_error_Ok) {
      _last_error = "Heif encoding creation: "
        + std::string(error.message);
      return false;
    }

    std::unique_ptr<heif_encoding_options,
      decltype(&heif_encoding_options_free)>
      options(nullptr, &heif_encoding_options_free);
    std::unique_ptr<heif_color_profile_nclx,
      decltype(&heif_nclx_color_profile_free)>
      nclx(nullptr, &heif_nclx_color_profile_free);

    if (lossless) {
      heif_encoder_set_lossless(encoder.get(), 1);
      heif_encoder_set_parameter(encoder.get(), "chroma", "444");

      nclx.reset(heif_nclx_color_profile_alloc());
      // Only set version 1 fields
      nclx->matrix_coefficients = heif_matrix_coefficients_RGB_GBR;
      nclx->transfer_characteristics =
        heif_transfer_characteristic_unspecified;
      nclx->color_primaries = heif_color_primaries_unspecified;
      nclx->full_range_flag = 1;

      options.reset(heif_encoding_options_alloc());
      options->output_nclx_profile = nclx.get();

    }
    else {
      int quality = -1 == _compression_quality?
        AVIF_DEFAULT_QUALITY : _compression_quality;
      heif_encoder_set_lossy_quality(encoder.get(), quality);
    }

    heif_colorspace colorspace = _img.channels() >= 3?
      heif_colorspace_RGB : heif_colorspace_monochrome;
    heif_chroma chroma = _img.channels() >= 3?
      heif_chroma_444 : heif_chroma_monochrome;

    heif_channel channel_map[][4] = {
      {heif_channel_Y},
      {heif_channel_Y, heif_channel_Alpha},
      {heif_channel_B, heif_channel_G, heif_channel_R},
      {heif_channel_B, heif_channel_G, heif_channel_R, heif_channel_Alpha},
    };

    std::unique_ptr<heif_image, decltype(&heif_image_release)> image(
      nullptr,
      &heif_image_release);
    heif_image *raw_image;
    error = heif_image_create(_img.cols,
        _img.rows,
        colorspace,
        chroma,
        &raw_image);
    image.reset(raw_image);
    if (error.code != heif_error_Ok) {
      _last_error = "Heif image creation: "
        + std::string(error.message);
      return false;
    }

    std::vector<cv::Mat> channel_mats;
    for (int i = 0; i < _img.channels(); i++) {
      heif_channel channel_type = channel_map[_img.channels()-1][i];

      error = heif_image_add_plane(image.get(),
          channel_type,
          _img.cols,
          _img.rows,
          8);
      if (error.code != heif_error_Ok) {
        _last_error = "Heif plane addition: "
          + std::string(error.message);
        return false;
      }

      int stride;
      uint8_t *data = heif_image_get_plane(image.get(), channel_type, &stride);
      channel_mats.emplace_back(_img.rows, _img.cols, CV_8UC1, data, stride);
    }

    int trivial_fromto[] = {0, 0, 1, 1, 2, 2, 3, 3};
    cv::mixChannels(&_img,
        1,
        channel_mats.data(),
        channel_mats.size(),
        trivial_fromto,
        _img.channels());

    error = heif_context_encode_image(context.get(),
        image.get(),
        encoder.get(),
        options.get(),
        nullptr);
    if (error.code != heif_error_Ok) {
      _last_error = "Heif image encoding: "
        + std::string(error.message);
      return false;
    }

    heif_writer simple_ram_copier;
    simple_ram_copier.writer_api_version = 1;
    simple_ram_copier.write = []
      (heif_context *ctx, const void *data, size_t size, void *userdata) {
        (void) ctx;

        std::vector<uint8_t> *buffer = (std::vector<uint8_t> *) userdata;
        buffer->resize(size);
        std::memcpy(buffer->data(), data, size);

        heif_error error;
        error.code = heif_error_Ok;
        return error;
      };

    error = heif_context_write(context.get(),
        &simple_ram_copier,
        &output_buffer);
    if (error.code != heif_error_Ok) {
      _last_error = "Heif context writing: "
        + std::string(error.message);
      return false;
    }

    return true;
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
    if (_img.empty()) {
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
    if (_img.empty()) {
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

    return (_img.empty()? _header_width : _img.cols);
  }

  Php::Value getimageheight() {
    _checkimageloaded();
    return (_img.empty()? _header_height : _img.rows);
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

    if (!_maybedecodeimage()) {
      return;
    }

    _format = new_format;
  }

  void setimageoption(Php::Parameters &params) {
    std::string format = params[0].stringValue();
    std::transform(format.begin(), format.end(), format.begin(), ::tolower);

    std::string key = params[1].stringValue();
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);

    std::string value = params[2].stringValue();

    std::string real_key = format + ":" + key;
    auto option = _image_options.find(real_key);
    if (option == _image_options.end() || option->second != value) {
      if (!_maybedecodeimage()) {
        return;
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

    long orientation = params[0];

    // Orientation follows exif specs, valid values are in [1,8], 0 is undef
    if (orientation < 0 || orientation > 8) {
      throw Php::Exception("Invalid orientation requested");
    }

    if (ORIENTATION_UNDEFINED == orientation) {
      orientation = _original_orientation.get()?
        _original_orientation.get()->toLong() : ORIENTATION_TOPLEFT;
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
        rotation = -1;
        break;

      case ORIENTATION_TOPLEFT:
      default:
        // noop
        return;
    }

    if (!_maybedecodeimage()) {
      return;
    }

    if (-1 != rotation) {
      cv::rotate(_img, _img, rotation);
    }
    if (ORIENTATION_TOPRIGHT == orientation
        || ORIENTATION_BOTTOMLEFT == orientation
        || ORIENTATION_LEFTTOP == orientation
        || ORIENTATION_RIGHTBOTTOM == orientation) {
      cv::flip(_img, _img, 1);
    }

    // Exif not reset intentionally, GraphicsMagick doesn't support it for Jpeg
  }

  void setimageprofile(Php::Parameters &params) {
    std::string name = params[0].stringValue();
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    if (!_maybedecodeimage()) {
      return;
    }

    if ("exif" == name) {
      if (!params[1].isNull()) {
        throw Php::Exception("Exif replacement unimplemented, only removal");
      }
      _original_orientation.release();
    }
    else if ("icc" == name) {
      if (params[1].isNull()) {
        _icc_profile.clear();
      }
      else {
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

  void setimagetype() {
    // Unimplemented
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
    int img_width = _img.empty()? _header_width : _img.cols;
    int img_height = _img.empty()? _header_height : _img.rows ;
    if (width == img_width && height == img_height) {
      return;
    }

    if (!_maybedecodeimage()) {
      return;
    }

    _transparencysaferesize(width, height,
        _gmagickfilter2opencvinter(filter, default_filter));
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
    int img_width = _img.empty()? _header_width : _img.cols;
    int img_height = _img.empty()? _header_height : _img.rows ;
    if (width == img_width && height == img_height) {
      return;
    }

    if (!_maybedecodeimage()) {
      return;
    }

    _transparencysaferesize(width, height, cv::INTER_AREA);
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

    int width = _img.empty()? _header_width : _img.cols;
    int height = _img.empty()? _header_height : _img.rows ;
    x = std::max(0, std::min(x, width));
    y = std::max(0, std::min(y, height));
    x2 = std::max(0, std::min(x2, width));
    y2 = std::max(0, std::min(y2, height));

    /* Prevent image from being loaded if it's a noop */
    if (!x && !y && x2 == _header_width && y2 == _header_height) {
      return;
    }

    if (!_maybedecodeimage()) {
      return;
    }

    _img = _img(cv::Rect(x, y, x2-x, y2-y));
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

    if (!_maybedecodeimage()) {
      return;
    }

    cv::rotate(_img, _img, rotation_constant);
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

    if (!_maybedecodeimage()) {
      return;
    }

    cv::Mat dst(_img.rows + height*2,
      _img.cols + width*2,
      _img.type(),
      _bgrtoloadedimagetype(bgr_color));

    _img.copyTo(dst(cv::Rect(width, height, _img.cols, _img.rows)));
    _img = dst;
  }
};
cmsHPROFILE Photon_OpenCV::_srgb_profile = nullptr;
const heif_encoder_descriptor *Photon_OpenCV::_aom_descriptor = nullptr;

extern "C" {
  PHPCPP_EXPORT void *get_module() {
    static Php::Extension extension("photon-opencv", "0.2.13");

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

    photon_opencv.method<&Photon_OpenCV::setimageprofile>(
      "setimageprofile", {
      Php::ByVal("name", Php::Type::String),
      Php::ByVal("profile", Php::Type::Null),
    });

    photon_opencv.method<&Photon_OpenCV::getimagetype>("getimagetype");
    photon_opencv.method<&Photon_OpenCV::setimagetype>("setimagetype");

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

    extension.add(std::move(photon_opencv));

    return extension;
  }
}

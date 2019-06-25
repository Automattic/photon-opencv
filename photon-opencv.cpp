#include <phpcpp.h>
#include <string>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include <lcms2.h>
#include <zend.h>
#include <zend_constants.h>
#include "srgb.icc.h"

#define check_image_loaded() { \
  if (_img.empty()) { \
    _last_error = "No image loaded"; \
    return false; \
  } \
}

class Photon_OpenCV : public Php::Base {
protected:
  cv::Mat _img;
  std::string _last_error;
  std::string _format;
  int _type;
  int _compression_quality = 80;
  std::vector<uint8_t> _icc_profile;

  static cmsHPROFILE _srgb_profile;

  /* Cached Gmagick constants */
  static int _gmagick_channel_opacity;
  static int _gmagick_filter_lanczos;
  static int _gmagick_filter_cubic;
  static int _gmagick_filter_triangle;
  static int _gmagick_filter_point;
  static int _gmagick_filter_box;

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
      return false;
    }

    int storage_format;
    switch (_img.channels()) {
      case 1:
        storage_format = TYPE_GRAY_8;
        break;

      case 2:
        storage_format = TYPE_GRAYA_8;
        break;

      case 3:
        storage_format = TYPE_BGR_8;
        break;

      case 4:
        storage_format = TYPE_BGRA_8;
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

    cmsCloseProfile(embedded_profile);

    cv::Mat transformed_img = cv::Mat(_img.rows, _img.cols, CV_8UC3);
    cmsDoTransformLineStride(
        transform,
        _img.data, transformed_img.data,
        _img.cols, _img.rows,
        _img.step, transformed_img.step,
        0, 0
    );
    _img = transformed_img;

    cmsDeleteTransform(transform);

    _icc_profile.clear();
    return true;
  }

  static int _gmagickfilter2opencvinter(int filter, int default_filter) {
    int opencv_filter = default_filter;

    if (filter == _gmagick_filter_lanczos) {
      opencv_filter = cv::INTER_LANCZOS4;
    }
    else if (filter == _gmagick_filter_point) {
      opencv_filter = cv::INTER_NEAREST;
    }
    else if (filter == _gmagick_filter_box) {
      opencv_filter = cv::INTER_AREA;
    }
    else if (filter == _gmagick_filter_triangle) {
      opencv_filter = cv::INTER_LINEAR;
    }
    else if (filter == _gmagick_filter_cubic) {
      opencv_filter = cv::INTER_CUBIC;
    }

    return opencv_filter;
  }

  /* Workaround for a bug in php-cpp.
     See https://github.com/CopernicaMarketingSoftware/PHP-CPP/issues/423
     If this is fixed and removed, it's safe to remove the zend includes */
  static Php::Value _getconstantex(const char *name) {
    zend_string *zstr = zend_string_init(name, ::strlen(name), 1);
    auto result = zend_get_constant_ex(zstr, nullptr, ZEND_FETCH_CLASS_SILENT);
    zend_string_release(zstr);

    return result;
  }

  static void _initialize() {
    /* Load constants */
    _gmagick_channel_opacity = _getconstantex("Gmagick::CHANNEL_OPACITY");

    _gmagick_filter_lanczos = _getconstantex("Gmagick::FILTER_LANCZOS");
    _gmagick_filter_cubic = _getconstantex("Gmagick::FILTER_CUBIC");
    _gmagick_filter_triangle = _getconstantex("Gmagick::FILTER_TRIANGLE");
    _gmagick_filter_point = _getconstantex("Gmagick::FILTER_POINT");
    _gmagick_filter_box = _getconstantex("Gmagick::FILTER_BOX");

    /* Load default sRGB profile */
    _srgb_profile = cmsOpenProfileFromMem(srgb_icc, sizeof(srgb_icc)-1);
    if (!_srgb_profile) {
      throw Php::Exception("Failed to decode builtin sRGB profile");
    }
  }

public:
  Photon_OpenCV() {
    /* Static local intilization is thread safe */
    static std::once_flag initialized;
    std::call_once(initialized, _initialize);
  }

  Php::Value readimageblob(Php::Parameters &params) {
    std::string raw_image_data = params[0];

    if (raw_image_data.empty()) {
      _last_error = "Input buffer is empty";
      return false;
    }

    cv::Mat raw_data(1, raw_image_data.size(), CV_8UC1,
      (void *) raw_image_data.data());
    _img = cv::imdecode(raw_data, cv::IMREAD_UNCHANGED);

    if (_img.empty()) {
      _last_error = "Failed to decode image";
      return false;
    }

    /* Use any number of channels, but enforce 8 bits per channel */
    _enforce8u();

    Exiv2::Image::AutoPtr exiv_img;
    try {
      exiv_img = Exiv2::ImageFactory::open(
        (Exiv2::byte *) raw_image_data.data(), raw_image_data.size());
      exiv_img->readMetadata();
    }
    catch (Exiv2::Error &error) {
      _last_error = error.what();
      return false;
    }
    switch (exiv_img->imageType()) {
      case Exiv2::ImageType::png:
        _format = "png";
        break;

      default:
        _format = "jpeg";
    }

    if (exiv_img->iccProfileDefined()) {
      const Exiv2::DataBuf *profile = exiv_img->iccProfile();
      _icc_profile.resize(profile->size_);
      std::memcpy(_icc_profile.data(), profile->pData_, profile->size_);
    }

    // TODO: also implement this
    _type = 0;

    return true;
  }

  Php::Value writeimage(Php::Parameters &params) {
    check_image_loaded();

    /* OpenCV strips the profile, we need to apply it first */
    _converttosrgb();

    std::string wanted_output = params[0];

    std::vector<int> img_parameters;
    if ("jpeg" == _format) {
      img_parameters.push_back(cv::IMWRITE_JPEG_QUALITY);
      img_parameters.push_back(_compression_quality);
    }
    else if ("png" == _format) {
      /* GMagick uses a single scalar for storing two values:
        _compressioN_quality = compression_level*10 + filter_type */
      img_parameters.push_back(cv::IMWRITE_PNG_COMPRESSION);
      img_parameters.push_back(_compression_quality/10);
      /* OpenCV does not support setting the filters like GMagick,
         instead we get to pick the strategy, so we ignore it
         https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html */
    }

    /* OpenCV looks at the extension to determine the format.
       We make sure it's there, then rename it to the expected filename. */
    std::string actual_output = wanted_output + "." + _format;
    if (!cv::imwrite(actual_output, _img, img_parameters)) {
      _last_error = "Failed to encode image";
      return false;
    }
    if (std::rename(actual_output.c_str(), wanted_output.c_str())) {
      _last_error = "Failed to rename generated image";
      return false;
    }

    return true;
  }


  Php::Value getlasterror() {
    return _last_error;
  }

  Php::Value getimagewidth() {
    check_image_loaded();
    return _img.cols;
  }

  Php::Value getimageheight() {
    check_image_loaded();
    return _img.rows;
  }

  Php::Value getimageformat() {
    check_image_loaded();
    return _format;
  }
  Php::Value setimageformat(Php::Parameters &params) {
    _format = std::string(params[0]);
    std::transform(_format.begin(), _format.end(),
      _format.begin(), ::tolower);
    return true;
  }

  Php::Value setcompressionquality(Php::Parameters &params) {
    _compression_quality = params[0];
    return true;
  }

  Php::Value getimagetype() {
    check_image_loaded();
    return _type;
  }

  Php::Value resizeimage(Php::Parameters &params) {
    check_image_loaded();

    /* Blur is ignored */
    int width = std::max(1, (int) params[0]);
    int height = std::max(1, (int) params[1]);
    int filter = params.size() > 2? (int) params[2] : -1;
    /* AREA looks excellent when downsampling and is super fast, but
       is visually awful when upscaling  */
    int default_filter = width > _img.cols || height > _img.rows?
      cv::INTER_LANCZOS4 : cv::INTER_AREA;

    cv::resize(_img, _img, cv::Size(width, height), 0, 0,
        _gmagickfilter2opencvinter(filter, default_filter));

    return true;
  }

  Php::Value cropimage(Php::Parameters &params) {
    check_image_loaded();

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
    x = std::max(0, std::min(x, _img.cols-1));
    y = std::max(0, std::min(y, _img.rows-1));
    x2 = std::max(0, std::min(x2, _img.cols-1));
    y2 = std::max(0, std::min(y2, _img.rows-1));

    _img = _img(cv::Rect(x, y, x2-x+1, y2-y+1));

    return true;
  }

  Php::Value getimagechanneldepth(Php::Parameters &params) {
    check_image_loaded();

    int channel = params[0];

    /* Gmagick's channels don't map directly to OpenCV's, but Photon is only
       interested in the opacity */
    if (channel != _gmagick_channel_opacity) {
      return 8;
    }

    /* Opacity channel is present with an even number number of channels */
    return (_img.channels() & 1? 0 : 8);
  }
};
cmsHPROFILE Photon_OpenCV::_srgb_profile = NULL;

int Photon_OpenCV::_gmagick_channel_opacity = -1;
int Photon_OpenCV::_gmagick_filter_lanczos = -1;
int Photon_OpenCV::_gmagick_filter_cubic = -1;
int Photon_OpenCV::_gmagick_filter_triangle = -1;
int Photon_OpenCV::_gmagick_filter_point = -1;
int Photon_OpenCV::_gmagick_filter_box = -1;

extern "C" {
  PHPCPP_EXPORT void *get_module() {
    static Php::Extension extension("photon-opencv", "0.1");

    Php::Class<Photon_OpenCV> photon_opencv("Photon_OpenCV");

    photon_opencv.method<&Photon_OpenCV::readimageblob>("readimageblob", {
      Php::ByVal("raw_image_data", Php::Type::String),
    });
    photon_opencv.method<&Photon_OpenCV::writeimage>("writeimage", {
      Php::ByVal("output", Php::Type::String),
    });

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

    photon_opencv.method<&Photon_OpenCV::getimagetype>("getimagetype");

    photon_opencv.method<&Photon_OpenCV::resizeimage>("resizeimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
      Php::ByVal("filter", Php::Type::Numeric, false),
    });
    photon_opencv.method<&Photon_OpenCV::resizeimage>("scaleimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
      Php::ByVal("filter", Php::Type::Numeric, false),
    });

    photon_opencv.method<&Photon_OpenCV::cropimage>("cropimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
    });

    extension.add(std::move(photon_opencv));

    return extension;
  }
}

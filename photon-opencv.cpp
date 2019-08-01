#include <phpcpp.h>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include <lcms2.h>
#include <zend.h>
#include <zend_constants.h>
#include "srgb.icc.h"

#define _checkimageloaded() { \
  if (_raw_image_data.empty()) { \
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
  std::string _raw_image_data;
  int _header_width;
  int _header_height;
  Exiv2::Value::AutoPtr _original_orientation;

  static cmsHPROFILE _srgb_profile;

  /* Cached Gmagick constants */
  static int _gmagick_channel_opacity;
  static int _gmagick_filter_lanczos;
  static int _gmagick_filter_cubic;
  static int _gmagick_filter_triangle;
  static int _gmagick_filter_point;
  static int _gmagick_filter_box;

  static int _gmagick_imgtype_grayscale;
  static int _gmagick_imgtype_grayscalematte;
  static int _gmagick_imgtype_truecolor;
  static int _gmagick_imgtype_truecolormatte;

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
    int alpha_channel;
    switch (_img.channels()) {
      case 1:
      case 2:
        storage_format = TYPE_GRAY_8;
        alpha_channel = 1;
        break;

      case 3:
      case 4:
        storage_format = TYPE_BGR_8;
        alpha_channel = 3;
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
    cv::Mat no_alpha_img = _img.reshape(1, _img.rows*_img.cols).
      colRange(0, alpha_channel);
    cv::Mat no_alpha_transformed_img = transformed_img.
      reshape(1, transformed_img.rows*transformed_img.cols).
      colRange(0, 3);

    cmsDoTransformLineStride(
        transform,
        no_alpha_img.data, no_alpha_transformed_img.data,
        no_alpha_img.cols/3, no_alpha_img.rows,
        no_alpha_img.step, no_alpha_transformed_img.step,
        0, 0
    );
    cmsDeleteTransform(transform);

    if (_imagehasalpha()) {
      /* Copy the original alpha information */
      cv::Mat alpha_only_img = _img.reshape(1, _img.rows*_img.cols).
        colRange(alpha_channel, alpha_channel+1);
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

    _gmagick_imgtype_grayscale = _getconstantex(
        "Gmagick::IMGTYPE_GRAYSCALE");
    _gmagick_imgtype_grayscalematte = _getconstantex(
        "Gmagick::IMGTYPE_GRAYSCALEMATTE");
    _gmagick_imgtype_truecolor = _getconstantex(
        "Gmagick::IMGTYPE_TRUECOLOR");
    _gmagick_imgtype_truecolormatte = _getconstantex(
        "Gmagick::IMGTYPE_TRUECOLORMATTE");

    /* Load default sRGB profile */
    _srgb_profile = cmsOpenProfileFromMem(srgb_icc, sizeof(srgb_icc)-1);
    if (!_srgb_profile) {
      throw Php::Exception("Failed to decode builtin sRGB profile");
    }
  }

  bool _imagehasalpha() {
    /* Even number of channels means it's either GA or BGRA */
    return !(_img.channels() & 1);
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

  bool _maybedecodeimage() {
    _checkimageloaded();

    if (!_img.empty()) {
      return true;
    }

    cv::Mat raw_data(1, _raw_image_data.size(), CV_8UC1,
      (void *) _raw_image_data.data());
    _img = cv::imdecode(raw_data, cv::IMREAD_UNCHANGED);

    if (_img.empty()) {
      _last_error = "Failed to decode image";
      return false;
    }

    _enforce8u();

    return true;
  }

  bool _loadimagefromrawdata() {
    /* Clear image in case object is being reused */
    _img = cv::Mat();

    Exiv2::Image::AutoPtr exiv_img;
    try {
      exiv_img = Exiv2::ImageFactory::open(
        (Exiv2::byte *) _raw_image_data.data(), _raw_image_data.size());
      exiv_img->readMetadata();
    }
    catch (Exiv2::Error &error) {
      _last_error = error.what();
      _raw_image_data.clear();
      return false;
    }

    switch (exiv_img->imageType()) {
      case Exiv2::ImageType::png:
        _format = "png";
        break;

      case Exiv2::ImageType::jpeg:
        _format = "jpeg";
        break;

      default:
        /* Default to jpeg, but disable lazy loading */
        _format = "jpeg";
        if (!_maybedecodeimage()) {
            return false;
        }
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
    switch (_img.channels()) {
      case 1:
        _type = _gmagick_imgtype_grayscale;
        break;

      case 2:
        _type = _gmagick_imgtype_grayscalematte;
        break;

      case 3:
        _type = _gmagick_imgtype_truecolor;
        break;

      case 4:
        _type = _gmagick_imgtype_truecolormatte;
        break;

      default:
        _raw_image_data.clear();
        _last_error = "Invalid number of channels";
        return false;
    }

    return true;
  }

public:
  Photon_OpenCV() {
    /* Static local intilization is thread safe */
    static std::once_flag initialized;
    std::call_once(initialized, _initialize);
  }

  Php::Value readimageblob(Php::Parameters &params) {
    _raw_image_data = params[0].stringValue();
    return _loadimagefromrawdata();
  }

  Php::Value readimage(Php::Parameters &params) {
    std::fstream input(params[0].stringValue(),
        std::ios::in | std::ios::binary);

    if (input.is_open()) {
      input.seekg(0, std::ios::end);
      _raw_image_data.resize(input.tellg());
      input.seekg(0, std::ios::beg);
      input.read(_raw_image_data.data(), _raw_image_data.size());
    }
    else {
      _raw_image_data.resize(0);
    }

    return _loadimagefromrawdata();
  }

  Php::Value writeimage(Php::Parameters &params) {
    _checkimageloaded();

    std::string wanted_output = params[0].stringValue();

    /* No ops were performed */
    if (_img.empty()) {
      std::ofstream output(wanted_output,
          std::ios::out | std::ios::binary);
      output << _raw_image_data;
      return true;
    }

    /* OpenCV strips the profile, we need to apply it first */
    _converttosrgb();

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

    /* Manually reinsert orientation exif data */
    if (_original_orientation.get()) {
      Exiv2::Image::AutoPtr exiv_img;
      try {
        exiv_img = Exiv2::ImageFactory::open(wanted_output, false);
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
      }
      catch (Exiv2::Error &error) {
        _last_error = error.what();
        return false;
      }
    }

    return true;
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

  Php::Value setimageformat(Php::Parameters &params) {
    std::string new_format = params[0].stringValue();
    std::transform(new_format.begin(), new_format.end(),
      new_format.begin(), ::tolower);

    if (new_format == _format) {
      return true;
    }

    if (!_maybedecodeimage()) {
      return false;
    }

    _format = new_format;

    return true;
  }

  Php::Value setcompressionquality(Php::Parameters &params) {
    _compression_quality = params[0];
    return true;
  }

  Php::Value getimagetype() {
    _checkimageloaded();
    return _type;
  }

  Php::Value setimagetype() {
    _last_error = "setimagetype() is not implemented";
    return false;
  }

  Php::Value resizeimage(Php::Parameters &params) {
    _checkimageloaded();

    /* Blur is ignored */
    int width = std::max(1, (int) params[0]);
    int height = std::max(1, (int) params[1]);
    int filter = params.size() > 2? (int) params[2] : -1;
    /* AREA is fast, looks excellent when downsampling, good when upscaling */
    const int default_filter = cv::INTER_AREA;

    /* Explicitly skip if it's a noop. */
    int img_width = _img.empty()? _header_width : _img.cols;
    int img_height = _img.empty()? _header_height : _img.rows ;
    if (width == img_width && height == img_height) {
      return true;
    }

    if (!_maybedecodeimage()) {
      return false;
    }

    _transparencysaferesize(width, height,
        _gmagickfilter2opencvinter(filter, default_filter));

    return true;
  }

  /* Documentation is lacking, but scaleimage is resizeimage with
    filter=Gmagick::FILTER_BOX and blur=1.0 */
  Php::Value scaleimage(Php::Parameters &params) {
    _checkimageloaded();

    int width = std::max(1, (int) params[0]);
    int height = std::max(1, (int) params[1]);

    /* Explicitly skip if it's a noop. */
    int img_width = _img.empty()? _header_width : _img.cols;
    int img_height = _img.empty()? _header_height : _img.rows ;
    if (width == img_width && height == img_height) {
      return true;
    }

    if (!_maybedecodeimage()) {
      return false;
    }

    _transparencysaferesize(width, height, cv::INTER_AREA);

    return true;
  }

  Php::Value cropimage(Php::Parameters &params) {
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
      return true;
    }

    if (!_maybedecodeimage()) {
      return false;
    }

    _img = _img(cv::Rect(x, y, x2-x, y2-y));

    return true;
  }

  Php::Value rotateimage(Php::Parameters &params) {
    _checkimageloaded();

    int degrees = params[1];
    int rotation_constant = -1;
    switch (degrees) {
      case 0:
        return true;

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
        _last_error = "Unsuported rotation angle";
        return false;
    }

    if (!_maybedecodeimage()) {
      return false;
    }

    printf("%dx%d\n", _img.rows, _img.cols);
    cv::rotate(_img, _img, rotation_constant);
    printf("%dx%d\n", _img.rows, _img.cols);

    return true;
  }

  Php::Value getimagechanneldepth(Php::Parameters &params) {
    _checkimageloaded();

    int channel = params[0];

    /* If the image has not been decoded, pretend all channels exist.
       Gmagick's channels don't map directly to OpenCV's, but Photon is only
       interested in the opacity */
    if (_img.empty() || channel != _gmagick_channel_opacity) {
      return 8;
    }

    /* Opacity channel is present with an even number number of channels */
    return _imagehasalpha()? 8 : 0;
  }
};
cmsHPROFILE Photon_OpenCV::_srgb_profile = NULL;

int Photon_OpenCV::_gmagick_channel_opacity = -1;
int Photon_OpenCV::_gmagick_filter_lanczos = -1;
int Photon_OpenCV::_gmagick_filter_cubic = -1;
int Photon_OpenCV::_gmagick_filter_triangle = -1;
int Photon_OpenCV::_gmagick_filter_point = -1;
int Photon_OpenCV::_gmagick_filter_box = -1;

int Photon_OpenCV::_gmagick_imgtype_grayscale = -1;
int Photon_OpenCV::_gmagick_imgtype_grayscalematte = -1;
int Photon_OpenCV::_gmagick_imgtype_truecolor = -1;
int Photon_OpenCV::_gmagick_imgtype_truecolormatte = -1;

extern "C" {
  PHPCPP_EXPORT void *get_module() {
    static Php::Extension extension("photon-opencv", "0.1");

    Php::Class<Photon_OpenCV> photon_opencv("Photon_OpenCV");

    photon_opencv.method<&Photon_OpenCV::readimageblob>("readimageblob", {
      Php::ByRef("raw_image_data", Php::Type::String),
    });
    photon_opencv.method<&Photon_OpenCV::readimage>("readimage", {
      Php::ByVal("filepath", Php::Type::String),
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
    photon_opencv.method<&Photon_OpenCV::setimagetype>("setimagetype");

    photon_opencv.method<&Photon_OpenCV::resizeimage>("resizeimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
      Php::ByVal("filter", Php::Type::Numeric, false),
    });
    photon_opencv.method<&Photon_OpenCV::scaleimage>("scaleimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
    });

    photon_opencv.method<&Photon_OpenCV::rotateimage>("rotateimage", {
      Php::ByVal("background", Php::Type::String),
      Php::ByVal("degrees", Php::Type::Numeric),
    });

    photon_opencv.method<&Photon_OpenCV::cropimage>("cropimage", {
      Php::ByVal("width", Php::Type::Numeric),
      Php::ByVal("height", Php::Type::Numeric),
    });

    extension.add(std::move(photon_opencv));

    return extension;
  }
}

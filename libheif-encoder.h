class Libheif_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;
  static const heif_encoder_descriptor *_aom_descriptor;

  static void _initialize() {
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
  
  
public:
  Libheif_Encoder(const std::string &format,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output) {
    /* Static local intilization is thread safe */
    static std::once_flag initialized;
    std::call_once(initialized, _initialize);

    _options = options;
    _quality = quality;
    _format = format;
    _output = output;

    _output->clear();
  }

  bool add_frame(const cv::Mat &frame, int delay) {
    (void) delay;

    // Only one format supported for now
    if (_output->size() || "avif" != _format) {
      return false;
    }
    const heif_compression_format heif_format = heif_compression_AV1;

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
      return false;
    }

    std::unique_ptr<heif_encoding_options,
      decltype(&heif_encoding_options_free)>
      options(nullptr, &heif_encoding_options_free);
    std::unique_ptr<heif_color_profile_nclx,
      decltype(&heif_nclx_color_profile_free)>
      nclx(nullptr, &heif_nclx_color_profile_free);

    auto lossless_option = _options->find(_format + ":lossless");
    if (lossless_option != _options->end()
        && "true" == lossless_option->second) {
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
      heif_encoder_set_lossy_quality(encoder.get(), _quality);
    }

    heif_colorspace colorspace = frame.channels() >= 3?
      heif_colorspace_RGB : heif_colorspace_monochrome;
    heif_chroma chroma = frame.channels() >= 3?
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
    error = heif_image_create(frame.cols,
        frame.rows,
        colorspace,
        chroma,
        &raw_image);
    image.reset(raw_image);
    if (error.code != heif_error_Ok) {
      return false;
    }

    std::vector<cv::Mat> channel_mats;
    for (int i = 0; i < frame.channels(); i++) {
      heif_channel channel_type = channel_map[frame.channels()-1][i];

      error = heif_image_add_plane(image.get(),
          channel_type,
          frame.cols,
          frame.rows,
          8);
      if (error.code != heif_error_Ok) {
        return false;
      }

      int stride;
      uint8_t *data = heif_image_get_plane(image.get(), channel_type, &stride);
      channel_mats.emplace_back(frame.rows,
          frame.cols,
          CV_8UC1,
          data,
          stride);
    }

    int trivial_fromto[] = {0, 0, 1, 1, 2, 2, 3, 3};
    cv::mixChannels(&frame,
        1,
        channel_mats.data(),
        channel_mats.size(),
        trivial_fromto,
        frame.channels());

    error = heif_context_encode_image(context.get(),
        image.get(),
        encoder.get(),
        options.get(),
        nullptr);
    if (error.code != heif_error_Ok) {
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
        _output);
    if (error.code != heif_error_Ok) {
      return false;
    }

    return true;
  }

  bool finalize() {
    return _output->size();
  }
};

const heif_encoder_descriptor *Libheif_Encoder::_aom_descriptor = nullptr;

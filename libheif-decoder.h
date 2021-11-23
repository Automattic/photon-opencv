class Libheif_Decoder : public Decoder {
protected:
  const std::string *_data;
  cv::Mat _frame;
  bool _ok;
  std::vector<uint8_t> _icc_profile;
  
public:
  Libheif_Decoder(const std::string *data) {
    _data = data;
    reset();
  }

  bool loaded() {
    return _ok;
  }

  void reset() {
    _ok = false;
    _frame = cv::Mat();

    std::unique_ptr<heif_context, decltype(&heif_context_free)> context(
      heif_context_alloc(), &heif_context_free);

    heif_error error;

    error = heif_context_read_from_memory_without_copy(context.get(),
      (void *) _data->data(),
      _data->size(),
      nullptr);
    if (error.code) {
      return;
    }

    std::unique_ptr<heif_image_handle,
      decltype(&heif_image_handle_release)>
      handle(nullptr, &heif_image_handle_release);
    heif_image_handle *raw_handle = nullptr;
    error = heif_context_get_primary_image_handle(context.get(),
      &raw_handle);
    handle.reset(raw_handle);
    if (error.code) {
      return;
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
      return;
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
        _frame,
        has_alpha? cv::COLOR_RGBA2BGRA : cv::COLOR_RGB2BGR);

    // This redundand ICC profile extraction code can be removed when
    // exiv2 0.27.4 is released, as it should support the new formats
    size_t profile_size = heif_image_get_raw_color_profile_size(
      h_image.get());
    if (profile_size) {
      _icc_profile.resize(profile_size);
      error = heif_image_handle_get_raw_color_profile(handle.get(),
        _icc_profile.data());
      if (error.code) {
        _icc_profile.clear();
      }
    }

    _ok = true;
  }

  bool get_next_frame(Frame &dst) {
    dst.img = _frame;
    _frame = cv::Mat();
    dst.delay = 0;
    dst.x = 0;
    dst.y = 0;
    dst.canvas_width = dst.img.cols;
    dst.canvas_height = dst.img.rows;
    dst.empty = dst.img.empty();

    return !dst.empty;
  }

  bool get_icc_profile(std::vector<uint8_t> &dst) {
    dst.assign(_icc_profile.begin(), _icc_profile.end());
    return true;
  }
};

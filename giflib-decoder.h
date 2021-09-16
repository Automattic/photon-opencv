class Giflib_Decoder : public Decoder {
 private:
  const std::string *_data;
  std::unique_ptr<GifFileType, void (*) (GifFileType *)> _gif;
  std::vector<uint8_t> _line;
  int _last_disposal;
  cv::Mat _canvas;
  cv::Mat _previous_cache;
  static const cv::Vec4b _bg_scalar;
  
 public:
  Giflib_Decoder(const std::string *data) :
    _gif(nullptr, [] (GifFileType *gif) { DGifCloseFile(gif, nullptr); }) {

    _data = data;
    reset();
  }

  bool loaded() {
    return _gif.get();
  }

  void reset() {
    int error = GIF_OK;
    std::pair<int, const std::string *> offset_and_data(0, _data);
    GifFileType *raw_gif = DGifOpen(&offset_and_data,
        [] (GifFileType *gif, GifByteType *buffer, int size) {
          std::pair<int, const std::string *> *user =
            (std::pair<int, const std::string *> *) gif->UserData;
          int offset = user->first;
          const std::string *data = user->second;
          int reading = std::min(size, (int) data->size() - offset);
          if (reading <= 0) {
            return 0;
          }
          memcpy(buffer, data->data() + offset, reading);
          user->first += reading;
          return reading;
        },
        &error);

    if (GIF_OK != error) {
      return;
    }

    _gif.reset(raw_gif);

    _line.resize(_gif->SWidth);
    _last_disposal = DISPOSAL_UNSPECIFIED;

    _canvas = cv::Mat(_gif->SHeight,
        _gif->SWidth,
        CV_8UC4,
        _bg_scalar);
  }

  bool get_next_frame(cv::Mat &dst, int &delay) {
    dst = cv::Mat();

    GraphicsControlBlock gcb;
    gcb.DisposalMode = DISPOSAL_UNSPECIFIED;
    gcb.UserInputFlag = false;
    gcb.DelayTime = 0;
    gcb.TransparentColor = NO_TRANSPARENT_COLOR;

    // Decodes trying not to fail even if the file is malformed
    GifRecordType type;
    do {
      if (GIF_ERROR == DGifGetRecordType(_gif.get(), &type)) {
        break;
      }

      if (EXTENSION_RECORD_TYPE == type) {
        int code;
        GifByteType *data;
        if (GIF_ERROR == DGifGetExtension(_gif.get(), &code, &data)) {
          continue;
        }

        if (GRAPHICS_EXT_FUNC_CODE == code && data) {
          // GCB doesn't get overwritten if this fails
          DGifExtensionToGCB(data[0], data+1, &gcb);
        }

        while (data && GIF_ERROR != DGifGetExtensionNext(_gif.get(), &data));
      }
      else if (IMAGE_DESC_RECORD_TYPE == type) {
        if (GIF_ERROR == DGifGetImageDesc(_gif.get())) {
          continue;
        }

        SavedImage *si = _gif->SavedImages + _gif->ImageCount - 1;
        const auto &desc = si->ImageDesc;
        if (desc.Width + desc.Left > _gif->SWidth
            || desc.Height + desc.Top > _gif->SHeight
            || desc.Width <= 0
            || desc.Height <= 0
            || desc.Left < 0
            || desc.Top < 0 ) {
          continue;
        }

        if (_last_disposal == DISPOSE_BACKGROUND) {
          if (_gif->ImageCount > 1) {
            const auto &p_desc =
              _gif->SavedImages[_gif->ImageCount-2].ImageDesc;
            cv::rectangle(_canvas,
                cv::Rect(p_desc.Left, p_desc.Top, p_desc.Width, p_desc.Height),
                _bg_scalar,
                -1);
          }
          _canvas.copyTo(_previous_cache);
        }
        else if (_last_disposal == DISPOSE_PREVIOUS) {
          _previous_cache.copyTo(_canvas);
        }
        else {
          _canvas.copyTo(_previous_cache);
        }

        cv::Mat roi(_canvas,
            cv::Rect(desc.Left, desc.Top, desc.Width, desc.Height));
        auto *color_map = desc.ColorMap? desc.ColorMap : _gif->SColorMap;
        if (!color_map) {
          continue;
        }

        // Initialize with interlaced values
        int row_offsets[] = {0, 4, 2, 1};
        int row_jumps[] = {8, 8, 4, 2};
        if (!desc.Interlace) {
          row_offsets[0] = 0;
          row_offsets[1] = row_offsets[2] = row_offsets[3] = desc.Height;
          row_jumps[0] = row_jumps[1] = row_jumps[2] = row_jumps[3] = 1;
        }

        for (int i = 0; i < 4; i++) {
          for (int y = row_offsets[i]; y < desc.Height; y += row_jumps[i]) {
            // Silently ignore failed reads
            DGifGetLine(_gif.get(), _line.data(), desc.Width);
            for (int x = 0; x < desc.Width; x++) {
              if (_line[x] == gcb.TransparentColor) {
                continue;
              }

              auto c = color_map->Colors[_line[x]];
              roi.at<cv::Vec4b>(y, x) = cv::Vec4b(c.Blue, c.Green, c.Red, 255);
            }
          }
        }

        // Success. Break early so next calls gets the next frame
        _canvas.copyTo(dst);
        break;
      }
    } while (TERMINATE_RECORD_TYPE != type);

    if (!dst.empty()) {
      _last_disposal = gcb.DisposalMode;
    }

    delay = gcb.DelayTime * 10;
    return !dst.empty();
  }
};

// Browsers ignore the background color and use transparent instead
const cv::Vec4b Giflib_Decoder::_bg_scalar(0, 0, 0, 0);

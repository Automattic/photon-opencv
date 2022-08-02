class Giflib_Decoder : public Decoder {
protected:
  const std::string *_data;
  std::unique_ptr<GifFileType, void (*) (GifFileType *)> _gif;
  std::pair<int, const std::string *> _offset_and_data;
  std::shared_ptr<Gif_Palette> _global_palette;
  int _loops;
  bool _can_read_loops;
  bool _may_dispose_to_previous;

  bool _has_previous_disposal() {
    GraphicsControlBlock gcb;
    gcb.DisposalMode = DISPOSAL_UNSPECIFIED;

    // Do minimal parsing to find a GCB with disposal set to previous
    GifRecordType type;
    do {
      if (GIF_ERROR == DGifGetRecordType(_gif.get(), &type)) {
        break;
      }

      GifByteType *data = nullptr;
      if (EXTENSION_RECORD_TYPE == type) {
        int code;
        if (GIF_ERROR == DGifGetExtension(_gif.get(), &code, &data)) {
          continue;
        }

        if (GRAPHICS_EXT_FUNC_CODE == code && data) {
          DGifExtensionToGCB(data[0], data+1, &gcb);

          if (gcb.DisposalMode == DISPOSE_PREVIOUS) {
            return true;
          }
        }
        while (data && GIF_ERROR != DGifGetExtensionNext(_gif.get(), &data));
      }
      else if (IMAGE_DESC_RECORD_TYPE == type) {
        if (GIF_ERROR == DGifGetImageDesc(_gif.get()) ) {
          continue;
        }

        int size;
        if (GIF_ERROR == DGifGetCode(_gif.get(), &size, &data)) {
          continue;
        }
        while (data && GIF_ERROR != DGifGetCodeNext(_gif.get(), &data));
      }
    } while (TERMINATE_RECORD_TYPE != type);

    return false;
  }

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
    _offset_and_data = std::make_pair(0, _data);

    int error = GIF_OK;
    GifFileType *raw_gif = DGifOpen(&_offset_and_data,
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

    if (raw_gif->SColorMap) {
      ColorMapObject *raw_palette = GifMakeMapObject(
          raw_gif->SColorMap->ColorCount,
          raw_gif->SColorMap->Colors);

      if (!raw_palette) {
        return;
      }

      _global_palette.reset(new Gif_Palette(raw_palette));
    }

    _gif.reset(raw_gif);
    _can_read_loops = true;
    _loops = 1;

    // _has_previous_disposal consumes the data. Rewind when done
    int original_offset = _offset_and_data.first;
    _may_dispose_to_previous = _has_previous_disposal();
    _offset_and_data.first = original_offset;
  }

  bool get_next_frame(Frame &dst) {
    dst.reset();

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
        GifByteType *data = nullptr;
        if (GIF_ERROR == DGifGetExtension(_gif.get(), &code, &data)) {
          continue;
        }

        if (GRAPHICS_EXT_FUNC_CODE == code && data) {
          // GCB doesn't get overwritten if this fails
          DGifExtensionToGCB(data[0], data+1, &gcb);
        }
        else if (APPLICATION_EXT_FUNC_CODE == code) {
          if (_can_read_loops &&
              11 == data[0] &&
              !memcmp("NETSCAPE2.0", data+1, 11)) {
            if (GIF_ERROR != DGifGetExtensionNext(_gif.get(), &data) &&
                3 == data[0] &&
                1 == data[1]) {
              _loops = data[2] | (data[3] << 8);
            }
          }
        }

        while (data && GIF_ERROR != DGifGetExtensionNext(_gif.get(), &data));
      }
      else if (IMAGE_DESC_RECORD_TYPE == type) {
        _can_read_loops = false;

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

        dst.img = cv::Mat(desc.Height,
            desc.Width,
            CV_8UC4,
            cv::Vec4b(0, 0, 0, 0));

        auto *color_map = desc.ColorMap? desc.ColorMap : _gif->SColorMap;
        if (!color_map) {
          continue;
        }

        dst.gif_global_palette = _global_palette;
        if (desc.ColorMap) {
          auto raw_palette = GifMakeMapObject(desc.ColorMap->ColorCount,
              desc.ColorMap->Colors);

          if (!raw_palette) {
            return false;
          }

          dst.gif_frame_palette.reset(new Gif_Palette(raw_palette));
        }

        // Initialize with interlaced values
        int row_offsets[] = {0, 4, 2, 1};
        int row_jumps[] = {8, 8, 4, 2};
        if (!desc.Interlace) {
          row_offsets[0] = 0;
          row_offsets[1] = row_offsets[2] = row_offsets[3] = desc.Height;
          row_jumps[0] = row_jumps[1] = row_jumps[2] = row_jumps[3] = 1;
        }

        std::vector<uint8_t> line(desc.Width);
        for (int i = 0; i < 4; i++) {
          for (int y = row_offsets[i]; y < desc.Height; y += row_jumps[i]) {
            cv::Vec4b *row = (cv::Vec4b *) (dst.img.data + y * dst.img.step);

            // Silently ignore failed reads
            DGifGetLine(_gif.get(), line.data(), desc.Width);
            for (int x = 0; x < desc.Width; x++) {
              if (line[x] == gcb.TransparentColor) {
                continue;
              }

              auto c = color_map->Colors[line[x]];
              row[x] = cv::Vec4b(c.Blue, c.Green, c.Red, 255);
            }
          }
        }

        dst.x = desc.Left;
        dst.y = desc.Top;

        // Success. Break early so next calls gets the next frame
        break;
      }
    } while (TERMINATE_RECORD_TYPE != type);

    dst.delay = gcb.DelayTime * 10;
    dst.canvas_width = _gif->SWidth;
    dst.canvas_height = _gif->SHeight;
    dst.empty = dst.img.empty();
    dst.loops = _loops;
    dst.blending = Frame::BLENDING_BLEND;
    dst.may_dispose_to_previous = _may_dispose_to_previous;
    switch (gcb.DisposalMode) {
      case DISPOSE_BACKGROUND:
        dst.disposal = Frame::DISPOSAL_BACKGROUND;
        break;

      case DISPOSE_PREVIOUS:
        dst.disposal = Frame::DISPOSAL_PREVIOUS;
        break;

      default:
        dst.disposal = Frame::DISPOSAL_NONE;
        break;
    }
    dst.gif_transparent_index = gcb.TransparentColor;

    return !dst.empty;
  }

  bool provides_optimized_frames() {
    return true;
  }

  bool provides_animation() {
    return true;
  }
};

class OpenCV_Decoder : public Decoder {
protected:
  const std::string *_data;
  cv::Mat _frame;
  bool _ok;
  
public:
  OpenCV_Decoder(const std::string *data) {
    _data = data;
    reset();
  }

  bool loaded() {
    return _ok;
  }

  void reset() {
    /*
     * Ideally, we would decode directly from memory. However, opencv is more
     * strict when imdecode is used. This results in jpegs that are missing
     * bytes and pngs that have broken exif information to only be parsed by
     * imread. In order to support more files, we write the files to the
     * filesystem so that imread can be used.
    */
    TempFile temp_image_file(*_data);
    _frame = cv::imread(temp_image_file.get_path(), cv::IMREAD_UNCHANGED);
    _ok = !_frame.empty();
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
};

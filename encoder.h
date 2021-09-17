class Encoder {
public:
  Encoder() {};
  virtual bool add_frame(const cv::Mat &frame, int delay) = 0;
  virtual bool finalize() = 0;
};

class Libheif_Encoder : public Encoder {
protected:
  const std::map<std::string, std::string> *_options;
  std::string _format;
  int _quality;
  std::vector<uint8_t> *_output;
  static const heif_encoder_descriptor *_aom_descriptor;

  static void _initialize();

public:
  Libheif_Encoder(const std::string &format,
      int quality,
      const std::map<std::string, std::string> *options,
      std::vector<uint8_t> *output);
  bool add_frame(const Frame &frame);
  bool finalize();
};

class Gif_Palette {
private:
  std::unique_ptr<ColorMapObject, decltype(&GifFreeMapObject)> _color_map;
  std::unordered_multimap<uint32_t, int> _color_to_index;

public:
  Gif_Palette(ColorMapObject *color_map);
  ColorMapObject *get_color_map();
  int get_index(uint8_t b, uint8_t g, uint8_t r, int transparent_index);
};


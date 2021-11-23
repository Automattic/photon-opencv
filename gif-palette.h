class Gif_Palette {
private:
  std::unique_ptr<ColorMapObject, decltype(&GifFreeMapObject)> _color_map;
  std::map<uint32_t, int> _color_to_index;

public:

  Gif_Palette(ColorMapObject *color_map) :
    _color_map(color_map, GifFreeMapObject) {

    for (int i = 0; i < _color_map->ColorCount; i++) {
      uint32_t key = (_color_map->Colors[i].Blue << 16) |
        (_color_map->Colors[i].Green << 8) |
        _color_map->Colors[i].Red;

      _color_to_index[key] = i;
    }
  }

  ColorMapObject *get_color_map() {
    return _color_map.get();
  }

  int get_index(uint8_t b, uint8_t g, uint8_t r) {
    uint32_t key = (b << 16) | (g << 8) | r;
    return _color_to_index.at(key);
  }
};


#include <unordered_map>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <gif_lib.h>
#include "gif-palette.h"

Gif_Palette::Gif_Palette(ColorMapObject *color_map) :
  _color_map(color_map, GifFreeMapObject) {

  for (int i = 0; i < _color_map->ColorCount; i++) {

    uint32_t key = (_color_map->Colors[i].Blue << 16) |
      (_color_map->Colors[i].Green << 8) |
      _color_map->Colors[i].Red;

    _color_to_index.insert(std::make_pair(key, i));
  }
}

ColorMapObject *Gif_Palette::get_color_map() {
  return _color_map.get();
}

int Gif_Palette::get_index(uint8_t b,
    uint8_t g,
    uint8_t r,
    int transparent_index) {
  uint32_t key = (b << 16) | (g << 8) | r;

  int index = -1;
  for (auto hit = _color_to_index.find(key);
      hit != _color_to_index.end();
      hit++) {
    if (hit->second != transparent_index) {
      index = hit->second;
      break;
    }
  }

  if (-1 == index) {
    throw std::out_of_range("No matching visible color in palette");
  }

  return index;
}

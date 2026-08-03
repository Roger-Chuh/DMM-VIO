#pragma once
#include "NumType.h"
#include <string>
#include <vector>

namespace dso {
/// @brief A simple color map implementation inspired by
/// github.com/yuki-koyama/tinycolormap/blob/master/include/tinycolormap.hpp
class ColorMap {
 public:
  using Rgb = Vec3;

  ColorMap() = default;
  ColorMap(std::string name, const aligned_vector<Vec3>& colors);

  /// @brief Map input x to color rgb/bgr, assumes x is in [0, 1]
  Vec3 GetRgb(number_t x) const noexcept;
  Vec3 GetBgr(number_t x) const noexcept { return GetRgb(x).reverse(); }

  bool Ok() const noexcept { return !data_.empty(); }
  const std::string& name() const noexcept { return name_; }
  int size() const noexcept { return static_cast<int>(data_.size()); }

 private:
  std::string name_;
  number_t step_{};
  aligned_vector<Vec3> data_;
};

/// @brief Factory
ColorMap MakeCmapJet();
ColorMap MakeCmapHeat();
ColorMap MakeCmapTurbo();
ColorMap MakeCmapPlasma();
ColorMap GetColorMap(const std::string& name);

}  // namespace dso

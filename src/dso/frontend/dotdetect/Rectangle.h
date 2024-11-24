#pragma once

#include <Eigen/Dense>

namespace dso {

namespace DotDetect {

class IRectangle {
public:
  IRectangle() {}
  IRectangle(int x1, int y1) : x1(x1), y1(y1), x2(x1), y2(y1) {}

  IRectangle(const IRectangle &r) {
    this->x1 = r.x1;
    this->x2 = r.x2;
    this->y2 = r.y2;
    this->y1 = r.y1;
  }

  IRectangle(int x1, int y1, int x2, int y2) {
    this->x1 = x1;
    this->x2 = x2;
    this->y2 = y2;
    this->y1 = y1;
  }

  int x1, y1, x2, y2;

  int Width() const { return std::max(0, x2 + 1 - x1); }

  int Height() const { return std::max(0, y2 + 1 - y1); }

  int Area() const { return Width() * Height(); }

  bool IntersectsWith(const IRectangle &other) const {
    if (y2 < other.y1)
      return false;
    if (y1 > other.y2)
      return false;
    if (x2 < other.x1)
      return false;
    if (x1 > other.x2)
      return false;
    return true;
  }

  bool Contains(const IRectangle &other) const {
    if (y1 >= other.y1)
      return false;
    if (x1 >= other.x1)
      return false;
    if (x2 <= other.x2)
      return false;
    if (y2 <= other.y2)
      return false;
    return true;
  }

  void Insert(int x, int y) {
    x1 = std::min(x1, x);
    x2 = std::max(x2, x);
    y1 = std::min(y1, y);
    y2 = std::max(y2, y);
  }

  void Insert(const IRectangle &d) {
    x1 = std::min(x1, d.x1);
    x2 = std::max(x2, d.x2);
    y1 = std::min(y1, d.y1);
    y2 = std::max(y2, d.y2);
  }

  IRectangle Grow(int r) const {
    IRectangle ret(*this);
    ret.x1 -= r;
    ret.x2 += r;
    ret.y1 -= r;
    ret.y2 += r;
    return ret;
  }

  void Grow(int r, bool) {
    this->x1 -= r;
    this->x2 += r;
    this->y1 -= r;
    this->y2 += r;
  }

  IRectangle Clamp(int minx, int miny, int maxx, int maxy) const {
    IRectangle ret(*this);
    ret.x1 = std::max(minx, ret.x1);
    ret.y1 = std::max(miny, ret.y1);
    ret.x2 = std::min(maxx, ret.x2);
    ret.y2 = std::min(maxy, ret.y2);
    return ret;
  }

  bool Contains(int x, int y) const {
    return x1 <= x && x <= x2 && y1 <= y && y <= y2;
  }

  Eigen::Vector2d Center() const {
    return Eigen::Vector2d((x2 + x1) / 2.0, (y2 + y1) / 2.0);
  }
};

} // namespace DotDetect
} // namespace dso
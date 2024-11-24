//
// Created by zk on 24-4-19.
//

#ifndef YVR_CALIB_BASICSTRUCT_H
#define YVR_CALIB_BASICSTRUCT_H

#include "LineGroup.h"
#include <Eigen/Core>
namespace dso::DotDetect {

struct lessCompare {
  bool operator()(const Eigen::Vector2i &lhs,
                  const Eigen::Vector2i &rhs) const {
    return (lhs[0] < rhs[0]) || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
  }
};

struct Dist {
  Vertex *v = nullptr;
  double dist = -1;
};

inline bool operator<(const Dist &lhs, const Dist &rhs) {
  return lhs.dist < rhs.dist;
}

struct compareMat01 {
  bool operator()(int a) const { return a >= 0; }
};

struct compareMatZero {
  bool operator()(int a) const { return a == 0; }
};

} // namespace dso::DotDetect
#endif // YVR_CALIB_BASICSTRUCT_H

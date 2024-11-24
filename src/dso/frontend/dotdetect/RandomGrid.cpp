#include "RandomGrid.h"

#include <iostream>

namespace dso::DotDetect {

Eigen::MatrixXi MakePattern(int r, int c, uint32_t seed) {
  Eigen::MatrixXi M(r, c);

  std::mt19937 rng(seed); // if fix seed, then M is fixed; other fixed by system

  for (int r = 0; r < M.rows(); ++r) {
    for (int c = 0; c < M.cols(); ++c) {
      M(r, c) = rng() % 2;
    }
  }
  //  std::cerr<<"M:\n"<<M<<std::endl;
  return M;
}

std::array<Eigen::MatrixXi, 4> MakePatternGroup(int r, int c, uint32_t seed) {
  return FillGroup(MakePattern(r, c, seed));
}

std::array<Eigen::MatrixXi, 4> FillGroup(const Eigen::MatrixXi &m) {
  std::array<Eigen::MatrixXi, 4> patterns;
  patterns[0] = m;

  // Found in this awesome answer http://stackoverflow.com/a/3488737/505049
  patterns[1] = m.transpose().colwise().reverse().eval(); // Rotate 90 CW
  patterns[2] = m.reverse().eval(); // Rotate 180. Not in the S.O. post
  patterns[3] = m.transpose().rowwise().reverse().eval(); // Rotate 270 CW
  return patterns;
}

} // namespace dso::DotDetect
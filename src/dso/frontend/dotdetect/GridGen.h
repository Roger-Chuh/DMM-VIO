//
// Created by zk on 24-5-10.
//

#ifndef YVR_CALIB_GRIDGEN_H
#define YVR_CALIB_GRIDGEN_H
#include <Eigen/Core>
#include <array>
namespace dso::DotDetect {

bool isFixAreaUnique(const std::array<Eigen::MatrixXi, 4> &PG, int min_r,
                     int min_c);

bool Area9Same(const Eigen::MatrixXi &mat);

int FindBestSeed(int r, int c, int uniquesize);

void PrintPattern(const Eigen::MatrixXi &M);

} // namespace dso::DotDetect

#endif // YVR_CALIB_GRIDGEN_H

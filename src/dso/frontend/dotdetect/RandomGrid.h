#pragma once

#include <Eigen/Eigen>
#include <array>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

namespace dso::DotDetect {

Eigen::MatrixXi MakePattern(int r, int c, uint32_t seed = 0);

std::array<Eigen::MatrixXi, 4> MakePatternGroup(int r, int c, uint32_t seed);

std::array<Eigen::MatrixXi, 4> FillGroup(const Eigen::MatrixXi &m);

} // namespace dso::DotDetect
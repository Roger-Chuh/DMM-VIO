#pragma once

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <array>
#include <map>
#include <opencv2/core/mat.hpp>
#include <unordered_map>

#include "BasicStruct.h"
#include "Conic.h"
#include "ImageProcessing.h"
#include "LineGroup.h"
#include "RandomGrid.h"
#include "TargetGridInfo.h"
namespace dso::DotDetect {

class TargetGridDot {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  TargetGridDot() = default;

  static bool FindTarget(std::vector<Conic> &conics,
                         std::map<int, std::vector<Conic *>> &plateConics,
                         const TargetGridInfo &boardInfo,
                         const ParamsImageProcessing &params_,
                         const cv::Mat *img = nullptr);

  static bool FindTarget(std::set<Vertex *> &vertex,
                         std::map<int, std::vector<Conic *>> &plateConics,
                         TargetGridInfo boardInfo,
                         const ParamsImageProcessing &params,
                         const cv::Mat *img = nullptr);

  static bool FindBoardsKd(std::vector<Vertex> &conics,
                           std::vector<std::set<Vertex *>> &multiPlate,
                           const ParamsImageProcessing &params,
                           const cv::Mat *img = nullptr);
};

} // namespace dso::DotDetect
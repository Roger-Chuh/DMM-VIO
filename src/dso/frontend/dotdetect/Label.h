#pragma once

#include "../../camera_model/vio_math_0.h"
#include <opencv2/core/mat.hpp>
#include <set>
#include <vector>

#include "Rectangle.h"
namespace dso {
namespace DotDetect {

struct PixelClass {
  IRectangle bbox;
  std::set<int> ellipse_pixel_set;
  std::set<int> cluster_pixel_set;
  Vec2 cluster_center;
  int size;
  bool is_near_boarder;
  PixelClass(int x, int y) : bbox(x, y), size(0), is_near_boarder(false) {}
  PixelClass(int x1, int y1, int x2, int y2, int size)
      : bbox(x1, y1, x2, y2), size(size), is_near_boarder(false) {}
};

void DFSSearchMat(const cv::Mat &treshPic, cv::Mat &binaryPic,
                  PixelClass &curTag, const int &currow, const int &curcol,
                  const int &pointValue, int boarder);

void LabelTreshPic(const cv::Mat &pic, std::vector<PixelClass> &allBbox,
                   double minArea, double maxArea, double minDensity,
                   double minAspect, int pointValue = 0, int boarder = 2,
                   cv::Mat *p_binary_mat = nullptr);

} // namespace DotDetect
} // namespace dso
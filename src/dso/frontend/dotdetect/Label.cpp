//#define DEBUGTRESH

#include "Label.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <vector>

using namespace std;
using namespace Eigen;
namespace dso {
namespace DotDetect {

static std::vector<std::vector<int>> dfsDirs = {
    {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
void DFSSearchMat(const cv::Mat &treshPic, cv::Mat &binaryPic,
                  PixelClass &curbbox, const int &currow, const int &curcol,
                  const int &pointValue, int boarder) {
  int img_width = treshPic.cols;
  binaryPic.at<unsigned char>(currow, curcol) = 1;
  curbbox.bbox.Insert(curcol, currow);
  std::queue<std::vector<int>> searchQue;
  searchQue.emplace(std::vector<int>{curcol, currow});
  while (!searchQue.empty()) {
    for (auto &dir : dfsDirs) {
      int nextCol = searchQue.front()[0] + dir[0];
      int nextRow = searchQue.front()[1] + dir[1];

      if (nextRow < boarder || nextRow >= (binaryPic.rows - boarder) ||
          nextCol < boarder || nextCol >= (binaryPic.cols - boarder)) {
        curbbox.is_near_boarder = true;
        continue;
      }

      if (binaryPic.at<unsigned char>(nextRow, nextCol) == 0 &&
          treshPic.at<unsigned char>(nextRow, nextCol) == pointValue) {
        binaryPic.at<unsigned char>(nextRow, nextCol) = 255;
        searchQue.emplace(std::vector<int>{nextCol, nextRow});
        curbbox.bbox.Insert(nextCol, nextRow);
        curbbox.size++;
      }
    }
    curbbox.cluster_pixel_set.insert(searchQue.front()[0] +
                                     img_width * searchQue.front()[1]);
    searchQue.pop();
  }
}

bool boxSymmetry(const IRectangle &bbox, const cv::Mat &treshPic,
                 double ratio) {
  int syscount = 0;
  for (int idx = 0; idx < bbox.Width(); ++idx) {
    for (int idy = 0; idy < bbox.Height(); ++idy) {
      if (treshPic.at<unsigned char>(bbox.y1 + idy, bbox.x1 + idx) !=
          treshPic.at<unsigned char>(bbox.y2 - idy, bbox.x2 - idx)) {
        syscount++;
      }
    }
  }
  double sys = double(syscount) / bbox.Area() / 2.;
  return (sys < ratio);
}

void LabelTreshPic(const cv::Mat &treshPic, std::vector<PixelClass> &allBbox,
                   double minArea, double maxArea, double conic_symmetry,
                   double minAspect, int pointValue, int boarder,
                   cv::Mat *p_binary_mat) {
  cv::Mat binaryPic;
  if (p_binary_mat) {
    binaryPic = *p_binary_mat;
  } else {
    binaryPic = cv::Mat::zeros(treshPic.rows, treshPic.cols, CV_8UC1);
  }
  for (int currow = boarder; currow < binaryPic.rows - boarder; ++currow) {
    for (int curcol = boarder; curcol < binaryPic.cols - boarder; ++curcol) {
      if (binaryPic.at<unsigned char>(currow, curcol) == 0 &&
          treshPic.at<unsigned char>(currow, curcol) == pointValue) {
        PixelClass curTag(curcol, currow);
        DFSSearchMat(treshPic, binaryPic, curTag, currow, curcol, pointValue,
                     boarder);
        const double aspect =
            (double)curTag.bbox.Width() / (double)curTag.bbox.Height();
        double area = curTag.bbox.Width() * curTag.bbox.Height();

#ifdef DEBUGTRESH
        cv::Mat showConic;
        cv::cvtColor(treshPic, showConic, cv::COLOR_GRAY2BGR);
        cv::rectangle(showConic, cv::Point(curTag.bbox.x1, curTag.bbox.y1),
                      cv::Point(curTag.bbox.x2, curTag.bbox.y2),
                      cv::Scalar(0, 0, 255));
        cv::imshow("one label", showConic);
        cv::waitKey(0);
#endif

        if ((curTag.size >= minArea) && (curTag.size < maxArea) &&
            (curTag.size / area > 0.4) && (minAspect < aspect) &&
            (aspect < 1.0 / minAspect) &&
            boxSymmetry(curTag.bbox, treshPic, conic_symmetry)) {
          curTag.bbox.Grow(2, true);
          allBbox.emplace_back(curTag);
        }
#ifdef DEBUGTRESH
        else {
          std::cerr << "point: " << curTag.bbox.Center().transpose()
                    << std::endl;
          if (curTag.size < minArea)
            std::cerr << "minArea not match" << std::endl;
          if (curTag.size >= maxArea)
            std::cerr << "maxArea not match" << std::endl;
          if (curTag.size / area <= 0.4)
            std::cerr << "curTag.size / area <= 0.4 not match" << std::endl;
          if (curTag.size / area >= 0.90)
            std::cerr << "curTag.size / area >= 0.90 not match" << std::endl;
          if (minAspect >= aspect)
            std::cerr << "minAspect >= aspect not match" << std::endl;
          if (aspect >= 1.0 / minAspect)
            std::cerr << "aspect >= 1.0 / minAspect not match" << std::endl;
          if (!boxSymmetry(curTag.bbox, treshPic, conic_symmetry))
            std::cerr << "boxSymmetry not match" << std::endl;
        }
#endif
      }
      binaryPic.at<unsigned char>(currow, curcol) = 255;
    }
  }
}

} // namespace DotDetect
} // namespace dso
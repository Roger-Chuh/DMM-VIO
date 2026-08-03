#pragma once

#include <opencv2/opencv.hpp>

#include "../FrameData.h"
#include "Conic.h"
#include "ImagesProcessParams.h"
#include "Label.h"
#include "TargetGridInfo.h"

namespace dso {
namespace DotDetect {

class ImageProcessing {
 public:
  ImageProcessing(double grid_spacing, const Eigen::Vector2i& grid_size, uint32_t seed, int plate_num = 3,
                  int unique_size = 6);

  void ProcessPic(const cv::Mat& grayscale_image, const ParamsImageProcessing& params, CalibIO::CurFrameRes& res,
                  cv::Mat* p_binary_mat = nullptr) const;

 private:
  // Influenced by libCVD gradient method
  template <typename TI, typename TD>
  void gradient(const int w, const int h, const TI* image, TD* grad) {
    const TI* pI = image + w + 1;
    const TI* pEnd = image + w * h - w - 1;
    TD* pOut = grad + w + 1;

    while (pI != pEnd) {
      (*pOut)[0] = *(pI + 1) - *(pI - 1);
      (*pOut)[1] = *(pI + w) - *(pI - w);
      pI++;
      pOut++;
    }
  }

  static void gradientMat(const cv::Mat& I, std::vector<Vec2>& grad) {
    for (int curRow = 1; curRow < I.rows - 1; ++curRow) {
      for (int curCol = 1; curCol < I.cols - 1; ++curCol) {
        Vec2& pOut = grad[curRow * I.cols + curCol];
        pOut[0] = number_t(I.at<unsigned char>(curRow, curCol + 1)) - number_t(I.at<unsigned char>(curRow, curCol - 1));
        pOut[1] = number_t(I.at<unsigned char>(curRow + 1, curCol)) - number_t(I.at<unsigned char>(curRow - 1, curCol));
      }
    }
  }

  template <typename TI, typename TintI, typename Tout>
  void AdaptiveThreshold(int w, int h, const TI* image, const TintI* intImage, Tout* out, float threshold, int rad,
                         int min_diff, Tout pass, Tout fail) {
    // Adaptive Thresholding Using the Integral Image
    // Derek Bradley, Gerhard Roth

    // With min diff trick to make it less sensitive in homogeneous regions:
    // http://homepages.inf.ed.ac.uk/rbf/HIPR2/adpthrsh.htm

    for (int j = 0; j < h; ++j) {
      const int y1 = std::max(1, j - rad);
      const int y2 = std::min(h - 1, j + rad);

      for (int i = 0; i < w; ++i) {
        const int x1 = std::max(1, i - rad);
        const int x2 = std::min(w - 1, i + rad);
        const int count = (x2 - x1) * (y2 - y1);
        const TintI* intIy2 = intImage + y2 * w;
        const TintI* intIy1m1 = intImage + (y1 - 1) * w;
        const TintI sum = intIy2[x2] - intIy1m1[x2] - intIy2[x1 - 1] + intIy1m1[x1 - 1];
        const float avg = sum / count;
        unsigned id = j * w + i;
        out[id] = (image[id] < threshold * (avg - (float)min_diff)) ? pass : fail;
      }
    }
  }

  // Influenced by libCVD integral_image
  template <typename TI, typename TO>
  void integral_image(const int w, const int h, const TI* in, TO* out) {
    out[0] = in[0];

    // Do the first row.
    for (int x = 1; x < w; x++) out[x] = out[x - 1] + in[x];

    // Do the first column.
    for (int y = 1; y < h; y++) out[y * w] = out[(y - 1) * w] + in[y * w];

    // Do the remainder of the image
    for (int y = 1; y < h; y++) {
      TO sum = in[y * w];

      for (int x = 1; x < w; x++) {
        sum += in[y * w + x];
        out[y * w + x] = sum + out[(y - 1) * w + x];
      }
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 public:
  const TargetGridInfo boardInfo_;
  bool verbose = false;
};

}  // namespace DotDetect
}  // namespace dso
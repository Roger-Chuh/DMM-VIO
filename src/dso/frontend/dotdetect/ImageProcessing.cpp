#include "ImageProcessing.h"
#include "TargetGridDot.h"
namespace dso::DotDetect {

ImageProcessing::ImageProcessing(double grid_spacing,
                                 const Eigen::Vector2i &grid_size,
                                 uint32_t seed, int plate_num, int unique_size)
    : boardInfo_(grid_spacing, grid_size, seed, plate_num, unique_size) {}

void ImageProcessing::ProcessPic(const cv::Mat &grayscale_image,
                                 const ParamsImageProcessing &params,
                                 CalibIO::CurFrameRes &res,
                                 cv::Mat *p_binary_mat) const {
  // 将仅有两个邻居的点删除
  cv::Mat grayImg;
  if (params.black_on_white) {
    grayImg = grayscale_image;
  } else {
    grayImg = 255 - grayscale_image;
  }

  // Process image
  cv::Mat threshPic;
  cv::adaptiveThreshold(grayImg, threshPic, 255,
                        cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C,
                        cv::THRESH_BINARY, params.window_size,
                        params.at_threshold * 25);
#ifndef __ANDROID__
  if (verbose) {
    cv::imshow("thresh cv:", threshPic);
    cv::waitKey(0);
  }
#endif

  // Label image (connected components)
  //  dblabels.clear();
  std::vector<PixelClass> labels;
  std::vector<Conic> conics;
  LabelTreshPic(threshPic, labels, params.conic_min_area, params.conic_max_area,
                params.conic_symmetry, params.conic_min_aspect, 0, 2,
                p_binary_mat);
  //  std::vector<Conic> conics;
  std::vector<Vec2> imgD(grayscale_image.rows * grayscale_image.cols, {0, 0});
  gradientMat(grayImg, imgD);

  //  std::cerr << "labels count:" << labels.size() << std::endl;
  FindConics(threshPic.cols, threshPic.rows, labels, &imgD[0], conics,
             params.black_on_white);

  std::map<int, std::vector<Conic *>> plateConics;

#ifndef __ANDROID__
  if (verbose) {
    cv::Mat boxMat, fit_pixel_mat;
    cv::cvtColor(grayscale_image, boxMat, cv::COLOR_GRAY2BGR);
    cv::cvtColor(grayscale_image, fit_pixel_mat, cv::COLOR_GRAY2BGR);
    for (const auto &label : labels) {
      auto point_color = cv::Scalar(
          int(label.cluster_center.x() * 10) % 255,
          int(label.cluster_center.y() * 10) % 255,
          int(label.cluster_center.x() * label.cluster_center.y() * 10) % 255);
      for (const auto &pixel_id : label.ellipse_pixel_set) {
        int col = pixel_id % threshPic.cols;
        int row = pixel_id / threshPic.cols;
        cv::circle(fit_pixel_mat, cv::Point2i(col, row), 0, point_color);
      }
    }

    for (const auto &curconic : conics) {
      cv::rectangle(boxMat, cv::Point2i(curconic.bbox.x1, curconic.bbox.y1),
                    cv::Point2i(curconic.bbox.x2, curconic.bbox.y2),
                    cv::Scalar(0, 0, 255));
      cv::rectangle(fit_pixel_mat,
                    cv::Point2i(curconic.bbox.x1, curconic.bbox.y1),
                    cv::Point2i(curconic.bbox.x2, curconic.bbox.y2),
                    cv::Scalar(0, 0, 255));
      /*** computer ellipse parameters ***/
      // C=
      //  A   B/2   D/2
      //  B/2   C    E/2
      //  D/2  E/2    F
      Mat3 C(curconic.C);
      Vec2 ellipse_center;
      number_t ellipse_a, ellipse_b, ellipse_theta;
      C /= C(2, 2);
      double m_A = C(0, 0);
      double m_B = C(0, 1) * 2.0;
      double m_C = C(1, 1);
      double m_D = C(0, 2) * 2.0;
      double m_E = C(1, 2) * 2.0;

      ellipse_center.x() =
          (m_B * m_E - 2 * m_C * m_D) / (4 * m_A * m_C - m_B * m_B);
      ellipse_center.y() =
          (m_B * m_D - 2 * m_A * m_E) / (4 * m_A * m_C - m_B * m_B);
      double a_a = 2 *
                   (m_A * ellipse_center.x() * ellipse_center.x() +
                    m_C * ellipse_center.y() * ellipse_center.y() +
                    m_B * ellipse_center.x() * ellipse_center.y() - 1) /
                   (m_A + m_C + sqrt((m_A - m_C) * (m_A - m_C) + m_B * m_B));
      double b_b = 2 *
                   (m_A * ellipse_center.x() * ellipse_center.x() +
                    m_C * ellipse_center.y() * ellipse_center.y() +
                    m_B * ellipse_center.x() * ellipse_center.y() - 1) /
                   (m_A + m_C - sqrt((m_A - m_C) * (m_A - m_C) + m_B * m_B));
      if (a_a > 0 && b_b > 0) {
        ellipse_a = sqrt(a_a);
        ellipse_b = sqrt(b_b);
        ellipse_theta = 0.5 * atan2(m_B, m_A - m_C);
        cv::ellipse(boxMat, cv::Point2d(ellipse_center.x(), ellipse_center.y()),
                    cv::Size(ellipse_a, ellipse_b),
                    ellipse_theta * 180.0 / M_PI, 0, 360,
                    cv::Scalar(255, 255, 0));
        cv::circle(boxMat, cv::Point2d(ellipse_center.x(), ellipse_center.y()),
                   0, cv::Scalar(255, 255, 0));
      }
    }

    cv::imshow("cv conics box", boxMat);
    cv::imshow("fit pixel mat", fit_pixel_mat);
    cv::waitKey(0);
  }
#endif
  //  if (params.skip_detection) {
  //    res.enough_points = true;
  //    return;
  //  }
  TargetGridDot::FindTarget(conics, plateConics, boardInfo_, params,
                            (verbose ? &grayscale_image : nullptr));

  res.enough_points = plateConics.size() > params.min_dot_num;

  for (const auto &plateData : plateConics) {
    res.mGridId.emplace(plateData.first, std::vector<int>{});
    res.mImagePointSets.emplace(plateData.first,
                                std::vector<Eigen::Vector2d>{});
    res.mObjectPointSets.emplace(plateData.first,
                                 std::vector<Eigen::Vector3d>{});

    for (const auto &oneConic : plateData.second) {
      res.mGridId.at(plateData.first).emplace_back(oneConic->label);
      res.mImagePointSets.at(plateData.first).emplace_back(oneConic->center);
      res.mObjectPointSets.at(plateData.first).emplace_back(oneConic->pos);
    }
  }
}

} // namespace dso::DotDetect
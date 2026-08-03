/**
 * This file is part of DSO, written by Jakob Engel.
 * It has been modified by Lukas von Stumberg for the inclusion in DM-VIO
 * (http://vision.in.tum.de/dm-vio).
 *
 * Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../../camera_model/calib_def.h"
#include "IOWrapper/ImageRW.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
namespace dso {

namespace IOWrap {
MinimalImageB* readImageBW_8U(std::string filename) {
  cv::Mat m = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (m.rows * m.cols == 0) {
    printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
    return 0;
  }
  if (m.type() == CV_8UC3) {
    // can happen for webp
    cv::cvtColor(m, m, cv::COLOR_BGR2GRAY);
  }
  if (m.type() != CV_8U) {
    printf("cv::imread did something strange! this may segfault. %i \n", m.type());
    return 0;
  }
  MinimalImageB* img = new MinimalImageB(m.cols, m.rows);
  memcpy(img->data, m.data, m.rows * m.cols);
  return img;
}
void VigCorrection(cv::Mat& image, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& vig_mat) {
  uint8_t raw_val;
  float viged_val;
  cv::Mat img_cv_after_vig = cv::Mat(image.rows, image.cols, CV_8UC1);
  for (size_t col = 0; col < img_cv_after_vig.cols; ++col) {
    for (size_t row = 0; row < img_cv_after_vig.rows; ++row) {
      float vig = vig_mat(row, col);
      // vig = 1.0;
      raw_val = image.at<uint8_t>(row, col);

      if (vig < 0.15) {
        viged_val = 0;
      } else {
        viged_val = static_cast<float>(raw_val) / vig;
        if (viged_val >= 255) {
          viged_val = 255;
        }
      }
      //      std::cout << "raw_val: " << static_cast<int>(raw_val) << ",
      //      viged_val: " << viged_val << ", vig: " << vig
      //                << std::endl;
      img_cv_after_vig.at<uint8_t>(row, col) = static_cast<uint8_t>(viged_val);
    }
  }

  image = img_cv_after_vig.clone();
}
MinimalImageB* readImageBW_8U2(int fid, aligned_vector<dso::CalibFrame>* p_input_data,
                               std::array<std::pair<cv::Mat, cv::Mat>, kCameraNumUsed>* p_cid_to_undist_map,
                               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>* p_vig_mat) {
  // cv::Mat m = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  std::array<cv::Mat, 4> show_mat_vec;
  for (int cam_id = 0; cam_id < kCameraNumUsed; ++cam_id) {
    std::string image_path = (*p_input_data)[fid].cid_to_img_file_path.at(cam_id);
    // cerr << "Reading..." << image_path << endl;
    // if (files[i].back() == '.') continue;  // skip . and ..
    cv::Mat image = cv::imread(image_path, 0);
    // cv::Mat image_before = image.clone();
    VigCorrection(image, (*p_vig_mat));
#ifdef USE_EDGE_ALIGN
    // cv::GaussianBlur(image, image, {5, 5}, 0);
#endif
    cv::remap(image, image, (*p_cid_to_undist_map)[cam_id].first, (*p_cid_to_undist_map)[cam_id].second,
              cv::INTER_CUBIC);
    // cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    show_mat_vec[cam_id] = image.clone();
  }
  //        if (m.rows * m.cols == 0) {
  //            printf("cv::imread could not read image %s! this may segfault.
  //            \n",
  //                   filename.c_str());
  //            return 0;
  //        }
  //        if (m.type() == CV_8UC3) {
  //            // can happen for webp
  //            cv::cvtColor(m, m, cv::COLOR_BGR2GRAY);
  //        }
  //        if (m.type() != CV_8U) {
  //            printf("cv::imread did something strange! this may segfault. %i
  //            \n",
  //                   m.type());
  //            return 0;
  //        }

  //        MinimalImageB *img_target = new MinimalImageB(, hG[0]);
  //
  //        for (int i = 0; i < wG[0] * hG[0]; i++) {
  //            // BRIGHTNESS TRANSFER
  //            float colL = host_dIl[i][0];
  //            if (colL < 0)
  //                colL = 0;
  //            if (colL > 255)
  //                colL = 255;
  //            img_host->at(i) = Vec3b(colL, colL, colL);
  //            colL = dIl[i][0];
  //            if (colL < 0)
  //                colL = 0;
  //            if (colL > 255)
  //                colL = 255;
  //            img_target->at(i) = Vec3b(colL, colL, colL);
  //        }
  MinimalImageB* img = new MinimalImageB(show_mat_vec[0].cols, show_mat_vec[0].rows);
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    ;
    for (int i = 0; i < show_mat_vec[0].cols * show_mat_vec[0].rows; i++) {
      img->data[i + show_mat_vec[0].cols * show_mat_vec[0].rows * cid] = show_mat_vec[cid].data[i];
    }
  }
  // memcpy(img->data, m.data, m.rows * m.cols);
  return img;
}

MinimalImageB3* readImageRGB_8U(std::string filename) {
  cv::Mat m = cv::imread(filename, cv::IMREAD_COLOR);
  if (m.rows * m.cols == 0) {
    printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
    return 0;
  }
  if (m.type() != CV_8UC3) {
    printf("cv::imread did something strange! this may segfault. \n");
    return 0;
  }
  MinimalImageB3* img = new MinimalImageB3(m.cols, m.rows);
  memcpy(img->data, m.data, 3 * m.rows * m.cols);
  return img;
}

MinimalImage<unsigned short>* readImageBW_16U(std::string filename) {
  cv::Mat m = cv::imread(filename, cv::IMREAD_UNCHANGED);
  if (m.rows * m.cols == 0) {
    printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
    return 0;
  }
  if (m.type() != CV_16U) {
    printf(
        "readImageBW_16U called on image that is not a 16bit grayscale "
        "image. this may segfault. \n");
    return 0;
  }
  MinimalImage<unsigned short>* img = new MinimalImage<unsigned short>(m.cols, m.rows);
  memcpy(img->data, m.data, 2 * m.rows * m.cols * kCameraNumUsed);
  return img;
}

MinimalImageB* readStreamBW_8U(char* data, int numBytes) {
  cv::Mat m = cv::imdecode(cv::Mat(numBytes, 1, CV_8U, data), cv::IMREAD_GRAYSCALE);
  if (m.rows * m.cols == 0) {
    printf("cv::imdecode could not read stream (%d bytes)! this may segfault. \n", numBytes);
    return 0;
  }
  if (m.type() != CV_8U) {
    printf("cv::imdecode did something strange! this may segfault. \n");
    return 0;
  }
  MinimalImageB* img = new MinimalImageB(m.cols, m.rows);
  memcpy(img->data, m.data, m.rows * m.cols);
  return img;
}

void writeImage(std::string filename, MinimalImageB* img) {
  cv::imwrite(filename, cv::Mat(img->h * kCameraNumUsed, img->w, CV_8U, img->data));
}

void writeImage(std::string filename, MinimalImageB3* img) {
  cv::imwrite(filename, cv::Mat(img->h * kCameraNumUsed, img->w, CV_8UC3, img->data));
}

void writeImage(std::string filename, MinimalImageF* img) {
  cv::imwrite(filename, cv::Mat(img->h * kCameraNumUsed, img->w, CV_32F, img->data));
}

void writeImage(std::string filename, MinimalImageF3* img) {
  cv::imwrite(filename, cv::Mat(img->h * kCameraNumUsed, img->w, CV_32FC3, img->data));
}

}  // namespace IOWrap

}  // namespace dso

/**
 * This file is part of DSO.
 *
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

#pragma once

#include "../camera_model/calib_def.h"
#include "util/MinimalImage.h"
#include "util/NumType.h"
namespace dso {
namespace IOWrap {
struct CalibFrame;
MinimalImageB *readImageBW_8U(std::string filename);

MinimalImageB *readImageBW_8U2(
    int fid, aligned_vector<dso::CalibFrame> *p_input_data,
    std::array<std::pair<cv::Mat, cv::Mat>, kCameraNumUsed>
        *p_cid_to_undist_map,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> *p_vig_mat);

void VigCorrection(
    cv::Mat &image,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &vig_mat);

MinimalImageB3 *readImageRGB_8U(std::string filename);

MinimalImage<unsigned short> *readImageBW_16U(std::string filename);

MinimalImageB *readStreamBW_8U(char *data, int numBytes);

void writeImage(std::string filename, MinimalImageB *img);

void writeImage(std::string filename, MinimalImageB3 *img);

void writeImage(std::string filename, MinimalImageF *img);

void writeImage(std::string filename, MinimalImageF3 *img);

} // namespace IOWrap
} // namespace dso

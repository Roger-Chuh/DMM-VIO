/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#include "pinhole_camera.h"
using namespace dso;

bool PinholeCamera::Project(const Vec3& p_3d, Vec2& p_img, LinearAlgebraLib::Matrix<number_t, 2, 3>* d_img_d_p3d,
                            LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>* d_img_d_param) const {
  const number_t& fx = parameters_[0];
  const number_t& fy = parameters_[1];
  const number_t& cx = parameters_[2];
  const number_t& cy = parameters_[3];

  const number_t& x = p_3d[0];
  const number_t& y = p_3d[1];
  const number_t& z = p_3d[2];

  p_img[0] = fx * x / z + cx;
  p_img[1] = fy * y / z + cy;

  if (d_img_d_p3d) {
    d_img_d_p3d->setZero();
    const number_t z2 = z * z;

    (*d_img_d_p3d)(0, 0) = fx / z;
    (*d_img_d_p3d)(0, 2) = -fx * x / z2;
    (*d_img_d_p3d)(1, 1) = fy / z;
    (*d_img_d_p3d)(1, 2) = -fy * y / z2;
  }

  if (d_img_d_param) {
    d_img_d_param->resize(2, 4);
    d_img_d_param->setZero();

    (*d_img_d_param)(0, 0) = x / z;
    (*d_img_d_param)(0, 2) = 1.0;
    (*d_img_d_param)(1, 1) = y / z;
    (*d_img_d_param)(1, 3) = 1.0;
  }
  return true;
}

bool PinholeCamera::UnProject(const Vec2& p_img, Vec3& p_3d, LinearAlgebraLib::Matrix<number_t, 3, 2>* d_p3d_d_img,
                              LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>* d_p3d_d_param) const {
  const number_t& fx = parameters_[0];
  const number_t& fy = parameters_[1];
  const number_t& cx = parameters_[2];
  const number_t& cy = parameters_[3];

  const number_t mx = (p_img[0] - cx) / fx;
  const number_t my = (p_img[1] - cy) / fy;

  const number_t r2 = mx * mx + my * my;

  const number_t norm = std::sqrt(number_t(1.0) + r2);
  const number_t norm_inv = number_t(1.0) / norm;

  p_3d[0] = mx * norm_inv;
  p_3d[1] = my * norm_inv;
  p_3d[2] = norm_inv;

  if (d_p3d_d_img || d_p3d_d_param) {
    const number_t d_norm_inv_d_r2 = number_t(-0.5) * norm_inv * norm_inv * norm_inv;

    Vec3 c0, c1;
    c0(0) = (norm_inv + 2 * mx * mx * d_norm_inv_d_r2) / fx;
    c0(1) = (2 * my * mx * d_norm_inv_d_r2) / fx;
    c0(2) = 2 * mx * d_norm_inv_d_r2 / fx;

    c1(0) = (2 * my * mx * d_norm_inv_d_r2) / fy;
    c1(1) = (norm_inv + 2 * my * my * d_norm_inv_d_r2) / fy;
    c1(2) = 2 * my * d_norm_inv_d_r2 / fy;

    if (d_p3d_d_img) {
      d_p3d_d_img->setZero();

      d_p3d_d_img->col(0) = c0;
      d_p3d_d_img->col(1) = c1;
    }

    if (d_p3d_d_param) {
      d_p3d_d_param->resize(3, 4);
      d_p3d_d_param->setZero();

      d_p3d_d_param->col(2) = -c0;
      d_p3d_d_param->col(3) = -c1;
      d_p3d_d_param->col(0) = -c0 * mx;
      d_p3d_d_param->col(1) = -c1 * my;
    }
  }
  return true;
}

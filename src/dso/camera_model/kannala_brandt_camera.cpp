/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#include "kannala_brandt_camera.h"
#include <cmath>

using namespace dso;

inline number_t FastArcTan(const number_t &x) {
  return M_PI_4 * x - x * (fabs(x) - 1) * (0.2447 + 0.0663 * fabs(x));
}
inline number_t FastRealArcTan(const number_t &x) {
  if (x <= 0)
    return -1;
  if (x < 1)
    return FastArcTan(x);
  return M_PI_4 * 2 - FastArcTan(1 / x);
}

bool KB8Camera::Project(
    const Vec3 &p_3d, Vec2 &p_img,
    LinearAlgebraLib::Matrix<number_t, 2, 3> *d_img_d_p3d,
    LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>
        *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];
  const number_t &k1 = parameters_[4];
  const number_t &k2 = parameters_[5];
  const number_t &k3 = parameters_[6];
  const number_t &k4 = parameters_[7];

  const number_t &x = p_3d[0];
  const number_t &y = p_3d[1];
  const number_t &z = p_3d[2];

  const number_t r2 = x * x + y * y;
  const number_t r = std::sqrt(r2);
  // Avoid overflow from divison by r
  if (r > std::sqrt(1e-10)) {
    const number_t theta = std::atan2(r, z);
    //    const number_t theta = FastRealArcTan(r / z);
    const number_t theta2 = theta * theta;

    // 使用下面的因式分解实现多项式，提高效率
    // r_theta = theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + k4 *
    // theta2))))
    number_t r_theta = k4 * theta2;
    r_theta += k3;
    r_theta *= theta2;
    r_theta += k2;
    r_theta *= theta2;
    r_theta += k1;
    r_theta *= theta2;
    r_theta += 1;
    r_theta *= theta;

    const number_t mx = x * r_theta / r;
    const number_t my = y * r_theta / r;

    p_img[0] = fx * mx + cx;
    p_img[1] = fy * my + cy;

    if (d_img_d_p3d) {
      const number_t d_r_d_x = x / r;
      const number_t d_r_d_y = y / r;

      const number_t tmp = (z * z + r2);
      const number_t d_theta_d_x = d_r_d_x * z / tmp;
      const number_t d_theta_d_y = d_r_d_y * z / tmp;
      const number_t d_theta_d_z = -r / tmp;

      number_t d_r_theta_d_theta = number_t(9) * k4 * theta2;
      d_r_theta_d_theta += number_t(7) * k3;
      d_r_theta_d_theta *= theta2;
      d_r_theta_d_theta += number_t(5) * k2;
      d_r_theta_d_theta *= theta2;
      d_r_theta_d_theta += number_t(3) * k1;
      d_r_theta_d_theta *= theta2;
      d_r_theta_d_theta += number_t(1);

      (*d_img_d_p3d)(0, 0) =
          fx *
          (r_theta * r + x * r * d_r_theta_d_theta * d_theta_d_x -
           x * x * r_theta / r) /
          r2;
      (*d_img_d_p3d)(1, 0) =
          fy * y * (d_r_theta_d_theta * d_theta_d_x * r - x * r_theta / r) / r2;

      (*d_img_d_p3d)(0, 1) =
          fx * x * (d_r_theta_d_theta * d_theta_d_y * r - y * r_theta / r) / r2;

      (*d_img_d_p3d)(1, 1) =
          fy *
          (r_theta * r + y * r * d_r_theta_d_theta * d_theta_d_y -
           y * y * r_theta / r) /
          r2;

      (*d_img_d_p3d)(0, 2) = fx * x * d_r_theta_d_theta * d_theta_d_z / r;
      (*d_img_d_p3d)(1, 2) = fy * y * d_r_theta_d_theta * d_theta_d_z / r;
    }

    if (d_img_d_param) {
      d_img_d_param->resize(2, 8);
      (*d_img_d_param).setZero();
      (*d_img_d_param)(0, 0) = mx;
      (*d_img_d_param)(0, 2) = number_t(1);
      (*d_img_d_param)(1, 1) = my;
      (*d_img_d_param)(1, 3) = number_t(1);

      (*d_img_d_param)(0, 4) = fx * x * theta * theta2 / r;
      (*d_img_d_param)(1, 4) = fy * y * theta * theta2 / r;

      d_img_d_param->col(5) = d_img_d_param->col(4) * theta2;
      d_img_d_param->col(6) = d_img_d_param->col(5) * theta2;
      d_img_d_param->col(7) = d_img_d_param->col(6) * theta2;
    }
  } else {
    // degenerate to pinhole at (0, 0) nearby
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
      d_img_d_param->resize(2, 8);
      d_img_d_param->setZero();
      (*d_img_d_param)(0, 0) = x / z;
      (*d_img_d_param)(0, 2) = number_t(1);
      (*d_img_d_param)(1, 1) = y / z;
      (*d_img_d_param)(1, 3) = number_t(1);
    }
  }
  return true;
}

bool KB8Camera::Project(
    const Vec3 &p_3d, LinearAlgebraLib::Ref<Vec2> &p_img,
    LinearAlgebraLib::Matrix<number_t, 2, 3> *d_img_d_p3d,
    LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>
        *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];
  const number_t &k1 = parameters_[4];
  const number_t &k2 = parameters_[5];
  const number_t &k3 = parameters_[6];
  const number_t &k4 = parameters_[7];

  const number_t &x = p_3d[0];
  const number_t &y = p_3d[1];
  const number_t &z = p_3d[2];

  const number_t r2 = x * x + y * y;
  const number_t r = std::sqrt(r2);
  // Avoid overflow from divison by r
  if (r > std::sqrt(1e-10)) {
    const number_t theta = std::atan2(r, z);
    const number_t theta2 = theta * theta;

    // 使用下面的因式分解实现多项式，提高效率
    // r_theta = theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + k4 *
    // theta2))))
    number_t r_theta = k4 * theta2;
    r_theta += k3;
    r_theta *= theta2;
    r_theta += k2;
    r_theta *= theta2;
    r_theta += k1;
    r_theta *= theta2;
    r_theta += 1;
    r_theta *= theta;

    const number_t mx = x * r_theta / r;
    const number_t my = y * r_theta / r;

    p_img[0] = fx * mx + cx;
    p_img[1] = fy * my + cy;

    if (d_img_d_p3d) {
      const number_t d_r_d_x = x / r;
      const number_t d_r_d_y = y / r;

      const number_t tmp = (z * z + r2);
      const number_t d_theta_d_x = d_r_d_x * z / tmp;
      const number_t d_theta_d_y = d_r_d_y * z / tmp;
      const number_t d_theta_d_z = -r / tmp;

      number_t d_r_theta_d_theta = number_t(9) * k4 * theta2;
      d_r_theta_d_theta += number_t(7) * k3;
      d_r_theta_d_theta *= theta2;
      d_r_theta_d_theta += number_t(5) * k2;
      d_r_theta_d_theta *= theta2;
      d_r_theta_d_theta += number_t(3) * k1;
      d_r_theta_d_theta *= theta2;
      d_r_theta_d_theta += number_t(1);

      (*d_img_d_p3d)(0, 0) =
          fx *
          (r_theta * r + x * r * d_r_theta_d_theta * d_theta_d_x -
           x * x * r_theta / r) /
          r2;
      (*d_img_d_p3d)(1, 0) =
          fy * y * (d_r_theta_d_theta * d_theta_d_x * r - x * r_theta / r) / r2;

      (*d_img_d_p3d)(0, 1) =
          fx * x * (d_r_theta_d_theta * d_theta_d_y * r - y * r_theta / r) / r2;

      (*d_img_d_p3d)(1, 1) =
          fy *
          (r_theta * r + y * r * d_r_theta_d_theta * d_theta_d_y -
           y * y * r_theta / r) /
          r2;

      (*d_img_d_p3d)(0, 2) = fx * x * d_r_theta_d_theta * d_theta_d_z / r;
      (*d_img_d_p3d)(1, 2) = fy * y * d_r_theta_d_theta * d_theta_d_z / r;
    }

    if (d_img_d_param) {
      d_img_d_param->resize(2, 8);
      (*d_img_d_param).setZero();
      (*d_img_d_param)(0, 0) = mx;
      (*d_img_d_param)(0, 2) = number_t(1);
      (*d_img_d_param)(1, 1) = my;
      (*d_img_d_param)(1, 3) = number_t(1);

      (*d_img_d_param)(0, 4) = fx * x * theta * theta2 / r;
      (*d_img_d_param)(1, 4) = fy * y * theta * theta2 / r;

      d_img_d_param->col(5) = d_img_d_param->col(4) * theta2;
      d_img_d_param->col(6) = d_img_d_param->col(5) * theta2;
      d_img_d_param->col(7) = d_img_d_param->col(6) * theta2;
    }
  } else {
    // degenerate to pinhole at (0, 0) nearby
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
      d_img_d_param->resize(2, 8);
      d_img_d_param->setZero();
      (*d_img_d_param)(0, 0) = x / z;
      (*d_img_d_param)(0, 2) = number_t(1);
      (*d_img_d_param)(1, 1) = y / z;
      (*d_img_d_param)(1, 3) = number_t(1);
    }
  }
  return true;
}

bool KB8Camera::UnProject(
    const Vec2 &p_img, Vec3 &p_3d,
    LinearAlgebraLib::Matrix<number_t, 3, 2> *d_p3d_d_img,
    LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>
        *d_p3d_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];

  const number_t mx = (p_img[0] - cx) / fx;
  const number_t my = (p_img[1] - cy) / fy;

  number_t theta = 0, sin_theta = 0, cos_theta = 1, thetad, scaling;
  number_t d_func_d_theta = 0;

  scaling = 1.0;
  thetad = std::sqrt(mx * mx + my * my);

  if (thetad > std::sqrt(1e-10)) {
    theta = SolveTheta(thetad, d_func_d_theta);

    sin_theta = std::sin(theta);
    cos_theta = std::cos(theta);
    scaling = sin_theta / thetad;
  }

  p_3d[0] = mx * scaling;
  p_3d[1] = my * scaling;
  p_3d[2] = cos_theta;

  if (d_p3d_d_img || d_p3d_d_param) {
    number_t d_thetad_d_mx = 0;
    number_t d_thetad_d_my = 0;
    number_t d_scaling_d_thetad = 0;
    number_t d_cos_d_thetad = 0;

    number_t d_scaling_d_k1 = 0;
    number_t d_cos_d_k1 = 0;

    number_t theta2 = 0;

    if (thetad > std::sqrt(1e-10)) {
      d_thetad_d_mx = mx / thetad;
      d_thetad_d_my = my / thetad;

      theta2 = theta * theta;

      d_scaling_d_thetad =
          (thetad * cos_theta / d_func_d_theta - sin_theta) / (thetad * thetad);

      d_cos_d_thetad = sin_theta / d_func_d_theta;

      d_scaling_d_k1 = -cos_theta * theta * theta2 / (d_func_d_theta * thetad);

      d_cos_d_k1 = d_cos_d_thetad * theta * theta2;
    }

    const number_t d_res0_d_mx =
        scaling + mx * d_scaling_d_thetad * d_thetad_d_mx;
    const number_t d_res0_d_my = mx * d_scaling_d_thetad * d_thetad_d_my;

    const number_t d_res1_d_mx = my * d_scaling_d_thetad * d_thetad_d_mx;
    const number_t d_res1_d_my =
        scaling + my * d_scaling_d_thetad * d_thetad_d_my;

    const number_t d_res2_d_mx = -d_cos_d_thetad * d_thetad_d_mx;
    const number_t d_res2_d_my = -d_cos_d_thetad * d_thetad_d_my;

    Vec3 c0, c1;

    c0(0) = d_res0_d_mx / fx;
    c0(1) = d_res1_d_mx / fx;
    c0(2) = d_res2_d_mx / fx;

    c1(0) = d_res0_d_my / fy;
    c1(1) = d_res1_d_my / fy;
    c1(2) = d_res2_d_my / fy;

    if (d_p3d_d_img) {
      d_p3d_d_img->setZero();
      d_p3d_d_img->col(0) = c0;
      d_p3d_d_img->col(1) = c1;
    }

    if (d_p3d_d_param) {
      d_p3d_d_param->resize(3, 8);
      d_p3d_d_param->setZero();

      d_p3d_d_param->col(2) = -c0;
      d_p3d_d_param->col(3) = -c1;

      d_p3d_d_param->col(0) = -c0 * mx;
      d_p3d_d_param->col(1) = -c1 * my;

      (*d_p3d_d_param)(0, 4) = mx * d_scaling_d_k1;
      (*d_p3d_d_param)(1, 4) = my * d_scaling_d_k1;
      (*d_p3d_d_param)(2, 4) = d_cos_d_k1;

      d_p3d_d_param->col(5) = d_p3d_d_param->col(4) * theta2;
      d_p3d_d_param->col(6) = d_p3d_d_param->col(5) * theta2;
      d_p3d_d_param->col(7) = d_p3d_d_param->col(6) * theta2;
    }
  }
  return true;
}

number_t KB8Camera::SolveTheta(const number_t &r_theta,
                               number_t &d_func_d_theta) const {
  const number_t &k1 = parameters_[4];
  const number_t &k2 = parameters_[5];
  const number_t &k3 = parameters_[6];
  const number_t &k4 = parameters_[7];

  number_t theta = r_theta;
  for (int i = 3; i > 0; --i) {
    number_t theta2 = theta * theta;
    number_t func = k4 * theta2;
    func += k3;
    func *= theta2;
    func += k2;
    func *= theta2;
    func += k1;
    func *= theta2;
    func += 1;
    func *= theta;

    d_func_d_theta = 9 * k4 * theta2;
    d_func_d_theta += 7 * k3;
    d_func_d_theta *= theta2;
    d_func_d_theta += 5 * k2;
    d_func_d_theta *= theta2;
    d_func_d_theta += 3 * k1;
    d_func_d_theta *= theta2;
    d_func_d_theta += 1;

    // Iteration of Newton method
    theta += (r_theta - func) / d_func_d_theta;
  }
  return theta;
}

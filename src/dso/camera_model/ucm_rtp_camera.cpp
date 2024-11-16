#include "ucm_rtp_camera.h"

#include <iostream>
using namespace dso;

bool UCMRTPCamera::Project(
    const Vec3 &p_3d, Vec2 &p_img, Eigen::Matrix<number_t, 2, 3> *d_img_d_p3d,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];
  number_t alpha = parameters_[4];
  number_t d_alpha_d_temp = 1.;
  if (use_exp_) {
    const number_t &temp = parameters_[4];
    alpha = 1. / (1. + exp(-temp));
    d_alpha_d_temp = alpha * alpha * exp(-temp);
  }
  const number_t &x = p_3d[0];
  const number_t &y = p_3d[1];
  const number_t &z = p_3d[2];

  const number_t &rho2 = p_3d.squaredNorm();
  const number_t &rho = std::sqrt(rho2);

  const number_t &norm = alpha * rho + (1.0 - alpha) * z;

  number_t mx = x / norm;
  number_t my = y / norm;

  number_t ux, uy;
  Mat2 d_uv_d_mxmy;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_uv_d_param;

  // todo: add d_img_d_param
  Distortion(mx, my, ux, uy, (d_img_d_p3d ? &d_uv_d_mxmy : nullptr),
             (d_img_d_param ? &d_uv_d_param : nullptr));
  mx = ux;
  my = uy;

  p_img[0] = fx * mx + cx;
  p_img[1] = fy * my + cy;
  if (p_img.hasNaN()) {
    return false;
  }
  // todo: add d_img_d_param, high level jacobi
  if (d_img_d_p3d) {
    if (level_ != 0) {
      printf("Err: can't get jacobi in high level\n");
      std::abort();
    }

    const number_t &norm2 = norm * norm;
    const number_t &dnorm_dx = alpha * x / rho;
    const number_t &dnorm_dy = alpha * y / rho;
    const number_t &dnorm_dz = alpha * z / rho + (1.0 - alpha);

    const number_t &dmx_dx = (norm - x * dnorm_dx) / norm2;
    const number_t &dmx_dy = (-x * dnorm_dy) / norm2;
    const number_t &dmx_dz = (-x * dnorm_dz) / norm2;

    const number_t &dmy_dx = (-y * dnorm_dx) / norm2;
    const number_t &dmy_dy = (norm - y * dnorm_dy) / norm2;
    const number_t &dmy_dz = (-y * dnorm_dz) / norm2;

    d_img_d_p3d->setZero();
    (*d_img_d_p3d)(0, 0) =
        fx * (d_uv_d_mxmy(0, 0) * dmx_dx + d_uv_d_mxmy(0, 1) * dmy_dx);
    (*d_img_d_p3d)(0, 1) =
        fx * (d_uv_d_mxmy(0, 0) * dmx_dy + d_uv_d_mxmy(0, 1) * dmy_dy);
    (*d_img_d_p3d)(0, 2) =
        fx * (d_uv_d_mxmy(0, 0) * dmx_dz + d_uv_d_mxmy(0, 1) * dmy_dz);

    (*d_img_d_p3d)(1, 0) =
        fy * (d_uv_d_mxmy(1, 0) * dmx_dx + d_uv_d_mxmy(1, 1) * dmy_dx);
    (*d_img_d_p3d)(1, 1) =
        fy * (d_uv_d_mxmy(1, 0) * dmx_dy + d_uv_d_mxmy(1, 1) * dmy_dy);
    (*d_img_d_p3d)(1, 2) =
        fy * (d_uv_d_mxmy(1, 0) * dmx_dz + d_uv_d_mxmy(1, 1) * dmy_dz);
    if (d_img_d_param) {
      d_img_d_param->resize(2, 17);
      d_img_d_param->setZero();
      const number_t dmx_dalpha = (-x * (rho - z)) / norm2;
      const number_t dmy_dalpha = (-y * (rho - z)) / norm2;

      (*d_img_d_param)(0, 0) = mx;
      (*d_img_d_param)(1, 1) = my;
      (*d_img_d_param)(0, 2) = 1;
      (*d_img_d_param)(1, 3) = 1;

      (*d_img_d_param)(0, 4) =
          fx *
          (d_uv_d_mxmy(0, 0) * dmx_dalpha + d_uv_d_mxmy(0, 1) * dmy_dalpha) *
          d_alpha_d_temp;
      (*d_img_d_param)(1, 4) =
          fy *
          (d_uv_d_mxmy(1, 0) * dmx_dalpha + d_uv_d_mxmy(1, 1) * dmy_dalpha) *
          d_alpha_d_temp;
      Eigen::Matrix<number_t, 2, 2> k_fix;
      k_fix << fx, 0, 0, fy;
      d_img_d_param->block<2, 12>(0, 5) = k_fix * d_uv_d_param; // k s p
    }
  }
  return true;
}

bool UCMRTPCamera::Project(
    const Vec3 &p_3d, Eigen::Ref<Vec2> &p_img,
    Eigen::Matrix<number_t, 2, 3> *d_img_d_p3d,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];

  number_t alpha = parameters_[4];
  number_t d_alpha_d_temp = 1.;
  if (use_exp_) {
    const number_t &temp = parameters_[4];
    alpha = 1. / (1. + exp(-temp));
    d_alpha_d_temp = alpha * alpha * exp(-temp);
  }

  const number_t &x = p_3d[0];
  const number_t &y = p_3d[1];
  const number_t &z = p_3d[2];

  const number_t &rho2 = p_3d.squaredNorm();
  const number_t &rho = std::sqrt(rho2);

  const number_t &norm = alpha * rho + (1.0 - alpha) * z;

  number_t mx = x / norm;
  number_t my = y / norm;

  number_t ux, uy;
  Mat2 d_uv_d_mxmy;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_uv_d_param;

  //  std::cout << "pre d_uv_d_mxmy\n" << d_uv_d_mxmy << std::endl;

  // todo: add d_img_d_param
  Distortion(mx, my, ux, uy, (d_img_d_p3d ? &d_uv_d_mxmy : nullptr),
             (d_img_d_param ? &d_uv_d_param : nullptr));
  mx = ux;
  my = uy;

  p_img[0] = fx * mx + cx;
  p_img[1] = fy * my + cy;
  if (p_img.hasNaN()) {
    return false;
  }
  // todo: add d_img_d_param, high level jacobi
  if (d_img_d_p3d) {
    if (level_ != 0) {
      printf("Err: can't get jacobi in high level\n");
      std::abort();
    }

    const number_t &norm2 = norm * norm;
    const number_t &dnorm_dx = alpha * x / rho;
    const number_t &dnorm_dy = alpha * y / rho;
    const number_t &dnorm_dz = alpha * z / rho + (1.0 - alpha);

    const number_t &dmx_dx = (norm - x * dnorm_dx) / norm2;
    const number_t &dmx_dy = (-x * dnorm_dy) / norm2;
    const number_t &dmx_dz = (-x * dnorm_dz) / norm2;

    const number_t &dmy_dx = (-y * dnorm_dx) / norm2;
    const number_t &dmy_dy = (norm - y * dnorm_dy) / norm2;
    const number_t &dmy_dz = (-y * dnorm_dz) / norm2;

    d_img_d_p3d->setZero();
    (*d_img_d_p3d)(0, 0) =
        fx * (d_uv_d_mxmy(0, 0) * dmx_dx + d_uv_d_mxmy(0, 1) * dmy_dx);
    (*d_img_d_p3d)(0, 1) =
        fx * (d_uv_d_mxmy(0, 0) * dmx_dy + d_uv_d_mxmy(0, 1) * dmy_dy);
    (*d_img_d_p3d)(0, 2) =
        fx * (d_uv_d_mxmy(0, 0) * dmx_dz + d_uv_d_mxmy(0, 1) * dmy_dz);

    (*d_img_d_p3d)(1, 0) =
        fy * (d_uv_d_mxmy(1, 0) * dmx_dx + d_uv_d_mxmy(1, 1) * dmy_dx);
    (*d_img_d_p3d)(1, 1) =
        fy * (d_uv_d_mxmy(1, 0) * dmx_dy + d_uv_d_mxmy(1, 1) * dmy_dy);
    (*d_img_d_p3d)(1, 2) =
        fy * (d_uv_d_mxmy(1, 0) * dmx_dz + d_uv_d_mxmy(1, 1) * dmy_dz);
    if (d_img_d_param) {
      d_img_d_param->resize(2, 17);
      d_img_d_param->setZero();
      const number_t dmx_dalpha = (-x * (rho - z)) / norm2;
      const number_t dmy_dalpha = (-y * (rho - z)) / norm2;

      (*d_img_d_param)(0, 0) = mx;
      (*d_img_d_param)(1, 1) = my;
      (*d_img_d_param)(0, 2) = 1;
      (*d_img_d_param)(1, 3) = 1;

      (*d_img_d_param)(0, 4) =
          fx *
          (d_uv_d_mxmy(0, 0) * dmx_dalpha + d_uv_d_mxmy(0, 1) * dmy_dalpha) *
          d_alpha_d_temp;
      (*d_img_d_param)(1, 4) =
          fy *
          (d_uv_d_mxmy(1, 0) * dmx_dalpha + d_uv_d_mxmy(1, 1) * dmy_dalpha) *
          d_alpha_d_temp;
      Eigen::Matrix<number_t, 2, 2> k_fix;
      k_fix << fx, 0, 0, fy;
      d_img_d_param->block<2, 12>(0, 5) = k_fix * d_uv_d_param; // k s p
    }
  }
  return true;
}

bool UCMRTPCamera::UnProject(
    const Vec2 &p_img, Vec3 &p_3d, Eigen::Matrix<number_t, 3, 2> *d_p3d_d_img,
    Eigen::Matrix<number_t, 3, Eigen::Dynamic> *d_p3d_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];

  number_t alpha = parameters_[4];
  number_t d_alpha_d_temp = 1.;
  if (use_exp_) {
    const number_t &temp = parameters_[4];
    alpha = 1. / (1. + exp(-temp));
    d_alpha_d_temp = alpha * alpha * exp(-temp);
  }

  number_t mx = (p_img[0] - cx) / fx;
  number_t my = (p_img[1] - cy) / fy;
  number_t res_x, res_y;
  UnDistortion(mx, my, res_x, res_y);
  mx = res_x, my = res_y;

  mx = (1.0 - alpha) * mx;
  my = (1.0 - alpha) * my;

  const number_t &r2 = mx * mx + my * my;

  const number_t xi = alpha / (1.0 - alpha);
  const number_t xi2 = xi * xi;

  const number_t n = std::sqrt(1.0 + (1.0 - xi2) * r2);
  const number_t m = (1.0 + r2);

  const number_t &k = (xi + n) / m;

  p_3d[0] = k * mx;
  p_3d[1] = k * my;
  p_3d[2] = k - xi;
  p_3d.normalize();
  if (p_3d.hasNaN()) {
    return false;
  }
  // todo: d_p3d_d_img, d_p3d_d_param
  if (d_p3d_d_img || d_p3d_d_param) {
    Vec3 wbar = p_3d;
    wbar = wbar / wbar[2];
    Mat3 dh3 =
        (Mat3::Identity() - wbar * wbar.transpose() / (wbar.squaredNorm())) /
        wbar.norm();
    Mat32 dh = dh3.block<3, 2>(0, 0);

    Vec3 xyz = p_3d;
    Vec2 uv = Vec2::Zero();
    Mat23 df = Mat23::Zero();
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_img_d_param;
    Project(xyz, uv, &df, &d_img_d_param);
    Mat2 dfdh = df * dh;
    Mat2 dg2 = dfdh.inverse();
    if (d_p3d_d_img)
      *d_p3d_d_img = dh * dg2;
    if (d_p3d_d_param)
      *d_p3d_d_param = -dh * dg2 * d_img_d_param.leftCols(17);
  }
  return true;
}

void UCMRTPCamera::Distortion(
    const number_t &p_x, const number_t &p_y, number_t &d_u, number_t &d_v,
    Mat2 *p_d_uv_d_xy,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_uv_d_params) const {
  const number_t &m_k1_ = GetParamByIndex(5);
  const number_t &m_k2_ = GetParamByIndex(6);
  const number_t &m_k3_ = GetParamByIndex(7);
  const number_t &m_k4_ = GetParamByIndex(8);
  const number_t &m_k5_ = GetParamByIndex(9);
  const number_t &m_k6_ = GetParamByIndex(10);
  const number_t &m_p1_ = GetParamByIndex(11);
  const number_t &m_p2_ = GetParamByIndex(12);
  const number_t &m_s1_ = GetParamByIndex(13);
  const number_t &m_s2_ = GetParamByIndex(14);
  const number_t &m_s3_ = GetParamByIndex(15);
  const number_t &m_s4_ = GetParamByIndex(16);

  number_t mx2, my2, mxy, rho2, rho4, rho8, rad_dist;
  mx2 = p_x * p_x;
  my2 = p_y * p_y;
  mxy = p_x * p_y;
  rho2 = mx2 + my2;
  rho4 = rho2 * rho2;
  rho8 = rho4 * rho4;

  rad_dist = 1.0 + m_k1_ * rho2 + m_k2_ * rho2 * rho2 + m_k3_ * rho4 * rho2 +
             m_k4_ * rho4 * rho4 + m_k5_ * rho8 * rho2 + m_k6_ * rho8 * rho4;
  d_u = p_x * rad_dist + 2.0 * m_p1_ * mxy + m_p2_ * (rho2 + 2.0 * mx2) +
        m_s1_ * rho2 + m_s3_ * rho4;
  d_v = p_y * rad_dist + 2.0 * m_p2_ * mxy + m_p1_ * (rho2 + 2.0 * my2) +
        m_s2_ * rho2 + m_s4_ * rho4;

  if (p_d_uv_d_xy) {
    const number_t &drad_dr2 =
        (m_k1_ + 2 * m_k2_ * rho2 + 3 * m_k3_ * rho4 + 4 * m_k4_ * rho4 * rho2 +
         5 * m_k5_ * rho8 + 6 * m_k6_ * rho8 * rho2);
    const number_t &drad_dx = drad_dr2 * 2 * p_x;
    const number_t &drad_dy = drad_dr2 * 2 * p_y;

    (*p_d_uv_d_xy)(0, 0) = rad_dist + 2.0 * m_p1_ * p_y + 6.0 * m_p2_ * p_x +
                           2 * m_s1_ * p_x + 4 * m_s3_ * p_x * rho2 +
                           p_x * drad_dx;

    (*p_d_uv_d_xy)(0, 1) = 2.0 * m_p1_ * p_x + 2 * m_p2_ * p_y +
                           2 * m_s1_ * p_y + 4 * m_s3_ * p_y * rho2 +
                           p_x * drad_dy;
    (*p_d_uv_d_xy)(1, 0) = 2 * m_p1_ * p_x + 2.0 * m_p2_ * p_y +
                           2 * m_s2_ * p_x + 4 * m_s4_ * p_x * rho2 +
                           p_y * drad_dx;
    (*p_d_uv_d_xy)(1, 1) = rad_dist + 6.0 * m_p1_ * p_y + 2.0 * m_p2_ * p_x +
                           2 * m_s2_ * p_y + 4 * m_s4_ * p_y * rho2 +
                           p_y * drad_dy;
  }

  if (d_uv_d_params) {
    d_uv_d_params->resize(2, 12);
    // du_dk1
    (*d_uv_d_params)(0, 0) = p_x * rho2;
    (*d_uv_d_params)(1, 0) = p_y * rho2;
    // du_dk2
    (*d_uv_d_params)(0, 1) = p_x * rho4;
    (*d_uv_d_params)(1, 1) = p_y * rho4;
    // du_dk3
    (*d_uv_d_params)(0, 2) = p_x * rho2 * rho4;
    (*d_uv_d_params)(1, 2) = p_y * rho2 * rho4;
    // du_dk4
    (*d_uv_d_params)(0, 3) = p_x * rho8;
    (*d_uv_d_params)(1, 3) = p_y * rho8;
#ifdef UCMK56
    // du_dk5
    (*d_uv_d_params)(0, 4) = p_x * rho2 * rho8;
    (*d_uv_d_params)(1, 4) = p_y * rho2 * rho8;
    // du_dk6
    (*d_uv_d_params)(0, 5) = p_x * rho4 * rho8;
    (*d_uv_d_params)(1, 5) = p_y * rho4 * rho8;
#else
    // du_dk5
    (*d_uv_d_params)(0, 4) = 0;
    (*d_uv_d_params)(1, 4) = 0;
    // du_dk6
    (*d_uv_d_params)(0, 5) = 0;
    (*d_uv_d_params)(1, 5) = 0;
#endif
    // du_dp1
    (*d_uv_d_params)(0, 6) = 2.0 * p_x * p_y;
    (*d_uv_d_params)(1, 6) = mx2 + 3.0 * my2;
    // du_dp2
    (*d_uv_d_params)(0, 7) = 3.0 * mx2 + my2;
    (*d_uv_d_params)(1, 7) = 2.0 * p_x * p_y;
    // du_ds1
    (*d_uv_d_params)(0, 8) = rho2;
    (*d_uv_d_params)(1, 8) = 0;
    // du_ds2
    (*d_uv_d_params)(0, 9) = 0;
    (*d_uv_d_params)(1, 9) = rho2;
    // du_ds3
    (*d_uv_d_params)(0, 10) = rho4;
    (*d_uv_d_params)(1, 10) = 0;
    // du_ds4
    (*d_uv_d_params)(0, 11) = 0;
    (*d_uv_d_params)(1, 11) = rho4;
  }
}

void UCMRTPCamera::UnDistortion(
    const number_t &d_u, const number_t &d_v, number_t &p_x, number_t &p_y,
    Mat2 *d_xy_d_uv,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_xy_d_params) const {
  number_t res_x = d_u, res_y = d_v, dis_u = 0, dis_v = 0, cost = 0;
  int maxIter = 20;
  Mat2 duv_dxy = Mat2::Zero();
  Vec2 error = Vec2::Zero();

  for (int i = 0; i < maxIter; i++) {
    Distortion(res_x, res_y, dis_u, dis_v, &duv_dxy);
    error(0, 0) = d_u - dis_u;
    error(1, 0) = d_v - dis_v;

    Vec2 J1(duv_dxy(0, 0), duv_dxy(0, 1));
    Vec2 J2(duv_dxy(1, 0), duv_dxy(1, 1));
    Mat2 H = J1 * J1.transpose() + J2 * J2.transpose();
    Vec2 b = error(0, 0) * J1 + error(1, 0) * J2;
    Vec2 delta = H.ldlt().solve(-b);
    res_x -= delta[0];
    res_y -= delta[1];
    cost = error(0, 0) * error(0, 0) + error(1, 0) * error(1, 0);
    if (cost < 1e-15)
      break;
  }
  p_x = res_x;
  p_y = res_y;

  // d_xy_d_uv, d_xy_d_params; not need this
  if (d_xy_d_uv || d_xy_d_params) {
    Mat2 d_uv_d_xy;
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_uv_d_params;
    Distortion(res_x, res_y, dis_u, dis_v, &d_uv_d_xy, &d_uv_d_params);
    if (d_xy_d_uv) {
      *d_xy_d_uv = d_uv_d_xy.inverse();
    }
    if (d_xy_d_params) {
      d_xy_d_params->resize(2, 12);
      (*d_xy_d_params) = d_uv_d_xy.inverse() * d_uv_d_params;
    }
  }
}
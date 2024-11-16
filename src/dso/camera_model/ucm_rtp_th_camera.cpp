#include "ucm_rtp_th_camera.h"

#include <iostream>

using namespace dso;

bool UCMRTPTHCamera::Project(
    const Vec3 &p_3d, Vec2 &p_img, Eigen::Matrix<number_t, 2, 3> *d_img_d_p3d,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];
  const number_t &m_alpha_ = parameters_[4];

  const number_t &real_alpha = 1.0 / (1.0 + exp(-m_alpha_));

  number_t d = p_3d.norm();
  Vec3 p_norm = p_3d.normalized();

  const number_t fi = 1.0 / (real_alpha + (1.0 - real_alpha) * p_norm[2]);
  Vec2 x_undistort = Vec2(p_norm[0], p_norm[1]) * fi;

  Vec2 x_distort;
  Mat2 d_xd_xu;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_xd_param;

  Distortion(
      x_undistort, x_distort,
      (d_img_d_p3d == nullptr && d_img_d_param == nullptr ? nullptr : &d_xd_xu),
      (d_img_d_param == nullptr ? nullptr : &d_xd_param));

  p_img[0] = fx * x_distort[0] + cx;
  p_img[1] = fy * x_distort[1] + cy;

  if (d_img_d_p3d || d_img_d_param) {
    Mat2 d_img_xd;
    d_img_xd << fx, 0, 0, fy;

    Mat23 d_xu_pnorm;
    ComputeDxuDpnorm(p_norm, d_xu_pnorm);

    Mat3 d_pnorm_p = 1.0 / d * (Mat3::Identity() - p_norm * p_norm.transpose());

    if (d_img_d_p3d) {
      (*d_img_d_p3d) = d_img_xd * d_xd_xu * d_xu_pnorm * d_pnorm_p;
    }

    if (d_img_d_param) {
      d_img_d_param->resize(2, kParamLength);
      d_img_d_param->setZero();
      (*d_img_d_param) = d_img_xd * d_xd_param;
      Eigen::Matrix<number_t, 2, 4> d_img_dfc;
      d_img_dfc << x_distort[0], 0, 1, 0, 0, x_distort[1], 0,
          1; // d fx_fy_cx_cy
      (*d_img_d_param).block<2, 4>(0, 0) = d_img_dfc;

      number_t e_alpha = std::exp(-m_alpha_);
      number_t d_alpha_ralpha = e_alpha / (1 + e_alpha) / (1 + e_alpha);

      const number_t fi2 = fi * fi;
      Vec2 d_xu_alpha =
          Vec2(p_norm[0] * (p_norm[2] - 1), p_norm[1] * (p_norm[2] - 1)) * fi2;
      (*d_img_d_param).block<2, 1>(0, 4) =
          d_img_xd * d_xd_xu * d_xu_alpha * d_alpha_ralpha;
    }
  }
  return true;
}

bool UCMRTPTHCamera::Project(
    const Vec3 &p_3d, Eigen::Ref<Vec2> &p_img,
    Eigen::Matrix<number_t, 2, 3> *d_img_d_p3d,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];
  const number_t &m_alpha_ = parameters_[4];

  const number_t &real_alpha = 1.0 / (1.0 + exp(-m_alpha_));

  number_t d = p_3d.norm();
  Vec3 p_norm = p_3d.normalized();

  const number_t fi = 1.0 / (real_alpha + (1.0 - real_alpha) * p_norm[2]);
  Vec2 x_undistort = Vec2(p_norm[0], p_norm[1]) * fi;

  Vec2 x_distort;
  Mat2 d_xd_xu;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_xd_param;

  Distortion(
      x_undistort, x_distort,
      (d_img_d_p3d == nullptr && d_img_d_param == nullptr ? nullptr : &d_xd_xu),
      (d_img_d_param == nullptr ? nullptr : &d_xd_param));

  p_img[0] = fx * x_distort[0] + cx;
  p_img[1] = fy * x_distort[1] + cy;

  if (d_img_d_p3d || d_img_d_param) {
    Mat2 d_img_xd;
    d_img_xd << fx, 0, 0, fy;

    Mat23 d_xu_pnorm;
    ComputeDxuDpnorm(p_norm, d_xu_pnorm);

    Mat3 d_pnorm_p = 1.0 / d * (Mat3::Identity() - p_norm * p_norm.transpose());

    if (d_img_d_p3d) {
      (*d_img_d_p3d) = d_img_xd * d_xd_xu * d_xu_pnorm * d_pnorm_p;
    }

    if (d_img_d_param) {
      d_img_d_param->resize(2, kParamLength);
      d_img_d_param->setZero();
      (*d_img_d_param) = d_img_xd * d_xd_param;
      Eigen::Matrix<number_t, 2, 4> d_img_dfc;
      d_img_dfc << x_distort[0], 0, 1, 0, 0, x_distort[1], 0,
          1; // d fx_fy_cx_cy
      (*d_img_d_param).block<2, 4>(0, 0) = d_img_dfc;

      number_t e_alpha = std::exp(-m_alpha_);
      number_t d_alpha_ralpha = e_alpha / (1 + e_alpha) / (1 + e_alpha);

      const number_t fi2 = fi * fi;
      Vec2 d_xu_alpha =
          Vec2(p_norm[0] * (p_norm[2] - 1), p_norm[1] * (p_norm[2] - 1)) * fi2;
      (*d_img_d_param).block<2, 1>(0, 4) =
          d_img_xd * d_xd_xu * d_xu_alpha * d_alpha_ralpha;
    }
  }
  return true;
}

bool UCMRTPTHCamera::UnProject(
    const Vec2 &p_img, Vec3 &p_3d, Eigen::Matrix<number_t, 3, 2> *d_p3d_d_img,
    Eigen::Matrix<number_t, 3, Eigen::Dynamic> *d_p3d_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];
  const number_t &m_alpha_ = parameters_[4];

  const number_t &real_alpha = 1.0 / (1.0 + exp(-m_alpha_));

  Vec2 x_distort;
  x_distort[0] = (p_img[0] - cx) / fx;
  x_distort[1] = (p_img[1] - cy) / fy;

  Vec2 x_undistort;
  Mat2 d_xu_xd;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_xu_params;
  UnDistortion(
      x_distort, x_undistort,
      (d_p3d_d_img == nullptr && d_p3d_d_param == nullptr ? nullptr : &d_xu_xd),
      (d_p3d_d_param == nullptr ? nullptr : &d_xu_params));

  x_undistort = (1.0 - real_alpha) * x_undistort;
  const number_t &r2 = x_undistort.squaredNorm();

  const number_t &xi = real_alpha / (1.0 - real_alpha);
  const number_t &xi2 = xi * xi;
  const number_t n = std::sqrt(1.0 + (1.0 - xi2) * r2);
  const number_t m = (1.0 + r2);
  const number_t &k = (xi + n) / m;

  p_3d = Vec3(k * x_undistort[0], k * x_undistort[1], k - xi);

  //  if (p_3d.hasNaN()) {
  //    std::cout << "real alpha " << real_alpha << std::endl;
  //    std::cout << "x_distort " << x_distort.transpose() << " p_img " <<
  //    p_img.transpose() << " f " << fx << " " << fy
  //              << " cxy " << cx << " " << cy;
  //    std::cout << " x_undistort " << x_undistort.transpose() << " k " << k <<
  //    " xi " << xi << std::endl;
  //  }

  if (d_p3d_d_img || d_p3d_d_param) {
    Mat2 d_xd_uv;
    d_xd_uv << 1.0 / fx, 0, 0, 1.0 / fy;

    Mat23 d_xu_p3d;
    ComputeDxuDpnorm(p_3d, d_xu_p3d);

    // todo check pseudoInverse
    Mat32 d_p3d_xu =
        d_xu_p3d.transpose() * (d_xu_p3d * d_xu_p3d.transpose()).inverse();

    if (d_p3d_d_img) {
      (*d_p3d_d_img) = d_p3d_xu * d_xu_xd * d_xd_uv;
    }

    if (d_p3d_d_param) {
      d_p3d_d_param->resize(3, kParamLength);
      d_p3d_d_param->setZero();

      (*d_p3d_d_param) = (d_p3d_xu * d_xu_params);

      Eigen::Matrix<number_t, 2, 4> d_xd_fc;
      d_xd_fc << -x_distort[0] / fx, 0, -1.0 / fx, 0, 0, -x_distort[1] / fy, 0,
          -1.0 / fy;

      (*d_p3d_d_param).block<3, 4>(0, 0) = d_p3d_xu * d_xu_xd * d_xd_fc;

      const number_t fi = 1.0 / (real_alpha + (1.0 - real_alpha) * p_3d[2]);
      const number_t fi2 = fi * fi;
      Vec2 d_xu_alpha =
          Vec2(p_3d[0] * (p_3d[2] - 1), p_3d[1] * (p_3d[2] - 1)) * fi2;

      number_t e_alpha = std::exp(-m_alpha_);
      number_t d_alpha_ralpha = e_alpha / (1 + e_alpha) / (1 + e_alpha);

      (*d_p3d_d_param).block<3, 1>(0, 4) =
          -d_p3d_xu * d_xu_alpha * d_alpha_ralpha;

      //      std::cout << "d_param\n" << (*d_p3d_d_param) << std::endl;
    }
  }
  return true;
}

void UCMRTPTHCamera::Distortion(
    const Vec2 &xy, Vec2 &uv, Mat2 *d_uv_xy,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_uv_params) const {
  const Vec6 k_vec = Eigen::Map<const Vec6>(parameters_ + k_start);
  const Vec2 p_vec = Eigen::Map<const Vec2>(parameters_ + p_start);
  const Vec4 s_vec = Eigen::Map<const Vec4>(parameters_ + s_start);

  Vec2 xy_squared = xy.array().square().matrix();
  const number_t &r_sq = xy_squared[0] + xy_squared[1];
  const number_t &r = std::sqrt(r_sq);
  const number_t &th = std::atan(r);
  const number_t &thetaSq = th * th;

  // radial distortion
  number_t th_radial = 1.0;
  number_t theta2is = thetaSq;
  for (int i = 0; i < k_vec.size(); ++i) {
    th_radial += theta2is * k_vec[i];
    theta2is *= thetaSq;
  }
  const number_t th_divr =
      (r < std::numeric_limits<number_t>::epsilon()) ? 1.0 : th / r;
  Vec2 xr_yr = (th_radial * th_divr) * xy;
  const number_t xr_yr_squareNorm = xr_yr.squaredNorm();
  uv = xr_yr;

  // tangent distortion
  const number_t &temp = 2.0 * xr_yr.dot(p_vec);
  uv += temp * xr_yr + xr_yr_squareNorm * p_vec;

  // thin prism distortion
  Vec2 radialPowers2And4;
  radialPowers2And4[0] = xr_yr_squareNorm;
  radialPowers2And4[1] = xr_yr_squareNorm * xr_yr_squareNorm;
  uv[0] += s_vec.topRows(2).dot(radialPowers2And4);
  uv[1] += s_vec.bottomRows(2).dot(radialPowers2And4);

  if (d_uv_xy || d_uv_params) {
    // radial + tangent
    Mat2 d_uv_xryr;
    ComputeDuvDxryr(xr_yr, xr_yr_squareNorm, d_uv_xryr);

    if (d_uv_xy) {
      d_uv_xy->setZero();
      if (r != 0.0) {
        number_t dthD_dth = 1.0;
        number_t theta2i = thetaSq;
        for (size_t i = 0; i < k_vec.size(); ++i) {
          dthD_dth += (2.0 * i + 3.0) * k_vec[i] * theta2i;
          theta2i *= thetaSq;
        }

        const number_t &w1 = dthD_dth / (r_sq + r_sq * r_sq);
        const number_t &w2 = th_radial * th_divr / r_sq;
        const number_t &ab10 = xy[0] * xy[1];
        Mat2 temp3;
        temp3(0, 0) = w1 * xy_squared[0] + w2 * xy_squared[1];
        temp3(0, 1) = (w1 - w2) * ab10;
        temp3(1, 0) = temp3(0, 1);
        temp3(1, 1) = w1 * xy_squared[1] + w2 * xy_squared[0];
        (*d_uv_xy) = d_uv_xryr * temp3;
      }
    }

    if (d_uv_params) {
      d_uv_params->resize(2, kParamLength);
      d_uv_params->setZero();
      // radial distrotion
      Vec2 temp;
      temp = th_divr * d_uv_xryr * xy;
      number_t theta2i = thetaSq;
      for (size_t i = 0; i < k_vec.size(); ++i) {
        d_uv_params->col(k_start + i) = theta2i * temp;
        theta2i *= thetaSq;
      }

      // tangent distortion
      d_uv_params->block<2, 2>(0, p_start) = 2.0 * xr_yr * xr_yr.transpose();
      (*d_uv_params)(0, p_start) += xr_yr_squareNorm;
      (*d_uv_params)(1, p_start + 1) += xr_yr_squareNorm;

      // thin prism distortion
      d_uv_params->block<1, 2>(0, s_start) = radialPowers2And4;
      d_uv_params->block<1, 2>(1, s_start).setZero();

      d_uv_params->block<1, 2>(0, s_start + 2).setZero();
      d_uv_params->block<1, 2>(1, s_start + 2) = radialPowers2And4;
    }
  }
}

void UCMRTPTHCamera::UnDistortion(
    const Vec2 &uv, Vec2 &xy, Mat2 *d_xy_uv,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_xy_params) const {
  const Vec6 k_vec = Eigen::Map<const Vec6>(parameters_ + k_start);
  const Vec2 p_vec = Eigen::Map<const Vec2>(parameters_ + p_start);
  const Vec4 s_vec = Eigen::Map<const Vec4>(parameters_ + s_start);

  int kMaxIterNum = 50;
  number_t kTolerance = 1e-7;

  // get xr_yr by uv
  Vec2 xr_yr = uv;

  int j = 0;
  for (; j < kMaxIterNum; ++j) {
    Vec2 uv_est = xr_yr;
    const number_t &xr_yr_squaredNorm = xr_yr.squaredNorm();

    // tangent distortion
    const number_t &temp = 2.0 * xr_yr.dot(p_vec);
    uv_est += temp * xr_yr + xr_yr_squaredNorm * p_vec;

    // thin prism distortion
    Vec2 radialPowers2And4;
    radialPowers2And4[0] = xr_yr_squaredNorm;
    radialPowers2And4[1] = xr_yr_squaredNorm * xr_yr_squaredNorm;
    uv_est[0] += s_vec.topRows(2).dot(radialPowers2And4);
    uv_est[1] += s_vec.bottomRows(2).dot(radialPowers2And4);

    Mat2 duv_dxryr;
    ComputeDuvDxryr(xr_yr, xr_yr_squaredNorm, duv_dxryr);

    Vec2 dx = duv_dxryr.inverse() * (uv - uv_est);

    xr_yr += dx;

    if (dx.squaredNorm() < kTolerance * kTolerance) {
      break;
    }
  }

  // printf("tangent and thin prism iter %d\n", j);

  const number_t &xr_yrNorm = xr_yr.norm();
  if (xr_yrNorm == 0) {
    // if point is in the center of the image
    xy = xr_yr;
  } else {
    // get th by xr_yrNorm
    number_t th = xr_yrNorm;

    j = 0;
    for (; j < kMaxIterNum; ++j) {
      const number_t &thetaSq = th * th;
      number_t th_radial = 1.0;
      number_t dthD_dth = 1.0;

      number_t theta2is = thetaSq;
      for (int i = 0; i < k_vec.size(); ++i) {
        th_radial += theta2is * k_vec[i];
        dthD_dth += (2.0 * i + 3) * k_vec[i] * theta2is;
        theta2is *= thetaSq;
      }

      th_radial *= th;

      number_t step;
      // make sure we don't divide by zero:
      if (std::abs(dthD_dth) > std::numeric_limits<number_t>::epsilon()) {
        step = (xr_yrNorm - th_radial) / dthD_dth;
      } else {
        // if derivative is close to zero, apply small correction in the
        // appropriate direction to avoid numerical explosions
        step = (xr_yrNorm - th_radial) * dthD_dth > 0.0
                   ? 10.0 * std::numeric_limits<number_t>::epsilon()
                   : -10.0 * std::numeric_limits<number_t>::epsilon();
      }

      th += step;

      if (std::abs(step) < kTolerance) {
        break;
      }

      // revert to within 180 degrees FOV to avoid numerical overflow
      if (abs(th) >= M_PI / 2.0) {
        // the exact value we choose here is not really important, we'll iterate
        // again over it.
        th = 0.999 * M_PI / 2.0;
      }
    }
    // printf("radial distort iter %d\n", j);

    //[ x_r ]  =  xr_yrNorm * [ X / r ]
    //[ y_r ]                 [ Y / r ],   r = sqrt(X*X + Y*Y), r = tan(theta)
    xy = std::tan(th) / xr_yrNorm * xr_yr;
  }

  if (d_xy_params || d_xy_uv) {
    Vec2 uv_temp;
    Mat2 d_uv_xy;
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_uv_param;
    Distortion(xy, uv_temp, &d_uv_xy, &d_uv_param);

    if (d_xy_uv) {
      (*d_xy_uv) = d_uv_xy.inverse();
    }

    if (d_xy_params) {
      d_xy_params->resize(2, kParamLength);
      // std::cout << "duv_xy\n" << d_uv_xy << "\nd_uv_xy.inverse\n" <<
      // d_uv_xy.inverse() << std::endl;
      (*d_xy_params) = -d_uv_xy.inverse() * d_uv_param;
    }
  }
}

void UCMRTPTHCamera::ComputeDuvDxryr(const Vec2 &xr_yr,
                                     const number_t &xr_yr_squaredNorm,
                                     Mat2 &d_uv_xryr) const {
  const Vec2 p_vec = Eigen::Map<const Vec2>(parameters_ + p_start);
  const Vec4 s_vec = Eigen::Map<const Vec4>(parameters_ + s_start);
  d_uv_xryr(0, 0) = 1.0 + 6.0 * xr_yr[0] * p_vec[0] + 2.0 * xr_yr[1] * p_vec[1];
  const number_t &offdiag = 2.0 * (xr_yr[0] * p_vec[1] + xr_yr[1] * p_vec[0]);
  d_uv_xryr(0, 1) = offdiag;
  d_uv_xryr(1, 0) = offdiag;
  d_uv_xryr(1, 1) = 1.0 + 6.0 * xr_yr[1] * p_vec[1] + 2.0 * xr_yr[0] * p_vec[0];
  // thin prism
  const number_t &temp1 = 2.0 * (s_vec[0] + 2.0 * s_vec[1] * xr_yr_squaredNorm);
  d_uv_xryr(0, 0) += xr_yr[0] * temp1;
  d_uv_xryr(0, 1) += xr_yr[1] * temp1;
  const number_t &temp2 = 2.0 * (s_vec[2] + 2.0 * s_vec[3] * xr_yr_squaredNorm);
  d_uv_xryr(1, 0) += xr_yr[0] * temp2;
  d_uv_xryr(1, 1) += xr_yr[1] * temp2;
}

void UCMRTPTHCamera::ComputeDxuDpnorm(const Vec3 &p_norm,
                                      Mat23 &d_xu_pnorm) const {
  const number_t &m_alpha_ = parameters_[4];
  const number_t &real_alpha = 1.0 / (1.0 + exp(-m_alpha_));

  const number_t xi = 1.0 / (real_alpha + (1.0 - real_alpha) * p_norm[2]);
  const number_t xi2 = xi * xi;

  d_xu_pnorm << xi, 0, -p_norm[0] * (1.0 - real_alpha) * xi2, 0, xi,
      -p_norm[1] * (1.0 - real_alpha) * xi2;
}
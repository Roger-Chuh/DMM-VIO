#include "kb16_camera.h"
#include "macro_define.h"
#include <cmath>

// #define _USE_FAST_ARCTAN_IN_KB16_CAM_MODEL_

using namespace dso;

bool KB16Camera::Project(
    const Vec3 &p_3d, Vec2 &p_img, Eigen::Matrix<number_t, 2, 3> *d_img_d_p3d,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];

  Vec2 xy_undistort = Vec2(p_3d[0] / p_3d[2], p_3d[1] / p_3d[2]);
  Vec2 uv_distort;

  Mat2 d_uvd_xy;

  Distortion(xy_undistort, uv_distort,
             (d_img_d_p3d == nullptr ? nullptr : &d_uvd_xy), d_img_d_param);

  p_img[0] = fx * uv_distort[0] + cx;
  p_img[1] = fy * uv_distort[1] + cy;
  if (p_img.hasNaN()) {
    return false;
  }
  if (d_img_d_p3d || d_img_d_param) {
    Mat2 d_img_uvd;
    d_img_uvd << fx, 0, 0, fy;
    if (d_img_d_p3d) {
      Mat23 d_xy_xyz;
      d_xy_xyz << 1.0 / p_3d[2], 0, -p_3d[0] / p_3d[2] / p_3d[2], 0,
          1.0 / p_3d[2], -p_3d[1] / p_3d[2] / p_3d[2];

      (*d_img_d_p3d) = d_img_uvd * d_uvd_xy * d_xy_xyz;
    }

    if (d_img_d_param) {
      (*d_img_d_param) = (d_img_uvd * (*d_img_d_param)).eval();
      Eigen::Matrix<number_t, 2, 4> d_img_dfc;
      d_img_dfc << uv_distort[0], 0, 1, 0, 0, uv_distort[1], 0,
          1; // d fx_fy_cx_cy
      (*d_img_d_param).block<2, 4>(0, 0) = d_img_dfc;
    }
  }
  return true;
}

bool KB16Camera::Project(
    const Vec3 &p_3d, Eigen::Ref<Vec2> &p_img,
    Eigen::Matrix<number_t, 2, 3> *d_img_d_p3d,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];

  Vec2 xy_undistort = Vec2(p_3d[0] / p_3d[2], p_3d[1] / p_3d[2]);
  Vec2 uv_distort;

  Mat2 d_uvd_xy;

  Distortion(xy_undistort, uv_distort,
             (d_img_d_p3d == nullptr ? nullptr : &d_uvd_xy), d_img_d_param);

  p_img[0] = fx * uv_distort[0] + cx;
  p_img[1] = fy * uv_distort[1] + cy;

  if (p_img.hasNaN()) {
    return false;
  }

  if (d_img_d_p3d || d_img_d_param) {
    Mat2 d_img_uvd;
    d_img_uvd << fx, 0, 0, fy;
    if (d_img_d_p3d) {
      Mat23 d_xy_xyz;
      d_xy_xyz << 1.0 / p_3d[2], 0, -p_3d[0] / p_3d[2] / p_3d[2], 0,
          1.0 / p_3d[2], -p_3d[1] / p_3d[2] / p_3d[2];

      (*d_img_d_p3d) = d_img_uvd * d_uvd_xy * d_xy_xyz;
    }

    if (d_img_d_param) {
      (*d_img_d_param) = (d_img_uvd * (*d_img_d_param)).eval();
      Eigen::Matrix<number_t, 2, 4> d_img_dfc;
      d_img_dfc << uv_distort[0], 0, 1, 0, 0, uv_distort[1], 0,
          1; // d fx_fy_cx_cy
      (*d_img_d_param).block<2, 4>(0, 0) = d_img_dfc;
    }
  }
  return true;
}

bool KB16Camera::UnProject(
    const Vec2 &p_img, Vec3 &p_3d, Eigen::Matrix<number_t, 3, 2> *d_p3d_d_img,
    Eigen::Matrix<number_t, 3, Eigen::Dynamic> *d_p3d_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];

  Vec2 uv_distort;
  uv_distort[0] = (p_img[0] - cx) / fx;
  uv_distort[1] = (p_img[1] - cy) / fy;

  Mat2 d_xy_uvd;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_xy_params;
  Vec2 uv_undistort;
  UnDistortion(uv_distort, uv_undistort,
               (d_p3d_d_img == nullptr && d_p3d_d_param == nullptr ? nullptr
                                                                   : &d_xy_uvd),
               (d_p3d_d_param == nullptr ? nullptr : &d_xy_params));

  p_3d = Vec3(uv_undistort[0], uv_undistort[1], 1.0);
  number_t d = p_3d.norm();
  p_3d.normalize();
  if (p_3d.hasNaN()) {
    return false;
  }
  if (d_p3d_d_img || d_p3d_d_param) {
    Mat3 d_p3d_norm = 1.0 / d * (Mat3::Identity() - p_3d * p_3d.transpose());
    Mat32 d_xyz_xy = d_p3d_norm.block<3, 2>(0, 0);

    Mat2 d_uv_cxy;
    d_uv_cxy << -1.0 / fx, 0, 0, -1.0 / fy;

    if (d_p3d_d_param) {
      d_xy_params.block<2, 2>(0, 2) = d_xy_uvd * d_uv_cxy;

      Mat2 d_uv_fxy;
      d_uv_fxy << -uv_distort[0] / fx, 0, 0, -uv_distort[1] / fy;
      d_xy_params.block<2, 2>(0, 0) = d_xy_uvd * d_uv_fxy;

      d_p3d_d_param->resize(3, kParamLength);
      (*d_p3d_d_param) = d_xyz_xy * d_xy_params;
    }

    if (d_p3d_d_img) {
      (*d_p3d_d_img) = d_xyz_xy * d_xy_uvd * (-d_uv_cxy);
    }
  }
  return true;
}

#ifdef _USE_FAST_ARCTAN_IN_KB16_CAM_MODEL_
inline number_t FastArcTan(const number_t &x,
                           number_t *p_d_arctan_x = nullptr) {
  if (p_d_arctan_x) {
    *p_d_arctan_x =
        M_PI_4 - (2 * x - 1) * (0.2447 + 0.0663 * x) - 0.0663 * x * (x - 1);
  }
  return M_PI_4 * x - x * (x - 1) * (0.2447 + 0.0663 * x);
}
inline number_t FastRealArcTan(const number_t &x,
                               number_t *p_d_arctan_x = nullptr) {
  if (x <= 0)
    return -1;
  if (x < 1)
    return FastArcTan(x, p_d_arctan_x);
  else {
    number_t tmp = FastArcTan(1 / x, p_d_arctan_x);

    if (p_d_arctan_x) {
      *p_d_arctan_x = *p_d_arctan_x / (x * x);
    }
    return M_PI_2 - tmp;
  }
  //  return M_PI_4 * 2 - FastArcTan(1 / x);
}
#endif

void KB16Camera::Distortion(
    const Vec2 &xy, Vec2 &uv, Mat2 *d_uv_xy,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_uv_params) const {
  const Vec6 k_vec = Eigen::Map<const Vec6>(parameters_ + k_start);
  const Vec2 p_vec = Eigen::Map<const Vec2>(parameters_ + p_start);
  const Vec4 s_vec = Eigen::Map<const Vec4>(parameters_ + s_start);

  Vec2 xy_squared = xy.array().square().matrix();
  const number_t &r_sq = xy_squared[0] + xy_squared[1];
  const number_t &r = std::sqrt(r_sq);
#ifdef _USE_FAST_ARCTAN_IN_KB16_CAM_MODEL_
  number_t d_th_d_r;
  const number_t &th = FastRealArcTan(r, (d_uv_xy ? &d_th_d_r : nullptr));
#else
  const number_t &th = std::atan(r);
#endif
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

#ifdef _USE_FAST_ARCTAN_IN_KB16_CAM_MODEL_
        const number_t &w1 = dthD_dth * d_th_d_r / r_sq;
#else
        const number_t &w1 = dthD_dth / (r_sq + r_sq * r_sq);
#endif

        const number_t &w2 = th_radial * th_divr / r_sq;
        const number_t &ab10 = xy[0] * xy[1];
        Mat2 temp3;
        temp3(0, 0) = w1 * xy_squared[0] + w2 * xy_squared[1];
        temp3(0, 1) = (w1 - w2) * ab10;
        temp3(1, 0) = temp3(0, 1);
        temp3(1, 1) = w1 * xy_squared[1] + w2 * xy_squared[0];
        (*d_uv_xy) = d_uv_xryr * temp3;
      } else {
        d_uv_xy->setIdentity();
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

void KB16Camera::UnDistortion(
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
    if (th < 0) {
      // LOG_Calib_WARN("th < 0 in KB16 Undistortion\n");
    }
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

void KB16Camera::ComputeDuvDxryr(const Vec2 &xr_yr,
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

void KB16Camera::DistortionLength(
    const Vec2 &xy, Vec2 &uv, Mat2 *d_uv_xy,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_uv_params) const {
  const number_t &p_x = xy[0];
  const number_t &p_y = xy[1];
  number_t &d_u = uv[0];
  number_t &d_v = uv[1];

  const Vec6 k_vec = Eigen::Map<const Vec6>(parameters_ + k_start);
  const Vec2 p_vec = Eigen::Map<const Vec2>(parameters_ + p_start);
  const Vec4 s_vec = Eigen::Map<const Vec4>(parameters_ + s_start);

  const number_t &m_k1_ = k_vec[0];
  const number_t &m_k2_ = k_vec[1];
  const number_t &m_k3_ = k_vec[2];
  const number_t &m_k4_ = k_vec[3];
  const number_t &m_k5_ = k_vec[4];
  const number_t &m_k6_ = k_vec[5];

  const number_t &m_p1_ = p_vec[0];
  const number_t &m_p2_ = p_vec[1];

  const number_t &m_s1_ = s_vec[0];
  const number_t &m_s2_ = s_vec[1];
  const number_t &m_s3_ = s_vec[2];
  const number_t &m_s4_ = s_vec[3];

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

  if (d_uv_xy) {
    const number_t &drad_dr2 =
        (m_k1_ + 2 * m_k2_ * rho2 + 3 * m_k3_ * rho4 + 4 * m_k4_ * rho4 * rho2 +
         5 * m_k5_ * rho8 + 6 * m_k6_ * rho8 * rho2);
    const number_t &drad_dx = drad_dr2 * 2 * p_x;
    const number_t &drad_dy = drad_dr2 * 2 * p_y;

    (*d_uv_xy)(0, 0) = rad_dist + 2.0 * m_p1_ * p_y + 6.0 * m_p2_ * p_x +
                       2 * m_s1_ * p_x + 4 * m_s3_ * p_x * rho2 + p_x * drad_dx;

    (*d_uv_xy)(0, 1) = 2.0 * m_p1_ * p_x + 2 * m_p2_ * p_y + 2 * m_s1_ * p_y +
                       4 * m_s3_ * p_y * rho2 + p_x * drad_dy;
    (*d_uv_xy)(1, 0) = 2 * m_p1_ * p_x + 2.0 * m_p2_ * p_y + 2 * m_s2_ * p_x +
                       4 * m_s4_ * p_x * rho2 + p_y * drad_dx;
    (*d_uv_xy)(1, 1) = rad_dist + 6.0 * m_p1_ * p_y + 2.0 * m_p2_ * p_x +
                       2 * m_s2_ * p_y + 4 * m_s4_ * p_y * rho2 + p_y * drad_dy;
  }

  if (d_uv_params) {
    d_uv_params->resize(2, kParamLength);
    d_uv_params->setZero();
    Eigen::Matrix<number_t, 2, 12> d_uv_params_tmp;
    d_uv_params_tmp.setZero();
    // du_dk1
    (d_uv_params_tmp)(0, 0) = p_x * rho2;
    (d_uv_params_tmp)(1, 0) = p_y * rho2;
    // du_dk2
    (d_uv_params_tmp)(0, 1) = p_x * rho4;
    (d_uv_params_tmp)(1, 1) = p_y * rho4;
    // du_dk3
    (d_uv_params_tmp)(0, 2) = p_x * rho2 * rho4;
    (d_uv_params_tmp)(1, 2) = p_y * rho2 * rho4;
    // du_dk4
    (d_uv_params_tmp)(0, 3) = p_x * rho8;
    (d_uv_params_tmp)(1, 3) = p_y * rho8;
    // du_dk5
    (d_uv_params_tmp)(0, 4) = p_x * rho2 * rho8;
    (d_uv_params_tmp)(1, 4) = p_y * rho2 * rho8;
    // du_dk6
    (d_uv_params_tmp)(0, 5) = p_x * rho4 * rho8;
    (d_uv_params_tmp)(1, 5) = p_y * rho4 * rho8;

    // du_dp1
    (d_uv_params_tmp)(0, 6) = 2.0 * p_x * p_y;
    (d_uv_params_tmp)(1, 6) = mx2 + 3.0 * my2;
    // du_dp2
    (d_uv_params_tmp)(0, 7) = 3.0 * mx2 + my2;
    (d_uv_params_tmp)(1, 7) = 2.0 * p_x * p_y;
    // du_ds1
    (d_uv_params_tmp)(0, 8) = rho2;
    (d_uv_params_tmp)(1, 8) = 0;
    // du_ds2
    (d_uv_params_tmp)(0, 9) = 0;
    (d_uv_params_tmp)(1, 9) = rho2;
    // du_ds3
    (d_uv_params_tmp)(0, 10) = rho4;
    (d_uv_params_tmp)(1, 10) = 0;
    // du_ds4
    (d_uv_params_tmp)(0, 11) = 0;
    (d_uv_params_tmp)(1, 11) = rho4;

    d_uv_params->block<2, 12>(0, k_start) = d_uv_params_tmp;
  }
}

void KB16Camera::UnDistortionLength(
    const Vec2 &uv, Vec2 &xy, Mat2 *d_xy_uv,
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_xy_params) const {
  Vec2 res = uv;
  Vec2 dis;

  number_t cost = 0;
  int maxIter = 20;
  Mat2 duv_dxy = Mat2::Zero();
  Vec2 error = Vec2::Zero();

  for (int i = 0; i < maxIter; i++) {
    DistortionLength(res, dis, &duv_dxy);
    error = uv - dis;

    Vec2 J1(duv_dxy(0, 0), duv_dxy(0, 1));
    Vec2 J2(duv_dxy(1, 0), duv_dxy(1, 1));
    Mat2 H = J1 * J1.transpose() + J2 * J2.transpose();
    Vec2 b = error(0, 0) * J1 + error(1, 0) * J2;
    Vec2 delta = H.ldlt().solve(-b);
    res -= delta;
    cost = error(0, 0) * error(0, 0) + error(1, 0) * error(1, 0);
    if (cost < 1e-15)
      break;
  }

  xy = res;

  if (d_xy_params || d_xy_uv) {
    Vec2 uv_temp;
    Mat2 d_uv_xy;
    Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_uv_param;
    DistortionLength(xy, uv_temp, &d_uv_xy, &d_uv_param);

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

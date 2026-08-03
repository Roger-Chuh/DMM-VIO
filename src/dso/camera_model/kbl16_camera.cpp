//
// Created by qy on 24-6-18.
//

#include "kbl16_camera.h"
#include "macro_define.h"
#include <cmath>

using namespace dso;

bool KBL16Camera::Project(const Vec3& p_3d, Vec2& p_img, Eigen::Matrix<number_t, 2, 3>* d_img_d_p3d,
                          Eigen::Matrix<number_t, 2, Eigen::Dynamic>* d_img_d_param) const {
  const number_t& fx = parameters_[0];
  const number_t& fy = parameters_[1];
  const number_t& cx = parameters_[2];
  const number_t& cy = parameters_[3];

  Vec2 xy_undistort = Vec2(p_3d[0] / p_3d[2], p_3d[1] / p_3d[2]);
  Vec2 uv_distort;

  Mat2 d_uvd_xy;

  Distortion(xy_undistort, uv_distort, (d_img_d_p3d == nullptr ? nullptr : &d_uvd_xy), d_img_d_param);

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
      d_xy_xyz << 1.0 / p_3d[2], 0, -p_3d[0] / p_3d[2] / p_3d[2], 0, 1.0 / p_3d[2], -p_3d[1] / p_3d[2] / p_3d[2];

      (*d_img_d_p3d) = d_img_uvd * d_uvd_xy * d_xy_xyz;
    }

    if (d_img_d_param) {
      (*d_img_d_param) = (d_img_uvd * (*d_img_d_param)).eval();
      Eigen::Matrix<number_t, 2, 4> d_img_dfc;
      d_img_dfc << uv_distort[0], 0, 1, 0, 0, uv_distort[1], 0,
          1;  // d fx_fy_cx_cy
      (*d_img_d_param).block<2, 4>(0, 0) = d_img_dfc;
    }
  }
  return true;
}

bool KBL16Camera::Project(const Vec3& p_3d, Eigen::Ref<Vec2>& p_img, Eigen::Matrix<number_t, 2, 3>* d_img_d_p3d,
                          Eigen::Matrix<number_t, 2, Eigen::Dynamic>* d_img_d_param) const {
  const number_t& fx = parameters_[0];
  const number_t& fy = parameters_[1];
  const number_t& cx = parameters_[2];
  const number_t& cy = parameters_[3];

  Vec2 xy_undistort = Vec2(p_3d[0] / p_3d[2], p_3d[1] / p_3d[2]);
  Vec2 uv_distort;

  Mat2 d_uvd_xy;

  Distortion(xy_undistort, uv_distort, (d_img_d_p3d == nullptr ? nullptr : &d_uvd_xy), d_img_d_param);

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
      d_xy_xyz << 1.0 / p_3d[2], 0, -p_3d[0] / p_3d[2] / p_3d[2], 0, 1.0 / p_3d[2], -p_3d[1] / p_3d[2] / p_3d[2];

      (*d_img_d_p3d) = d_img_uvd * d_uvd_xy * d_xy_xyz;
    }

    if (d_img_d_param) {
      (*d_img_d_param) = (d_img_uvd * (*d_img_d_param)).eval();
      Eigen::Matrix<number_t, 2, 4> d_img_dfc;
      d_img_dfc << uv_distort[0], 0, 1, 0, 0, uv_distort[1], 0,
          1;  // d fx_fy_cx_cy
      (*d_img_d_param).block<2, 4>(0, 0) = d_img_dfc;
    }
  }
  return true;
}

bool KBL16Camera::UnProject(const Vec2& p_img, Vec3& p_3d, Eigen::Matrix<number_t, 3, 2>* d_p3d_d_img,
                            Eigen::Matrix<number_t, 3, Eigen::Dynamic>* d_p3d_d_param) const {
  const number_t& fx = parameters_[0];
  const number_t& fy = parameters_[1];
  const number_t& cx = parameters_[2];
  const number_t& cy = parameters_[3];

  Vec2 uv_distort;
  uv_distort[0] = (p_img[0] - cx) / fx;
  uv_distort[1] = (p_img[1] - cy) / fy;

  Mat2 d_xy_uvd;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> d_xy_params;
  Vec2 uv_undistort;
  UnDistortion(uv_distort, uv_undistort, (d_p3d_d_img == nullptr && d_p3d_d_param == nullptr ? nullptr : &d_xy_uvd),
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

// ------------------------------

inline Vec2 J_length_xy(const Vec2& xy) {
  // Define the optical axis
  const Vec3 optical_axis(0, 0, 1);

  // Compute the bearing vector
  Vec3 bearing;
  bearing << xy[0], xy[1], 1;
  bearing.normalize();

  // Compute the Jacobian analytically
  number_t norm_v = std::sqrt(xy[0] * xy[0] + xy[1] * xy[1] + 1);
  Vec3 v = Vec3(xy[0], xy[1], 1);
  Vec3 unit_v = v / norm_v;
  Vec3 delta = unit_v - optical_axis;
  number_t delta_norm = delta.norm();
  Vec3 delta_normalized = delta / delta_norm;

  Eigen::Matrix<number_t, 1, 3> jacobian_v =
      delta_normalized.transpose() * (Mat3::Identity() - unit_v * unit_v.transpose()) / norm_v;
  Vec2 jacobian = jacobian_v.head<2>();

  return jacobian;
}

void KBL16Camera::Distortion(const Vec2& xy, Vec2& uv, Mat2* d_uv_xy,
                             Eigen::Matrix<number_t, 2, Eigen::Dynamic>* d_uv_params) const {
  const Vec6 k_vec = Eigen::Map<const Vec6>(parameters_ + k_start);
  const Vec2 p_vec = Eigen::Map<const Vec2>(parameters_ + p_start);
  const Vec4 s_vec = Eigen::Map<const Vec4>(parameters_ + s_start);

  Vec2 xy_squared = xy.array().square().matrix();
  const number_t& r_sq = xy_squared[0] + xy_squared[1];
  const number_t& r = std::sqrt(r_sq);

  // radial distortion
  const Vec3 optical_axis = {0, 0, 1};
  Vec3 bearing;
  bearing << xy[0], xy[1], 1;
  bearing.normalize();
  const number_t length = (bearing - optical_axis).norm();
  const number_t lengthSq = length * length;

  number_t length_radial = 1.0;
  number_t length2is = lengthSq;
  for (int i = 0; i < k_vec.size(); ++i) {
    length_radial += length2is * k_vec[i];
    length2is *= lengthSq;
  }
  const number_t length_divr = (r < std::numeric_limits<number_t>::epsilon()) ? 1.0 : length / r;
  Vec2 xr_yr = (length_radial * length_divr) * xy;
  const number_t xr_yr_squareNorm = xr_yr.squaredNorm();
  uv = xr_yr;

  // tangent distortion
  const number_t& temp = 2.0 * xr_yr.dot(p_vec);
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
        number_t dlengthD_dlength = 1.0;
        number_t length2i = lengthSq;
        for (size_t i = 0; i < k_vec.size(); ++i) {
          dlengthD_dlength += (2.0 * i + 3.0) * k_vec[i] * length2i;
          length2i *= lengthSq;
        }

        number_t lengthd = length_radial * length;
        Mat2 d_xryr_xy;
        Mat2 d_xryr_xy_p1 = lengthd / r * Mat2::Identity();
        Mat2 d_xryr_xy_p2 =
            xy * (dlengthD_dlength * J_length_xy(xy).transpose() / r - lengthd / r_sq / r * xy.transpose());
        d_xryr_xy = d_xryr_xy_p1 + d_xryr_xy_p2;

        (*d_uv_xy) = d_uv_xryr * d_xryr_xy;
      } else {
        d_uv_xy->setIdentity();
      }
    }

    if (d_uv_params) {
      d_uv_params->resize(2, kParamLength);
      d_uv_params->setZero();
      // radial distrotion
      Vec2 temp;
      temp = length_divr * d_uv_xryr * xy;
      number_t length2i = lengthSq;
      for (size_t i = 0; i < k_vec.size(); ++i) {
        d_uv_params->col(k_start + i) = length2i * temp;
        length2i *= lengthSq;
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

void KBL16Camera::UnDistortion(const Vec2& uv, Vec2& xy, Mat2* d_xy_uv,
                               Eigen::Matrix<number_t, 2, Eigen::Dynamic>* d_xy_params) const {
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
    const number_t& xr_yr_squaredNorm = xr_yr.squaredNorm();

    // tangent distortion
    const number_t& temp = 2.0 * xr_yr.dot(p_vec);
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

  const number_t& xr_yrNorm = xr_yr.norm();
  if (xr_yrNorm == 0) {
    // if point is in the center of the image
    xy = xr_yr;
  } else {
    // init length = lenthd by xr_yrNorm
    number_t length = xr_yrNorm;

    j = 0;
    for (; j < kMaxIterNum; ++j) {
      const number_t& lengthSq = length * length;
      number_t length_radial = 1.0;
      number_t dlengthD_dlength = 1.0;

      number_t length2is = lengthSq;
      for (int i = 0; i < k_vec.size(); ++i) {
        length_radial += length2is * k_vec[i];
        dlengthD_dlength += (2.0 * i + 3) * k_vec[i] * length2is;
        length2is *= lengthSq;
      }

      length_radial *= length;

      number_t step;
      // make sure we don't divide by zero:
      if (std::abs(dlengthD_dlength) > std::numeric_limits<number_t>::epsilon()) {
        step = (xr_yrNorm - length_radial) / dlengthD_dlength;
      } else {
        // if derivative is close to zero, apply small correction in the
        // appropriate direction to avoid numerical explosions
        step = (xr_yrNorm - length_radial) * dlengthD_dlength > 0.0 ? 10.0 * std::numeric_limits<number_t>::epsilon()
                                                                    : -10.0 * std::numeric_limits<number_t>::epsilon();
      }

      length += step;

      if (std::abs(step) < kTolerance) {
        break;
      }

      // revert to within 180 degrees FOV to avoid numerical overflow
      if (abs(length) >= std::sqrt(2)) {
        // the exact value we choose here is not really important, we'll iterate
        // again over it.
        length = 0.999 * std::sqrt(2);
      }
    }
    // printf("radial distort iter %d\n", j);

    //[ x_r ]  =  xr_yrNorm * [ X / r ]
    //[ y_r ]                 [ Y / r ],   r = sqrt(X*X + Y*Y), r = tan(theta)
    if (length < 0) {
      // LOG_Calib_WARN("length < 0 in lenght624 Undistortion\n");
    }
    number_t cos_theta = 1 - 2 * (0.5 * length) * (0.5 * length);
    number_t sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    xy = sin_theta / cos_theta / xr_yrNorm * xr_yr;
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

void KBL16Camera::ComputeDuvDxryr(const Vec2& xr_yr, const number_t& xr_yr_squaredNorm, Mat2& d_uv_xryr) const {
  const Vec2 p_vec = Eigen::Map<const Vec2>(parameters_ + p_start);
  const Vec4 s_vec = Eigen::Map<const Vec4>(parameters_ + s_start);
  d_uv_xryr(0, 0) = 1.0 + 6.0 * xr_yr[0] * p_vec[0] + 2.0 * xr_yr[1] * p_vec[1];
  const number_t& offdiag = 2.0 * (xr_yr[0] * p_vec[1] + xr_yr[1] * p_vec[0]);
  d_uv_xryr(0, 1) = offdiag;
  d_uv_xryr(1, 0) = offdiag;
  d_uv_xryr(1, 1) = 1.0 + 6.0 * xr_yr[1] * p_vec[1] + 2.0 * xr_yr[0] * p_vec[0];
  // thin prism
  const number_t& temp1 = 2.0 * (s_vec[0] + 2.0 * s_vec[1] * xr_yr_squaredNorm);
  d_uv_xryr(0, 0) += xr_yr[0] * temp1;
  d_uv_xryr(0, 1) += xr_yr[1] * temp1;
  const number_t& temp2 = 2.0 * (s_vec[2] + 2.0 * s_vec[3] * xr_yr_squaredNorm);
  d_uv_xryr(1, 0) += xr_yr[0] * temp2;
  d_uv_xryr(1, 1) += xr_yr[1] * temp2;
}

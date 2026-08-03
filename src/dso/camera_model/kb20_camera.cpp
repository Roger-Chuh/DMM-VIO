/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#include "kb20_camera.h"
#include "vio_math_0.h"
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace dso;

bool KB20Camera::Project(const Vec3& p_3d22, Vec2& p_img, Eigen::Matrix<number_t, 2, 3>* d_img_d_p3d,
                         Eigen::Matrix<number_t, 2, Eigen::Dynamic>* d_img_d_param) const {
  const number_t fx = parameters_[0];
  const number_t fy = parameters_[1];
  const number_t cx = parameters_[2];
  const number_t cy = parameters_[3];

  const number_t k1 = parameters_[4];
  const number_t k2 = parameters_[5];
  const number_t k3 = parameters_[6];
  number_t k4 = 0, k5 = 0, k6 = 0, s5 = 0, s6 = 0, t1 = 0, t2 = 0, alpha = 0, beta = 0, rotx = 0, roty = 0, ofst_x0 = 0,
           ofst_y0 = 0, ofst_x1 = 0, ofst_y1 = 0, ofst_x2 = 0, ofst_y2 = 0, q1 = 0, q2 = 0, q3 = 0, p1_1 = 0, p2_1 = 0,
           p1_2 = 0, p2_2 = 0;
  if (k_nums_used >= 4) {
    k4 = parameters_[7];
  }
  if (k_nums_used >= 5) {
    k5 = parameters_[8];
  }
  if (k_nums_used >= 6) {
    k6 = parameters_[9];
  }
  const number_t p1 = parameters_[p0_start_idx];
  const number_t p2 = parameters_[p0_start_idx + 1];

  const number_t s1 = parameters_[s_start_idx];
  const number_t s2 = parameters_[s_start_idx + 1];
  const number_t s3 = parameters_[s_start_idx + 2];
  const number_t s4 = parameters_[s_start_idx + 3];
  if (opt_s5s6t1t2) {
    s5 = parameters_[s_start_idx + 4];
    s6 = parameters_[s_start_idx + 5];
    t1 = parameters_[tilt_start_idx];
    t2 = parameters_[tilt_start_idx + 1];
  }
  if (extra_param) {
    if (opt_eucm) {
#ifdef USE_EXP_IN_KB20
      alpha = 1. / (1. + exp(-parameters_[eucm_start_idx]));
      beta = exp(parameters_[eucm_start_idx + 1]);
#else
      alpha = parameters_[eucm_start_idx];
      beta = parameters_[eucm_start_idx + 1];
#endif
    }
    if (opt_rot) {
      rotx = parameters_[rot_start_idx];
      roty = parameters_[rot_start_idx + 1];
    }
    if (opt_ofst_xy0) {
      ofst_x0 = parameters_[ofst0_start_idx];
      ofst_y0 = parameters_[ofst0_start_idx + 1];
    }
    if (opt_ofst_xy1) {
      ofst_x1 = parameters_[ofst1_start_idx];
      ofst_y1 = parameters_[ofst1_start_idx + 1];
    }
    if (opt_ofst_xy2) {
      ofst_x2 = parameters_[ofst2_start_idx];
      ofst_y2 = parameters_[ofst2_start_idx + 1];
    }
    if (opt_extra_p) {
      q1 = parameters_[extra_p_start_idx];
      q2 = parameters_[extra_p_start_idx + 1];
      q3 = parameters_[extra_p_start_idx + 2];
    }
    if (opt_p1) {
      p1_1 = parameters_[p1_start_idx];
      p2_1 = parameters_[p1_start_idx + 1];
    }
    if (opt_p2) {
      p1_2 = parameters_[p2_start_idx];
      p2_2 = parameters_[p2_start_idx + 1];
    }
  }
  // const number_t rotz = 0;

  Vec3 p_3d2, p_3d2_temp, p_3d1_temp;
  // Mat3 d_bering_d_p_3d22 = Mat3::Identity();
  Mat3 d_noamalized_z_d_p_3d22 = Mat3::Identity();
  Mat3 d_uv_w_p_d_xy_wo_p2 = Mat3::Identity(), d_uv_w_p_d_xy_wo_p1 = Mat3::Identity();
  Mat32 d_uv_w_p_d_p2, d_uv_w_p_d_p1;
  p_3d2 = p_3d22;
  if (!extra_param) {
    p_3d2 = p_3d22;
  } else {
    // d_bering_d_p_3d22 = ComputeBearingJac(p_3d22);
    // p_3d2 = p_3d22.normalized();
    if (opt_ofst_xy2 || opt_p2) {
      d_noamalized_z_d_p_3d22 = ComputeNormalizedZJac(p_3d22);
      p_3d2 /= p_3d2(2);
      if (opt_ofst_xy2) {
        p_3d2(0) += ofst_x2;
        p_3d2(1) += ofst_y2;
      }
      if (opt_p2) {
        compute_d_uv_w_p_d_xy_wo_p(p_3d2, p_3d2_temp, d_uv_w_p_d_xy_wo_p2, d_uv_w_p_d_p2, p1_2, p2_2);
        p_3d2 = p_3d2_temp;
      }
    } else {
      d_noamalized_z_d_p_3d22.setIdentity();
      p_3d2 = p_3d22;
    }
  }

  number_t z_old = p_3d2(2);
  number_t d = std::sqrt(beta * p_3d2.head(2).squaredNorm() + z_old * z_old);
  number_t d_inv = 1 / d;
  number_t z_new = alpha * d + (1 - alpha) * z_old;
  Vec3 p_3d1 = Vec3(p_3d2(0), p_3d2(1), z_new);
#ifndef USE_TILT_IN_KB20
  Vec3 p_3d1_orig = p_3d1;
  if (opt_ofst_xy1 || opt_p1) {
    p_3d1 /= p_3d1(2);
  }
  if (opt_ofst_xy1) {
    p_3d1(0) += ofst_x1;
    p_3d1(1) += ofst_y1;
  }
  if (opt_p1) {
    compute_d_uv_w_p_d_xy_wo_p(p_3d1, p_3d1_temp, d_uv_w_p_d_xy_wo_p1, d_uv_w_p_d_p1, p1_1, p2_1);
    p_3d1 = p_3d1_temp;
  }
  Vec3 rot_vec = Vec3(rotx, roty, 0);
  Mat3 rotMat = ExpSO3(rot_vec);
  // Mat3 d_normalized_z_d_p3d1 = Mat3::Identity();
  Vec3 p_3d = rotMat * p_3d1;
  Vec3 p_3d0_orig = p_3d;
  if (opt_ofst_xy0) {
    p_3d /= p_3d(2);
  }
#else
  Mat3 maskTilt, invMaskTilt;
  computeTiltProjectionMatrix(rotx, roty, maskTilt, invMaskTilt);
  Vec3 p_3d1_orig = p_3d1;
  p_3d1 /= p_3d1(2);
  if (opt_ofst_xy1) {
    p_3d1(0) += ofst_x1;
    p_3d1(1) += ofst_y1;
  }
  if (opt_p1) {
    compute_d_uv_w_p_d_xy_wo_p(p_3d1, p_3d1_temp, d_uv_w_p_d_xy_wo_p1, d_uv_w_p_d_p1, p1_1, p2_1);
    p_3d1 = p_3d1_temp;
  }
  Vec3 p_3d = maskTilt * p_3d1;
  p_3d /= p_3d(2);
  Vec3 p_3d_orig = p_3d;
#endif
  if (opt_ofst_xy0) {
    p_3d(0) += ofst_x0;
    p_3d(1) += ofst_y0;
  }
  if (!extra_param) {
    p_3d = p_3d2;
  }

  number_t X = p_3d(0);
  number_t Y = p_3d(1);
  number_t Z = p_3d(2);
  // Transform to model plane
  number_t a = X / Z;
  number_t b = Y / Z;
  number_t r = sqrt(a * a + b * b);
  number_t th = atan(r);
  number_t th2 = th * th;
  number_t th4 = th2 * th2;
  number_t th6 = th2 * th4;
  number_t th8 = th4 * th4;
  number_t th10 = th2 * th8;
  number_t th12 = th6 * th6;

  number_t a2 = a * a;
  number_t b2 = b * b;
  number_t r2 = r * r;
  number_t r3 = r * r2;
  number_t r4 = r2 * r2;

  number_t thd = th * (1.0 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8 + k5 * th10 + k6 * th12);

  number_t x_r = a / r * thd;
  number_t y_r = b / r * thd;

  number_t r_d = sqrt(x_r * x_r + y_r * y_r);
  number_t r_d2 = r_d * r_d;
  number_t r_d4 = r_d2 * r_d2;
  number_t r_d6 = r_d2 * r_d4;
  number_t x_r2 = x_r * x_r;
  number_t y_r2 = y_r * y_r;

  number_t q_coeff = 1 + q1 * r_d2 + q2 * r_d4 + q3 * r_d6;
  number_t u_distorted =
      x_r + (p1 * (2.0 * x_r2 + r_d2) + 2.0 * x_r * y_r * p2) * (q_coeff) + s1 * r_d2 + s2 * r_d4 + s5 * r_d6;
  number_t v_distorted =
      y_r + (p2 * (2.0 * y_r2 + r_d2) + 2.0 * x_r * y_r * p1) * (q_coeff) + s3 * r_d2 + s4 * r_d4 + s6 * r_d6;

  Mat3 matTilt, invMatTilt;
  computeTiltProjectionMatrix(t1, t2, matTilt, invMatTilt);

  Vec3 uvDistorted;
  uvDistorted << u_distorted, v_distorted, 1.0;
  Vec3 uvTilted = matTilt * uvDistorted;
  uvTilted /= uvTilted(2);

  number_t u = fx * uvTilted(0) + cx;
  number_t v = fy * uvTilted(1) + cy;
  p_img[0] = u, p_img[1] = v;
  if (p_img.hasNaN()) {
    return false;
  }
  if (d_img_d_p3d || d_img_d_param) {
    Vec2 xr_yr(x_r, y_r);

    Mat2 duvDistorted_dxryr, d_xr_yr_d_ab, d_uv_d_uvDistorted, d_uv_d_uvTilted;
    Mat23 duvDistorted_dq;

    compute_duvDistorted_dxryr(xr_yr, duvDistorted_dxryr, duvDistorted_dq, r_d2, p1, p2, s1, s2, s3, s4, s5, s6, q1, q2,
                               q3);

    number_t d_thd_d_th =
        1.0 + 3.0 * k1 * th2 + 5.0 * k2 * th4 + 7.0 * k3 * th6 + 9.0 * k4 * th8 + 11.0 * k5 * th10 + 13.0 * k6 * th12;
    number_t d_x_r_d_a = b2 / r3 * thd + a2 / (r2 + r4) * d_thd_d_th;
    number_t d_x_r_d_b = -a * b / r3 * thd + a * b / (r2 + r4) * d_thd_d_th;
    number_t d_y_r_d_a = -a * b / r3 * thd + a * b / (r2 + r4) * d_thd_d_th;
    number_t d_y_r_d_b = a2 / r3 * thd + b2 / (r2 + r4) * d_thd_d_th;

    d_xr_yr_d_ab << d_x_r_d_a, d_x_r_d_b, d_y_r_d_a, d_y_r_d_b;
    d_uv_d_uvTilted << fx, 0, 0, fy;

    Mat23 d_ab_d_xyz;
    d_ab_d_xyz << 1 / Z, 0, -X / Z / Z, 0, 1 / Z, -Y / Z / Z;
    ///

    number_t stx = sin(t1);
    number_t ctx = cos(t1);
    number_t sty = sin(t2);
    number_t cty = cos(t2);

    number_t tt1 = ctx;
    number_t tt4 = -stx * sty;
    number_t tt5 = cty;
    number_t tt7 = sty;
    number_t tt8 = -stx * cty;
    number_t tt9 = ctx * cty;

    number_t u_d = uvDistorted(0);
    number_t v_d = uvDistorted(1);

    number_t d_ut_d_ud = tt1 / (tt7 * u_d + tt8 * v_d + tt9) -
                         tt1 * u_d * tt7 / ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_ut_d_vd = -tt1 * u_d * tt8 / ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_vt_d_ud =
        tt4 / (tt7 * u_d + tt8 * v_d + tt9) -
        (tt4 * u_d + tt5 * v_d) * tt7 / ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_vt_d_vd =
        tt5 / (tt7 * u_d + tt8 * v_d + tt9) -
        (tt4 * u_d + tt5 * v_d) * tt8 / ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));

    Mat2 d_uvTilted_d_uvDistorted;
    d_uvTilted_d_uvDistorted << d_ut_d_ud, d_ut_d_vd, d_vt_d_ud, d_vt_d_vd;

    number_t temp = sty * u_d - stx * cty * v_d + ctx * cty;

    number_t d_ut_d_tx = -stx * u_d / temp - ctx * u_d * (-ctx * cty * v_d - stx * cty) / (temp * temp);

    number_t d_ut_d_ty = -ctx * u_d * (cty * u_d + stx * sty * v_d - ctx * sty) / (temp * temp);

    number_t d_vt_d_tx = (-ctx * sty * u_d + cty * v_d) / temp -
                         (cty * v_d - stx * sty * u_d) * (-ctx * cty * v_d - stx * cty) / (temp * temp);

    number_t d_vt_d_ty = (-stx * cty * u_d - sty * v_d) / temp -
                         (cty * v_d - stx * sty * u_d) * (cty * u_d + stx * sty * v_d - ctx * sty) / (temp * temp);

    Mat2 d_uvTilted_d_t1t2;
    d_uvTilted_d_t1t2 << d_ut_d_tx, d_ut_d_ty, d_vt_d_tx, d_vt_d_ty;

    Mat2 d_uv_d_t1t2 = d_uv_d_uvTilted * d_uvTilted_d_t1t2;

    ///
    Mat24 d_uv_d_fxfycxcy;
    d_uv_d_fxfycxcy << uvTilted(0), 0, 1, 0, 0, uvTilted(1), 0, 1;

    Vec2 d_xr_yr_d_thd;
    d_xr_yr_d_thd << a / r, b / r;

    Mat16 d_thd_d_k1k2k3k4k5k6;
    d_thd_d_k1k2k3k4k5k6 << th2, th4, th6, th8, th10, th12;
    d_thd_d_k1k2k3k4k5k6 *= th;
    Mat26 d_uv_d_k1k2k3k4k5k6 =
        d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_dxryr * d_xr_yr_d_thd * d_thd_d_k1k2k3k4k5k6;

    Mat2 duvDistorted_d_p1p2;
    duvDistorted_d_p1p2 << (2.0 * x_r2 + r_d2), 2.0 * x_r * y_r, 2.0 * x_r * y_r, (2.0 * y_r2 + r_d2);
    duvDistorted_d_p1p2 *= q_coeff;

    Mat26 duvDistorted_d_s1s2s3s4s5s6;
    duvDistorted_d_s1s2s3s4s5s6 << r_d2, r_d4, 0, 0, r_d6, 0, 0, 0, r_d2, r_d4, 0, r_d6;
    Mat2 d_uv_d_p1p2 = d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_d_p1p2;
    Mat26 d_uv_d_s1s2s3s4s5s6 = d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_d_s1s2s3s4s5s6;

    Mat23 d_uv_d_q = d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_dq;

#ifndef USE_TILT_IN_KB20
    Mat3 d_p3d0_un_ofsted_normalized_z_d_p3d0_orig = ComputeNormalizedZJac(p_3d0_orig);
    Mat3 d_p3d1_un_ofsted_normalized_z_d_p3d1_orig = ComputeNormalizedZJac(p_3d1_orig);
    if (!opt_ofst_xy0) {
      d_p3d0_un_ofsted_normalized_z_d_p3d0_orig.setIdentity();
    }
    if (!(opt_ofst_xy1 || opt_p1)) {
      d_p3d1_un_ofsted_normalized_z_d_p3d1_orig.setIdentity();
    }
    Mat3 d_p3d1_ofsted_normalized_z_d_p3d1_un_ofsted_normalized = Mat3::Identity();
    Mat3 d_p3d0_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z = Mat3::Identity();
    Mat3 d_p3d0_ofsted_normalized_z_d_p3d1_orig =
        d_p3d0_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z * d_p3d0_un_ofsted_normalized_z_d_p3d0_orig * rotMat *
        d_uv_w_p_d_xy_wo_p1 * d_p3d1_ofsted_normalized_z_d_p3d1_un_ofsted_normalized *
        d_p3d1_un_ofsted_normalized_z_d_p3d1_orig;
    Mat3 d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z =
        d_p3d0_un_ofsted_normalized_z_d_p3d0_orig * rotMat * d_uv_w_p_d_xy_wo_p1;
    Mat3 d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z_with_p1 =
        d_p3d0_un_ofsted_normalized_z_d_p3d0_orig * rotMat;
#else
    Mat3 d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z, d_p3d0_ofsted_normalized_z_d_p3d1_orig;
    Mat32 d_p3d0_ofsted_normalized_z_d_t1t2, d_p3d0_un_ofsted_normalized_z_d_t1t2;
    // Mat3 d_p3d1_d_p3d1_with_ofst = Mat3::Identity();
    ComputeForwardTiltJac(rotx, roty, p_3d1, d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z,
                          d_p3d0_un_ofsted_normalized_z_d_t1t2);
    Mat3 d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z_with_p1 =
        d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z;
    d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z *= d_uv_w_p_d_xy_wo_p1;
    Mat3 d_p3d1_un_ofsted_normalized_z_d_p3d1_orig = ComputeNormalizedZJac(p_3d1_orig);
    if (!opt_ofst_xy1) {
      /// 只要USE_TILT的宏打开了，无论opt_ofst_xy1是否打开，pt3d1永远要求对归一化z的雅可比
      // d_p3d1_un_ofsted_normalized_z_d_p3d1_orig.setIdentity();
    }
    Mat3 d_p3d1_ofsted_normalized_z_d_p3d1_un_ofsted_normalized_z = Mat3::Identity();
    Mat3 d_p3d0_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z = Mat3::Identity();
    d_p3d0_ofsted_normalized_z_d_p3d1_orig = d_p3d0_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z *
                                             d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z *
                                             d_p3d1_ofsted_normalized_z_d_p3d1_un_ofsted_normalized_z *
                                             d_p3d1_un_ofsted_normalized_z_d_p3d1_orig;
    d_p3d0_ofsted_normalized_z_d_t1t2 =
        d_p3d0_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z * d_p3d0_un_ofsted_normalized_z_d_t1t2;
#endif
    Mat3 d_p3d1_orig_d_p3d2_ofsted_normalized_z = Mat3::Identity(),
         d_p3d1_orig_d_p3d2_ofsted_normalized_z_with_p2 = Mat3::Identity();
    d_p3d1_orig_d_p3d2_ofsted_normalized_z(2, 0) = alpha * d_inv * beta * p_3d2(0);
    d_p3d1_orig_d_p3d2_ofsted_normalized_z(2, 1) = alpha * d_inv * beta * p_3d2(1);
    d_p3d1_orig_d_p3d2_ofsted_normalized_z(2, 2) = alpha * d_inv * beta * z_old + 1 - alpha;

    d_p3d1_orig_d_p3d2_ofsted_normalized_z_with_p2 = d_p3d1_orig_d_p3d2_ofsted_normalized_z;
    d_p3d1_orig_d_p3d2_ofsted_normalized_z *= d_uv_w_p_d_xy_wo_p2;

    Mat23 d_img_d_p3d0_ofsted_normalized_z, d_img_d_p3d1_orig;

    d_img_d_p3d0_ofsted_normalized_z.setZero();
    d_img_d_p3d0_ofsted_normalized_z.topLeftCorner<2, 3>() =
        d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_dxryr * d_xr_yr_d_ab * d_ab_d_xyz;
    d_img_d_p3d1_orig = d_img_d_p3d0_ofsted_normalized_z * d_p3d0_ofsted_normalized_z_d_p3d1_orig;

    if (d_img_d_p3d) {
      d_img_d_p3d->resize(2, 3);
      (*d_img_d_p3d).setZero();
      (*d_img_d_p3d) = d_img_d_p3d0_ofsted_normalized_z;
      if (extra_param) {
        Mat3 d_p3d2_ofsted_normalized_z_d_p3d2_un_ofsted_normalized_z = Mat3::Identity();
        (*d_img_d_p3d) *=
            d_p3d0_ofsted_normalized_z_d_p3d1_orig *
            (d_p3d1_orig_d_p3d2_ofsted_normalized_z)*d_p3d2_ofsted_normalized_z_d_p3d2_un_ofsted_normalized_z *
            d_noamalized_z_d_p_3d22;
      }
    }

    if (d_img_d_param) {
      d_img_d_param->resize(2, kParamLength);
      (*d_img_d_param).setZero();
      (*d_img_d_param).leftCols(4) = d_uv_d_fxfycxcy;
      (*d_img_d_param).block(0, 4, 2, k_nums_used) = d_uv_d_k1k2k3k4k5k6.leftCols(k_nums_used);
      (*d_img_d_param).block<2, 2>(0, p0_start_idx) = d_uv_d_p1p2;
      (*d_img_d_param).block(0, s_start_idx, 2, s_size) = d_uv_d_s1s2s3s4s5s6.leftCols(s_size);
      if (opt_s5s6t1t2) {
        (*d_img_d_param).block<2, 2>(0, tilt_start_idx) = d_uv_d_t1t2;
      }
#ifdef USE_EXP_IN_KB20
      number_t d_alpha_d_alpha_true = alpha * alpha * exp(-parameters_[eucm_start_idx]);
      number_t d_beta_d_beta_true = beta;
#else
      number_t d_alpha_d_alpha_true = 1;
      number_t d_beta_d_beta_true = 1;
#endif
      Vec3 d_p3d1_orig_d_alpha, d_p3d1_orig_d_alpha_true;
      Vec3 d_p3d1_orig_d_beta, d_p3d1_orig_d_beta_true;
      d_p3d1_orig_d_alpha << 0, 0, d - z_old;
      d_p3d1_orig_d_beta << 0, 0, alpha * 0.5 * d_inv * p_3d2.head(2).squaredNorm();

      d_p3d1_orig_d_alpha_true = d_p3d1_orig_d_alpha * d_alpha_d_alpha_true;
      d_p3d1_orig_d_beta_true = d_p3d1_orig_d_beta * d_beta_d_beta_true;
      if (extra_param && opt_eucm) {
        (*d_img_d_param).block<2, 1>(0, eucm_start_idx) = d_img_d_p3d1_orig * d_p3d1_orig_d_alpha_true;
        (*d_img_d_param).block<2, 1>(0, eucm_start_idx + 1) = d_img_d_p3d1_orig * d_p3d1_orig_d_beta_true;
      }

#ifndef USE_TILT_IN_KB20
#ifndef USE_JR
      Mat3 d_p3d0_orig_d_rot = -Skew(rotMat * p_3d1);
#else
      Mat3 d_p3d0_orig_d_rot = -rotMat * Skew(p_3d1) * Jr(rot_vec);
#endif
      Mat3 d_p3d0_ofsted_normalized_z_d_rot = d_p3d0_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z *
                                              d_p3d0_un_ofsted_normalized_z_d_p3d0_orig * d_p3d0_orig_d_rot;
      if (extra_param && opt_rot) {
        (*d_img_d_param).block<2, 2>(0, rot_start_idx) =
            d_img_d_p3d0_ofsted_normalized_z * d_p3d0_ofsted_normalized_z_d_rot.leftCols(2);
      }
#else
      (*d_img_d_param).block<2, 2>(0, rot_start_idx) =
          d_img_d_p3d0_ofsted_normalized_z * d_p3d0_ofsted_normalized_z_d_t1t2;
#endif
      //(*d_img_d_param).block<2, 2>(0, 20).setZero();
      //(*d_img_d_param).block<2, 2>(0, 22).setZero();
      // std::cout << "extra_param: "<<extra_param<< ", (*d_img_d_param): \n" <<
      // (*d_img_d_param) << std::endl;

      Mat32 d_p3d0_ofsted_normalized_z_d_ofst_xy0 = Mat32::Zero();
      d_p3d0_ofsted_normalized_z_d_ofst_xy0.topLeftCorner<2, 2>() = Mat2::Identity();

      Mat32 d_p3d1_ofsted_normalized_z_d_ofst_xy1 = Mat32::Zero();
      d_p3d1_ofsted_normalized_z_d_ofst_xy1.topLeftCorner<2, 2>() = Mat2::Identity();

      Mat32 d_p3d2_ofsted_normalized_z_d_ofst_xy2 = Mat32::Zero();
      d_p3d2_ofsted_normalized_z_d_ofst_xy2.topLeftCorner<2, 2>() = Mat2::Identity();
      if (extra_param && opt_ofst_xy0) {
        (*d_img_d_param).block<2, 2>(0, ofst0_start_idx) =
            d_img_d_p3d0_ofsted_normalized_z * d_p3d0_ofsted_normalized_z_d_ofst_xy0;
      }
      if (extra_param && opt_ofst_xy1) {
        (*d_img_d_param).block<2, 2>(0, ofst1_start_idx) =
            d_img_d_p3d0_ofsted_normalized_z * d_p3d0_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z *
            d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z * d_p3d1_ofsted_normalized_z_d_ofst_xy1;
      }
      if (extra_param && opt_ofst_xy2) {
        (*d_img_d_param).block<2, 2>(0, ofst2_start_idx) =
            d_img_d_p3d1_orig * d_p3d1_orig_d_p3d2_ofsted_normalized_z * d_p3d2_ofsted_normalized_z_d_ofst_xy2;
      }

#ifdef FIX_BETA_IN_KB20
      (*d_img_d_param).block<2, 1>(0, eucm_start_idx + 1).setZero();
#endif
      if (extra_param && opt_extra_p) {
        (*d_img_d_param).block<2, 3>(0, extra_p_start_idx) = d_uv_d_q;
      }

      if (extra_param && opt_p1) {
        (*d_img_d_param).block<2, 2>(0, p1_start_idx) =
            d_img_d_p3d0_ofsted_normalized_z * d_p3d0_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z *
            d_p3d0_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z_with_p1 * d_uv_w_p_d_p1;
      }
      if (extra_param && opt_p2) {
        (*d_img_d_param).block<2, 2>(0, p2_start_idx) =
            d_img_d_p3d1_orig * d_p3d1_orig_d_p3d2_ofsted_normalized_z_with_p2 * d_uv_w_p_d_p2;
      }
      if (fix_fc) {
        (*d_img_d_param).leftCols(4).setZero();
      }
      //      if (fix_k) {
      //        (*d_img_d_param).block<2, 6>(0, 4).setZero();
      //      } else {
      //        if (k_nums_used != 6) {
      //          (*d_img_d_param).block(0, 4 + k_nums_used, 2, 6 -
      //          k_nums_used).setZero();
      //        }
      //      }
      //      (*d_img_d_param).block<2, 24>(0, 4).setZero();
      //      PrintIntri();
      //      std::cout << "p3d: " << p_3d22.transpose() << ", extra_param: " <<
      //      extra_param << std::endl; std::cout << "d_uv:d_p3d:\n" <<
      //      (*d_img_d_p3d).transpose() << std::endl; std::cout <<
      //      "d_uv:d_param:\n" << (*d_img_d_param).transpose() << std::endl;
      //      std::exit(-1);
    }
  }
  return true;
}

bool KB20Camera::Project(const Vec3& p_3d22, Eigen::Ref<Vec2>& p_img, Eigen::Matrix<number_t, 2, 3>* d_img_d_p3d,
                         Eigen::Matrix<number_t, 2, Eigen::Dynamic>* d_img_d_param) const {
  const number_t fx = parameters_[0];
  const number_t fy = parameters_[1];
  const number_t cx = parameters_[2];
  const number_t cy = parameters_[3];

  const number_t k1 = parameters_[4];
  const number_t k2 = parameters_[5];
  const number_t k3 = parameters_[6];
  const number_t k4 = parameters_[7];
  const number_t k5 = parameters_[8];
  const number_t k6 = parameters_[9];

  const number_t p1 = parameters_[10];
  const number_t p2 = parameters_[11];

  const number_t s1 = parameters_[12];
  const number_t s2 = parameters_[13];
  const number_t s3 = parameters_[14];
  const number_t s4 = parameters_[15];
  //  std::cerr<<"kb20 proj ref fx:"<<fx<<" fy:"<<fy<<" cx:"<<cx<<"
  //  cy:"<<cy<<"k1:"<<k1<<" k2:"<<k2<<" k3:"<<k3<<" k4:"<<k4<<" k5:"<<k5<<"
  //  k6:"<<k6<<std::endl;
  const number_t s5 = parameters_[16];
  const number_t s6 = parameters_[17];
  const number_t t1 = parameters_[18];
  const number_t t2 = parameters_[19];

#ifdef USE_EXP_IN_KB20
  const number_t alpha = 1. / (1. + exp(-parameters_[20]));
  const number_t beta = exp(parameters_[21]);
#else
  const number_t alpha = parameters_[20];
  const number_t beta = parameters_[21];
#endif
  const number_t rotx = parameters_[22];
  const number_t roty = parameters_[23];

  const number_t ofst_x0 = parameters_[24];
  const number_t ofst_y0 = parameters_[25];
  const number_t ofst_x1 = parameters_[26];
  const number_t ofst_y1 = parameters_[27];
  const number_t ofst_x2 = parameters_[28];
  const number_t ofst_y2 = parameters_[29];

  const number_t q1 = parameters_[30];
  const number_t q2 = parameters_[31];
  const number_t q3 = parameters_[32];

  const number_t p1_1 = parameters_[33];
  const number_t p2_1 = parameters_[34];
  const number_t p1_2 = parameters_[35];
  const number_t p2_2 = parameters_[36];

  // const number_t rotz = 0;

  Vec3 p_3d2;
  // Mat3 d_bering_d_p_3d22 = Mat3::Identity();
  Mat3 d_noamalized_z_d_p_3d22 = Mat3::Identity();
  p_3d2 = p_3d22;
  if (!extra_param) {
    p_3d2 = p_3d22;
  } else {
    // d_bering_d_p_3d22 = ComputeBearingJac(p_3d22);
    // p_3d2 = p_3d22.normalized();
    d_noamalized_z_d_p_3d22 = ComputeNormalizedZJac(p_3d22);
  }

  number_t z_old = p_3d2(2);
  number_t d = std::sqrt(beta * p_3d2.head(2).squaredNorm() + z_old * z_old);
  number_t d_inv = 1 / d;
  number_t z_new = alpha * d + (1 - alpha) * z_old;
  Vec3 p_3d1 = Vec3(p_3d2(0), p_3d2(1), z_new);
#ifndef USE_TILT_IN_KB20
  Vec3 rot_vec = Vec3(rotx, roty, 0);
  Mat3 rotMat = ExpSO3(rot_vec);
  // Mat3 d_normalized_z_d_p3d1 = Mat3::Identity();
  Vec3 p_3d = rotMat * p_3d1;
#else
  Mat3 maskTilt, invMaskTilt;
  computeTiltProjectionMatrix(rotx, roty, maskTilt, invMaskTilt);
  Vec3 p_3d1_orig = p_3d1;
  p_3d1 /= p_3d1(2);
  Vec3 p_3d = maskTilt * p_3d1;
  p_3d /= p_3d(2);
#endif

  if (!extra_param) {
    p_3d = p_3d2;
  }

  number_t X = p_3d(0);
  number_t Y = p_3d(1);
  number_t Z = p_3d(2);
  // Transform to model plane
  number_t a = X / Z;
  number_t b = Y / Z;
  number_t r = sqrt(a * a + b * b);
  number_t th = atan(r);
  number_t th2 = th * th;
  number_t th4 = th2 * th2;
  number_t th6 = th2 * th4;
  number_t th8 = th4 * th4;
  number_t th10 = th2 * th8;
  number_t th12 = th6 * th6;

  number_t a2 = a * a;
  number_t b2 = b * b;
  number_t r2 = r * r;
  number_t r3 = r * r2;
  number_t r4 = r2 * r2;

  number_t thd = th * (1.0 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8 + k5 * th10 + k6 * th12);

  number_t x_r = a / r * thd;
  number_t y_r = b / r * thd;

  number_t r_d = sqrt(x_r * x_r + y_r * y_r);
  number_t r_d2 = r_d * r_d;
  number_t r_d4 = r_d2 * r_d2;
  number_t r_d6 = r_d2 * r_d4;
  number_t x_r2 = x_r * x_r;
  number_t y_r2 = y_r * y_r;

  number_t u_distorted = x_r + p1 * (2.0 * x_r2 + r_d2) + 2.0 * x_r * y_r * p2 + s1 * r_d2 + s2 * r_d4 + s5 * r_d6;
  number_t v_distorted = y_r + p2 * (2.0 * y_r2 + r_d2) + 2.0 * x_r * y_r * p1 + s3 * r_d2 + s4 * r_d4 + s6 * r_d6;

  Mat3 matTilt, invMatTilt;
  computeTiltProjectionMatrix(t1, t2, matTilt, invMatTilt);

  Vec3 uvDistorted;
  uvDistorted << u_distorted, v_distorted, 1.0;
  Vec3 uvTilted = matTilt * uvDistorted;
  uvTilted /= uvTilted(2);

  number_t u = fx * uvTilted(0) + cx;
  number_t v = fy * uvTilted(1) + cy;
  p_img[0] = u, p_img[1] = v;
  if (p_img.hasNaN()) {
    return false;
  }
  if (d_img_d_p3d || d_img_d_param) {
    Vec2 xr_yr(x_r, y_r);

    Mat2 duvDistorted_dxryr, d_xr_yr_d_ab, d_uv_d_uvDistorted, d_uv_d_uvTilted;
    Mat23 duvDistorted_dq;

    compute_duvDistorted_dxryr(xr_yr, duvDistorted_dxryr, duvDistorted_dq, r_d2, p1, p2, s1, s2, s3, s4, s5, s6, q1, q2,
                               q3);

    number_t d_thd_d_th =
        1.0 + 3.0 * k1 * th2 + 5.0 * k2 * th4 + 7.0 * k3 * th6 + 9.0 * k4 * th8 + 11.0 * k5 * th10 + 13.0 * k6 * th12;
    number_t d_x_r_d_a = b2 / r3 * thd + a2 / (r2 + r4) * d_thd_d_th;
    number_t d_x_r_d_b = -a * b / r3 * thd + a * b / (r2 + r4) * d_thd_d_th;
    number_t d_y_r_d_a = -a * b / r3 * thd + a * b / (r2 + r4) * d_thd_d_th;
    number_t d_y_r_d_b = a2 / r3 * thd + b2 / (r2 + r4) * d_thd_d_th;

    d_xr_yr_d_ab << d_x_r_d_a, d_x_r_d_b, d_y_r_d_a, d_y_r_d_b;
    d_uv_d_uvTilted << fx, 0, 0, fy;

    Mat23 d_ab_d_xyz;
    d_ab_d_xyz << 1 / Z, 0, -X / Z / Z, 0, 1 / Z, -Y / Z / Z;
    ///

    number_t stx = sin(t1);
    number_t ctx = cos(t1);
    number_t sty = sin(t2);
    number_t cty = cos(t2);

    number_t tt1 = ctx;
    number_t tt4 = -stx * sty;
    number_t tt5 = cty;
    number_t tt7 = sty;
    number_t tt8 = -stx * cty;
    number_t tt9 = ctx * cty;

    number_t u_d = uvDistorted(0);
    number_t v_d = uvDistorted(1);

    number_t d_ut_d_ud = tt1 / (tt7 * u_d + tt8 * v_d + tt9) -
                         tt1 * u_d * tt7 / ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_ut_d_vd = -tt1 * u_d * tt8 / ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_vt_d_ud =
        tt4 / (tt7 * u_d + tt8 * v_d + tt9) -
        (tt4 * u_d + tt5 * v_d) * tt7 / ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_vt_d_vd =
        tt5 / (tt7 * u_d + tt8 * v_d + tt9) -
        (tt4 * u_d + tt5 * v_d) * tt8 / ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));

    Mat2 d_uvTilted_d_uvDistorted;
    d_uvTilted_d_uvDistorted << d_ut_d_ud, d_ut_d_vd, d_vt_d_ud, d_vt_d_vd;

    number_t temp = sty * u_d - stx * cty * v_d + ctx * cty;

    number_t d_ut_d_tx = -stx * u_d / temp - ctx * u_d * (-ctx * cty * v_d - stx * cty) / (temp * temp);

    number_t d_ut_d_ty = -ctx * u_d * (cty * u_d + stx * sty * v_d - ctx * sty) / (temp * temp);

    number_t d_vt_d_tx = (-ctx * sty * u_d + cty * v_d) / temp -
                         (cty * v_d - stx * sty * u_d) * (-ctx * cty * v_d - stx * cty) / (temp * temp);

    number_t d_vt_d_ty = (-stx * cty * u_d - sty * v_d) / temp -
                         (cty * v_d - stx * sty * u_d) * (cty * u_d + stx * sty * v_d - ctx * sty) / (temp * temp);

    Mat2 d_uvTilted_d_t1t2;
    d_uvTilted_d_t1t2 << d_ut_d_tx, d_ut_d_ty, d_vt_d_tx, d_vt_d_ty;

    Mat2 d_uv_d_t1t2 = d_uv_d_uvTilted * d_uvTilted_d_t1t2;

    ///
    Mat24 d_uv_d_fxfycxcy;
    d_uv_d_fxfycxcy << uvTilted(0), 0, 1, 0, 0, uvTilted(1), 0, 1;

    Vec2 d_xr_yr_d_thd;
    d_xr_yr_d_thd << a / r, b / r;

    Mat16 d_thd_d_k1k2k3k4k5k6;
    d_thd_d_k1k2k3k4k5k6 << th2, th4, th6, th8, th10, th12;
    d_thd_d_k1k2k3k4k5k6 *= th;
    Mat26 d_uv_d_k1k2k3k4k5k6 =
        d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_dxryr * d_xr_yr_d_thd * d_thd_d_k1k2k3k4k5k6;

    Mat2 duvDistorted_d_p1p2;
    duvDistorted_d_p1p2 << (2.0 * x_r2 + r_d2), 2.0 * x_r * y_r, 2.0 * x_r * y_r, (2.0 * y_r2 + r_d2);

    Mat26 duvDistorted_d_s1s2s3s4s5s6;
    duvDistorted_d_s1s2s3s4s5s6 << r_d2, r_d4, 0, 0, r_d6, 0, 0, 0, r_d2, r_d4, 0, r_d6;
    Mat2 d_uv_d_p1p2 = d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_d_p1p2;
    Mat26 d_uv_d_s1s2s3s4s5s6 = d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_d_s1s2s3s4s5s6;

#ifndef USE_TILT_IN_KB20
    Mat3 d_p3d0_d_p3d1 = rotMat;
#else
    Mat3 d_p3d0_d_p3d1_normalized_z, d_p3d0_d_p3d1;
    Mat32 d_p3d0_d_t1t2;
    ComputeForwardTiltJac(rotx, roty, p_3d1, d_p3d0_d_p3d1_normalized_z, d_p3d0_d_t1t2);
    Mat3 d_normalized_z_d_p3d1 = ComputeNormalizedZJac(p_3d1_orig);
    d_p3d0_d_p3d1 = d_p3d0_d_p3d1_normalized_z * d_normalized_z_d_p3d1;
#endif
    Mat3 d_p3d1_d_p3d2 = Mat3::Identity();
    d_p3d1_d_p3d2(2, 0) = alpha * d_inv * beta * p_3d2(0);
    d_p3d1_d_p3d2(2, 1) = alpha * d_inv * beta * p_3d2(1);
    d_p3d1_d_p3d2(2, 2) = alpha * d_inv * beta * z_old + 1 - alpha;

    Mat23 d_img_d_p3d0, d_img_d_p3d1;

    d_img_d_p3d0.setZero();
    d_img_d_p3d0.topLeftCorner<2, 3>() =
        d_uv_d_uvTilted * d_uvTilted_d_uvDistorted * duvDistorted_dxryr * d_xr_yr_d_ab * d_ab_d_xyz;
    d_img_d_p3d1 = d_img_d_p3d0 * d_p3d0_d_p3d1;

    if (d_img_d_p3d) {
      d_img_d_p3d->resize(2, 3);
      (*d_img_d_p3d).setZero();
      (*d_img_d_p3d) = d_img_d_p3d0;
      if (extra_param) {
        (*d_img_d_p3d) *= d_p3d0_d_p3d1 * (d_p3d1_d_p3d2)*d_noamalized_z_d_p_3d22;
      }
    }

    if (d_img_d_param) {
      d_img_d_param->resize(2, kParamLength);
      (*d_img_d_param).setZero();
      (*d_img_d_param).leftCols(4) = d_uv_d_fxfycxcy;
      (*d_img_d_param).block<2, 6>(0, 4) = d_uv_d_k1k2k3k4k5k6;
      (*d_img_d_param).block<2, 2>(0, 10) = d_uv_d_p1p2;
      (*d_img_d_param).block<2, 6>(0, 12) = d_uv_d_s1s2s3s4s5s6;
      (*d_img_d_param).block<2, 2>(0, 18) = d_uv_d_t1t2;
#ifdef USE_EXP_IN_KB20
      number_t d_alpha_d_alpha_true = alpha * alpha * exp(-parameters_[20]);
      number_t d_beta_d_beta_true = beta;
#else
      number_t d_alpha_d_alpha_true = 1;
      number_t d_beta_d_beta_true = 1;
#endif
      Vec3 d_p3d1_d_alpha, d_p3d1_d_alpha_true;
      Vec3 d_p3d1_d_beta, d_p3d1_d_beta_true;
      d_p3d1_d_alpha << 0, 0, d - z_old;
      d_p3d1_d_beta << 0, 0, alpha * 0.5 * d_inv * p_3d2.head(2).squaredNorm();

      d_p3d1_d_alpha_true = d_p3d1_d_alpha * d_alpha_d_alpha_true;
      d_p3d1_d_beta_true = d_p3d1_d_beta * d_beta_d_beta_true;

      (*d_img_d_param).block<2, 1>(0, 20) = d_img_d_p3d1 * d_p3d1_d_alpha_true;
      (*d_img_d_param).block<2, 1>(0, 21) = d_img_d_p3d1 * d_p3d1_d_beta_true;
#ifndef USE_TILT_IN_KB20
#ifndef USE_JR
      Mat3 d_p3d0_d_rot = -Skew(rotMat * p_3d1);
#else
      Mat3 d_p3d0_d_rot = -rotMat * Skew(p_3d1) * Jr(rot_vec);
#endif
      (*d_img_d_param).block<2, 2>(0, 22) = d_img_d_p3d0 * d_p3d0_d_rot.leftCols(2);
#else
      (*d_img_d_param).block<2, 2>(0, 22) = d_img_d_p3d0 * d_p3d0_d_t1t2;
#endif
      // (*d_img_d_param).block<2, 2>(0, 20).setZero();
      // (*d_img_d_param).block<2, 2>(0, 22).setZero();
#ifdef FIX_BETA_IN_KB20
      (*d_img_d_param).block<2, 1>(0, 21).setZero();
#endif
      if (!opt_s5s6t1t2) {
        (*d_img_d_param).block<2, 4>(0, 16).setZero();
      }
      if (!extra_param) {
        (*d_img_d_param).block<2, 17>(0, 20).setZero();
      }
      if (fix_k) {
        (*d_img_d_param).block<2, 6>(0, 4).setZero();
      } else {
        if (k_nums_used != 6) {
          (*d_img_d_param).block(0, 4 + k_nums_used, 2, 6 - k_nums_used).setZero();
        }
      }
    }
  }
  return true;
}

bool KB20Camera::UnProject(const Vec2& p_img, Vec3& p_3d2, Eigen::Matrix<number_t, 3, 2>* d_p3d2_d_img,
                           Eigen::Matrix<number_t, 3, Eigen::Dynamic>* d_p3d2_d_param) const {
  Vec3 p_3d, p_3d1;
  Mat3 d_p3d1_un_ofsted_normalized_z_d_p3d2_orig;
  const number_t fx = parameters_[0];
  const number_t fy = parameters_[1];
  const number_t cx = parameters_[2];
  const number_t cy = parameters_[3];

  const number_t k1 = parameters_[4];
  const number_t k2 = parameters_[5];
  const number_t k3 = parameters_[6];
  number_t k4 = 0, k5 = 0, k6 = 0, s5 = 0, s6 = 0, t1 = 0, t2 = 0, alpha = 0, beta = 0, rotx = 0, roty = 0, ofst_x0 = 0,
           ofst_y0 = 0, ofst_x1 = 0, ofst_y1 = 0, ofst_x2 = 0, ofst_y2 = 0, q1 = 0, q2 = 0, q3 = 0, p1_1 = 0, p2_1 = 0,
           p1_2 = 0, p2_2 = 0;
  if (k_nums_used >= 4) {
    k4 = parameters_[7];
  }
  if (k_nums_used >= 5) {
    k5 = parameters_[8];
  }
  if (k_nums_used >= 6) {
    k6 = parameters_[9];
  }

  const number_t p1 = parameters_[p0_start_idx];
  const number_t p2 = parameters_[p0_start_idx + 1];

  const number_t s1 = parameters_[s_start_idx];
  const number_t s2 = parameters_[s_start_idx + 1];
  const number_t s3 = parameters_[s_start_idx + 2];
  const number_t s4 = parameters_[s_start_idx + 3];
  //  std::cerr<<"kb20 unproj fx:"<<fx<<" fy:"<<fy<<" cx:"<<cx<<"
  //  cy:"<<cy<<"k1:"<<k1<<" k2:"<<k2<<" k3:"<<k3<<" k4:"<<k4<<" k5:"<<k5<<"
  //  k6:"<<k6<<std::endl;
  if (opt_s5s6t1t2) {
    s5 = parameters_[s_start_idx + 4];
    s6 = parameters_[s_start_idx + 5];
    t1 = parameters_[tilt_start_idx];
    t2 = parameters_[tilt_start_idx + 1];
  }
  if (extra_param) {
    if (opt_eucm) {
#ifdef USE_EXP_IN_KB20
      alpha = 1. / (1. + exp(-parameters_[eucm_start_idx]));
      beta = exp(parameters_[eucm_start_idx + 1]);
#else
      alpha = parameters_[eucm_start_idx];
      beta = parameters_[eucm_start_idx + 1];
#endif
    }
    if (opt_rot) {
      rotx = parameters_[rot_start_idx];
      roty = parameters_[rot_start_idx + 1];
    }
    if (opt_ofst_xy0) {
      ofst_x0 = parameters_[ofst0_start_idx];
      ofst_y0 = parameters_[ofst0_start_idx + 1];
    }
    if (opt_ofst_xy1) {
      ofst_x1 = parameters_[ofst1_start_idx];
      ofst_y1 = parameters_[ofst1_start_idx + 1];
    }
    if (opt_ofst_xy2) {
      ofst_x2 = parameters_[ofst2_start_idx];
      ofst_y2 = parameters_[ofst2_start_idx + 1];
    }
    if (opt_extra_p) {
      q1 = parameters_[extra_p_start_idx];
      q2 = parameters_[extra_p_start_idx + 1];
      q3 = parameters_[extra_p_start_idx + 2];
    }
    if (opt_p1) {
      p1_1 = parameters_[p1_start_idx];
      p2_1 = parameters_[p1_start_idx + 1];
    }
    if (opt_p2) {
      p1_2 = parameters_[p2_start_idx];
      p2_2 = parameters_[p2_start_idx + 1];
    }
  }
  // const number_t rotz = 0;
#ifndef USE_TILT_IN_KB20
  Vec3 rot_vec = Vec3(rotx, roty, 0);
  Mat3 rotMat = ExpSO3(rot_vec);
#else
  Mat3 maskTilt, invMaskTilt;
  computeTiltProjectionMatrix(rotx, roty, maskTilt, invMaskTilt);
#endif
  Vec2 xr_yr;

  Mat2 duvDistorted_dxryr;
  Mat23 duvDistorted_dq;

  number_t uTilted = (p_img(0) - cx) / fx;
  number_t vTilted = (p_img(1) - cy) / fy;

  Mat3 matTilt, invMatTilt;
  computeTiltProjectionMatrix(t1, t2, matTilt, invMatTilt);

  Vec3 uvTilted;
  uvTilted << uTilted, vTilted, 1.0;
  Vec3 uvDistorted = invMatTilt * uvTilted;
  uvDistorted /= uvDistorted(2);
  bool suc_distort = compute_xr_yr_from_uvDistorted(uvDistorted.head(2), xr_yr, duvDistorted_dxryr, duvDistorted_dq, p1,
                                                    p2, s1, s2, s3, s4, s5, s6, q1, q2, q3);

  number_t xr_yrNorm = xr_yr.norm();

  number_t theta, thd, scaling;
  number_t dthD_dth;
  bool success = true, suc_theta = true;
  if (xr_yrNorm <= 1e-20) {
    p_3d << 0.0, 0.0, 1.0;
    if (extra_param) {
#ifndef USE_TILT_IN_KB20
      p_3d1 = rotMat.transpose() * p_3d;
#else
      Vec3 p_3d_orig = p_3d;
      p_3d /= p_3d(2);
      Vec3 p_3d1 = invMaskTilt * p_3d;
      p_3d1 /= p_3d1(2);

      //      Mat3 d_normlized_z_d_p3d0 = ComputeNormalizedZJac(p_3d_orig);
      //      Mat3 d_p3d1_d_p3d0_normalized_z, d_p3d1_d_p3d0;
      //      Mat32 d_p3d1_d_t1t2;
      //      ComputeBackwardTiltJac(rotx, roty, p_3d,
      //      d_p3d1_d_p3d0_normalized_z, d_p3d1_d_t1t2);

#endif
      success = compute_p3d2_from_p3d1(p_3d1, p_3d2, d_p3d1_un_ofsted_normalized_z_d_p3d2_orig, alpha, beta, p_img);
      p_3d2.normalize();
    } else {
      p_3d2 = p_3d;
    }
    if (d_p3d2_d_img || d_p3d2_d_param) {
      if (d_p3d2_d_img) {
        d_p3d2_d_img->resize(3, 2);
        (*d_p3d2_d_img).setZero();
      }
      if (d_p3d2_d_param) {
        d_p3d2_d_param->resize(3, kParamLength);
        (*d_p3d2_d_param).setZero();
      }
    }
    return false;  // true;
  } else {
    suc_theta = getThetaFromNorm_xr_yr(xr_yrNorm, theta, dthD_dth, k1, k2, k3, k4, k5, k6);
    //    theta = SolveTheta(xr_yrNorm, dthD_dth, k1, k2, k3, k4);
  }

  thd = xr_yrNorm;
  scaling = std::sin(theta) / thd;
  p_3d << xr_yr(0) * scaling, xr_yr(1) * scaling, std::cos(theta);

  Vec3 p_3d2_orig, p_3d1_orig, p_3d1_, p_3d2_temp, p_3d1_temp, p_3d0_temp;
  Mat3 d_bearing_d_p3d2_un_ofsted_normalized_z;
  Mat3 d_uv_w_p_d_xy_wo_p2 = Mat3::Identity(), d_uv_w_p_d_xy_wo_p1 = Mat3::Identity();
  Mat3 d_uv_w_p_d_xy_wo_p2_inv = Mat3::Identity(), d_uv_w_p_d_xy_wo_p1_inv = Mat3::Identity();
  Mat32 d_uv_w_p_d_p2, d_uv_w_p_d_p1;
  Vec3 p_3d0_orig = p_3d;
  success = true;
  bool suc1 = true, suc2 = true;
  if (extra_param) {
    if (opt_ofst_xy0) {
      p_3d /= p_3d(2);
      p_3d(0) -= ofst_x0;
      p_3d(1) -= ofst_y0;
    }
#ifndef USE_TILT_IN_KB20
    p_3d1 = rotMat.transpose() * p_3d;
    p_3d1_orig = p_3d1;
    if (opt_ofst_xy1 || opt_p1) {
      p_3d1 /= p_3d1(2);
    }
#else
    p_3d /= p_3d(2);
    p_3d1 = invMaskTilt * p_3d;
    p_3d1 /= p_3d1(2);
    p_3d1_orig = p_3d1;
#endif
    if (opt_p1) {
      suc1 = compute_xy_wo_p_from_uv_w_p(p_3d1, p_3d1_temp, d_uv_w_p_d_xy_wo_p1, d_uv_w_p_d_p1, p1_1, p2_1);
      d_uv_w_p_d_xy_wo_p1_inv.setZero();
      d_uv_w_p_d_xy_wo_p1_inv.topLeftCorner<2, 2>() = d_uv_w_p_d_xy_wo_p1.topLeftCorner<2, 2>().inverse();
      p_3d1 = p_3d1_temp;
    }
    if (opt_ofst_xy1) {
      p_3d1(0) -= ofst_x1;
      p_3d1(1) -= ofst_y1;
    }
    success = compute_p3d2_from_p3d1(p_3d1, p_3d2, d_p3d1_un_ofsted_normalized_z_d_p3d2_orig, alpha, beta, p_img);
    p_3d2_orig = p_3d2;
    if (opt_ofst_xy2 || opt_p2) {
      p_3d2 /= p_3d2(2);
    }
    if (opt_p2) {
      suc2 = compute_xy_wo_p_from_uv_w_p(p_3d2, p_3d2_temp, d_uv_w_p_d_xy_wo_p2, d_uv_w_p_d_p2, p1_2, p2_2);
      d_uv_w_p_d_xy_wo_p2_inv.setZero();
      d_uv_w_p_d_xy_wo_p2_inv.topLeftCorner<2, 2>() = d_uv_w_p_d_xy_wo_p2.topLeftCorner<2, 2>().inverse();
      p_3d2 = p_3d2_temp;
    }
    if (opt_ofst_xy2) {
      p_3d2(0) -= ofst_x2;
      p_3d2(1) -= ofst_y2;
    }
    //    if (opt_p2) {
    //      compute_xy_wo_p_from_uv_w_p(p_3d2, p_3d2_temp, d_uv_w_p_d_xy_wo_p2,
    //      d_uv_w_p_d_p2, p1_2, p2_2); d_uv_w_p_d_xy_wo_p2_inv.setZero();
    //      d_uv_w_p_d_xy_wo_p2_inv.topLeftCorner<2, 2>() =
    //      d_uv_w_p_d_xy_wo_p2.topLeftCorner<2, 2>().inverse(); p_3d2 =
    //      p_3d2_temp;
    //    }
    d_bearing_d_p3d2_un_ofsted_normalized_z = ComputeBearingJac(p_3d2);
    p_3d2.normalize();
  } else {
    p_3d2 = p_3d;
  }

  // std::cout << "p_3d2: " << p_3d2.transpose() << std::endl;
  if (p_3d.hasNaN() /*|| !success || !suc1 || !suc2 || !suc_distort || !suc_theta*/) {
    return false;
  }
  //  std::cerr<<"p_3d:"<<p_3d.transpose()<<std::endl;
  // p_3d.normalize();
  if (d_p3d2_d_img || d_p3d2_d_param) {
    number_t sin_theta = sin(theta);
    number_t cos_theta = cos(theta);
    number_t mx = xr_yr(0);
    number_t my = xr_yr(1);

    number_t x_r = mx;
    number_t y_r = my;
    number_t x_r2 = x_r * x_r;
    number_t y_r2 = y_r * y_r;
    number_t r_d = sqrt(x_r2 + y_r2);
    number_t r_d2 = r_d * r_d;
    number_t r_d4 = r_d2 * r_d2;
    number_t r_d6 = r_d2 * r_d4;

    number_t q_coeff = 1 + q1 * r_d2 + q2 * r_d4 + q3 * r_d6;

    number_t d_thetad_d_mx = mx / thd;
    number_t d_thetad_d_my = my / thd;

    number_t theta2 = theta * theta;
    number_t d_scaling_d_thetad = (thd * cos_theta / dthD_dth - sin_theta) / (thd * thd);
    number_t d_cos_d_thetad = -sin_theta / dthD_dth;
    number_t d_scaling_d_k1 = -cos_theta * theta * theta2 / (dthD_dth * thd);
    number_t d_cos_d_k1 = -d_cos_d_thetad * theta * theta2;

    Vec3 d_pt3d_d_k1;
    d_pt3d_d_k1 << mx * d_scaling_d_k1, my * d_scaling_d_k1, d_cos_d_k1;

    Mat3 d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z = Mat3::Identity();
    Mat3 d_p3d2_ofsted_normalized_z_d_p3d2_orig = ComputeNormalizedZJac(p_3d2_orig);
    Mat3 d_p3d1_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z = Mat3::Identity();
    Mat3 d_p3d1_ofsted_normalized_z_d_p3d1_orig = ComputeNormalizedZJac(p_3d1_orig);
    Mat3 d_p3d0_un_ofsted_normalized_z_d_p3d0_ofsted_normalized_z = Mat3::Identity();
    Mat3 d_p3d0_ofsted_normalized_z_d_p3d0_orig = ComputeNormalizedZJac(p_3d0_orig);
#ifndef USE_TILT_IN_KB20
    if (!opt_ofst_xy0) {
      d_p3d0_ofsted_normalized_z_d_p3d0_orig.setIdentity();
    }
    if (!(opt_ofst_xy1 || opt_p1)) {
      d_p3d1_ofsted_normalized_z_d_p3d1_orig.setIdentity();
    }
    if (!(opt_ofst_xy2 || opt_p2)) {
      d_p3d2_ofsted_normalized_z_d_p3d2_orig.setIdentity();
    }
    Mat3 d_p3d1_orig_d_p3d0_un_ofsted_normalized_z = rotMat.transpose();
    Mat3 d_p3d1_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z =
        d_p3d1_ofsted_normalized_z_d_p3d1_orig * d_p3d1_orig_d_p3d0_un_ofsted_normalized_z;
#else
    if (!(opt_ofst_xy2 || opt_p2)) {
      d_p3d2_ofsted_normalized_z_d_p3d2_orig.setIdentity();
    }
    Mat3 d_p3d1_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z, d_p3d1_d_p3d0_orig;
    Mat32 d_p3d1_ofsted_normalized_z_d_t1t2;
    ComputeBackwardTiltJac(rotx, roty, p_3d, d_p3d1_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z,
                           d_p3d1_ofsted_normalized_z_d_t1t2);
#endif

    if (extra_param) {
      d_pt3d_d_k1 = d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
                    (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
                    (d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) *
                    d_p3d1_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z *
                    (d_uv_w_p_d_xy_wo_p1_inv)*d_p3d1_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z *
                    d_p3d0_un_ofsted_normalized_z_d_p3d0_ofsted_normalized_z * d_p3d0_ofsted_normalized_z_d_p3d0_orig *
                    d_pt3d_d_k1;
    }
    Vec3 d_pt3d_d_k2 = d_pt3d_d_k1 * theta2;
    Vec3 d_pt3d_d_k3 = d_pt3d_d_k2 * theta2;
    Vec3 d_pt3d_d_k4 = d_pt3d_d_k3 * theta2;
    Vec3 d_pt3d_d_k5 = d_pt3d_d_k4 * theta2;
    Vec3 d_pt3d_d_k6 = d_pt3d_d_k5 * theta2;

    number_t d_X_d_mx = scaling + mx * d_scaling_d_thetad * d_thetad_d_mx;
    number_t d_X_d_my = mx * d_scaling_d_thetad * d_thetad_d_my;

    number_t d_Y_d_mx = my * d_scaling_d_thetad * d_thetad_d_mx;
    number_t d_Y_d_my = scaling + my * d_scaling_d_thetad * d_thetad_d_my;

    number_t d_Z_d_mx = d_cos_d_thetad * d_thetad_d_mx;
    number_t d_Z_d_my = d_cos_d_thetad * d_thetad_d_my;

    Mat32 d_pt3d_d_mxmy, d_pt3d_d_mxmy0;
    d_pt3d_d_mxmy << d_X_d_mx, d_X_d_my, d_Y_d_mx, d_Y_d_my, d_Z_d_mx, d_Z_d_my;
    d_pt3d_d_mxmy0 = d_pt3d_d_mxmy;
    if (extra_param) {
      d_pt3d_d_mxmy = d_bearing_d_p3d2_un_ofsted_normalized_z *
                      d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
                      (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
                      (d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) *
                      d_p3d1_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z *
                      (d_uv_w_p_d_xy_wo_p1_inv)*d_p3d1_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z *
                      d_p3d0_un_ofsted_normalized_z_d_p3d0_ofsted_normalized_z *
                      d_p3d0_ofsted_normalized_z_d_p3d0_orig * d_pt3d_d_mxmy;
    }

    Mat2 d_xryr_duvDistorted = duvDistorted_dxryr.inverse();

    Mat24 d_utvt_d_fxfycxcy;
    d_utvt_d_fxfycxcy << -uvTilted(0) / fx, 0, -1 / fx, 0, 0, -uvTilted(1) / fy, 0, -1 / fy;

    Mat2 d_utvt_d_uv;
    d_utvt_d_uv << 1 / fx, 0, 0, 1 / fy;
    ///
    number_t stx = sin(t1);
    number_t ctx = cos(t1);
    number_t sty = sin(t2);
    number_t cty = cos(t2);
    number_t ttx = tan(t1);
    number_t tty = tan(t2);

    number_t tt1 = 1 / ctx;
    number_t tt4 = ttx * tty;
    number_t tt5 = 1 / cty;
    number_t tt7 = -tty;
    number_t tt8 = ttx / cty;
    number_t tt9 = 1 / ctx / cty;

    number_t u_t = uvTilted(0);
    number_t v_t = uvTilted(1);

    number_t d_ud_d_ut = tt1 / (tt7 * u_t + tt8 * v_t + tt9) -
                         tt1 * u_t * tt7 / ((tt7 * u_t + tt8 * v_t + tt9) * (tt7 * u_t + tt8 * v_t + tt9));

    number_t d_ud_d_vt = -tt1 * u_t * tt8 / ((tt7 * u_t + tt8 * v_t + tt9) * (tt7 * u_t + tt8 * v_t + tt9));

    number_t d_vd_d_ut =
        tt4 / (tt7 * u_t + tt8 * v_t + tt9) -
        (tt4 * u_t + tt5 * v_t) * tt7 / ((tt7 * u_t + tt8 * v_t + tt9) * (tt7 * u_t + tt8 * v_t + tt9));

    number_t d_vd_d_vt =
        tt5 / (tt7 * u_t + tt8 * v_t + tt9) -
        (tt4 * u_t + tt5 * v_t) * tt8 / ((tt7 * u_t + tt8 * v_t + tt9) * (tt7 * u_t + tt8 * v_t + tt9));

    Mat2 d_uvDistorted_d_uvTilted;
    d_uvDistorted_d_uvTilted << d_ud_d_ut, d_ud_d_vt, d_vd_d_ut, d_vd_d_vt;

    Mat2 d_udvd_d_tilt = d_uvDistorted_d_uvTilted;

    number_t temp = -tty * u_t + ttx / cty * v_t + 1 / ctx / cty;

    number_t d_ud_d_tx =
        (stx / (ctx * ctx) * u_t / temp) -
        (1 / ctx * u_t * (v_t / (ctx * ctx) / cty + cty * stx / ((ctx * cty) * (ctx * cty))) / (temp * temp));

    number_t d_ud_d_ty =
        -u_t / ctx * (-u_t / (cty * cty) + ttx * sty * v_t / (cty * cty) + ctx * sty / ((ctx * cty) * (ctx * cty))) /
        (temp * temp);

    number_t d_vd_d_tx = (1 / (ctx * ctx) * tty * u_t / temp) -
                         ((ttx * tty * u_t + 1 / cty * v_t) *
                          (1 / cty / (ctx * ctx) * v_t + cty * stx / ((ctx * cty) * (ctx * cty))) / (temp * temp));

    number_t d_vd_d_ty =
        ((ttx * u_t / (cty * cty) + sty * v_t / (cty * cty)) / temp) -
        ((ttx * tty * u_t + 1 / cty * v_t) *
         (-u_t / (cty * cty) + ttx * sty * v_t / (cty * cty) + ctx * sty / ((ctx * cty) * (ctx * cty))) /
         (temp * temp));

    Mat2 d_udvd_d_t1t2;
    d_udvd_d_t1t2 << d_ud_d_tx, d_ud_d_ty, d_vd_d_tx, d_vd_d_ty;

    Mat2 duvDistorted_d_p1p2;
    duvDistorted_d_p1p2 << (2.0 * x_r2 + r_d2), 2.0 * x_r * y_r, 2.0 * x_r * y_r, (2.0 * y_r2 + r_d2);
    duvDistorted_d_p1p2 *= q_coeff;
    duvDistorted_d_p1p2 *= -1;

    Mat26 duvDistorted_d_s1s2s3s4s5s6;
    duvDistorted_d_s1s2s3s4s5s6 << r_d2, r_d4, 0, 0, r_d6, 0, 0, 0, r_d2, r_d4, 0, r_d6;
    duvDistorted_d_s1s2s3s4s5s6 *= -1;

    duvDistorted_dq *= -1;

    Mat2 d_xryr_d_uv = d_xryr_duvDistorted * d_udvd_d_tilt * d_utvt_d_uv;
    Mat24 d_xryr_d_fxfycxcy = d_xryr_duvDistorted * d_udvd_d_tilt * d_utvt_d_fxfycxcy;
    Mat2 d_xryr_d_p1p2 = d_xryr_duvDistorted * duvDistorted_d_p1p2;
    Mat26 d_xryr_d_s1s2s3s4s5s6 = d_xryr_duvDistorted * duvDistorted_d_s1s2s3s4s5s6;
    Mat2 d_xryr_d_t1t2 = d_xryr_duvDistorted * d_udvd_d_t1t2;
    Mat23 d_xryr_d_q = d_xryr_duvDistorted * duvDistorted_dq;

    Mat36 d_pt3d_d_k1k2k3k4k5k6;
    d_pt3d_d_k1k2k3k4k5k6 << d_pt3d_d_k1, d_pt3d_d_k2, d_pt3d_d_k3, d_pt3d_d_k4, d_pt3d_d_k5, d_pt3d_d_k6;

    if (d_p3d2_d_param) {
      d_p3d2_d_param->resize(3, kParamLength);
      (*d_p3d2_d_param).setZero();
      (*d_p3d2_d_param).leftCols(4) = d_pt3d_d_mxmy * d_xryr_d_fxfycxcy;
      (*d_p3d2_d_param).block(0, 4, 3, k_nums_used) = d_pt3d_d_k1k2k3k4k5k6.leftCols(k_nums_used);
      (*d_p3d2_d_param).block<3, 2>(0, p0_start_idx) = d_pt3d_d_mxmy * d_xryr_d_p1p2;
      (*d_p3d2_d_param).block(0, s_start_idx, 3, s_size) = d_pt3d_d_mxmy * d_xryr_d_s1s2s3s4s5s6.leftCols(s_size);
      if (opt_s5s6t1t2) {
        (*d_p3d2_d_param).block<3, 2>(0, tilt_start_idx) = d_pt3d_d_mxmy * d_xryr_d_t1t2;
      }
      Vec3 d_p3d1_un_ofsted_normalized_z_d_alpha, d_p3d1_un_ofsted_normalized_z_d_alpha_true;
      Vec3 d_p3d1_un_ofsted_normalized_z_d_beta, d_p3d1_un_ofsted_normalized_z_d_beta_true;
      {
#ifdef USE_EXP_IN_KB20
        number_t d_alpha_d_alpha_true = alpha * alpha * exp(-parameters_[eucm_start_idx]);
        number_t d_beta_d_beta_true = beta;
#else
        number_t d_alpha_d_alpha_true = 1;
        number_t d_beta_d_beta_true = 1;
#endif
        number_t z_old2 = p_3d2_orig(2);
        number_t d = std::sqrt(beta * p_3d2_orig.head(2).squaredNorm() + z_old2 * z_old2);
        number_t d_inv = 1 / d;

        d_p3d1_un_ofsted_normalized_z_d_alpha << 0, 0, d - z_old2;
        d_p3d1_un_ofsted_normalized_z_d_beta << 0, 0, alpha * 0.5 * d_inv * p_3d2_orig.head(2).squaredNorm();

        d_p3d1_un_ofsted_normalized_z_d_alpha_true = d_p3d1_un_ofsted_normalized_z_d_alpha * d_alpha_d_alpha_true;
        d_p3d1_un_ofsted_normalized_z_d_beta_true = d_p3d1_un_ofsted_normalized_z_d_beta * d_beta_d_beta_true;
      }

#ifndef USE_TILT_IN_KB20
      if (extra_param && opt_eucm) {
        (*d_p3d2_d_param).block<3, 1>(0, eucm_start_idx) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (-d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) * d_p3d1_un_ofsted_normalized_z_d_alpha_true;
        (*d_p3d2_d_param).block<3, 1>(0, eucm_start_idx + 1) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (-d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) * d_p3d1_un_ofsted_normalized_z_d_beta_true;
      }
#ifndef USE_JR
      Mat3 d_p3d1_orig_d_rot = rotMat.transpose() * Skew(p_3d);
#else
      Mat3 d_p3d1_orig_d_rot = Skew(rotMat.transpose() * p_3d) * Jr(rot_vec);
#endif
      if (extra_param && opt_rot) {
        (*d_p3d2_d_param).block<3, 2>(0, rot_start_idx) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) *
            d_p3d1_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p1_inv)*d_p3d1_ofsted_normalized_z_d_p3d1_orig * d_p3d1_orig_d_rot.leftCols(2);
      }
#else
      if (extra_param && opt_eucm) {
        (*d_p3d2_d_param).block<3, 1>(0, eucm_start_idx) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (-d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) * d_p3d1_un_ofsted_normalized_z_d_alpha_true;
        (*d_p3d2_d_param).block<3, 1>(0, eucm_start_idx + 1) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (-d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) * d_p3d1_un_ofsted_normalized_z_d_beta_true;
      }
      if (extra_param && opt_rot) {
        (*d_p3d2_d_param).block<3, 2>(0, rot_start_idx) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) *
            d_p3d1_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p1_inv)*d_p3d1_ofsted_normalized_z_d_t1t2;
      }
#endif

      // (*d_p3d2_d_param).block<3, 2>(0, 20).setZero();
      // (*d_p3d2_d_param).block<3, 2>(0, 22).setZero();
      Mat32 d_p3d0_un_ofsted_normalized_z_d_ofst_xy0 = Mat32::Zero();
      d_p3d0_un_ofsted_normalized_z_d_ofst_xy0.topLeftCorner<2, 2>() = -Mat2::Identity();

      Mat32 d_p3d1_un_ofsted_normalized_z_d_ofst_xy1 = Mat32::Zero();
      d_p3d1_un_ofsted_normalized_z_d_ofst_xy1.topLeftCorner<2, 2>() = -Mat2::Identity();

      Mat32 d_p3d2_un_ofsted_normalized_z_d_ofst_xy2 = Mat32::Zero();
      d_p3d2_un_ofsted_normalized_z_d_ofst_xy2.topLeftCorner<2, 2>() = -Mat2::Identity();

      if (extra_param && opt_ofst_xy0) {
        (*d_p3d2_d_param).block<3, 2>(0, ofst0_start_idx) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) *
            d_p3d1_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p1_inv)*d_p3d1_ofsted_normalized_z_d_p3d0_un_ofsted_normalized_z *
            d_p3d0_un_ofsted_normalized_z_d_ofst_xy0;
      }
      if (extra_param && opt_ofst_xy1) {
        (*d_p3d2_d_param).block<3, 2>(0, ofst1_start_idx) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) * d_p3d1_un_ofsted_normalized_z_d_ofst_xy1;
      }
      if (extra_param && opt_ofst_xy2) {
        (*d_p3d2_d_param).block<3, 2>(0, ofst2_start_idx) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_ofst_xy2;
      }

#ifdef FIX_BETA_IN_KB20
      (*d_p3d2_d_param).block<3, 1>(0, eucm_start_idx + 1).setZero();
#endif
      if (extra_param && opt_extra_p) {
        (*d_p3d2_d_param).block<3, 3>(0, extra_p_start_idx) = d_pt3d_d_mxmy * d_xryr_d_q;
      }
      if (extra_param && opt_p1) {
        (*d_p3d2_d_param).block<3, 2>(0, p1_start_idx) =
            d_bearing_d_p3d2_un_ofsted_normalized_z * d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
            (d_uv_w_p_d_xy_wo_p2_inv)*d_p3d2_ofsted_normalized_z_d_p3d2_orig *
            (d_p3d1_un_ofsted_normalized_z_d_p3d2_orig.inverse()) *
            d_p3d1_un_ofsted_normalized_z_d_p3d1_ofsted_normalized_z * (-d_uv_w_p_d_xy_wo_p1_inv) * d_uv_w_p_d_p1;
      }
      // (*d_p3d2_d_param).block<3, 33>(0, 0).setZero();
      if (extra_param && opt_p2 /*&& false*/) {
        (*d_p3d2_d_param).block<3, 2>(0, p2_start_idx) = d_bearing_d_p3d2_un_ofsted_normalized_z *
                                                         d_p3d2_un_ofsted_normalized_z_d_p3d2_ofsted_normalized_z *
                                                         (-d_uv_w_p_d_xy_wo_p2_inv) * d_uv_w_p_d_p2;
      }
      if (fix_fc) {
        (*d_p3d2_d_param).leftCols(4).setZero();
      }
      //      if (!opt_s5s6t1t2) {
      //        (*d_p3d2_d_param).block<3, 4>(0, 16).setZero();
      //      }
      //      if (!extra_param) {
      //        (*d_p3d2_d_param).block<3, 17>(0, 20).setZero();
      //      }
      //      // std::cout << "(*d_p3d2_d_param).block<3, 4>(0, 20):\n" <<
      //      (*d_p3d2_d_param).block<3, 4>(0, 20) << std::endl; if (fix_k) {
      //        (*d_p3d2_d_param).block<3, 6>(0, 4).setZero();
      //      } else {
      //        if (k_nums_used != 6) {
      //          (*d_p3d2_d_param).block(0, 4 + k_nums_used, 3, 6 -
      //          k_nums_used).setZero();
      //        }
      //      }
    }
    // (*d_p3d2_d_param).setZero();
    if (d_p3d2_d_img) {
      d_p3d2_d_img->resize(3, 2);
      (*d_p3d2_d_img).setZero();
      (*d_p3d2_d_img) = d_pt3d_d_mxmy * d_xryr_d_uv;
    }
    //    std::cout << "d_bearing_uv:\n" << (*d_p3d2_d_img).transpose() <<
    //    std::endl;
    // std::cout << "d_bearing_d_param:\n" << (*d_p3d2_d_param).block<3,4>(0,
    // 33).transpose() << std::endl; std::exit(-1);
  }
  return true;
}
/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#pragma once
#include "camera_base.h"
#include <iostream>
namespace dso {
class KB20Camera : public CameraBase {
public:
  using Ptr = std::shared_ptr<KB20Camera>;
  bool opt_s5s6t1t2 = true;
  int k_start_idx = 4;
  int p0_start_idx;
  int s_start_idx;
  int s_size;
  int tilt_start_idx = 0;
  int eucm_start_idx = 0;
  int rot_start_idx = 0;
  int ofst0_start_idx = 0;
  int ofst1_start_idx = 0;
  int ofst2_start_idx = 0;
  int extra_p_start_idx = 0;
  int p1_start_idx = 0;
  int p2_start_idx = 0;
  bool opt_eucm = true;
  bool opt_rot = true;
  explicit KB20Camera(CamId camera_id, int width, int height,
                      bool opt_s5s6t1t2_ = true, bool extra_param_ = false)
      : CameraBase(camera_id, width, height) {
    camera_model_ = CameraModel::kKB20;
    opt_s5s6t1t2 = opt_s5s6t1t2_;
    extra_param = extra_param_;
    kParamLength = 37;
    opt_ofst_xy0 = false;
    opt_ofst_xy1 = false;
    opt_ofst_xy2 = false;
    opt_p1 = false;
    opt_p2 = false;
    opt_extra_p = false;
    fix_k = false;
    k_nums_used = 6;
    if (extra_param) {
      opt_ofst_xy0 = true;
      opt_ofst_xy1 = true;
      opt_ofst_xy2 = true;
      opt_p1 = true;
      opt_p2 = true;
      opt_extra_p = true;
    }
    SetParamSize();
  }

  // parameters: fx, fy, cx, cy, k1, k2, k3, k4
  KB20Camera(CamId camera_id, int width, int height, const number_t *parameters,
             bool opt_s5s6t1t2_ = true, bool extra_param_ = false)
      : CameraBase(camera_id, width, height, parameters, 37) {
    // TODO: add assert
    camera_model_ = CameraModel::kKB20;
    opt_s5s6t1t2 = opt_s5s6t1t2_;
    extra_param = extra_param_;
  }

  virtual void SetParamSize() override {
    p0_start_idx = k_start_idx + k_nums_used;
    s_start_idx = p0_start_idx + 2;
    kParamLength = 6 + k_nums_used;
    if (opt_s5s6t1t2) {
      s_size = 6;
    } else {
      s_size = 4;
    }
    tilt_start_idx = s_start_idx + s_size;
    kParamLength += 4;
    if (opt_s5s6t1t2) {
      eucm_start_idx = tilt_start_idx + 2;
      kParamLength += 4;
    } else {
      eucm_start_idx = tilt_start_idx;
    }

    if (extra_param) {
      if (opt_eucm) {
        rot_start_idx = eucm_start_idx + 2;
        kParamLength += 2;
      } else {
        rot_start_idx = eucm_start_idx;
      }
      if (opt_rot) {
        ofst0_start_idx = rot_start_idx + 2;
        kParamLength += 2;
      } else {
        ofst0_start_idx = rot_start_idx;
      }
      if (opt_ofst_xy0) {
        ofst1_start_idx = ofst0_start_idx + 2;
        kParamLength += 2;
      } else {
        ofst1_start_idx = ofst0_start_idx;
      }
      if (opt_ofst_xy1) {
        ofst2_start_idx = ofst1_start_idx + 2;
        kParamLength += 2;
      } else {
        ofst2_start_idx = ofst1_start_idx;
      }
      if (opt_ofst_xy2) {
        extra_p_start_idx = ofst2_start_idx + 2;
        kParamLength += 2;
      } else {
        extra_p_start_idx = ofst2_start_idx;
      }
      if (opt_extra_p) {
        p1_start_idx = extra_p_start_idx + 3;
        kParamLength += 3;
      } else {
        p1_start_idx = extra_p_start_idx;
      }
      if (opt_p1) {
        p2_start_idx = p1_start_idx + 2;
        kParamLength += 2;
      } else {
        p2_start_idx = p1_start_idx;
      }
      if (opt_p2) {
        kParamLength += 2;
      }
    }
    printf("param_size: %d, k_start_idx: %d, p0_start_idx: %d, s_start_idx: "
           "%d, tilt_start_idx: %d, eucm_start_idx: %d, "
           "x, rot_start_idx: %d, ofst0_start_idx: %d, ofst1_start_idx: %d, "
           "ofst2_start_idx: %d, extra_p_start_idx: %d, "
           "p1_start_idx: %d, p2_start_idx: %d\n\n",
           kParamLength, k_start_idx, p0_start_idx, s_start_idx, tilt_start_idx,
           eucm_start_idx, rot_start_idx, ofst0_start_idx, ofst1_start_idx,
           ofst2_start_idx, extra_p_start_idx, p1_start_idx, p2_start_idx);
  }

  virtual bool Project(const Vec3 &p_3d22, Vec2 &p_img,
                       Eigen::Matrix<number_t, 2, 3> *d_img_d_p3d = nullptr,
                       Eigen::Matrix<number_t, 2, Eigen::Dynamic>
                           *d_img_d_param = nullptr) const override;

  virtual bool Project(const Vec3 &p_3d22, Eigen::Ref<Vec2> &p_img,
                       Eigen::Matrix<number_t, 2, 3> *d_img_d_p3d = nullptr,
                       Eigen::Matrix<number_t, 2, Eigen::Dynamic>
                           *d_img_d_param = nullptr) const override;

  virtual bool UnProject(const Vec2 &p_img, Vec3 &p_3d2,
                         Eigen::Matrix<number_t, 3, 2> *d_p3d2_d_img = nullptr,
                         Eigen::Matrix<number_t, 3, Eigen::Dynamic>
                             *d_p3d2_d_param = nullptr) const override;

private:
  void compute_duvDistorted_dxryr(
      const Vec2 &xr_yr, Mat2 &duvDistorted_dxryr, Mat23 &duvDistorted_dq,
      const number_t &xr_yr_squaredNorm, const number_t &p1, const number_t &p2,
      const number_t &s1, const number_t &s2, const number_t &s3,
      const number_t &s4, const number_t &s5, const number_t &s6,
      const number_t &q1, const number_t &q2, const number_t &q3) const {
    number_t p_coeff_x = (2 * xr_yr(0) * xr_yr(0) + xr_yr_squaredNorm) * p1 +
                         2.0 * xr_yr(0) * xr_yr(1) * p2;
    number_t p_coeff_y = (2 * xr_yr(1) * xr_yr(1) + xr_yr_squaredNorm) * p2 +
                         2.0 * xr_yr(0) * xr_yr(1) * p1;

    number_t q_coeff =
        1 + q1 * xr_yr_squaredNorm +
        q2 * xr_yr_squaredNorm * xr_yr_squaredNorm +
        q3 * xr_yr_squaredNorm * xr_yr_squaredNorm * xr_yr_squaredNorm;

    number_t d_q_coeff_d_x_r =
        (2.0 * q1 + 4.0 * q2 * xr_yr_squaredNorm +
         6.0 * q3 * xr_yr_squaredNorm * xr_yr_squaredNorm) *
        xr_yr(0);
    number_t d_q_coeff_d_y_r =
        (2.0 * q1 + 4.0 * q2 * xr_yr_squaredNorm +
         6.0 * q3 * xr_yr_squaredNorm * xr_yr_squaredNorm) *
        xr_yr(1);

    duvDistorted_dxryr(0, 0) =
        (1 + (6.0 * xr_yr(0) * p1 + 2.0 * xr_yr(1) * p2) * (q_coeff) +
         p_coeff_x * (d_q_coeff_d_x_r)) +
        2.0 *
            (s1 + 2.0 * s2 * xr_yr_squaredNorm +
             3.0 * s5 * xr_yr_squaredNorm * xr_yr_squaredNorm) *
            xr_yr(0);
    duvDistorted_dxryr(0, 1) =
        ((2.0 * xr_yr(1) * p1 + 2.0 * xr_yr(0) * p2) * (q_coeff) +
         p_coeff_x * (d_q_coeff_d_y_r)) +
        2.0 *
            (s1 + 2.0 * s2 * xr_yr_squaredNorm +
             3.0 * s5 * xr_yr_squaredNorm * xr_yr_squaredNorm) *
            xr_yr(1);
    duvDistorted_dxryr(1, 0) =
        ((2.0 * xr_yr(1) * p1 + 2.0 * xr_yr(0) * p2) * (q_coeff) +
         p_coeff_y * (d_q_coeff_d_x_r)) +
        2.0 *
            (s3 + 2.0 * s4 * xr_yr_squaredNorm +
             3.0 * s6 * xr_yr_squaredNorm * xr_yr_squaredNorm) *
            xr_yr(0);
    duvDistorted_dxryr(1, 1) =
        ((1 + 6.0 * xr_yr(1) * p2 + 2. * xr_yr(0) * p1) * (q_coeff) +
         p_coeff_y * (d_q_coeff_d_y_r)) +
        2.0 *
            (s3 + 2.0 * s4 * xr_yr_squaredNorm +
             3.0 * s6 * xr_yr_squaredNorm * xr_yr_squaredNorm) *
            xr_yr(1);

    duvDistorted_dq.setZero();
    duvDistorted_dq.row(0) << xr_yr_squaredNorm,
        xr_yr_squaredNorm * xr_yr_squaredNorm,
        xr_yr_squaredNorm * xr_yr_squaredNorm * xr_yr_squaredNorm;
    duvDistorted_dq.row(1) = duvDistorted_dq.row(0);
    duvDistorted_dq.row(0) *= p_coeff_x;
    duvDistorted_dq.row(1) *= p_coeff_y;
    return;
  }
  void compute_d_uv_w_p_d_xy_wo_p(const Vec3 &xy_wo_p, Vec3 &uv_w_p,
                                  Mat3 &d_uv_w_p_d_xy_wo_p, Mat32 &d_uv_w_p_d_p,
                                  const number_t &p1,
                                  const number_t &p2) const {
    number_t r2 = xy_wo_p.head(2).squaredNorm();
    number_t x_r2 = xy_wo_p(0) * xy_wo_p(0);
    number_t y_r2 = xy_wo_p(1) * xy_wo_p(1);
    number_t x_y = xy_wo_p(0) * xy_wo_p(1);

    uv_w_p.setOnes();
    uv_w_p(0) = xy_wo_p(0) + (2 * x_r2 + r2) * p1 + 2.0 * x_y * p2;
    uv_w_p(1) = xy_wo_p(1) + (2 * y_r2 + r2) * p2 + 2.0 * x_y * p1;

    d_uv_w_p_d_xy_wo_p.setZero();
    d_uv_w_p_d_xy_wo_p(0, 0) =
        1.0 + 6.0 * xy_wo_p(0) * p1 + 2.0 * xy_wo_p(1) * p2;
    d_uv_w_p_d_xy_wo_p(0, 1) = 2.0 * xy_wo_p(1) * p1 + 2.0 * xy_wo_p(0) * p2;
    d_uv_w_p_d_xy_wo_p(1, 0) = 2.0 * xy_wo_p(1) * p1 + 2.0 * xy_wo_p(0) * p2;
    d_uv_w_p_d_xy_wo_p(1, 1) =
        1.0 + 6.0 * xy_wo_p(1) * p2 + 2. * xy_wo_p(0) * p1;

    d_uv_w_p_d_p.setZero();

    d_uv_w_p_d_p.row(0) << (2.0 * x_r2 + r2), 2.0 * x_y;
    d_uv_w_p_d_p.row(1) << 2.0 * x_y, (2.0 * y_r2 + r2);
  }
  bool compute_xy_wo_p_from_uv_w_p(const Vec3 &uv_w_p, Vec3 &xy_wo_p,
                                   Mat3 &d_uv_w_p_d_xy_wo_p,
                                   Mat32 &d_uv_w_p_d_p, const number_t &p1,
                                   const number_t &p2) const {
    xy_wo_p = uv_w_p;
    Vec3 uv_w_p_est, correction;
    correction.setZero();
    int max_iter = 200;
    int count = 0;
    for (int i = 0; i < max_iter; i++) {
      // p3d1_est = p3d2;
      compute_d_uv_w_p_d_xy_wo_p(xy_wo_p, uv_w_p_est, d_uv_w_p_d_xy_wo_p,
                                 d_uv_w_p_d_p, p1, p2);
      correction.head(2) = d_uv_w_p_d_xy_wo_p.topLeftCorner<2, 2>().inverse() *
                           (uv_w_p.head(2) - uv_w_p_est.head(2));
      xy_wo_p += correction;
      // printf("tangential coeff, count : %d\n", count);
      count = i;
      if (correction.head(2).norm() < 1e-25) {
        break;
      }
      if (i == max_iter - 1 && extra_param) {
        return false;
        bool bad_param = false;
        for (int i = 0; i < kParamLength; ++i) {
          if (std::abs(parameters_[i]) > 1000) {
            bad_param = true;
            break;
          }
        }
        if (!bad_param) {
          printf("WARNING, max iter in tangential distortion!!!, dx: %f\n",
                 correction.head(2).norm());
        }
      }
    }
    return true;
  }
  void computeTiltProjectionMatrix(const number_t &tauX, const number_t &tauY,
                                   Mat3 &matTilt, Mat3 &invMatTilt) const {
    matTilt.setIdentity();
    invMatTilt.setIdentity();

    number_t cTauX = cos(tauX);
    number_t sTauX = sin(tauX);
    number_t cTauY = cos(tauY);
    number_t sTauY = sin(tauY);
    number_t tTauX = tan(tauX);
    number_t tTauY = tan(tauY);
    Mat3 matRotX, matRotY, matProjZ;
    matRotX << number_t(1.0), number_t(0.0), number_t(0.0), number_t(0.0),
        cTauX, sTauX, number_t(0.0), -sTauX, cTauX;
    matRotY << cTauY, number_t(0.0), -sTauY, number_t(0.0), number_t(1.0),
        number_t(0.0), sTauY, number_t(0.0), cTauY;
    Mat3 matRotXY = matRotY * matRotX;
    matProjZ << matRotXY(2, 2), number_t(0.0), -matRotXY(0, 2), number_t(0.0),
        matRotXY(2, 2), -matRotXY(1, 2), number_t(0.0), number_t(0.0),
        number_t(1.0);

    matTilt = matProjZ * matRotXY;
    invMatTilt << number_t(1.0) / cTauX, number_t(0.0), number_t(0.0),
        tTauX * tTauY, number_t(1.0) / cTauY, number_t(0.0), -tTauY,
        tTauX / cTauY, number_t(1.0) / (cTauX * cTauY);
  }

  bool compute_xr_yr_from_uvDistorted(
      const Vec2 &uvDistorted, Vec2 &xr_yr, Mat2 &duvDistorted_dxryr,
      Mat23 &duvDistorted_dq, const number_t &p1, const number_t &p2,
      const number_t &s1, const number_t &s2, const number_t &s3,
      const number_t &s4, const number_t &s5, const number_t &s6,
      const number_t &q1, const number_t &q2, const number_t &q3) const {
    xr_yr = uvDistorted;
    Vec2 uvDistorted_est, uvTan, uvPrism, correction;
    // Mat2 duvDistorted_dxryr;
    number_t x_r, y_r, r_d, x_r2, y_r2, r_d2, x_r_y_r, r_d4, r_d6, q_coeff;
    int max_iter = 200;
    for (int i = 0; i < max_iter; i++) {
      uvDistorted_est = xr_yr;
      number_t xr_yr_squaredNorm = xr_yr.squaredNorm();
      x_r = xr_yr(0);
      y_r = xr_yr(1);
      r_d = xr_yr.norm();
      x_r2 = x_r * x_r;
      y_r2 = y_r * y_r;
      r_d2 = r_d * r_d;
      r_d4 = r_d2 * r_d2;
      r_d6 = r_d2 * r_d4;
      x_r_y_r = x_r * y_r;
      q_coeff = 1.0 + q1 * r_d2 + q2 * r_d4 + q3 * r_d6;
      uvTan(0) = (p1 * (2.0 * x_r2 + r_d2) + 2.0 * x_r_y_r * p2) * (q_coeff);
      uvTan(1) = (p2 * (2.0 * y_r2 + r_d2) + 2.0 * x_r_y_r * p1) * (q_coeff);
      uvPrism(0) = s1 * r_d2 + s2 * r_d4 + s5 * r_d6;
      uvPrism(1) = s3 * r_d2 + s4 * r_d4 + s6 * r_d6;
      uvDistorted_est += uvTan + uvPrism;
      compute_duvDistorted_dxryr(xr_yr, duvDistorted_dxryr, duvDistorted_dq,
                                 xr_yr_squaredNorm, p1, p2, s1, s2, s3, s4, s5,
                                 s6, q1, q2, q3);
      correction =
          duvDistorted_dxryr.inverse() * (uvDistorted - uvDistorted_est);
      xr_yr = xr_yr + correction;
      if (correction.norm() < 1e-15) {
        break;
      }
      if (i == max_iter - 1 && extra_param) {
        return false;
        bool bad_param = false;
        for (int i = 0; i < kParamLength; ++i) {
          if (std::abs(parameters_[i]) > 1000) {
            bad_param = true;
            break;
          }
        }
        if (!bad_param) {
          printf("WARNING, max iter in p1p2 s1s2s3s4s5s6 !!!, dx: %f\n",
                 correction.norm());
        }
      }
    }
    return true;
  }
  bool getThetaFromNorm_xr_yr(const number_t &th_radialDesired, number_t &th,
                              number_t &dthD_dth, const number_t &k1,
                              const number_t &k2, const number_t &k3,
                              const number_t &k4, const number_t &k5,
                              const number_t &k6) const {
    std::vector<number_t> k;
    k.push_back(k1);
    k.push_back(k2);
    k.push_back(k3);
    k.push_back(k4);
    k.push_back(k5);
    k.push_back(k6);
    int startK = 0;
    int max_iter = 200;
    th = th_radialDesired;

    number_t thetaSq, th_radial, theta2is, step;

    for (int i = 0; i < max_iter; i++) {
      thetaSq = th * th;
      th_radial = 1;
      dthD_dth = 1;
      theta2is = thetaSq;
      for (int j = 0; j < k.size(); j++) {
        th_radial = th_radial + theta2is * k[startK + j];
        dthD_dth =
            dthD_dth + (2.0 * number_t(j) + 3.0) * k[startK + j] * theta2is;
        theta2is = theta2is * thetaSq;
      }
      th_radial = th_radial * th;
      if (std::fabs(dthD_dth) > 1e-20) {
        step = (th_radialDesired - th_radial) / dthD_dth;
      } else {
        if ((th_radialDesired - th_radial) * dthD_dth > 0.0) {
          step = 1e-19;
        } else {
          step = -1e-19;
        }
      }
      th += step;
      if (std::fabs(step) < 1e-15) {
        break;
      }
      if (std::fabs(th) >= 3.1415926 / 2.0) {
        th = (0.999) * 3.1415926 / 2.0;
      }
      if (i == max_iter - 1 && extra_param) {
        return false;
        bool bad_param = false;
        for (int i = 0; i < kParamLength; ++i) {
          if (std::abs(parameters_[i]) > 1000) {
            bad_param = true;
            break;
          }
        }
        if (!bad_param) {
          printf("WARNING, max iter in solve theta!!!, dx: %f\n", step);
        }
      }
    }
    return true;
  }
  bool compute_p3d2_from_p3d1(const Vec3 &p3d1, Vec3 &p3d2, Mat3 &d_p3d1_d_p3d2,
                              const number_t &alpha, const number_t &beta,
                              const Vec2 &p_img) const {
    p3d2 = p3d1;
    Vec3 p3d1_est, correction;
    number_t z_old, d, d_inv, z_new;
    int max_iter = 200;
    int count = 0;
    std::vector<Vec3> v_p3d2;
    for (int i = 0; i < max_iter; i++) {
      // p3d1_est = p3d2;
      z_old = p3d2(2);
      // v_p3d2.emplace_back(p3d2);
      d = std::sqrt(beta * p3d2.head(2).squaredNorm() + z_old * z_old);
      d_inv = 1 / d;
      z_new = alpha * d + (1 - alpha) * z_old;
      p3d1_est = Vec3(p3d2(0), p3d2(1), z_new);
      d_p3d1_d_p3d2.setIdentity();
      d_p3d1_d_p3d2(2, 0) = alpha * d_inv * beta * p3d2(0);
      d_p3d1_d_p3d2(2, 1) = alpha * d_inv * beta * p3d2(1);
      d_p3d1_d_p3d2(2, 2) = alpha * d_inv * beta * z_old + 1 - alpha;
      if (p3d1.cwiseAbs().maxCoeff() > 100) {
        // return false;
      }
      correction = d_p3d1_d_p3d2.inverse() * (p3d1 - p3d1_est);
      p3d2 += correction;
      if (correction.hasNaN()) {
        //        bool bad_param = false;
        //        for (int i = 0; i < kParamLength; ++i) {
        //          if (std::abs(parameters_[i]) > 1000) {
        //            bad_param = true;
        //            break;
        //          }
        //        }
        //        if (!bad_param) {
        //          std::cout << "d_p3d1_d_p3d2: \n" << d_p3d1_d_p3d2 <<
        //          std::endl; std::cout << "d_p3d1_d_p3d2_last: \n" <<
        //          d_p3d1_d_p3d2_last << std::endl; std::cout << "iter: " << i
        //          << ", p3d1: " << p3d1.transpose() << ", p3d2: " <<
        //          p3d2.transpose()
        //                    << ", p3d2_last: " << p3d2_last.transpose() << ",
        //                    correction_last: " << correction_last.transpose()
        //                    << ", correction: " << correction.transpose()
        //                    << ", d_inv: " << d_inv << ", alpha: " << alpha <<
        //                    ", beta: " << beta << std::endl;
        //          PrintIntri();
        //        }
      }
      count = i;
      if (correction.norm() < 1e-15) {
        break;
      }
      if (correction.hasNaN()) {
        // return false;
      }
      if (i == max_iter - 1 && extra_param) {
        return false;
        bool bad_param = false;
        for (int i = 0; i < kParamLength; ++i) {
          if (std::abs(parameters_[i]) > 1000) {
            bad_param = true;
            break;
          }
        }
        if (!bad_param) {
          printf("WARNING, max iter in alpha beta!!!, dx: [%f %f %f]\n",
                 correction(0), correction(1), correction(2));
          if (correction.hasNaN()) {
            std::cout << "d_p3d1_d_p3d2: \n" << d_p3d1_d_p3d2 << std::endl;
            std::cout << "p3d1: " << p3d1.transpose() << ", alpha: " << alpha
                      << ", beta: " << beta << std::endl;
            PrintIntri();
            std::cout << "p_img: " << p_img.transpose() << std::endl;
            for (const Vec3 pt : v_p3d2) {
              std::cout << "pt3d2: " << pt.transpose() << std::endl;
            }
          }
        }
        //        if (correction.hasNaN()) {
        //          // std::exit(-1);
        //        }
      }
    }
    // printf("count : %d\n", count);
    return true;
  }
  Mat3 ComputeBearingJac(const Vec3 &transedXYZ_c) const {
    number_t ptNorm = transedXYZ_c.norm();
    Vec3 ray_normalize = transedXYZ_c.normalized();

    // 归一化向量对原向量的导数
    Mat3 d_errornorm_d_xyz = Mat3::Identity() / ptNorm -
                             ray_normalize * ray_normalize.transpose() / ptNorm;

    return d_errornorm_d_xyz;
  }
  number_t SolveTheta(const number_t &r_theta, number_t &d_func_d_theta,
                      const number_t &k1, const number_t &k2,
                      const number_t &k3, const number_t &k4) const {
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
  Mat3 ComputeNormalizedZJac(const Vec3 &xyz) const {
    Mat3 Jac = Mat3::Zero();
    const number_t x = xyz(0);
    const number_t y = xyz(1);
    const number_t z = xyz(2);
    Jac(0, 0) = 1 / z;
    Jac(1, 1) = 1 / z;
    Jac(0, 2) = -x / z / z;
    Jac(1, 2) = -y / z / z;
    return Jac;
  }
  void ComputeForwardTiltJac(const number_t &t1, const number_t &t2,
                             const Vec3 &uvDistorted,
                             Mat3 &d_uvTilted_d_uvDistorted_full,
                             Mat32 &d_uvTilted_d_t1t2_full) const {
    /// part1
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

    number_t d_ut_d_ud =
        tt1 / (tt7 * u_d + tt8 * v_d + tt9) -
        tt1 * u_d * tt7 /
            ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_ut_d_vd =
        -tt1 * u_d * tt8 /
        ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_vt_d_ud =
        tt4 / (tt7 * u_d + tt8 * v_d + tt9) -
        (tt4 * u_d + tt5 * v_d) * tt7 /
            ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));
    number_t d_vt_d_vd =
        tt5 / (tt7 * u_d + tt8 * v_d + tt9) -
        (tt4 * u_d + tt5 * v_d) * tt8 /
            ((tt7 * u_d + tt8 * v_d + tt9) * ((tt7 * u_d + tt8 * v_d + tt9)));

    Mat2 d_uvTilted_d_uvDistorted;
    d_uvTilted_d_uvDistorted << d_ut_d_ud, d_ut_d_vd, d_vt_d_ud, d_vt_d_vd;

    d_uvTilted_d_uvDistorted_full.setZero();
    d_uvTilted_d_uvDistorted_full.topLeftCorner<2, 2>() =
        d_uvTilted_d_uvDistorted;

    /// part2
    number_t temp = sty * u_d - stx * cty * v_d + ctx * cty;

    number_t d_ut_d_tx =
        -stx * u_d / temp -
        ctx * u_d * (-ctx * cty * v_d - stx * cty) / (temp * temp);

    number_t d_ut_d_ty =
        -ctx * u_d * (cty * u_d + stx * sty * v_d - ctx * sty) / (temp * temp);

    number_t d_vt_d_tx = (-ctx * sty * u_d + cty * v_d) / temp -
                         (cty * v_d - stx * sty * u_d) *
                             (-ctx * cty * v_d - stx * cty) / (temp * temp);

    number_t d_vt_d_ty = (-stx * cty * u_d - sty * v_d) / temp -
                         (cty * v_d - stx * sty * u_d) *
                             (cty * u_d + stx * sty * v_d - ctx * sty) /
                             (temp * temp);

    Mat2 d_uvTilted_d_t1t2;
    d_uvTilted_d_t1t2 << d_ut_d_tx, d_ut_d_ty, d_vt_d_tx, d_vt_d_ty;

    d_uvTilted_d_t1t2_full.setZero();
    d_uvTilted_d_t1t2_full.topLeftCorner<2, 2>() = d_uvTilted_d_t1t2;
  }
  void ComputeBackwardTiltJac(const number_t &t1, const number_t &t2,
                              const Vec3 &uvTilted,
                              Mat3 &d_uvDistorted_d_uvTilted_full,
                              Mat32 &d_uvDistorted_d_t1t2_full) const {
    /// part1
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

    number_t d_ud_d_ut =
        tt1 / (tt7 * u_t + tt8 * v_t + tt9) -
        tt1 * u_t * tt7 /
            ((tt7 * u_t + tt8 * v_t + tt9) * (tt7 * u_t + tt8 * v_t + tt9));

    number_t d_ud_d_vt =
        -tt1 * u_t * tt8 /
        ((tt7 * u_t + tt8 * v_t + tt9) * (tt7 * u_t + tt8 * v_t + tt9));

    number_t d_vd_d_ut =
        tt4 / (tt7 * u_t + tt8 * v_t + tt9) -
        (tt4 * u_t + tt5 * v_t) * tt7 /
            ((tt7 * u_t + tt8 * v_t + tt9) * (tt7 * u_t + tt8 * v_t + tt9));

    number_t d_vd_d_vt =
        tt5 / (tt7 * u_t + tt8 * v_t + tt9) -
        (tt4 * u_t + tt5 * v_t) * tt8 /
            ((tt7 * u_t + tt8 * v_t + tt9) * (tt7 * u_t + tt8 * v_t + tt9));

    Mat2 d_uvDistorted_d_uvTilted;
    d_uvDistorted_d_uvTilted << d_ud_d_ut, d_ud_d_vt, d_vd_d_ut, d_vd_d_vt;

    d_uvDistorted_d_uvTilted_full.setZero();
    d_uvDistorted_d_uvTilted_full.topLeftCorner<2, 2>() =
        d_uvDistorted_d_uvTilted;

    /// part2
    number_t temp = -tty * u_t + ttx / cty * v_t + 1 / ctx / cty;

    number_t d_ud_d_tx =
        (stx / (ctx * ctx) * u_t / temp) -
        (1 / ctx * u_t *
         (v_t / (ctx * ctx) / cty + cty * stx / ((ctx * cty) * (ctx * cty))) /
         (temp * temp));

    number_t d_ud_d_ty = -u_t / ctx *
                         (-u_t / (cty * cty) + ttx * sty * v_t / (cty * cty) +
                          ctx * sty / ((ctx * cty) * (ctx * cty))) /
                         (temp * temp);

    number_t d_vd_d_tx = (1 / (ctx * ctx) * tty * u_t / temp) -
                         ((ttx * tty * u_t + 1 / cty * v_t) *
                          (1 / cty / (ctx * ctx) * v_t +
                           cty * stx / ((ctx * cty) * (ctx * cty))) /
                          (temp * temp));

    number_t d_vd_d_ty =
        ((ttx * u_t / (cty * cty) + sty * v_t / (cty * cty)) / temp) -
        ((ttx * tty * u_t + 1 / cty * v_t) *
         (-u_t / (cty * cty) + ttx * sty * v_t / (cty * cty) +
          ctx * sty / ((ctx * cty) * (ctx * cty))) /
         (temp * temp));

    Mat2 d_udvd_d_t1t2;
    d_udvd_d_t1t2 << d_ud_d_tx, d_ud_d_ty, d_vd_d_tx, d_vd_d_ty;

    d_uvDistorted_d_t1t2_full.setZero();
    d_uvDistorted_d_t1t2_full.topLeftCorner<2, 2>() = d_udvd_d_t1t2;
  }
};

} // namespace dso
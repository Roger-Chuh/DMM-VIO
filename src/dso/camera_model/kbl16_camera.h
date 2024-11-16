//
// Created by qy on 24-6-18.
//

#pragma once

#include "camera_base.h"

namespace dso {

class KBL16Camera : public CameraBase {
public:
  using Ptr = std::shared_ptr<KBL16Camera>;

  explicit KBL16Camera(CamId camera_id, int width, int height)
      : CameraBase(camera_id, width, height) {
    camera_model_ = CameraModel::kKBL16;
    kParamLength = 16;
  }

  // parameters: fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4
  KBL16Camera(CamId camera_id, int width, int height,
              const number_t *parameters)
      : CameraBase(camera_id, width, height, parameters, 16) {
    camera_model_ = CameraModel::kKBL16;
  }

  virtual bool
  Project(const Vec3 &p_3d, Vec2 &p_img,
          LinearAlgebraLib::Matrix<number_t, 2, 3> *d_img_d_p3d = nullptr,
          LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>
              *d_img_d_param = nullptr) const override;

  virtual bool
  Project(const Vec3 &p_3d, LinearAlgebraLib::Ref<Vec2> &p_img,
          LinearAlgebraLib::Matrix<number_t, 2, 3> *d_img_d_p3d = nullptr,
          LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>
              *d_img_d_param = nullptr) const override;

  virtual bool
  UnProject(const Vec2 &p_img, Vec3 &p_3d,
            LinearAlgebraLib::Matrix<number_t, 3, 2> *d_p3d_d_img = nullptr,
            LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>
                *d_p3d_d_param = nullptr) const override;
  virtual void SetParamSize() override {}

private:
  void Distortion(
      const Vec2 &xy, Vec2 &uv, Mat2 *d_uv_xy = nullptr,
      Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_uv_params = nullptr) const;

  void UnDistortion(
      const Vec2 &uv, Vec2 &xy, Mat2 *d_xy_uv = nullptr,
      Eigen::Matrix<number_t, 2, Eigen::Dynamic> *d_xy_params = nullptr) const;

  void ComputeDuvDxryr(const Vec2 &xr_yr, const number_t &xr_yr_squaredNorm,
                       Mat2 &d_uv_xryr) const;

public:
  static constexpr int k_start = 4;
  static constexpr int p_start = 10;
  static constexpr int s_start = 12;
};

} // namespace dso

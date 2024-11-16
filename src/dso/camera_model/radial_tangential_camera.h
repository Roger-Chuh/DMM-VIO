/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#pragma once
#include "camera_base.h"
namespace dso {
// TODO: RadTanCamera的投影和反投影测试
class RadtanCamera : public CameraBase {
public:
  using Ptr = std::shared_ptr<RadtanCamera>;

  explicit RadtanCamera(CamId camera_id, int width, int height)
      : CameraBase(camera_id, width, height) {
    camera_model_ = CameraModel::kRadialTangential;
    kParamLength = 8;
  }

  // parameters: fx, fy, cx, cy, k1, k2, p1, p2
  RadtanCamera(CamId camera_id, int width, int height,
               const number_t *parameters)
      : CameraBase(camera_id, width, height, parameters, 8) {
    // TODO: add assert
    camera_model_ = CameraModel::kRadialTangential;
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
              *d_img_d_param = nullptr) const override {
    return true;
  };

  virtual bool
  UnProject(const Vec2 &p_img, Vec3 &p_3d,
            LinearAlgebraLib::Matrix<number_t, 3, 2> *d_p3d_d_img = nullptr,
            LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>
                *d_p3d_d_param = nullptr) const override;
  virtual void SetParamSize() override {}
};
} // namespace dso
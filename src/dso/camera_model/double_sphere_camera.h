/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#pragma once
#include "camera_base.h"
namespace dso {
class DoubleSphereCamera : public CameraBase {
public:
  using Ptr = std::shared_ptr<DoubleSphereCamera>;

  explicit DoubleSphereCamera(CamId camera_id, int width, int height)
      : CameraBase(camera_id, width, height) {
    camera_model_ = CameraModel::kDoubleSphere;
    kParamLength = 6;
  }

  // parameters: fx, fy, cx, cy, xi, alpha
  DoubleSphereCamera(uint8_t camera_id, int width, int height,
                     const number_t *parameters)
      : CameraBase(camera_id, width, height, parameters, 6) {
    camera_model_ = CameraModel::kDoubleSphere;
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
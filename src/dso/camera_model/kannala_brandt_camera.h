/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#pragma once
#include "camera_base.h"
namespace dso {
class KB8Camera : public CameraBase {
 public:
  using Ptr = std::shared_ptr<KB8Camera>;

  explicit KB8Camera(CamId camera_id, int width, int height) : CameraBase(camera_id, width, height) {
    camera_model_ = CameraModel::kKB8;
    kParamLength = 8;
  }

  // parameters: fx, fy, cx, cy, k1, k2, k3, k4
  KB8Camera(CamId camera_id, int width, int height, const number_t* parameters)
      : CameraBase(camera_id, width, height, parameters, 8) {
    // TODO: add assert
    camera_model_ = CameraModel::kKB8;
  }

  virtual bool Project(
      const Vec3& p_3d, Vec2& p_img, LinearAlgebraLib::Matrix<number_t, 2, 3>* d_img_d_p3d = nullptr,
      LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>* d_img_d_param = nullptr) const override;

  virtual bool Project(
      const Vec3& p_3d, LinearAlgebraLib::Ref<Vec2>& p_img,
      LinearAlgebraLib::Matrix<number_t, 2, 3>* d_img_d_p3d = nullptr,
      LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>* d_img_d_param = nullptr) const override;

  virtual bool UnProject(
      const Vec2& p_img, Vec3& p_3d, LinearAlgebraLib::Matrix<number_t, 3, 2>* d_p3d_d_img = nullptr,
      LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>* d_p3d_d_param = nullptr) const override;
  virtual void SetParamSize() override {}

 private:
  //反投影时，使用迭代的方法，根据r_theta计算theta
  number_t SolveTheta(const number_t& r_theta, number_t& d_func_d_theta) const;
};

}  // namespace dso
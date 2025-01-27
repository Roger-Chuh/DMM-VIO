#pragma once
#include "../camera_model/calib_def.h"
#include "../camera_model/camera_base.h"
#include "../camera_model/vio_def.h"
//#include "opt_nodes_def.h"
#include "../util/settings.h"
#include <array>

namespace dso {
template <class Scalar> struct Pattern {
  // first pattern must be [0,0], for main dir
  //  static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-1, -1}, {1, -1},
  //  {-1, 1}, {0, -2}, {0, 2}, {-2, 0}, {2, 0}}; static constexpr Scalar
  //  pattern_raw[][2] = {{0, 0},   {-2, -2}, {2, -2}, {-2, 2}, {0, -4}, {0, 4},
  //  {-4, 0}, {4, 0},
  //                                              {-4, -4}, {4, -4},  {-4, 4},
  //                                              {4, 4},  {0, -6}, {0, 6}, {-6,
  //                                              0}, {6, 0}};
  //  static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-1, -1}, {-1, 1}, {1,
  //  -1}, {1, 1},  {-2, 0}, {2, 0},  {0, -2},
  //                                              {0, 2}, {-2, -2}, {2, -2},
  //                                              {-2, 2}, {0, -4}, {0, 4}, {-4,
  //                                              0}, {4, 0}};

  // static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-2, -2}, {2, -2}, {-2,
  // 2}, {0, -4}, {0, 4}, {-4, 0}, {4, 0}};

  //  static constexpr Scalar pattern_raw[][2] = {{0, 0},  {-2, -2}, {2, -2},
  //  {-2, 2}, {-2, -4}, {-4, -2}, {2, -4}, {4, -2},
  //                                              {-4, 2}, {-2, 4},  {2, 4}, {4,
  //                                              2},  {0, -4},  {0, 4},   {-4,
  //                                              0}, {4, 0}};

  //  static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-2, -2}, {-4, -4},
  //  {-2, 2}, {-4, 4}, {2, 0}, {-4, 0}, {6, 0}}; static constexpr Scalar
  //  pattern_raw[][2] = {{0, 0}, {-2, -2}, {0, -4}, {2, -2}, {-2, 2}, {0, 4},
  //  {-4, 0}, {4, 0}}; static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-2,
  //  -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {-2, -1},
  //                                              {-1, -1}, {0, -1},  {1, -1},
  //                                              {2, -1}, {-2, 0}, {-1, 0}, {1,
  //                                              0}, {2, 0},   {-2, 1},  {-1,
  //                                              1},  {0, 1},  {1, 1},  {2, 1},
  //                                              {-2, 2},
  //                                              {-1, 2},  {0, 2},   {1, 2},
  //                                              {2, 2}};
  //  static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-2, -2}, {0, -2}, {2,
  //  -2}, {-1, -1}, {1, -1}, {-2, 0},
  //                                              {2, 0}, {-1, 1},  {1, 1}, {-2,
  //                                              2}, {0, 2},   {2, 2}};
  //  static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-1, -1}, {-2, -2},
  //  {1, -1}, {2, -2}, {-1, 1}, {-2, 2}, {2, 2}}; static constexpr Scalar
  //  pattern_raw[][2] = {{0, 0},  {-1, -1}, {-2, -2}, {1, -1}, {2, -2}, {-1,
  //  1}, {-2, 2}, {2, 2},
  //                                              {0, -2}, {0, -4},  {0, 2}, {0,
  //                                              4},  {2, 0},  {4, 0},  {-2,
  //                                              0}, {-4, 0}};
  //  static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-1, -1}, {1, -1},
  //  {-1, 1}, {1, 1},  {-2, 0}, {0, -2}, {2, 0},
  //                                              {0, 2}, {-3, -3}, {-5, -5},
  //                                              {-5, 0}, {-3, 3}, {-5, 5}, {4,
  //                                              0},  {6, 0}};
  //  static constexpr Scalar pattern_raw[][2] = {{0, 0}, {-2, -2}, {2, -2},
  //  {-2, 2}, {2, 2},  {-4, 0}, {0, -4}, {4, 0},
  //                                              {0, 4}, {-4, -4}, {-6, -6},
  //                                              {-4, 4}, {-6, 6}, {6, 0}, {-6,
  //                                              0}, {8, 0}};

  //  static constexpr Scalar pattern_raw[][2] = {{0, 0},  {-8, -8}, {-4, -4},
  //  {-8, 8}, {-4, 4}, {4, 0}, {8, 0},  {12, 0},
  //                                              {-4, 0}, {-8, -4}, {-8, 0},
  //                                              {-8, 4}, {0, -4}, {0, 4}, {4,
  //                                              -4}, {4, 4}};
  //  static constexpr Scalar pattern_raw[][2] = {{0, 0},  {-8, -8}, {-4, -4},
  //  {-8, 8}, {-4, 4}, {-4, 0},
  //                                              {-8, 0}, {4, -4},  {4, 4}, {8,
  //                                              -8}, {8, 0},  {8, 8}};

  //  static constexpr Scalar pattern_raw[][2] = {{0, 0},  {-2, 0}, {-4, 0},
  //  {-2, -2}, {-4, -4}, {-4, -2},
  //                                              {-2, 2}, {-4, 4}, {-4, 2}, {2,
  //                                              0},   {4, 0},   {6, 0}};

  //  static constexpr Scalar pattern_raw[][2] = {
  //      {0, 0},   {-6, -6}, {-4, -6}, {-2, -6}, {0, -6},  {2, -6},  {4, -6},
  //      {6, -6},  {-6, -4}, {-4, -4},
  //      {-2, -4}, {0, -4},  {2, -4},  {4, -4},  {6, -4},  {-6, -2}, {-4, -2},
  //      {-2, -2}, {0, -2},  {2, -2}, {4, -2},  {6, -2},  {-6, -0}, {-4, -0},
  //      {-2, -0}, {2, -0},  {4, -0},  {6, -0},  {-6, 2},  {-4, 2},
  //      {-2, 2},  {0, 2},   {2, 2},   {4, 2},   {6, 2},   {-6, 4},  {-4, 4},
  //      {-2, 4},  {0, 4},   {2, 4}, {4, 4},   {6, 4},   {-6, 6},  {-4, 6},
  //      {-2, 6},  {0, 6},   {2, 6},   {4, 6},   {6, 6}};

  static constexpr Scalar pattern_raw_angle[][2] = {{0, 0},  {-1, -1}, {0, -1},
                                                    {1, -1}, {-1, 0},  {1, 0},
                                                    {-1, 1}, {0, 1},   {1, 1}};
  //  static constexpr Scalar pattern_raw_big[][2] = {{0, 0}, {-1, -1}, {1, -1},
  //  {-1, 1}, {-2, 0}, {0, -2}, {2, 0},
  //                                                  {0, 2}, {1, 1},   {4, 4},
  //                                                  {-4, 4}, {4, -4}, {-4,
  //                                                  -4}};

  //  static constexpr Scalar pattern_raw_big[][2] = {
  //      {0, 0},   {-6, -6}, {-4, -6}, {-2, -6}, {0, -6},  {2, -6},  {4, -6},
  //      {6, -6},  {-6, -4}, {-4, -4},
  //      {-2, -4}, {0, -4},  {2, -4},  {4, -4},  {6, -4},  {-6, -2}, {-4, -2},
  //      {-2, -2}, {0, -2},  {2, -2}, {4, -2},  {6, -2},  {-6, -0}, {-4, -0},
  //      {-2, -0}, {2, -0},  {4, -0},  {6, -0},  {-6, 2},  {-4, 2},
  //      {-2, 2},  {0, 2},   {2, 2},   {4, 2},   {6, 2},   {-6, 4},  {-4, 4},
  //      {-2, 4},  {0, 4},   {2, 4}, {4, 4},   {6, 4},   {-6, 6},  {-4, 6},
  //      {-2, 6},  {0, 6},   {2, 6},   {4, 6},   {6, 6}};

  static constexpr int PATTERN_BORDER = 2;

  // static constexpr int PATTERN_SIZE = sizeof(pattern_raw) / (2 *
  // sizeof(Scalar));
  static constexpr int PATTERN_SIZE_ANGLE =
      sizeof(pattern_raw_angle) / (2 * sizeof(Scalar));

  typedef LinearAlgebraLib::Matrix<Scalar, 2, PATTERN_SIZE_def> Matrix2P;
  // typedef LinearAlgebraLib::Matrix<Scalar, 2, PATTERN_SIZE_ANGLE> Matrix2P_B;
  // static const Matrix2P pattern2;
  // static const Matrix2P_B pattern2_B;
};

struct Patch {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr int PATCH_SIZE = PATTERN_SIZE_def;
  //  static constexpr int PATCH_SIZE = Pattern<number_t>::PATTERN_SIZE;
  static constexpr int VAL_SIZE = PATCH_SIZE;
  static constexpr int PATCH_SIZE_B = Pattern<number_t>::PATTERN_SIZE_ANGLE;
  static constexpr int VAL_SIZE_B = PATCH_SIZE_B;
  static constexpr int PATCH_BORDER = Pattern<number_t>::PATTERN_BORDER;
  static constexpr int HALF_PATCH_SIZE_B = 5;
  // static const Pattern<number_t>::Matrix2P pattern2;
  // static const Pattern<number_t>::Matrix2P_B pattern2_B;

  typedef Eigen::Matrix<number_t, 2, PATCH_SIZE> Matrix2P;
  typedef Eigen::Matrix<number_t, PATCH_SIZE, 2> MatrixP2;
  typedef Eigen::Matrix<number_t, 3, PATCH_SIZE> Matrix3P;
  typedef Eigen::Matrix<number_t, PATCH_SIZE, 3> MatrixP3;
  typedef Eigen::Matrix<number_t, PATCH_SIZE, 6> MatrixP6;
  typedef Eigen::Matrix<number_t, PATCH_SIZE, 1> VectorP;
  typedef Eigen::Matrix<number_t, PATCH_SIZE, PATCH_SIZE> MatrixP;
  typedef Eigen::Array<number_t, PATCH_SIZE, 1> ArrayP;
  typedef Eigen::Array<number_t, VAL_SIZE, 1> ArrayV;
  typedef Eigen::Matrix<number_t, 2, VAL_SIZE> Matrix2V;
  typedef Eigen::Matrix<number_t, VAL_SIZE, 2> MatrixV2;
  typedef Eigen::Matrix<number_t, VAL_SIZE, 3> MatrixV3;
  typedef Eigen::Matrix<number_t, VAL_SIZE, 6> MatrixV6;
  typedef Eigen::Matrix<number_t, VAL_SIZE, 1> VectorV;
  typedef Eigen::Matrix<number_t, VAL_SIZE, PATCH_SIZE> MatrixVP;
  typedef Eigen::Matrix<number_t, VAL_SIZE, VAL_SIZE> MatrixV;

  typedef Eigen::Array<number_t, PATCH_SIZE_B, 1> ArrayP_B;
  typedef Eigen::Array<number_t, PATCH_SIZE_B, 1> ArrayV_B;
  typedef Eigen::Matrix<number_t, 2, PATCH_SIZE_B> Matrix2P_B;

  bool SetFromImg(std::shared_ptr<AlgsImage> img, const int &level,
                  const Vec2 &px, bool &is_corner, CameraBase *p_simple_camera);

  //  bool SetFromImgBigPatch(std::shared_ptr<AlgsImage> img, const Vec2 &px,
  //  std::shared_ptr<CameraBase> p_simple_camera);

  //  bool SetValOnlyFromImg(std::shared_ptr<AlgsImage> img, const Vec2 &px,
  //  Vec2 *grad_pixel_0 = nullptr);

  void SetHPose(const Mat36 &dp_dx0);

  inline void TransformScaled(const Mat4 &T10, const number_t &idp,
                              Matrix3P &res) const;

  void ProjectPatchs(std::shared_ptr<CameraBase> simple_camera,
                     const Patch::Matrix3P &target_dir, bool &has_outlier,
                     const int &x_border_min, const int &x_border_max,
                     const int &y_border_min, const int &y_border_max,
                     Matrix2P &res) const;

  void GetPatchValues(const Patch::Matrix2P &uvs,
                      std::shared_ptr<AlgsImage> img, ArrayP &res,
                      Patch::Matrix2P *p_patch_grad = nullptr,
                      Vec2 *p_center_pixel_grad = nullptr) const;

  //  void GetBigPatchValues(const Patch::Matrix2P_B &uvs,
  //  std::shared_ptr<AlgsImage> img, ArrayP_B &res,
  //                         Patch::Matrix2P_B *p_patch_grad = nullptr) const;

  //  void GetWarpAffineMatrix(std::shared_ptr<CameraBase> target_camera, const
  //  Mat4 &T10, const number_t &rho,
  //                           Mat2 &A10) const;

  static void PatchInit(const number_t &z_thre);
  //  static number_t GetPatchSize(const Vec3 &point_A, const Vec3 &point_B,
  //  const Vec3 &point_C);

  //  ArrayV vals;
  ArrayV normalized_vals;
  //  ArrayV wgs;
  //  Matrix2P grads;
  number_t grads0_square_norm;
  //  Matrix2P coordinate;
  //  Matrix3P patch_dir;
  Vec3 dir0;

  MatrixV3 J_dir;
  //  MatrixV3 J_affine_dir;
  MatrixV2 J_ZNSSD_J_uv;
  MatrixV6 J_x0;

  Mat3 H_dir;
  Mat6 H_x0;

  Mat32 J_unproj;

  // patch for epipolar search
  //  Vec3 xyz_du_ref, xyz_dv_ref;
  //  ArrayV_B vals_Big;
  //  ArrayV_B normalized_vals_Big;
  //  Vec2 grad_dir;
  //  Vec2 grad_dir0;

  Vec3 grad_plane_dir;

  Vec2 sigma2_threshold_vec;

  //  number_t patch_size = 0;
  number_t R_;

private:
  //  bool GetRotMatrix(std::shared_ptr<AlgsImage> img, const Vec2 &px, Mat2
  //  &rot_mat);

  static number_t z_threshold;
  static MatrixV J_ZNSSD_mean;
  const static MatrixV Mat_ZNSSD_I;
};

inline void Patch::TransformScaled(const Mat4 &T10, const number_t &idp,
                                   Matrix3P &res) const {
  //  Matrix3P res_0 = T10.block<3, 3>(0, 0) * patch_dir;
  //  res = res_0.colwise() + T10.block<3, 1>(0, 3) * idp;
  //  res = (T10.block<3, 3>(0, 0) * patch_dir).colwise() + T10.block<3, 1>(0,
  //  3) * idp;
}

struct PyramidPatch {
  // todo: change camera input
  bool SetFromImg(std::shared_ptr<AlgsImage> img, const Vec2 &px,
                  const size_t &cid, bool &is_corner,
                  MultiCamera *p_simple_camera);
  void SetH(const Mat4 &Tcw0, const number_t &idp);

  Mat36 dp_dx0; // same in every level
  std::array<Patch, 1> patchs;
};

void GradValAtD(std::shared_ptr<AlgsImage> img, const Vec2 &px, number_t &res,
                Vec2 *p_grad = nullptr);

void GradValAtDSobel(std::shared_ptr<AlgsImage> img, const Vec2i &px,
                     number_t &res, Vec2 *p_grad = nullptr);

} // namespace dso
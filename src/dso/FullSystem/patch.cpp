#include "patch.h"
#include "../camera_model/vio_def.h"
#include "../camera_model/vio_math_0.h"
#include "opt_nodes_def.h"

#include <iostream>

// template <class Scalar>
// const typename dso::Pattern<Scalar>::Matrix2P dso::Pattern<Scalar>::pattern2
// =
//        Eigen::Map<Pattern<Scalar>::Matrix2P>((Scalar
//        *)Pattern<Scalar>::pattern_raw);
// const typename dso::Patch::Matrix2P dso::Patch::pattern2 =
// Pattern<number_t>::pattern2;

namespace dso {

//    template <class Scalar>
//    const typename dso::Pattern<Scalar>::Matrix2P
//    dso::Pattern<Scalar>::pattern2 =
//            Eigen::Map<Pattern<Scalar>::Matrix2P>((Scalar
//            *)Pattern<Scalar>::pattern_raw);
//    const typename dso::Patch::Matrix2P dso::Patch::pattern2 =
//    Pattern<number_t>::pattern2;

// template <class Scalar>
// const typename Pattern<Scalar>::Matrix2P_B Pattern<Scalar>::pattern2_B =
//    Eigen::Map<Pattern<Scalar>::Matrix2P_B>((Scalar
//    *)Pattern<Scalar>::pattern_raw_angle);
//
// const typename Patch::Matrix2P_B Patch::pattern2_B =
// Pattern<number_t>::pattern2_B;

const typename Patch::MatrixV Patch::Mat_ZNSSD_I = Patch::MatrixV::Identity();

number_t Patch::z_threshold = 0;

typename Patch::MatrixV Patch::J_ZNSSD_mean = Patch::MatrixV::Zero();

void Patch::PatchInit(const number_t &z_thre) {
  VectorV vec_1;
  vec_1.setOnes();
  J_ZNSSD_mean =
      Mat_ZNSSD_I - (vec_1 / (number_t)PATCH_SIZE) * vec_1.transpose();

  z_threshold = z_thre;
}

/*
number_t Patch::GetPatchSize(const Vec3 &point_A, const Vec3 &point_B, const
Vec3 &point_C) { Mat3 mat_point; mat_point.row(0) = point_A.transpose();
  mat_point.row(1) = point_B.transpose();
  mat_point.row(2) = point_C.transpose();
  return std::abs(mat_point.determinant());
}
 */

//  *  *  a  b  *  *
//  *  c  d  e  f  *
//  *  g  h  i  j  *
//  *  *  k  l  *  *
void GradValAtD(std::shared_ptr<AlgsImage> img, const Vec2 &px, number_t &res,
                Vec2 *p_grad) {
  // Note that we don't use ceil here, because we need to compute gradient from
  // these 4 pixels. Doing ceil would cause 0 and 1 to be the same pixel which
  // results in zero gradient

  const int x0i = static_cast<int>(std::floor(px.x()));
  const int x1i = x0i + 1;
  const int y0i = static_cast<int>(std::floor(px.y()));
  const int y1i = y0i + 1;

  int a, b, c, d, e, f, g, h, i, j, k, l;
  number_t x0, y0, x1, y1;

  img->GetTwoPixel(y0i - 1, x0i, a, b);
  img->GetTwoPixel(y0i, x0i - 1, c, d);
  img->GetTwoPixel(y0i, x1i, e, f);
  img->GetTwoPixel(y1i, x0i - 1, g, h);
  img->GetTwoPixel(y1i, x1i, i, j);
  img->GetTwoPixel(y1i + 1, x0i, k, l);

  // normalize coordinate to [0, 1]
  x0 = px.x() - x0i;
  y0 = px.y() - y0i;
  x1 = 1.0 - x0;
  y1 = 1.0 - y0;

  if (p_grad) {
    number_t gd, ge, gh, gi;
    gd = e - c;
    ge = f - d;
    gh = i - g;
    gi = j - h;
    (*p_grad)[0] =
        0.5 * (gd * x1 * y1 + ge * x0 * y1 + gh * x1 * y0 + gi * x0 * y0);

    gd = h - a;
    ge = i - b;
    gh = k - d;
    gi = l - e;
    (*p_grad)[1] =
        0.5 * (gd * x1 * y1 + ge * x0 * y1 + gh * x1 * y0 + gi * x0 * y0);
  }
  res = d * x1 * y1 + e * x0 * y1 + h * x1 * y0 + i * x0 * y0;
}

//  *  a  b  c *
//  *  d  e  f *
//  *  g  h  i *
void GradValAtDSobel(std::shared_ptr<AlgsImage> img, const Vec2i &px,
                     number_t &res, Vec2 *p_grad) {
  // Note that we don't use ceil here, because we need to compute gradient from
  // these 4 pixels. Doing ceil would cause 0 and 1 to be the same pixel which
  // results in zero gradient

  const int x0 = px[0];
  const int y0 = px[1];

  int e;
  img->GetPixel(y0, x0, e);

  if (p_grad) {
    int a, b, c, d, f, g, h, i;
    img->GetPixel(y0 - 1, x0 - 1, a);
    img->GetPixel(y0 - 1, x0, b);
    img->GetPixel(y0 - 1, x0 + 1, c);
    img->GetPixel(y0, x0 - 1, d);
    img->GetPixel(y0, x0 + 1, f);
    img->GetPixel(y0 + 1, x0 - 1, g);
    img->GetPixel(y0 + 1, x0, h);
    img->GetPixel(y0 + 1, x0 + 1, i);

    (*p_grad)[0] = (3 * c + 10 * f + 3 * i - 3 * a - 10 * d - 3 * g) / 32.0;
    (*p_grad)[1] = (3 * g + 10 * h + 3 * i - 3 * a - 10 * b - 3 * c) / 32.0;
  }

  res = e;
}

bool Patch::SetFromImg(std::shared_ptr<AlgsImage> img, const int &level,
                       const Vec2 &px, bool &is_corner,
                       CameraBase *p_simple_camera) {

  int row = img->height;
  int col = img->width;
  Vec2 cur_px;
  Vec2 grad;
  Matrix2P grads;
  int grad_border = 1;
  ArrayV vals;

  Matrix2P coordinate;

  number_t dxx = 0;
  number_t dyy = 0;
  number_t dxy = 0;

  for (size_t i = 0; i < PATCH_SIZE; ++i) {

    // cur_px = px + pattern2[i];
    cur_px = px + pattern2_def.col(i);

    coordinate.col(i) = cur_px;
    if (cur_px.x() < grad_border || cur_px.x() >= col - 1 - grad_border ||
        cur_px.y() < grad_border || cur_px.y() >= row - 1 - grad_border) {
      return false;
    }

    if (i == 0) {
      p_simple_camera->UnProject(cur_px, dir0, &J_unproj);

      if (dir0.hasNaN() || dir0.z() < 0.2) {
        return false;
      }
    }

    //    if (level == 0) {
    //      Vec2i cur_px_i = cur_px.cast<int>();
    //      GradValAtDSobel(img, cur_px_i, vals[i], &grad);
    //    } else {
    GradValAtD(img, cur_px, vals[i], &grad);
    //    }

    dxx += grad[0] * grad[0];
    dyy += grad[1] * grad[1];
    dxy += grad[0] * grad[1];

    if (i == 0) {
      grads0_square_norm = grad.squaredNorm();
    }

    //    if (vals[i] >= 250) {
    //      return false;
    //    }
    grads.col(i) = grad;
  }

  const number_t mean = vals.sum() / (number_t)Patch::PATCH_SIZE;
  normalized_vals = vals - mean;
  const number_t sigma2 = normalized_vals.square().sum();
  const number_t sigma = std::sqrt(sigma2);
  normalized_vals /= sigma;

  if (sigma < 10) {
    return false;
  }

  is_corner = false;

  sigma2_threshold_vec[0] = sigma2 / 16.0f;
  sigma2_threshold_vec[1] = sigma2 * 16.0f;

  //  p_simple_camera->UnProject(px + Vec2(HALF_PATCH_SIZE_B, 0), xyz_du_ref);
  //  p_simple_camera->UnProject(px + Vec2(0, HALF_PATCH_SIZE_B), xyz_dv_ref);

  MatrixV J_ZNSSD_J_I = (Mat_ZNSSD_I - (normalized_vals.matrix() *
                                        normalized_vals.matrix().transpose())) /
                        sigma * J_ZNSSD_mean;

  J_ZNSSD_J_uv = J_ZNSSD_J_I * grads.matrix().transpose();
  Mat2 H_uv = J_ZNSSD_J_uv.transpose() * J_ZNSSD_J_uv;

  //  wgs = 0.01 / (0.01 + J_ZNSSD_J_uv.rowwise().squaredNorm().array());
  //  std::cout << "grads square norm " <<
  //  J_ZNSSD_J_uv.rowwise().squaredNorm().transpose() << std::endl; std::cout
  //  << "wgs " << wgs.transpose() << std::endl;

  //  patch_size = GetPatchSize(patch_dir.col(10), patch_dir.col(13),
  //  patch_dir.col(15));

  //  grad_plane_dir = Skew(patch_dir.col(Patch::PATCH_SIZE - 1)) *
  //  patch_dir.col(Patch::PATCH_SIZE - 2);
  Vec2 grad_uv = coordinate.col(0) + grads.col(0).normalized() * 1.0;
  Vec3 grad_dir;
  p_simple_camera->UnProject(grad_uv, grad_dir);
  grad_plane_dir = Skew(dir0) * grad_dir;
  grad_plane_dir.normalize();

  Mat23 du_ddir;
  p_simple_camera->Project(dir0, cur_px, &du_ddir);
  H_dir = du_ddir.transpose() * H_uv * du_ddir;
  J_dir = -J_ZNSSD_J_uv * du_ddir;
  //  J_affine_dir = grads.matrix().transpose() * du_ddir;
  return true;
}

void Patch::SetHPose(const Mat36 &dp_dx0) {
  H_x0 = dp_dx0.transpose() * H_dir * dp_dx0;
  J_x0 = J_dir * dp_dx0;
}

inline number_t atan_scalar_approximation(number_t x) {
  number_t a1 = 0.99997726f;
  number_t a3 = -0.33262347f;
  number_t a5 = 0.19354346f;
  number_t a7 = -0.11643287f;
  number_t a9 = 0.05265332f;
  number_t a11 = -0.01172120f;

  number_t x_sq = x * x;
  return x *
         (a1 +
          x_sq * (a3 + x_sq * (a5 + x_sq * (a7 + x_sq * (a9 + x_sq * a11)))));
}

inline number_t atan2_auto_1(const number_t &y, const number_t &x) {
  // Ensure input is in [-1, +1]
  bool swap = fabs(x) < fabs(y);
  number_t atan_input = (swap ? x : y) / (swap ? y : x);

  // Approximate atan
  number_t res = atan_scalar_approximation(atan_input);

  // If swapped, adjust atan output
  res = swap ? (atan_input >= 0.0f ? M_PI_2 : -M_PI_2) - res : res;
  // Adjust quadrants
  if (x >= 0.0f && y >= 0.0f) {
  } // 1st quadrant
  else if (x < 0.0f && y >= 0.0f) {
    res = M_PI + res;
  } // 2nd quadrant
  else if (x < 0.0f && y < 0.0f) {
    res = -M_PI + res;
  } // 3rd quadrant
  else if (x >= 0.0f && y < 0.0f) {
  } // 4th quadrant

  return res;
}

inline number_t atan_auto_1(const number_t &y, const number_t &x) {
  // Ensure input >= 0
  bool swap = x < y;
  number_t atan_input = (swap ? x : y) / (swap ? y : x);
  // Approximate atan
  number_t res = atan_scalar_approximation(atan_input);
  // If swapped, adjust atan output
  res = swap ? M_PI_2 - res : res;

  return res;
}

void Patch::ProjectPatchs(std::shared_ptr<CameraBase> simple_camera,
                          const Patch::Matrix3P &target_dir, bool &has_outlier,
                          const int &x_border_min, const int &x_border_max,
                          const int &y_border_min, const int &y_border_max,
                          Matrix2P &res) const {
  for (size_t i = 0; i < target_dir.cols(); ++i) {
    Vec3 normalized_dir = target_dir.col(i).normalized();
    if (normalized_dir.z() < z_threshold) {
      has_outlier = true;
      return;
    }

    Eigen::Ref<Vec2> uv = Eigen::Ref<Vec2>(res.col(i));
    simple_camera->Project(target_dir.col(i), uv);

    if (uv.x() < x_border_min || uv.x() >= x_border_max ||
        uv.y() < y_border_min || uv.y() >= y_border_max) {
      has_outlier = true;
      return;
    }
  }
  has_outlier = false;
}

void Patch::GetPatchValues(const Patch::Matrix2P &uvs,
                           std::shared_ptr<AlgsImage> img, ArrayP &res,
                           Patch::Matrix2P *p_patch_grad,
                           Vec2 *p_center_pixel_grad) const {
  int x0i;
  int y0i;
  int y1i;
  number_t x0, y0, x1, y1;

  for (size_t i = 0; i < uvs.cols(); ++i) {
    const Vec2 &px = uvs.col(i);
    x0i = static_cast<int>(std::floor(px.x()));
    int x1i = x0i + 1;
    y0i = static_cast<int>(std::floor(px.y()));
    y1i = y0i + 1;

    int f00 = static_cast<int>((*img)(y0i, x0i));
    int f10 = static_cast<int>((*img)(y0i, x1i));
    int f01 = static_cast<int>((*img)(y1i, x0i));
    int f11 = static_cast<int>((*img)(y1i, x1i));

    // normalize coordinate to [0, 1]
    x0 = px.x() - x0i;
    y0 = px.y() - y0i;
    x1 = 1.0 - x0;
    y1 = 1.0 - y0;

    if (p_patch_grad) {
      //          Vec2 grad;
      (*p_patch_grad).col(i)[0] = (f10 - f00) * y1 + (f11 - f01) * y0;
      (*p_patch_grad).col(i)[1] = (f01 - f00) * x1 + (f11 - f10) * x0;
      //          target_g2[i] = grad.squaredNorm();
    }

    if (i == 0 && p_center_pixel_grad) {
      //  a      b        c
      //  d     f00     e(f10)
      //  f    g(f01)   h(f11)
      const int a = static_cast<int>((*img)(y0i - 1, x0i - 1));
      const int b = static_cast<int>((*img)(y0i - 1, x0i));
      const int c = static_cast<int>((*img)(y0i - 1, x0i + 1));
      const int d = static_cast<int>((*img)(y0i, x0i - 1));
      const int &e = f10;
      const int f = static_cast<int>((*img)(y0i + 1, x0i - 1));
      const int g = f01;
      const int h = f11;

      (*p_center_pixel_grad).col(i)[0] = 2.0 * (e - d) + (c - a) + (h - f);
      (*p_center_pixel_grad).col(i)[1] = 2.0 * (g - b) + (f - a) + (h - c);
    }

    res[i] = f00 * x1 * y1 + f10 * x0 * y1 + f01 * x1 * y0 + f11 * x0 * y0;
  }
}

bool PyramidPatch::SetFromImg(std::shared_ptr<AlgsImage> img, const Vec2 &px,
                              const size_t &cid, bool &is_corner,
                              MultiCamera *p_simple_camera) {
  Vec2 px_scaled;
  for (int level = 0; level < 1; ++level) {
    number_t scale = std::pow(2, -level);
    px_scaled = (scale * (px.array() + 0.5) - 0.5).matrix();
    if (!patchs[level].SetFromImg(img, level, px_scaled, is_corner,
                                  p_simple_camera->cid_to_cam.at(cid))) {
      return false;
    }

    // Only build big patch on level 0
    //    if (level == 0) {
    //      if (!patchs[level].SetFromImgBigPatch(img_pyramid[level], px_scaled,
    //      p_simple_camera[level]->GetCam(cid))) {
    //        return false;
    //      }
    //    }
  }
  return true;
}

void PyramidPatch::SetH(const Mat4 &Tcw0, const number_t &idp) {
  const Vec3 &n = patchs[0].dir0; // main dir, same in every level
  dp_dx0.leftCols<3>() =
      -LeftMultiSkew(n, Tcw0.block<3, 3>(0, 0)) +
      idp * LeftMultiSkew(Tcw0.block<3, 1>(0, 3), Tcw0.block<3, 3>(0, 0));
  dp_dx0.rightCols<3>() = idp * Tcw0.block<3, 3>(0, 0);

  for (int level = 0; level < 1; ++level) {
    patchs[level].SetHPose(dp_dx0);
  }
}

} // namespace dso
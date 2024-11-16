/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#include "double_sphere_camera.h"

using namespace dso;

bool DoubleSphereCamera::Project(
    const Vec3 &p_3d, Vec2 &p_img,
    LinearAlgebraLib::Matrix<number_t, 2, 3> *d_img_d_p3d,
    LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>
        *d_img_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];

  const number_t &xi = parameters_[4];
  const number_t &alpha = parameters_[5];

  const number_t &x = p_3d[0];
  const number_t &y = p_3d[1];
  const number_t &z = p_3d[2];

  const number_t xx = x * x;
  const number_t yy = y * y;
  const number_t zz = z * z;

  const number_t r2 = xx + yy;

  const number_t d1_2 = r2 + zz;
  const number_t d1 = sqrt(d1_2);

  //  const number_t w1 = alpha > number_t(0.5) ? (number_t(1) - alpha) / alpha
  //                                        : alpha / (number_t(1) - alpha);
  //  const number_t w2 =
  //      (w1 + xi) / sqrt(number_t(2) * w1 * xi + xi * xi + number_t(1));
  //  if (z <= -w2 * d1) return false;
  // TODO: asser z <= -w2 * d1

  const number_t k = xi * d1 + z;
  const number_t kk = k * k;

  const number_t d2_2 = r2 + kk;
  const number_t d2 = std::sqrt(d2_2);

  const number_t norm = alpha * d2 + (number_t(1) - alpha) * k;

  const number_t mx = x / norm;
  const number_t my = y / norm;

  p_img[0] = fx * mx + cx;
  p_img[1] = fy * my + cy;

  if (d_img_d_p3d || d_img_d_param) {
    const number_t norm2 = norm * norm;

    if (d_img_d_p3d) {
      const number_t xy = x * y;
      const number_t tt2 = xi * z / d1 + number_t(1);

      const number_t d_norm_d_r2 = (xi * (number_t(1) - alpha) / d1 +
                                    alpha * (xi * k / d1 + number_t(1)) / d2) /
                                   norm2;

      const number_t tmp2 =
          ((number_t(1) - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;

      (*d_img_d_p3d)(0, 0) = fx * (number_t(1) / norm - xx * d_norm_d_r2);
      (*d_img_d_p3d)(1, 0) = -fy * xy * d_norm_d_r2;

      (*d_img_d_p3d)(0, 1) = -fx * xy * d_norm_d_r2;
      (*d_img_d_p3d)(1, 1) = fy * (number_t(1) / norm - yy * d_norm_d_r2);

      (*d_img_d_p3d)(0, 2) = -fx * x * tmp2;
      (*d_img_d_p3d)(1, 2) = -fy * y * tmp2;
    }

    if (d_img_d_param) {
      d_img_d_param->resize(2, 6);
      (*d_img_d_param).setZero();
      (*d_img_d_param)(0, 0) = mx;
      (*d_img_d_param)(0, 2) = number_t(1);
      (*d_img_d_param)(1, 1) = my;
      (*d_img_d_param)(1, 3) = number_t(1);

      const number_t tmp4 = (alpha - number_t(1) - alpha * k / d2) * d1 / norm2;
      const number_t tmp5 = (k - d2) / norm2;

      (*d_img_d_param)(0, 4) = fx * x * tmp4;
      (*d_img_d_param)(1, 4) = fy * y * tmp4;

      (*d_img_d_param)(0, 5) = fx * x * tmp5;
      (*d_img_d_param)(1, 5) = fy * y * tmp5;
    }
  }
  return true;
}

bool DoubleSphereCamera::UnProject(
    const Vec2 &p_img, Vec3 &p_3d,
    LinearAlgebraLib::Matrix<number_t, 3, 2> *d_p3d_d_img,
    LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>
        *d_p3d_d_param) const {
  const number_t &fx = parameters_[0];
  const number_t &fy = parameters_[1];
  const number_t &cx = parameters_[2];
  const number_t &cy = parameters_[3];

  const number_t &xi = parameters_[4];
  const number_t &alpha = parameters_[5];

  const number_t mx = (p_img[0] - cx) / fx;
  const number_t my = (p_img[1] - cy) / fy;

  const number_t r2 = mx * mx + my * my;

  //  if (alpha > number_t(0.5)) {
  //    if (r2 >= number_t(1) / (number_t(2) * alpha - number_t(1))) return
  //    false;
  //  }
  // TODO: assert return false

  const number_t xi2_2 = alpha * alpha;
  const number_t xi1_2 = xi * xi;

  const number_t sqrt2 =
      std::sqrt(number_t(1) - (number_t(2) * alpha - number_t(1)) * r2);

  const number_t norm2 = alpha * sqrt2 + number_t(1) - alpha;

  const number_t mz = (number_t(1) - xi2_2 * r2) / norm2;
  const number_t mz2 = mz * mz;

  const number_t norm1 = mz2 + r2;
  const number_t sqrt1 = std::sqrt(mz2 + (number_t(1) - xi1_2) * r2);
  const number_t k = (mz * xi + sqrt1) / norm1;

  p_3d[0] = k * mx;
  p_3d[1] = k * my;
  p_3d[2] = k * mz - xi;

  if (d_p3d_d_img || d_p3d_d_param) {
    const number_t norm2_2 = norm2 * norm2;
    const number_t norm1_2 = norm1 * norm1;

    const number_t d_mz_d_r2 = (number_t(0.5) * alpha - xi2_2) *
                                   (r2 * xi2_2 - number_t(1)) /
                                   (sqrt2 * norm2_2) -
                               xi2_2 / norm2;

    const number_t d_mz_d_mx = 2 * mx * d_mz_d_r2;
    const number_t d_mz_d_my = 2 * my * d_mz_d_r2;

    const number_t d_k_d_mz =
        (norm1 * (xi * sqrt1 + mz) - 2 * mz * (mz * xi + sqrt1) * sqrt1) /
        (norm1_2 * sqrt1);

    const number_t d_k_d_r2 =
        (xi * d_mz_d_r2 +
         number_t(0.5) / sqrt1 *
             (number_t(2) * mz * d_mz_d_r2 + number_t(1) - xi1_2)) /
            norm1 -
        (mz * xi + sqrt1) * (number_t(2) * mz * d_mz_d_r2 + number_t(1)) /
            norm1_2;

    const number_t d_k_d_mx = d_k_d_r2 * 2 * mx;
    const number_t d_k_d_my = d_k_d_r2 * 2 * my;

    Vec3 c0, c1;

    c0[0] = (mx * d_k_d_mx + k);
    c0[1] = my * d_k_d_mx;
    c0[2] = (mz * d_k_d_mx + k * d_mz_d_mx);

    c0 /= fx;

    c1[0] = mx * d_k_d_my;
    c1[1] = (my * d_k_d_my + k);
    c1[2] = (mz * d_k_d_my + k * d_mz_d_my);

    c1 /= fy;

    if (d_p3d_d_img) {
      d_p3d_d_img->col(0) = c0;
      d_p3d_d_img->col(1) = c1;
    }

    if (d_p3d_d_param) {
      d_p3d_d_param->resize(3, 6);
      d_p3d_d_param->setZero();
      const number_t d_k_d_xi1 = (mz * sqrt1 - xi * r2) / (sqrt1 * norm1);

      const number_t d_mz_d_xi2 =
          (number_t(1) - r2 * xi2_2) *
              (r2 * alpha / sqrt2 - sqrt2 + number_t(1)) / norm2_2 -
          number_t(2) * r2 * alpha / norm2;

      const number_t d_k_d_xi2 = d_k_d_mz * d_mz_d_xi2;

      (*d_p3d_d_param).col(0) = -c0 * mx;
      (*d_p3d_d_param).col(1) = -c1 * my;

      (*d_p3d_d_param).col(2) = -c0;
      (*d_p3d_d_param).col(3) = -c1;

      (*d_p3d_d_param)(0, 4) = mx * d_k_d_xi1;
      (*d_p3d_d_param)(1, 4) = my * d_k_d_xi1;
      (*d_p3d_d_param)(2, 4) = mz * d_k_d_xi1 - 1;

      (*d_p3d_d_param)(0, 5) = mx * d_k_d_xi2;
      (*d_p3d_d_param)(1, 5) = my * d_k_d_xi2;
      (*d_p3d_d_param)(2, 5) = mz * d_k_d_xi2 + k * d_mz_d_xi2;
    }
  }
  return true;
}

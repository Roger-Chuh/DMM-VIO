/*******************************************************
 * This file is part of PISCES.
 * Author: Chence
 *******************************************************/
#include "radial_tangential_camera.h"
using namespace dso;

bool RadtanCamera::Project(const Vec3& p_3d, Vec2& p_img, LinearAlgebraLib::Matrix<number_t, 2, 3>* d_img_d_p3d,
                           LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>* d_img_d_param) const {
  const number_t& fx = parameters_[0];
  const number_t& fy = parameters_[1];
  const number_t& cx = parameters_[2];
  const number_t& cy = parameters_[3];

  const number_t& k1 = parameters_[4];
  const number_t& k2 = parameters_[5];
  const number_t& p1 = parameters_[6];
  const number_t& p2 = parameters_[7];

  number_t x = p_3d[0] / p_3d[2];
  number_t y = p_3d[1] / p_3d[2];

  number_t x2 = x * x;
  number_t y2 = y * y;
  number_t xy = x * y;
  number_t r2 = x2 + y2;
  number_t r4 = r2 * r2;

  // TODO: project jacobi
  if (d_img_d_p3d) {
    number_t du_dx =
        fx * (1 + k1 * r2 + k2 * r4 + 2 * k1 * x2 + 4 * k2 * x2 * r2 + 2 * p1 * y + 6 * p2 * x) * (1.0f / p_3d[2]);
    number_t du_dy = fx * (2 * k1 * xy + 4 * k2 * r2 * xy + 2 * p1 * x + 2 * p2 * y) * (1.0f / p_3d[2]);
    number_t du_dz = -1.0f * (x * du_dx + y * du_dy);
    number_t dv_dx = fy * (2 * k1 * xy + 4 * k2 * r2 * xy + 2 * p1 * x + 2 * p2 * y) * (1.0f / p_3d[2]);
    number_t dv_dy =
        fy * (1 + k1 * r2 + k2 * r4 + 2 * k1 * y2 + 4 * k2 * y2 * r2 + 6 * p1 * y + 2 * p2 * x) * (1.0f / p_3d[2]);
    number_t dv_dz = -1.0f * (x * dv_dx + y * dv_dy);

    (*d_img_d_p3d) << du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz;
  }

  if (d_img_d_param) {
    d_img_d_param->resize(2, 8);
    (*d_img_d_param) << x, 0.0f, 1.0f, 0.0f, fx * x * r2, fx * x * r4, 2 * fx * xy, fx * (r2 + 2 * x2), 0, y, 0, 1,
        fy * y * r2, fy * y * r4, fy * (r2 + 2 * y2), 2 * fy * xy;
  }

  number_t rad_dist_u = k1 * r2 + k2 * r2 * r2;
  x += x * rad_dist_u + number_t(2.0) * p1 * xy + p2 * (r2 + number_t(2.0) * x2);
  y += y * rad_dist_u + number_t(2.0) * p2 * xy + p1 * (r2 + number_t(2.0) * y2);

  p_img[0] = fx * x + cx;
  p_img[1] = fy * y + cy;
  return true;
}

// TODO: unproject jacobi
bool RadtanCamera::UnProject(const Vec2& p_img, Vec3& p_3d, LinearAlgebraLib::Matrix<number_t, 3, 2>* d_p3d_d_img,
                             LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>* d_p3d_d_param) const {
  const number_t& fx = parameters_[0];
  const number_t& fy = parameters_[1];
  const number_t& cx = parameters_[2];
  const number_t& cy = parameters_[3];

  const number_t& k1 = parameters_[4];
  const number_t& k2 = parameters_[5];
  const number_t& p1 = parameters_[6];
  const number_t& p2 = parameters_[7];

  Vec2 y = Vec2((p_img(0) - cx) / fx, (p_img(1) - cy) / fy);
  Vec2 y_bar = y;
  const int n = 5;
  LinearAlgebraLib::Matrix<number_t, 2, 2> F;

  Vec2 y_tmp;

  for (int i = 0; i < n; ++i) {
    y_tmp = y_bar;

    number_t mx2_u = y_tmp(0) * y_tmp(0);
    number_t my2_u = y_tmp(1) * y_tmp(1);
    number_t mxy_u = y_tmp(0) * y_tmp(1);
    number_t rho2_u = mx2_u + my2_u;

    number_t rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;

    F(0, 0) = 1 + rad_dist_u + k1 * 2.0 * mx2_u + k2 * rho2_u * 4 * mx2_u + 2.0 * p1 * y_tmp(1) + 6 * p2 * y_tmp(0);
    F(1, 0) = k1 * 2.0 * y_tmp(0) * y_tmp(1) + k2 * 4 * rho2_u * y_tmp(0) * y_tmp(1) + p1 * 2.0 * y_tmp(0) +
              2.0 * p2 * y_tmp(1);
    F(0, 1) = F(1, 0);
    F(1, 1) = 1 + rad_dist_u + k1 * 2.0 * my2_u + k2 * rho2_u * 4 * my2_u + 6 * p1 * y_tmp(1) + 2.0 * p2 * y_tmp(0);

    y_tmp(0) += y_tmp(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u);
    y_tmp(1) += y_tmp(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

    Vec2 e(y - y_tmp);
    Vec2 du = (F.transpose() * F).inverse() * F.transpose() * e;

    y_bar += du;

    if (e.dot(e) < 1e-15) break;
  }

  p_3d(0) = y_bar(0);
  p_3d(1) = y_bar(1);
  p_3d(2) = 1.0;
  p_3d.normalize();
  return true;
}
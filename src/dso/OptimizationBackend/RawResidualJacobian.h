/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "util/NumType.h"

namespace dso {
struct RawResidualJacobian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // ================== new structure: save independently =============.
  VecNRf resF; // TODO 加权后的光度残差 //!< 每个patch的8个残差

  // the two rows of d[x,y]/d[xi].
  Vec6f Jpdxi[2]; // 2x6 //TODO uv对pose的雅可比[2x6] //!< 点对位姿

  // the two rows of d[x,y]/d[C].
  VecCf Jpdc[2]; // 2x4 //TODO uv对intr的雅可比[2x4] //!< 点对相机参数

  // the two rows of d[x,y]/d[idepth].
  Vec2f Jpdd; // 2x1 //TODO uv对idp的雅可比[2x1] 	//!< 点对逆深度 //TODO
              // fej, 其实逆深度每次都会重新线性化，等于没用fej

  // the two columns of d[r]/d[x,y].
  VecNRf JIdx[2]; // 9x2 //TODO 残差对uv的雅可比（梯度） //!<
                  // patch光度误差对点(gradient), 8×2 //TODO gradient, not using
                  // fej

  // = the two columns of d[r] / d[ab]
  VecNRf JabF[2]; // 9x2 //TODO 残差对ab的雅可比[1x2] //!<
                  // patch光度误差对光度仿射， 8x2 //TODO affine correction, not
                  // using fej

  //!< 对应的小的hessian
  // = JIdx^T * JIdx (inner product). Only as a shorthand.
  Mat22f JIdx2; // 2x2 //TODO 梯度x梯度部分的小hessian
  // = Jab^T * JIdx (inner product). Only as a shorthand.
  Mat22f JabJIdx; // 2x2 //TODO 光度x梯度部分的小hessian
  // = Jab^T * Jab (inner product). Only as a shorthand.
  Mat22f Jab2; // 2x2 //TODO 光度x光度部分的小hessian
};
} // namespace dso

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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"


//#define USE_INVERSE_COMPOSITIONAL

namespace dso
{
int PointFrameResidual::instanceCounter = 0;


long runningResID=0;


PointFrameResidual::PointFrameResidual(){assert(false); instanceCounter++;}

PointFrameResidual::~PointFrameResidual(){assert(efResidual==0); instanceCounter--; delete J;}

PointFrameResidual::PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_) :
	point(point_),
	host(host_),
	target(target_)
{
	efResidual=0;
	instanceCounter++;
	resetOOB();
    //TODO 这时J只是开辟了空间，还没有赋值
	J = new RawResidualJacobian();// 各种雅克比
	assert(((long)J)%16==0); // 16位对齐

	isNew=true;
}



//@ 求对各个参数的导数, 和能量值
double PointFrameResidual::linearize(CalibHessian* HCalib)
{
    // printf("fx fy cx cy: [%f %f %f %f]\n", HCalib->fxl(), HCalib->fyl(), HCalib->cxl(), HCalib->cyl());
	state_NewEnergyWithOutlier=-1;

	if(state_state == ResState::OOB)
		{ state_NewState = ResState::OOB; return state_energy; }
//TODO 同一个host有多个target，合理
	FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);// 得到这个目标帧在主帧上的一些预计算参数
	float energyLeft=0;
	const Eigen::Vector3f* dIl = target->dI;
    const Eigen::Vector3f* host_dIl = host->dI;
	//const float* const Il = target->I;
    const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll; //todo relative pose after optimize
    const Vec3f &PRE_KtTll = precalc->PRE_KtTll;      //
    const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;   //todo relative pose before optimize
	const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
	const float * const color = point->color;// host帧上颜色
	const float * const weights = point->weights;

	Vec2f affLL = precalc->PRE_aff_mode;// 待优化的a和b, 就是host和target合的
	float b0 = precalc->PRE_b0_mode;// 主帧的单独 b

//! x=0时候求几何的导数, 使用FEJ!! ,逆深度没有使用FEJ
	Vec6f d_xi_x, d_xi_y;
	Vec4f d_C_x, d_C_y;
	float d_d_x, d_d_y;
    Eigen::Matrix<float, 2, 6> d_uv_d_pose, d_uv_d_pose_inverse_comp, d_uv_d_pose_fwd_jac;
    Eigen::Matrix<float, 2, 3> d_uv_d_pt3d, d_uv_host_d_n_host, d_uv_target_d_x_target_scaled;
    Eigen::Matrix<float, 3, 6> d_pt3d_d_pose, d_pt3d_d_pose_inverse_comp;
	{
		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;
        /// PRE_RTll_0 means FEJ
		if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
				PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{ state_NewState = ResState::OOB; return state_energy; }// 投影不在图像里, 则返回OOB

		centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

        Vec3f n_host = Vec3f(
                (point->u-HCalib->cxl())*HCalib->fxli(),
                (point->v-HCalib->cyl())*HCalib->fyli(),
                1);

        Vec3f X_target_scaled = PRE_RTll_0 * n_host + PRE_tTll_0*point->idepth_zero_scaled;




        Mat33f X_target_scaled_skew;
        X_target_scaled_skew <<  (0), -X_target_scaled(2),  X_target_scaled(1),
                X_target_scaled(2),    (0), -X_target_scaled(0),
                -X_target_scaled(1),  X_target_scaled(0),     (0);


        d_uv_d_pt3d<<HCalib->fxl()/X_target_scaled(2),0,-HCalib->fxl()*X_target_scaled(0)/X_target_scaled(2)/X_target_scaled(2),
                0, HCalib->fyl()/X_target_scaled(2),-HCalib->fyl()*X_target_scaled(1)/X_target_scaled(2)/X_target_scaled(2);
        d_pt3d_d_pose.leftCols(3) = point->idepth_zero_scaled*Mat33f::Identity();
        d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;

        d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;



        d_uv_host_d_n_host << HCalib->fxl(),           0,            -HCalib->fxl()*n_host(0),
                0,              HCalib->fyl(),      -HCalib->fyl()*n_host(1);

        //TODO inverse comp d_uv_d_pose
        d_uv_d_pose_inverse_comp = d_uv_host_d_n_host * PRE_RTll_0.transpose() * d_pt3d_d_pose;



        d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
        d_uv_d_pose_fwd_jac = d_uv_target_d_x_target_scaled * d_pt3d_d_pose;



		// diff d_idepth
        //TODO 这些在初始化都写过了又写一遍 !!! 放到一起好不好, ai
        //TODO same as CoarseInitializer.cpp -> CoarseInitializer::calcResAndGS
        //* 像素点对host上逆深度求导(由于乘了SCALE_IDEPTH倍, 因此乘上)
        /// d_idepth / d_x,  d_idepth / d_y
        //TODO 带0的都是使用FEJ的？ PRE_tTll_0, 并不是，带0只是表示这是优化之前的state estimation
        // drescale = 1/X_target_scaled(2);
        // drescale = rou2 / rou1
        // u v是target帧的归一化坐标
        Vec2f d_uv_d_d;
        d_uv_d_d = d_uv_host_d_n_host * PRE_RTll_0.transpose() * PRE_tTll_0;


        Vec2f d_uv_d_d_fwd_jac;
        d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled * PRE_tTll_0;

#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
		d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*SCALE_IDEPTH*HCalib->fxl();
		d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*SCALE_IDEPTH*HCalib->fyl();
#else
        d_d_x = d_uv_d_d_fwd_jac(0);
        d_d_y = d_uv_d_d_fwd_jac(1);
#endif
#else
        d_d_x = d_uv_d_d(0);
        d_d_y = d_uv_d_d(1);
#endif



        //* 像素点对相机内参fx fy cx cy的导数第一部分
        // diff calib
        //! [0]: 1/Pz'*Px*(R20*Px'/Pz' - R00)
        //! [1]: 1/Pz'*Py*fx/fy*(R21*Px'/Pz' - R01)
        //! [2]: 1/Pz'*(R20*Px'/Pz' - R00)
        //! [3]: 1/Pz'*fx/fy*(R21*Px'/Pz' - R01)
		d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));
		d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) * HCalib->fyli();
        //TODO KliP: host帧归一化坐标
		d_C_x[0] = KliP[0]*d_C_x[2];
		d_C_x[1] = KliP[1]*d_C_x[3];

        //! [0]: 1/Pz'*Px*fy/fy*(R20*Py'/Pz' - R10)
        //! [1]: 1/Pz'*Py*(R21*Py'/Pz' - R11)
        //! [2]: 1/Pz'*fy/fy*(R20*Py'/Pz' - R10)
        //! [3]: 1/Pz'*(R21*Py'/Pz' - R11)
		d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) * HCalib->fxli();
		d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
		d_C_y[0] = KliP[0]*d_C_y[2];
		d_C_y[1] = KliP[1]*d_C_y[3];



        Eigen::Matrix<float,2,4> d_uv_d_C, d_uv_d_C_inverse_comp;

        //* 第二部分 同样project时候一样使用了scaled的内参
        //! [Px'/Pz'  0  1  0;
        //!  0  Py'/Pz'  0  1]
#if 1
     //   #ifndef USE_INVERSE_COMPOSITIONAL
		d_C_x[0] = (d_C_x[0]+u)*SCALE_F;//TODO d_u2_d_fx
		d_C_x[1] *= SCALE_F;
		d_C_x[2] = (d_C_x[2]+1)*SCALE_C;//TODO d_u2_d_cx
		d_C_x[3] *= SCALE_C;

		d_C_y[0] *= SCALE_F;
		d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
		d_C_y[2] *= SCALE_C;
		d_C_y[3] = (d_C_y[3]+1)*SCALE_C;
#else
        d_C_x[0] = n_host(0)*SCALE_F;//TODO d_u2_d_fx
        d_C_x[1] = 0*SCALE_F;
        d_C_x[2] = 1*SCALE_C;//TODO d_u2_d_cx
        d_C_x[3] = 0*SCALE_C;

        d_C_y[0] = 0*SCALE_F;
        d_C_y[1] = n_host(1)*SCALE_F;
        d_C_y[2] = 0*SCALE_C;
        d_C_y[3] = 1*SCALE_C;
#endif

#ifndef USE_INVERSE_COMPOSITIONAL
        //* 像素点对位姿的导数, 位移在前!
        //! 公式见初始化那儿
        //TODO same as CoarseInitializer.cpp -> CoarseInitializer::calcResAndGS
#ifndef USE_ZNCC
		d_xi_x[0] = new_idepth*HCalib->fxl();
		d_xi_x[1] = 0;
		d_xi_x[2] = -new_idepth*u*HCalib->fxl();
		d_xi_x[3] = -u*v*HCalib->fxl();
		d_xi_x[4] = (1+u*u)*HCalib->fxl();
		d_xi_x[5] = -v*HCalib->fxl();

		d_xi_y[0] = 0;
		d_xi_y[1] = new_idepth*HCalib->fyl();
		d_xi_y[2] = -new_idepth*v*HCalib->fyl();
		d_xi_y[3] = -(1+v*v)*HCalib->fyl();
		d_xi_y[4] = u*v*HCalib->fyl();
		d_xi_y[5] = u*HCalib->fyl();
#else
        d_xi_x = d_uv_d_pose_fwd_jac.row(0).transpose();
        d_xi_y = d_uv_d_pose_fwd_jac.row(1).transpose();
#endif
#else
        d_xi_x = d_uv_d_pose_inverse_comp.row(0).transpose();
        d_xi_y = d_uv_d_pose_inverse_comp.row(1).transpose();
#endif
#if 0
        Eigen::Matrix<float,2,12> show;
        show.block<1,6>(0,0) = d_xi_x.transpose();
        show.block<1,6>(1,0) = d_xi_y.transpose();
        show.rightCols(6) = show.leftCols(6) - d_uv_d_pose;
        std::cout<<"pose jac diff:\n"<<show<<std::endl;
#endif
	}


	{
        //TODO 终于找到给J赋值的地方了
		J->Jpdxi[0] = d_xi_x;
		J->Jpdxi[1] = d_xi_y;

		J->Jpdc[0] = d_C_x;
		J->Jpdc[1] = d_C_y;

		J->Jpdd[0] = d_d_x;
		J->Jpdd[1] = d_d_y;
#if 0
        Eigen::Matrix<float, 2,2> d_uv_d_c_show;
        d_uv_d_c_show.col(0) = J->Jpdd;
        d_uv_d_c_show.col(1) = d_uv_d_c_show.col(0) - d_uv_d_pt3d*PRE_tTll_0;
        std::cout<<"d_uv_d_c_show:\n"<<d_uv_d_c_show<<std::endl;
#endif
	}






	float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
	float JabJIdx_00=0, JabJIdx_01=0, JabJIdx_10=0, JabJIdx_11=0;
	float JabJab_00=0, JabJab_01=0, JabJab_11=0;

	float wJI2_sum = 0;


    Eigen::MatrixXf Mat_ZNSSD_I;
    Eigen::MatrixXf J_ZNSSD_mean;
    Eigen::MatrixXf J_ZNSSD_J_I_host;
    Eigen::MatrixXf J_ZNSSD_J_I_target;
    Eigen::MatrixXf grad_new_host;
    Eigen::MatrixXf grad_new_target;
    float host_val_mean;
    float target_val_mean;
    Eigen::MatrixXf ones;
    float host_sigma, target_sigma;

    Eigen::MatrixXf host_info, target_info;
    size_t count = 0;
    for(int idx=0;idx<patternNum;idx++) {
        float Ku, Kv;
        //? 为啥这里使用idepth_scaled, 上面使用的是zero； 答： 其实和上面一样的....同时调用了setIdepth() setIdepthZero()
        //! 答: 这里是求图像导数, 由于线性误差大, 就不使用FEJ, 所以使用当前的状态
        //TODO  这里求残差用的是最新状态重投影，而不是fej状态重投影
        if (!projectPoint(point->u + patternP[idx][0], point->v + patternP[idx][1], point->idepth_scaled, PRE_KRKiTll,
                          PRE_KtTll, Ku, Kv)) {
            continue;
        }

        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        //float residual = hitColor[0] - (float) (affLL[0] * color[idx] + affLL[1]);
        Vec3f hostColor = (getInterpolatedElement33(host_dIl, point->u + patternP[idx][0], point->v + patternP[idx][1],
                                                    wG[0]));
        float host_value_corrected = (float)(affLL[0] * color[idx] + affLL[1]);

        if (!std::isfinite((float) hitColor[0])) {
            continue;
        }
        hostColor[0] = host_value_corrected;
        host_info.conservativeResize(count + 1, 3);
        target_info.conservativeResize(count + 1, 3);
        host_info.row(count) = hostColor.transpose();
        target_info.row(count) = hitColor.transpose();
        count++;
    }

    int patch_num = host_info.rows();
    float ws2 = 1;
    if(patch_num != 0) {

        host_val_mean = host_info.col(0).sum() / patch_num;
        target_val_mean = target_info.col(0).sum() / patch_num;

        ones.conservativeResize(patch_num, 1);
        ones.setOnes();
        host_info.col(0) = host_info.col(0) - host_val_mean * ones;
        target_info.col(0) = target_info.col(0) - target_val_mean * ones;
        host_sigma = host_info.col(0).norm();
        target_sigma = target_info.col(0).norm();
        host_info.col(0) /= host_sigma;
        target_info.col(0) /= target_sigma;

        Mat_ZNSSD_I.conservativeResize(patch_num, patch_num);
        Mat_ZNSSD_I.setIdentity();


        J_ZNSSD_mean = Mat_ZNSSD_I - (ones / static_cast<float>(patch_num)) * ones.transpose();

        J_ZNSSD_J_I_host = setting_variableScale *
                ((Mat_ZNSSD_I - (host_info.col(0) * host_info.col(0).transpose())) / host_sigma * J_ZNSSD_mean);
        J_ZNSSD_J_I_target = setting_variableScale *
                ((Mat_ZNSSD_I - (target_info.col(0) * target_info.col(0).transpose())) / target_sigma * J_ZNSSD_mean);

        grad_new_host = J_ZNSSD_J_I_host * host_info.rightCols(2);        // "new" gradient: 8x2
        grad_new_target = J_ZNSSD_J_I_target * target_info.rightCols(2);  // "new" gradient: 8x2


        float zncc = target_info.col(0).dot(host_info.col(0));
        float r2 = 2 - 2 * zncc;
        ws2 = 2.0 / (r2 + 2.0);

#ifdef USE_ZNCC_WEIGHT
        if (zncc < 0.8) {
            state_NewState = ResState::OOB; return state_energy;
        }
#endif
        host_info.col(0) *= setting_variableScale;
        target_info.col(0) *= setting_variableScale;
    }
//    std::cout << "lba, grad_new_host: \n" << grad_new_host << std::endl;
//    std::cout << "lba, grad_new_target: \n" << grad_new_target << std::endl;


    int cnt = 0;
	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
        //? 为啥这里使用idepth_scaled, 上面使用的是zero； 答： 其实和上面一样的....同时调用了setIdepth() setIdepthZero()
        //! 答: 这里是求图像导数, 由于线性误差大, 就不使用FEJ, 所以使用当前的状态
        //TODO  这里求残差用的是最新状态重投影，而不是fej状态重投影
		if(!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{ state_NewState = ResState::OOB; return state_energy; }

		// 像素坐标
		projectedTo[idx][0] = Ku;
		projectedTo[idx][1] = Kv;


        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        //* 残差对光度仿射a求导
        //! 光度参数使用固定线性化点了
        float drdA = (color[idx]-b0);
        if(!std::isfinite((float)hitColor[0]))
        { state_NewState = ResState::OOB; return state_energy; }



#ifndef USE_ZNCC
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);
#else
        float residual_bak = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);
        float residual = 1 * (target_info(cnt, 0) - host_info(cnt, 0));
        if (std::isnan(residual)) {
//            isGood = false;
//            break;
            state_NewState = ResState::OOB; return state_energy;
        }
#endif
        Vec3f hostColor = (getInterpolatedElement33(host_dIl, point->u+patternP[idx][0], point->v+patternP[idx][1], wG[0]));

        //printf("value1: %f, value check: %f\n", hostColor[0], color[idx]);
        //assert(hostColor[0] == color[idx]);
//        //* 残差对光度仿射a求导
//        //! 光度参数使用固定线性化点了
//		float drdA = (color[idx]-b0);
//		if(!std::isfinite((float)hitColor[0]))
//		{ state_NewState = ResState::OOB; return state_energy; }

#ifndef USE_ZNCC
		float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
#else
        //float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + grad_new_target.row(cnt).squaredNorm()));
        float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
#endif
#ifndef USE_ZNCC_WEIGHT
        w = 0.5f*(w + weights[idx]);
#else
        w = std::sqrt(ws2);
#endif

#ifndef USE_ZNCC
		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += w*w*hw *residual*residual*(2-hw);
#else
        float hw = fabsf(residual) < setting_huberTH_LBA ? 1 : setting_huberTH_LBA / fabsf(residual);
        energyLeft += w*w*hw *residual*residual*(2-hw);
#endif

		{
            //printf("weights: %f, w: %f, hw: %f, residual: %f\n", weights[idx], w, hw, residual);
		    //printf("hw: %f\n", hw);
			if(hw < 1) hw = sqrtf(hw);
			hw = hw*w;

			hitColor[1]*=hw;
			hitColor[2]*=hw;

            hostColor[1]*=hw;
            hostColor[2]*=hw;

            grad_new_target.row(cnt) *= hw;
            grad_new_host.row(cnt) *= hw;


            //! 残差 res*w*sqrt(hw)
			J->resF[idx] = residual*hw;

            //! 图像导数 dx dy
#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
			J->JIdx[0][idx] = hitColor[1];
			J->JIdx[1][idx] = hitColor[2];
#else
            J->JIdx[0][idx] = grad_new_target(cnt, 0);
            J->JIdx[1][idx] = grad_new_target(cnt, 1);
#endif
#else
#ifndef USE_ZNCC
            J->JIdx[0][idx] = affLL[0] * hostColor[1];
            J->JIdx[1][idx] = affLL[0] * hostColor[2];
#else
            J->JIdx[0][idx] = affLL[0] * grad_new_host(cnt, 0);
            J->JIdx[1][idx] = affLL[0] * grad_new_host(cnt, 1);
#endif
#endif

            //! 对光度合成后a b的导数 [Ii-b0  1]
            //! Ij - a*Ii - b  (a = tj*e^aj / ti*e^ai,   b = bj - a*bi) //TODO true dat
            //! Ij - [a*(Ii-b0) + b]
            //TODO bug 正负号有影响 ??? ab部分好确实差了一个负号
#ifndef USE_INVERSE_COMPOSITIONAL
			J->JabF[0][idx] = drdA*hw;
			J->JabF[1][idx] = hw;
#else
            J->JabF[0][idx] = drdA*hw;
            J->JabF[1][idx] = 1*hw;
#endif

#ifndef USE_INVERSE_COMPOSITIONAL
            //! dIdx&dIdx hessian block
            // Jt * W * J = [gx; gy] * [gx gy] = [gxgx gxgy; gxgy gygy]
#ifndef USE_ZNCC
			JIdxJIdx_00+=hitColor[1]*hitColor[1];
			JIdxJIdx_11+=hitColor[2]*hitColor[2];
			JIdxJIdx_10+=hitColor[1]*hitColor[2];
            //! dIdx&dIdab hessian block
			JabJIdx_00+= drdA*hw * hitColor[1];
			JabJIdx_01+= drdA*hw * hitColor[2];
			JabJIdx_10+= hw * hitColor[1];
			JabJIdx_11+= hw * hitColor[2];
#else
            JIdxJIdx_00+=grad_new_target(cnt, 0)*grad_new_target(cnt, 0);
            JIdxJIdx_11+=grad_new_target(cnt, 1)*grad_new_target(cnt, 1);
            JIdxJIdx_10+=grad_new_target(cnt, 0)*grad_new_target(cnt, 1);
            //! dIdx&dIdab hessian block
            //TODO 即使用了zncc，但关于ab的雅可比任然需要用梯度
#if 1
            JabJIdx_00+= drdA*hw * grad_new_target(cnt, 0);
            JabJIdx_01+= drdA*hw * grad_new_target(cnt, 1);
            JabJIdx_10+= hw * grad_new_target(cnt, 0);
            JabJIdx_11+= hw * grad_new_target(cnt, 1);
#else
            JabJIdx_00+= drdA*hw * hitColor(cnt, 0);
            JabJIdx_01+= drdA*hw * hitColor(cnt, 1);
            JabJIdx_10+= hw * hitColor(cnt, 0);
            JabJIdx_11+= hw * hitColor(cnt, 1);
#endif
#endif
            //! dIdab&dIdab hessian block
			JabJab_00+= drdA*drdA*hw*hw;
			JabJab_01+= drdA*hw*hw;
			JabJab_11+= hw*hw;
#else
            //! dIdx&dIdx hessian block
            // Jt * W * J = [gx; gy] * [gx gy] = [gxgx gxgy; gxgy gygy]
#ifndef USE_ZNCC
            JIdxJIdx_00+=affLL[0] *affLL[0] *hostColor[1]*hostColor[1];
            JIdxJIdx_11+=affLL[0] *affLL[0] *hostColor[2]*hostColor[2];
            JIdxJIdx_10+=affLL[0] *affLL[0] *hostColor[1]*hostColor[2];
            //! dIdx&dIdab hessian block
            JabJIdx_00+= drdA*hw * affLL[0] * hostColor[1];
            JabJIdx_01+= drdA*hw * affLL[0] * hostColor[2];
            JabJIdx_10+= hw * affLL[0] * hostColor[1];
            JabJIdx_11+= hw * affLL[0] * hostColor[2];
#else
            JIdxJIdx_00+=affLL[0] *affLL[0] *grad_new_host(cnt, 0)*grad_new_host(cnt, 0);
            JIdxJIdx_11+=affLL[0] *affLL[0] *grad_new_host(cnt, 1)*grad_new_host(cnt, 1);
            JIdxJIdx_10+=affLL[0] *affLL[0] *grad_new_host(cnt, 0)*grad_new_host(cnt, 1);
            //! dIdx&dIdab hessian block
            JabJIdx_00+= drdA*hw * affLL[0] * grad_new_host(cnt, 0);
            JabJIdx_01+= drdA*hw * affLL[0] * grad_new_host(cnt, 1);
            JabJIdx_10+= hw * affLL[0] * grad_new_host(cnt, 0);
            JabJIdx_11+= hw * affLL[0] * grad_new_host(cnt, 1);

#endif
            //! dIdab&dIdab hessian block
            JabJab_00+= drdA*drdA*hw*hw;
            JabJab_01+= drdA*hw*hw;
            JabJab_11+= hw*hw;
#endif
#ifndef USE_ZNCC
			wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]);
#else
            wJI2_sum += hw*hw*(grad_new_target.row(cnt).squaredNorm());
#endif
			if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;
			if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;

		}
        cnt++;
	}

	J->JIdx2(0,0) = JIdxJIdx_00;  //TODO gradient related 2x2, top left
	J->JIdx2(0,1) = JIdxJIdx_10;  //TODO 梯度x梯度部分的小hessian
	J->JIdx2(1,0) = JIdxJIdx_10;
	J->JIdx2(1,1) = JIdxJIdx_11;
	J->JabJIdx(0,0) = JabJIdx_00; //TODO buttom left
	J->JabJIdx(0,1) = JabJIdx_01; //TODO 光度x梯度部分的小hessian
	J->JabJIdx(1,0) = JabJIdx_10;
	J->JabJIdx(1,1) = JabJIdx_11;
	J->Jab2(0,0) = JabJab_00;     //TODO buttom right
	J->Jab2(0,1) = JabJab_01;     //TODO 光度x光度部分的小hessian
	J->Jab2(1,0) = JabJab_01;
	J->Jab2(1,1) = JabJab_11;

	state_NewEnergyWithOutlier = energyLeft;
//* 大于阈值则视为有外点
	if(energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) /*|| wJI2_sum < 2*/)
	{
		energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
		state_NewState = ResState::OUTLIER;
	}
	else
	{
		state_NewState = ResState::IN;
	}

	state_NewEnergy = energyLeft;
	return energyLeft;
}



void PointFrameResidual::debugPlot()
{
	if(state_state==ResState::OOB) return;
	Vec3b cT = Vec3b(0,0,0);

	if(freeDebugParam5==0)
	{
		float rT = 20*sqrt(state_energy/9);
		if(rT<0) rT=0; if(rT>255)rT=255;
		cT = Vec3b(0,255-rT,rT);
	}
	else
	{
		if(state_state == ResState::IN) cT = Vec3b(255,0,0);
		else if(state_state == ResState::OOB) cT = Vec3b(255,255,0);
		else if(state_state == ResState::OUTLIER) cT = Vec3b(0,0,255);
		else cT = Vec3b(255,255,255);
	}

	for(int i=0;i<patternNum;i++)
	{
		if((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0]-3 && projectedTo[i][1] < hG[0]-3 ))
			target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1],cT);
	}
}


//@ 把计算的残差,雅克比值给EFResidual, 更新残差的状态(好坏)
void PointFrameResidual::applyRes(bool copyJacobians)
{
	if(copyJacobians)
	{
		if(state_state == ResState::OOB)
		{
			assert(!efResidual->isActiveAndIsGoodNEW);
			return;	// can never go back from OOB
		}
		if(state_NewState == ResState::IN)// && )
		{
			efResidual->isActiveAndIsGoodNEW=true;
            //? 指针好恶心, 计算好了调用这个函数
			efResidual->takeDataF(); // 从当前取jacobian数据
		}
		else
		{
			efResidual->isActiveAndIsGoodNEW=false;
		}
	}

	setState(state_NewState);
	state_energy = state_NewEnergy;
}
}

/**
* This file is part of DSO, written by Jakob Engel.
* It has been modified by Lukas von Stumberg for the inclusion in DM-VIO (http://vision.in.tum.de/dm-vio).
*
* Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"

#include <opencv2/highgui/highgui.hpp>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh)
        : thisToNext_aff(0, 0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}


bool CoarseInitializer::trackFrame(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	newFrame = newFrameHessian;

    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = {5,5,10,30,50};

#ifndef USE_ZNCC
	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
#else
    alphaK = 2.5*2.5;//2.5*2.5;//0.0150*0.0150;//0.005*0.005;//0.010*0.010;//0.0150*0.0150;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
#endif
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

	if(!snapped)
	{
		thisToNext.translation().setZero();
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}


	SE3 refToNew_current = thisToNext;

	AffLight refToNew_aff_current = thisToNext_aff;

	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.


	Vec3f latestRes = Vec3f::Zero();
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{

		if(lvl<pyrLevelsUsed-1)
			propagateDown(lvl+1);

		Mat88f H,Hsc; Vec8f b,bsc;
		resetPoints(lvl);
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;

		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

		int iteration=0;
		while(true)
		{
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));


            Vec8f inc;
            SE3 refToNew_new;
            if (fixAffine)
            {
                // Note as we set the weights of rotation and translation to 1 the wM is just the identity in this case.
                inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
                                  (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                inc.tail<2>().setZero();
            } else
                inc = -(wM * (Hl.ldlt().solve(bl)));    //=-H^-1 * b.

            double incNorm = inc.norm();

            refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;

			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);


			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);

			std::cout << "resNew: " << resNew.transpose() << ", regEnergy: " << regEnergy.transpose() << std::endl;

			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);


			bool accept = eTotalOld > eTotalNew;

			printf("accept: %d, [eTotalOld / eTotalNew]: [%f / %f]\n", accept, eTotalOld, eTotalNew);
			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						incNorm);
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}

			if(accept)
			{
                printf("alphaK: %f, numPoints[lvl]: %d, resNew[1]: %f\n", alphaK, numPoints[lvl], resNew[1]);
				if(resNew[1] == alphaK*numPoints[lvl]) {
                    snapped = true;
                } else {
				}
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				optReg(lvl);
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;

			if(!(incNorm > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;

				quitOpt = true;
			}


			if(quitOpt) break;
			iteration++;
		}
		latestRes = resOld;

	}



	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i);




	frameID++;
	if(!snapped) snappedAt=0;

	if(snapped && snappedAt==0)
		snappedAt = frameID;



    debugPlot(0,wraps);



	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;


	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
{
	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	float fxl = fx[lvl];
	float fyl = fy[lvl];
    float fxli = 1/fx[lvl];
    float fyli = 1/fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];


	Accumulator11 E;
	acc9.initialize();
	E.initialize();


	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	for(int i=0;i<npts;i++)
	{
	    /// ptsl + i same as ptsl[i];
		Pnt* point = ptsl+i;

		point->maxstep = 1e10;
		if(!point->isGood)
		{
			E.updateSingle((float)(point->energy[0]));
			point->energy_new = point->energy;
			point->isGood_new = false;
			continue;
		}

        /// VecNeighbourResidualFloat
        /// dp here 0-5 is d_residual / d_SE3, 6-7 is d_residual / d_a and d_residual / d_b
        /// 6dof pose residual
        //TODO !< 用来计算Schur的 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry
        VecNRf dp0;
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        /// affine residual x 2
        VecNRf dp6;
        VecNRf dp7;

        VecNRf dd;
        VecNRf r;
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		bool isGood = true;
		float energy=0;
        Eigen::Matrix<float, 2, 6> d_uv_d_pose, d_uv_d_pose_inverse_comp, d_uv_d_pose_fwd_jac;
        Eigen::Matrix<float, 2, 3> d_uv_d_pt3d, d_uv_host_d_n_host, d_uv_target_d_x_target_scaled;
        Eigen::Matrix<float, 3, 6> d_pt3d_d_pose, d_pt3d_d_pose_inverse_comp;
        Eigen::Matrix<float,3,3> Rot;
        Eigen::Matrix<float,3,1> trans;
        Vec4f d_C_x, d_C_y;
        float drescale;
        int show_cnt = 0;


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

        int count = 0;
        for(int idx=0;idx<patternNum;idx++) {
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];

            //! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
            /// Pj = [x y z]
            /// Pj' * Z = Pj
            /// Pj' = [x/z y/z 1]
            Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;


            //Vec3f pt_scaled = refToNew.rotationMatrix().cast<float>() * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;

            Vec3f n_host = Vec3f(
                    (point->u + dx - cxl) * fxli,
                    (point->v + dy - cyl) * fyli,
                    1);
            Rot = refToNew.rotationMatrix().cast<float>();
            trans = refToNew.translation().cast<float>();
            Vec3f X_target_scaled = refToNew.rotationMatrix().cast<float>() * n_host +
                                    refToNew.translation().cast<float>() * point->idepth_new;
            Mat33f X_target_scaled_skew;
            X_target_scaled_skew << (0), -X_target_scaled(2), X_target_scaled(1),
                    X_target_scaled(2), (0), -X_target_scaled(0),
                    -X_target_scaled(1), X_target_scaled(0), (0);
            d_uv_d_pt3d << fxl / X_target_scaled(2), 0, -fxl * X_target_scaled(0) / X_target_scaled(2) /
                                                        X_target_scaled(2),
                    0, fyl / X_target_scaled(2), -fyl * X_target_scaled(1) / X_target_scaled(2) / X_target_scaled(2);
            d_pt3d_d_pose.leftCols(3) = point->idepth_new * Mat33f::Identity();
            d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;
            d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;
            d_uv_host_d_n_host << fxl, 0, -fxl * n_host(0),
                    0, fyl, -fyl * n_host(1);
            d_uv_d_pose_inverse_comp =
                    d_uv_host_d_n_host * refToNew.rotationMatrix().cast<float>().transpose() * d_pt3d_d_pose;

            Vec2f d_uv_d_d_inverse_comp;
            d_uv_d_d_inverse_comp = d_uv_host_d_n_host * refToNew.rotationMatrix().cast<float>().transpose() *
                                    refToNew.translation().cast<float>();



            d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
            d_uv_d_pose_fwd_jac = d_uv_target_d_x_target_scaled * d_pt3d_d_pose;

            Vec2f d_uv_d_d_fwd_jac;
            d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled * refToNew.translation().cast<float>();
            // printf("xyz: %f\n",pt[2]);
            // 归一化坐标 Pj



            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;

            float new_idepth = point->idepth_new/pt[2];
            if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
            {
//                isGood = false;
//                break;
                continue;
            }
            Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
            Vec3f hostColor = getInterpolatedElement33(colorRef, point->u+dx, point->v+dy, wl);
            //Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);
            float host_value_corrected = (float)(r2new_aff[0] * hostColor[0] + r2new_aff[1]);

            //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
            float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

            if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
            {
//                isGood = false;
//                break;
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
        if(patch_num != 0) {

            host_val_mean = host_info.col(0).sum() / patch_num;
            target_val_mean = target_info.col(0).sum() / patch_num;

            ones.conservativeResize(patch_num, 1);
            ones.setOnes();
#ifdef USE_ZNCC
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
            host_info.col(0) *= setting_variableScale;
            target_info.col(0) *= setting_variableScale;
#else
            grad_new_host = host_info.rightCols(2);        // "new" gradient: 8x2
            grad_new_target = target_info.rightCols(2);  // "new" gradient: 8x2
#endif
//            std::cout << "init, grad_new_host: \n" << grad_new_host << std::endl;
//            std::cout << "init, grad_new_target: \n" << grad_new_target << std::endl;
//
//            std::cout << "init, grad_old_host: \n" << host_info.rightCols(2) << std::endl;
//            std::cout << "init, grad_old_target: \n" << target_info.rightCols(2) << std::endl;
        }

        int cnt = 0;
		for(int idx=0;idx<patternNum;idx++)
		{
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];

            //! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
            /// Pj = [x y z]
            /// Pj' * Z = Pj
            /// Pj' = [x/z y/z 1]
			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;


            //Vec3f pt_scaled = refToNew.rotationMatrix().cast<float>() * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;

            Vec3f n_host = Vec3f(
                    (point->u+dx-cxl)*fxli,
                    (point->v+dy-cyl)*fyli,
                    1);
            Rot = refToNew.rotationMatrix().cast<float>();
            trans = refToNew.translation().cast<float>();
            Vec3f X_target_scaled = refToNew.rotationMatrix().cast<float>() * n_host + refToNew.translation().cast<float>()*point->idepth_new;
            Mat33f X_target_scaled_skew;
            X_target_scaled_skew <<  (0), -X_target_scaled(2),  X_target_scaled(1),
                    X_target_scaled(2),    (0), -X_target_scaled(0),
                    -X_target_scaled(1),  X_target_scaled(0),     (0);
            d_uv_d_pt3d<<fxl/X_target_scaled(2),0,-fxl*X_target_scaled(0)/X_target_scaled(2)/X_target_scaled(2),
                    0, fyl/X_target_scaled(2),-fyl*X_target_scaled(1)/X_target_scaled(2)/X_target_scaled(2);
            d_pt3d_d_pose.leftCols(3) = point->idepth_new*Mat33f::Identity();
            d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;
            d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;
            d_uv_host_d_n_host << fxl,           0,            -fxl*n_host(0),
                    0,              fyl,      -fyl*n_host(1);
            d_uv_d_pose_inverse_comp = d_uv_host_d_n_host * refToNew.rotationMatrix().cast<float>().transpose() * d_pt3d_d_pose;

            Vec2f d_uv_d_d_inverse_comp;
            d_uv_d_d_inverse_comp = d_uv_host_d_n_host * refToNew.rotationMatrix().cast<float>().transpose() * refToNew.translation().cast<float>();




            d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
            d_uv_d_pose_fwd_jac = d_uv_target_d_x_target_scaled * d_pt3d_d_pose;

            Vec2f d_uv_d_d_fwd_jac;
            d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled * refToNew.translation().cast<float>();

            // printf("xyz: %f\n",pt[2]);
            // 归一化坐标 Pj



			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
            // dpi/pz'
            /// 这2个相除应该没什么几何含义，相当于rou1/rou2吧，为了计算雅可比的
            /// idepth_new is the estimated z in host frame, and pt[2] is projected z in new frame.
			float new_idepth = point->idepth_new/pt[2];
#if 0
            drescale = 1/X_target_scaled(2);
            d_C_x[2] = drescale*(Rot(2,0)*u-Rot(0,0));
            d_C_x[3] = fxl * drescale*(Rot(2,1)*u-Rot(0,1)) * fyli;
            //TODO KliP: host帧归一化坐标
            d_C_x[0] = n_host[0]*d_C_x[2];
            d_C_x[1] = n_host[1]*d_C_x[3];

            d_C_y[2] = fyl * drescale*(Rot(2,0)*v-Rot(1,0)) * fxli;
            d_C_y[3] = drescale*(Rot(2,1)*v-Rot(1,1));
            d_C_y[0] = n_host[0]*d_C_y[2];
            d_C_y[1] = n_host[1]*d_C_y[3];

            d_C_x[0] = (d_C_x[0]+u);//TODO d_u2_d_fx
            d_C_x[1] *= 1;
            d_C_x[2] = (d_C_x[2]+1);//TODO d_u2_d_cx
            d_C_x[3] *= 1;

            d_C_y[0] *= 1;
            d_C_y[1] = (d_C_y[1]+v)*1;
            d_C_y[2] *= 1;
            d_C_y[3] = (d_C_y[3]+1)*1;
#endif
			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}

			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
            Vec3f hostColor = getInterpolatedElement33(colorRef, point->u+dx, point->v+dy, wl);
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}

#ifndef USE_ZNCC
			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);
#else
            float residual_bak = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
            float residual = 1 * (target_info(cnt, 0) - host_info(cnt, 0));
            //printf("residual: %f\n", residual);

            if (std::isnan(residual)) {
                isGood = false;
                break;
            }

            //printf("residual: %f\n", residual);
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw *residual*residual*(2-hw);
#endif



			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			if(hw < 1) hw = sqrtf(hw);
#ifndef USE_ZNCC
			float dxInterp = hw*hitColor[1]*fxl;
			float dyInterp = hw*hitColor[2]*fyl;
#else
            float dxInterp = hw*grad_new_target(cnt, 0)*fxl;
            float dyInterp = hw*grad_new_target(cnt, 1)*fyl;
#endif
            //TODO* 残差对 j(新状态) 位姿求导, 6

            Eigen::Matrix<float,3,6> show;
            show(0,0) = new_idepth*dxInterp;
            show(0,1) = new_idepth*dyInterp;
            show(0,2) = -new_idepth*(u*dxInterp + v*dyInterp);
            show(0,3) = -u*v*dxInterp - (1+v*v)*dyInterp;
            show(0,4) = (1+u*u)*dxInterp + u*v*dyInterp;
            show(0,5) = -v*dxInterp + u*dyInterp;

#ifndef USE_ZNCC
            show.row(1) = show.row(0) - hw*Vec2f(hitColor[1], hitColor[2]).transpose()*d_uv_d_pose;
            Vec6f d_res_d_pose_inverse_comp = hw*Vec2f(hostColor[1], hostColor[2]).transpose()*d_uv_d_pose_inverse_comp;
            Vec6f d_res_d_pose_fwd_jac = hw*Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1)).transpose()*d_uv_d_pose_fwd_jac;
            show.row(2) = show.row(0) - d_res_d_pose_fwd_jac.transpose();
#else
            show.row(1) = show.row(0) - hw*Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1)).transpose()*d_uv_d_pose;
            Vec6f d_res_d_pose_inverse_comp = hw*Vec2f(grad_new_host(cnt, 0), grad_new_host(cnt, 1)).transpose()*d_uv_d_pose_inverse_comp;
            Vec6f d_res_d_pose_fwd_jac = hw*Vec2f(grad_new_host(cnt, 0), grad_new_host(cnt, 1)).transpose()*d_uv_d_pose_fwd_jac;
#endif

            Vec3f d_uv_d_idp_show;
            d_uv_d_idp_show(0) = dxInterp * dxdd  + dyInterp * dydd;
#ifndef USE_ZNCC
            d_uv_d_idp_show(1) =d_uv_d_idp_show(0) - hw*Vec2f(hitColor[1], hitColor[2]).transpose()*d_uv_d_pt3d*trans;
            //std::cout<<"d_uv_d_c_show:\n"<<d_uv_d_c_show<<std::endl;
            float d_res_d_idp_inverse_comp = hw*Vec2f(hostColor[1], hostColor[2]).transpose() * d_uv_d_d_inverse_comp;
            float d_res_d_idp_fwd_jac = hw*Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1)).transpose() * d_uv_d_d_fwd_jac;
            d_uv_d_idp_show(2) = d_uv_d_idp_show(0) - d_res_d_idp_fwd_jac;
#else
            d_uv_d_idp_show(1) =d_uv_d_idp_show(0) - hw*Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1)).transpose()*d_uv_d_pt3d*trans;
            float d_res_d_idp_inverse_comp = hw*Vec2f(grad_new_host(cnt, 0), grad_new_host(cnt, 1)).transpose() * d_uv_d_d_inverse_comp;
            float d_res_d_idp_fwd_jac = hw*Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1)).transpose() * d_uv_d_d_fwd_jac;
#endif

            if(show_cnt < -10) {
                std::cout <<"[du dv hw]: ["<<dx<< ", "<<dy<<", "<<hw<<"], tracking pose jac diff: \n" << show << std::endl;
                std::cout<<"d_uv_d_idp_show: \n"<<d_uv_d_idp_show.transpose()<<std::endl<<std::endl<<std::endl;
                show_cnt++;
            }



#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
			dp0[idx] = new_idepth*dxInterp;
			dp1[idx] = new_idepth*dyInterp;
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;
			dp5[idx] = -v*dxInterp + u*dyInterp;
#else
            dp0[idx] = d_res_d_pose_fwd_jac(0);
            dp1[idx] = d_res_d_pose_fwd_jac(1);
            dp2[idx] = d_res_d_pose_fwd_jac(2);
            dp3[idx] = d_res_d_pose_fwd_jac(3);
            dp4[idx] = d_res_d_pose_fwd_jac(4);
            dp5[idx] = d_res_d_pose_fwd_jac(5);
#endif
#else
            dp0[idx] = d_res_d_pose_inverse_comp(0);
			dp1[idx] = d_res_d_pose_inverse_comp(1);
			dp2[idx] = d_res_d_pose_inverse_comp(2);
			dp3[idx] = d_res_d_pose_inverse_comp(3);
			dp4[idx] = d_res_d_pose_inverse_comp(4);
			dp5[idx] = d_res_d_pose_inverse_comp(5);
#endif
            //TODO* 残差对光度参数求导, 2
			dp6[idx] = - hw*r2new_aff[0] * rlR;
			dp7[idx] = - hw*1;
            //TODO* 残差对 i(旧状态) 逆深度求导, 1
#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;
#else
            dd[idx] = d_res_d_idp_fwd_jac;
#endif
#else
            dd[idx] = d_res_d_idp_inverse_comp;
#endif
            //TODO* 残差 res, 1
			r[idx] = hw*residual;


            //#else
//            // Pj 对 逆深度 di 求导
//			//! 1/Pz * (tx - u*tz), u = px/pz
//			float dxdd = (t[0]-t[2]*u)/pt[2];
//			//! 1/Pz * (ty - v*tz), u = py/pz
//			float dydd = (t[1]-t[2]*v)/pt[2];
//
//			if(hw < 1) hw = sqrtf(hw); //?? 为啥开根号, 答: 鲁棒核函数等价于加权最小二乘
//			//! dxfx, dyfy
//			float dxInterp = hw*hitColor[1]*fxl;
//			float dyInterp = hw*hitColor[2]*fyl;
//			//TODO* 残差对 j(新状态) 位姿求导, 6
//			dp0[idx] = new_idepth*dxInterp; //! dpi/pz' * dxfx
//			dp1[idx] = new_idepth*dyInterp; //! dpi/pz' * dyfy
//			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp); //! -dpi/pz' * (px'/pz'*dxfx + py'/pz'*dyfy)
//			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp; //! - px'py'/pz'^2*dxfy - (1+py'^2/pz'^2)*dyfy
//			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp; //! (1+px'^2/pz'^2)*dxfx + px'py'/pz'^2*dxfy
//			dp5[idx] = -v*dxInterp + u*dyInterp; //! -py'/pz'*dxfx + px'/pz'*dyfy
//			//TODO* 残差对光度参数求导, 2
//			dp6[idx] = - hw*r2new_aff[0] * rlR; //! exp(aj-ai)*I(pi)
//			dp7[idx] = - hw*1;	//! 对 b 导
//			//TODO* 残差对 i(旧状态) 逆深度求导, 1
//			dd[idx] = dxInterp * dxdd  + dyInterp * dydd; 	//! dxfx * 1/Pz * (tx - u*tz) +　dyfy * 1/Pz * (tx - u*tz)
//            //TODO* 残差 res, 1
//			r[idx] = hw*residual; //! 残差 res
//
//#endif



            //* 像素误差对逆深度的导数，取模倒数
#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
#else
            float maxstep = 1.0f / d_uv_d_d_fwd_jac.norm();
#endif
#else
            float maxstep = 1.0f / d_uv_d_d_inverse_comp.norm();  //? 为什么这么设置
#endif
			if(maxstep < point->maxstep) point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
            //* 计算Hessian的第一行(列), 及Jr 关于逆深度那一行
            // 用来计算舒尔补
            // |H11  H12| * |x1| = |b1|
            // |H21  H22|   |x2|   |b2|
            //
            //
            //
            //
            //
            //
            //
            //
            //
			JbBuffer_new[i][0] += dp0[idx]*dd[idx];
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			JbBuffer_new[i][8] += r[idx]*dd[idx];
            //TODO 10 = 6 + 2 + 1 + 1 = pose + affine + idepth + res
            //TODO H22 hessian约等于JTJ，当变量为1维时，hessian = J^2
			JbBuffer_new[i][9] += dd[idx]*dd[idx]; /// 1/(1+sum(dd*dd))=inverse depth hessian entry, while now is just sum(dd*dd), H_{\beta \beta}
            cnt++;
		}

		if(!isGood || energy > point->outlierTH*20)
		{
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}


		// add into energy.
        /// energy = sum(weight * residual * residual * (2 - weight));
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
        // update Hessian matrix.
        // update Hessian matrix.
        // acc += dp[0] + dp[4]
        // this trick unroll the loop into 4 blocks and speed up this for loop in x86 SSE instruction set
        //dp0 * dp0 + dp0*dp1 + .. + dp0*r
        //            dp1*dp1 + .. + dp1*r
        //                      ..
        //                           r * r
        // here ((float *) (&dp0)) + i we can see that &dp0 get the address of dp0
        // convert this address into float * which occupy 4 size_of space.
        // and i is the offsets, which shift size_of 4 for each loop
        // this acc9 is aggregating inside each point, this is just summing up the pattern, it will sum the points also
		for(int i=0;i+3<patternNum;i+=4)
			acc9.updateSSE(
					_mm_load_ps(((float*)(&dp0))+i),
					_mm_load_ps(((float*)(&dp1))+i),
					_mm_load_ps(((float*)(&dp2))+i),
					_mm_load_ps(((float*)(&dp3))+i),
					_mm_load_ps(((float*)(&dp4))+i),
					_mm_load_ps(((float*)(&dp5))+i),
					_mm_load_ps(((float*)(&dp6))+i),
					_mm_load_ps(((float*)(&dp7))+i),
					_mm_load_ps(((float*)(&r))+i));

        // 加0, 4, 8后面多余的值, 因为SSE2是以128为单位相加, 多余的单独加
        // ((patternNum >> 2) << 2) this will align the patternNum to be n*4 which is required by SSE.
        // ((8 >> 2) << 2) = 8 so, this will jump this for loop directly.
        // this loop is prepared for the patternNum more than 8
        // for example if it's 10, then ((10 >> 2) << 2) = 8. i then loop start from 8 to 10
        //TODO 这不是一路自加，pt+=4表示移位，其实算的就是Jt * J和-Jt*b
        // H += H_i
        // H = Jt * J,  b = -Jt * b
        // 老老实实对H和b进行累加
		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			acc9.updateSingle(
					(float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
					(float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
					(float)r[i]);


	}

	E.finish();
	acc9.finish();


	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
		{
			E.updateSingle((float)(point->energy[1]));
		}
		else
		{
            /// res = 1 - idepth_new
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


	// compute alpha opt.
	float alphaOpt;
	if(alphaEnergy > alphaK*npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW;
	}


	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

        /// hessian约等于JTJ，当变量为1维时，hessian = J^2
        /// 1/(1+sum(dd*dd))=inverse depth hessian entry, while now is just sum(dd*dd), H_{\beta \beta}
		point->lastHessian_new = JbBuffer_new[i][9];

        //? 这又是啥??? 对逆深度的值进行加权? 深度值归一化?
        // 前面Energe加上了（d-1)*(d-1), 所以dd = 1， r += (d-1)
        //TODO 因为初始化阶段idepth的目标值是1，所以idepth的残差就是 weight * (1-idepth)^2了;
        // res = (point->idepth_new - 1); J = 1
        // b = Jt*res = 1 * res = res;
		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		if(alphaOpt==0)
		{
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

        /// refer to the equation (17) in DSO, here JbBuffer_new[i][9] is H^{-1}_{\beta \beta}
		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
        //* 9做权重, 计算的是舒尔补项!
        //! dp*dd*(dd^2)^-1*dd*dp

        // H11_ = H11 - H12 * H22^-1 * H12.t ;
        // b1_ = b1 - H12 * H22^-1 * b2 ;
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;

    //??? 啥意思
    // t*t*ntps
    // 给 t 对应的Hessian, 对角线加上一个数, b也加上
    //TODO 加权重
	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;


	// Add zero prior to translation.
    // setting_weightZeroPriorDSOInitY is the squared weight of the prior residual.
    H_out(1, 1) += setting_weightZeroPriorDSOInitY;
    b_out(1) += setting_weightZeroPriorDSOInitY * refToNew.translation().y();

    H_out(0, 0) += setting_weightZeroPriorDSOInitX;
    b_out(0) += setting_weightZeroPriorDSOInitX * refToNew.translation().x();

    /// E.A is actually equal to residual + (1-d)^2, this (1-d)^2 came from all good tracking points
    /// return vector is total error, alphaEnergy, and the good tracking number point
    /// E.num will be npts + num_good_points
    /// since the E += energy is for every point, but E += (d-1)^2 is only when the point is good
    /// so E.A = sum_npts(energy) + sum_isgood((d-1)^2)
    // 能量值, ? , 使用的点的个数
	return Vec3f(E.A, alphaEnergy ,E.num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	if(!snapped)
	{
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		if(nnn > 2)
		{
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}



void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1);
}

void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target

	int nptst= numPoints[srcLvl-1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl-1];

	for(int i=0;i<nptst;i++)
	{
		Pnt* point = ptst+i;
		Pnt* parent = ptss+point->parent;

		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		if(!point->isGood)
		{
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0;
		}
		else
		{
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	optReg(srcLvl-1);
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}
void CoarseInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

	makeK(HCalib);
	firstFrame = newFrameHessian;

	PixelSelector sel(w[0],h[0]);

	float* statusMap = new float[w[0]*h[0]];
	bool* statusMapB = new bool[w[0]*h[0]];

	float densities[] = {0.03,0.05,0.15,0.5,1};
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		sel.currentPotential = 3;
		int npts;
		if(lvl == 0)
			npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
		else
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);



		if(points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		Pnt* pl = points[lvl];
		int nl = 0;
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
		{
			//if(x==2) printf("y=%d!\n",y);
			if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
			{
				//assert(patternNum==9);
				pl[nl].u = x+0.1;
				pl[nl].v = y+0.1;
				pl[nl].idepth = 1;
				pl[nl].iR = 1;
				pl[nl].isGood=true;
				pl[nl].energy.setZero();
				pl[nl].lastHessian=0;
				pl[nl].lastHessian_new=0;
				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
				float sumGrad2=0;
				for(int idx=0;idx<patternNum;idx++)
				{
					int dx = patternP[idx][0];
					int dy = patternP[idx][1];
					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
					sumGrad2 += absgrad;
				}

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
//

				pl[nl].outlierTH = patternNum*setting_outlierTH;



				nl++;
				assert(nl <= npts);
			}
		}


		numPoints[lvl]=nl;
	}
	delete[] statusMap;
	delete[] statusMapB;

	makeNN();

	thisToNext=SE3();
	snapped = false;
	frameID = snappedAt = 0;

	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();

}

void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;


		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			float snd=0, sn=0;
			for(int n = 0;n<10;n++)
			{
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if(sn > 0)
			{
				pts[i].isGood=true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;


		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}




void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			for(int k=0;k<nn;k++)
			{
				pts[i].neighbours[myidx]=ret_index[k];
				float df = expf(-ret_dist[k]*NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;


			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f);
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}


/**
 * This file is part of DSO, written by Jakob Engel.
 * It has been modified by Lukas von Stumberg for the inclusion in DM-VIO
 * (http://vision.in.tum.de/dm-vio).
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

#pragma once

#include <cmath>
#include <string.h>
#include <string>

namespace dso {
#define SOLVER_SVD (int)1
#define SOLVER_ORTHOGONALIZE_SYSTEM (int)2
#define SOLVER_ORTHOGONALIZE_POINTMARG (int)4
#define SOLVER_ORTHOGONALIZE_FULL (int)8
#define SOLVER_SVD_CUT7 (int)16
#define SOLVER_REMOVE_POSEPRIOR (int)32
#define SOLVER_USE_GN (int)64
#define SOLVER_FIX_LAMBDA (int)128
#define SOLVER_ORTHOGONALIZE_X (int)256
#define SOLVER_MOMENTUM (int)512
#define SOLVER_STEPMOMENTUM (int)1024
#define SOLVER_ORTHOGONALIZE_X_LATER (int)2048

// ============== PARAMETERS TO BE DECIDED ON COMPILE TIME =================
#define USE_MULTI_CAM
#ifdef USE_MULTI_CAM
//#define FIX_ZERO_TRANS_IN_INIT
//#define DISABLE_CROSS_CID_ALIGN
#ifndef DISABLE_CROSS_CID_ALIGN
//#define USE_HACK
#ifndef define USE_HACK
#define USE_BUNDLED_RES
#endif
#endif
#define kCameraNumUsed 4
#else
#define kCameraNumUsed 1
#endif
#define kCameraNum 4

#define kImageWidth 640
#define kImageHeight 480
// #define PYR_LEVELS 6 // 3
//#define USE_INVERSE_COMPOSITIONAL
#define USE_EDGE_ALIGN
#ifndef USE_EDGE_ALIGN
#define PYR_LEVELS 6  // 3
//#define USE_ZNCC
#else
#define PYR_LEVELS 3  // at least 3 levels
//#define USE_CENTER_PIXEL_ONLY
#endif

#ifdef USE_ZNCC
#define USE_ZNCC_WEIGHT
#endif

#define SHOW_ALIGN_FRAME
//#define SAVE_IMAGES

#define MIN_PATH_LENGTH_IN_ED 30

// ============== DEBUG: NaN detection switches =================
// Uncomment to enable NaN/Inf/condition-number logging throughout the optimizer
#define DEBUG_NAN

#ifdef DEBUG_NAN
#include <cmath>
#include <cstdio>

// Check scalar for NaN/Inf
#define NAN_CHECK_SCALAR(val, name)                                        \
  do {                                                                     \
    if (!std::isfinite(val)) {                                             \
      printf("[NAN_DETECT] %s = %g (NOT finite!)\n", name, (double)(val)); \
    }                                                                      \
  } while (0)

// Check Eigen vector/matrix for NaN/Inf (linear access)
#define NAN_CHECK_EIGEN(mat, name)                                                                           \
  do {                                                                                                       \
    int _nf = 0;                                                                                             \
    for (int _i = 0; _i < (mat).size(); _i++) {                                                              \
      if (!std::isfinite((double)((mat)(_i)))) _nf++;                                                        \
    }                                                                                                        \
    if (_nf > 0) {                                                                                           \
      printf("[NAN_DETECT] %s: %d/%d non-finite, norm=%g\n", name, _nf, (mat).size(), (double)(mat).norm()); \
    }                                                                                                        \
  } while (0)

// Log condition number from min/max singular values
#define NAN_LOG_COND(min_sv, max_sv, name)                                                                  \
  do {                                                                                                      \
    double _c = (min_sv > 0 && std::isfinite(min_sv) && std::isfinite(max_sv))                              \
                    ? ((double)(max_sv) / (double)(min_sv))                                                 \
                    : ((std::isfinite(max_sv) && max_sv > 0) ? 1.0 / 0.0 : 0.0 / 0.0);                      \
    printf("[COND_NUM] %s: min_sv=%g, max_sv=%g, cond=%g\n", name, (double)(min_sv), (double)(max_sv), _c); \
  } while (0)

#define NAN_PRINT(fmt, ...) printf("[NAN_DBG] " fmt, ##__VA_ARGS__)

// Compute condition number from eigenvalues (symmetric matrix only)
#define NAN_CHECK_COND(mat, name)                                                                                \
  do {                                                                                                           \
    if ((mat).rows() > 0 && (mat).cols() > 0) {                                                                  \
      Eigen::SelfAdjointEigenSolver<MatXX> _eig(mat);                                                            \
      if (_eig.info() == Eigen::Success) {                                                                       \
        double _minEv = _eig.eigenvalues().minCoeff();                                                           \
        double _maxEv = _eig.eigenvalues().maxCoeff();                                                           \
        double _cond = (std::abs(_minEv) > 1e-12) ? (_maxEv / _minEv) : std::numeric_limits<double>::infinity(); \
        printf("[COND_NUM] %s: min_ev=%g, max_ev=%g, cond=%g\n", name, _minEv, _maxEv, _cond);                   \
        int _nNeg = (_eig.eigenvalues().array() < 0).count();                                                    \
        if (_nNeg > 0) printf("[COND_NUM] %s: WARNING %d negative eigenvalues!\n", name, _nNeg);                 \
      }                                                                                                          \
    }                                                                                                            \
  } while (0)

#else
#define NAN_CHECK_SCALAR(val, name) ((void)0)
#define NAN_CHECK_EIGEN(mat, name) ((void)0)
#define NAN_LOG_COND(min_sv, max_sv, name) ((void)0)
#define NAN_CHECK_COND(mat, name) ((void)0)
#define NAN_PRINT(fmt, ...) ((void)0)
#endif

extern float setting_variableScale;
extern float setting_variableScale_edge;
// extern float setting_variableScale_edge_tracker;
// extern float setting_variableScale_edge_seed;

extern int setting_kfNumWithAffineFixed;

extern int setting_pyrLvlWithAffineFixed;

extern int pyrLevelsUsed;

extern bool setting_useIMU;
extern bool setting_useGTSAMIntegration;
extern double setting_weightZeroPriorDSOInitY;
extern double setting_weightZeroPriorDSOInitX;
extern double setting_forceNoKFTranslationThresh;

extern double setting_maxTimeBetweenKeyframes;

extern double setting_minFramesBetweenKeyframes;

extern float setting_minIdepth;

extern float setting_keyframesPerSecond;
extern bool setting_realTimeMaxKF;
extern float setting_maxShiftWeightT;
extern float setting_maxShiftWeightR;
extern float setting_maxShiftWeightRT;
extern float setting_maxAffineWeight;
extern float setting_kfGlobalWeight;

extern float setting_idepthFixPrior;
extern float setting_idepthFixPriorMargFac;
extern float setting_initialRotPrior;
extern float setting_initialTransPrior;
extern float setting_initialAffBPrior;
extern float setting_initialAffAPrior;
extern float setting_initialCalibHessian;

extern int setting_solverMode;
extern double setting_solverModeDelta;

extern float setting_minIdepthH_act;
extern float setting_minIdepthH_marg;

extern float setting_maxIdepth;
extern float setting_maxPixSearch;
extern float setting_desiredImmatureDensity;  // done
extern float setting_desiredPointDensity;     // done
extern float setting_minPointsRemaining;
extern float setting_maxLogAffFacInWindow;
extern int setting_minFrames;
extern int setting_maxFrames;
extern int setting_minFrameAge;
extern int setting_maxOptIterations;
extern int setting_minOptIterations;
extern float setting_thOptIterations;
// extern float setting_outlierTH;
extern float setting_outlierTHSumComponent;

extern int setting_pattern;
extern float setting_margWeightFac;
extern int setting_GNItsOnPointActivation;

extern float setting_minTraceQuality;
extern int setting_minTraceTestRadius;
extern float setting_reTrackThreshold;

extern int setting_minGoodActiveResForMarg;
extern int setting_minGoodResForMarg;
extern int setting_minInlierVotesForMarg;

extern int setting_photometricCalibration;
extern bool setting_useExposure;
extern float setting_affineOptModeA;
extern float setting_affineOptModeB;
extern float setting_affineOptModeA_huberTH;
extern float setting_affineOptModeB_huberTH;
extern int setting_gammaWeightsPixelSelect;

extern bool setting_forceAceptStep;

extern float setting_outlierTH_epi_trace_on;
extern float setting_outlierTH_epi_linearize;
extern float setting_outlierTH_zncc_angle_epi_trace_on;
extern float setting_outlierTH_zncc_angle_epi_linearize;
extern float setting_outlierTH_init;
extern float setting_outlierTH_zncc_init;
extern float setting_outlierTH_zncc_angle_init;
extern float setting_outlierTH_tracker;
extern float setting_outlierTH_loose_tracker;
extern float setting_outlierTH_zncc_tracker;
extern float setting_outlierTH_LBA;
extern float setting_outlierTH_zncc_LBA;
extern float setting_outlierTH_zncc_angle_LBA;

// extern float setting_huberTH;
// extern float setting_huberTH_loose;
extern float setting_huberTH_epi_trace_on;
extern float setting_huberTH_epi_linearize;
extern float setting_huberTH_init;
extern float setting_huberTH_zncc_init;
extern float setting_huberTH_zncc_angle_init;
extern float setting_huberTH_tracker;
extern float setting_huberTH_loose_tracker;
extern float setting_huberTH_zncc_tracker;
extern float setting_huberTH_LBA;
extern float setting_huberTH_zncc_LBA;
extern float setting_huberTH_zncc_angle_LBA;

extern float setting_energyTH_epi_trace_on;
extern float setting_energyTH_epi_linearize;
extern float setting_energyTH_init;
extern float setting_energyTH_zncc_init;
extern float setting_energyTH_zncc_angle_init;
extern float setting_energyTH_tracker;
extern float setting_energyTH_loose_tracker;
extern float setting_energyTH_zncc_tracker;
extern float setting_energyTH_LBA;
extern float setting_energyTH_zncc_LBA;
extern float setting_energyTH_zncc_angle_LBA;

extern bool setting_logStuff;
extern float benchmarkSetting_fxfyfac;
extern int benchmarkSetting_width;
extern int benchmarkSetting_height;
extern float benchmark_varNoise;
extern float benchmark_varBlurNoise;
extern int benchmark_noiseGridsize;
extern float benchmark_initializerSlackFactor;

extern float setting_frameEnergyTHConstWeight;
extern float setting_frameEnergyTHN;

extern float setting_frameEnergyTHFacMedian;
extern float setting_overallEnergyTHWeight;
extern float setting_coarseCutoffTH;
extern float setting_coarseCutoffTH_loose;

extern float setting_dtCutoffTH;
extern float setting_dtCutoffTH_loose;

extern float setting_minGradHistCut;
extern float setting_minGradHistAdd;
extern float setting_gradDownweightPerLevel;
extern bool setting_selectDirectionDistribution;

extern float setting_trace_stepsize;
extern int setting_trace_GNIterations;
extern float setting_trace_GNThreshold;
extern float setting_trace_extraSlackOnTH;
extern float setting_trace_slackInterval;
extern float setting_trace_minImprovementFactor;

extern bool setting_render_displayCoarseTrackingFull;
extern bool setting_render_renderWindowFrames;
extern bool setting_render_plotTrackingFull;
extern bool setting_render_display3D;
extern bool setting_render_displayResidual;
extern bool setting_render_displayVideo;
extern bool setting_render_displayDepth;

extern bool setting_fullResetRequested;

extern bool setting_debugout_runquiet;

extern bool disableAllDisplay;

extern bool debugSaveImages;

extern int sparsityFactor;
extern bool goStepByStep;
extern bool plotStereoImages;
extern bool multiThreading;

extern float freeDebugParam1;
extern float freeDebugParam2;
extern float freeDebugParam3;
extern float freeDebugParam4;
extern float freeDebugParam5;

void handleKey(char k);
#ifndef USE_EDGE_ALIGN
constexpr float pattern_scale = 2;  // 1;
constexpr float pattern_scale_extra_edge = 1.0f;
#else
constexpr float pattern_scale = 2.0f;
constexpr float pattern_scale_extra_edge = 0.1;  // 0.0f;
#endif
constexpr int pattern_index = 8;

constexpr float pattern_scale_seed_init = 2.0f;  // don't change
constexpr int pattern_index_seed = 9;

#ifndef USE_EDGE_ALIGN
constexpr float pattern_scale_seed_point_opt = pattern_scale_seed_init;
#else
constexpr float pattern_scale_seed_point_opt = 1.f;
#endif

extern float staticPattern[12][40][2];
constexpr int staticPatternNum[12] = {1, 5, 5, 9, 9, 13, 25, 21, 8, 24 /*25*/, 8, 8};
constexpr int staticPatternPadding[12] = {
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    2 * (int)(pattern_scale + 1.f),
    4 * (int)(pattern_scale_seed_init > pattern_scale_seed_point_opt ? pattern_scale_seed_init
                                                                     : pattern_scale_seed_point_opt + 1.f),
    2 * (int)(pattern_scale + 1.f),
    2 * (int)(pattern_scale + 1.f)};

// extern int staticPatternNum[10][1];
// extern int staticPatternPadding[10];

#define patternNum staticPatternNum[pattern_index]
#define patternP staticPattern[pattern_index]
#define patternPadding staticPatternPadding[pattern_index]

#define patternNumSeed staticPatternNum[pattern_index_seed]
#define patternPSeed staticPattern[pattern_index_seed]
#define patternPaddingSeed                                                          \
  ((staticPatternPadding[pattern_index_seed] > staticPatternPadding[pattern_index]) \
       ? staticPatternPadding[pattern_index_seed]                                   \
       : staticPatternPadding[pattern_index])

//
//#define patternNum 8
//#define patternP staticPattern[8]
//#define patternPadding 2
constexpr int cannyThreshold1 = 60;
constexpr int cannyThreshold2 = 90;

constexpr bool adaptiveCannyThreshold = true;

#if defined(USE_EDGE_ALIGN) || defined(USE_ZNCC)
#define eachErrDim 2
#else
#define eachErrDim 1
#endif
}  // namespace dso

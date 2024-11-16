#pragma once
// #include "../utils/orca_log.h"
namespace dso {
//#define LOG_Calib_DEBUG(...) yvr::LOG_Calib.Debug(__VA_ARGS__);
//#define LOG_Calib_INFO(...) yvr::LOG_Calib.Info(__VA_ARGS__);
//#define LOG_Calib_WARN(...) yvr::LOG_Calib.Warn(__VA_ARGS__);
//#define LOG_Calib_ERROR(...) yvr::LOG_Calib.Error(__VA_ARGS__);

#define USE_INTR_ROT_TRANS_FACTOR
#define USE_DIST_TO_PLANE_ERROR_IN_INTR_ROT_FACTOR
#define USE_2_DOF_GRAVITY_FACTOR

#define USE_HOMO_WARP
#ifdef USE_HOMO_WARP
#define DO_2ND_ROUND_CORNER_OPT
#endif

#define USE_HOMO_WARP_MARKER_MAP
#ifdef USE_HOMO_WARP_MARKER_MAP
#define WARP_FROM_DISTORTED_IMAGE_MARKER_MAP
#define DO_2ND_ROUND_CORNER_OPT_MARKER_MAP
#endif

#define USE_PIXEL_RES_SINGLE_CAM
#define USE_PIXEL_RES_MULTI_CAM

#define USE_LOCAL_SIZE_5_FOR_5DOF_TRIFOCAL_FACTOR

#define USE_DIST_2_PLANE_ERROR_FOR_TRIFOCAL_LINE_FACTOR

#define USE_HYBRID_POSE_NODE

#define _DETECT_LINES_
#ifdef _DETECT_LINES_
#define _SHOW_LINE_DETECT_RES_
#endif

//#define USE_SELF_PROJECTION_FACTOR
//#define SHOW_VALID_PROJECTION

//#define USE_BEARING_IN_VI_CALIB

//#define SLOW_VERSION

//#define __SHOW_CALIB_BOARDS__

//#define SHOW_LOG_IN_SELF_OPT

} // namespace dso
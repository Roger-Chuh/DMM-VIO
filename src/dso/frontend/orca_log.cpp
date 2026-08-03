#include "orca_log.h"

namespace dso {
const std::string calib_tag = "CalibLog: ";
ILOG LOG_Calib = ILOG(true, GLOBAL_LOG_LEVEL, calib_tag);

const std::string frontend_tag = "FrontEndLog: ";
ILOG LOG_FrontEnd = ILOG(true, GLOBAL_LOG_LEVEL, frontend_tag);

const std::string single_cam_multiboard_tag = "SCMBTLOG: ";
ILOG LOG_SCMBT = ILOG(true, GLOBAL_LOG_LEVEL, single_cam_multiboard_tag);

}  // namespace dso

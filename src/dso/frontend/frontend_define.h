#pragma once
#include "orca_log.h"
namespace dso {
#define LOG_Front_DEBUG(...) dso::LOG_FrontEnd.Debug(__VA_ARGS__);
#define LOG_FRONT_INFO(...) dso::LOG_FrontEnd.Info(__VA_ARGS__);
#define LOG_FRONT_WARN(...) dso::LOG_FrontEnd.Warn(__VA_ARGS__);
#define LOG_FRONT_ERROR(...) dso::LOG_FrontEnd.Error(__VA_ARGS__);

}  // namespace dso
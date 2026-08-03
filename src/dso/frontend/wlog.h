#pragma once
#include "orca_log.h"

#define EXTRACT_FILENAME(path) (strrchr(path, '/') ? strrchr(path, '/') + 1 : path)

#define LOG_COLOR_TRACE "\x1b[37m"  // White for trace, some terminals default
#define LOG_COLOR_DEBUG "\x1b[36m"  // Cyan for debug
#define LOG_COLOR_INFO "\x1b[32m"   // Green for info
#define LOG_COLOR_WARN "\x1b[33m"   // Yellow for warning
#define LOG_COLOR_ERROR "\x1b[31m"  // Red for error
#define LOG_COLOR_RESET "\x1b[0m"

// YLOG_TRACE macro for trace logs.
// using trace to replace Info whatever - -
#define YLOG_TRACE(fmt, ...)                                                                                     \
  dso::LOG_SCMBT.Debug("%s%s:%d [trace] %s: " fmt "\n%s", LOG_COLOR_TRACE, EXTRACT_FILENAME(__FILE__), __LINE__, \
                       __func__, ##__VA_ARGS__, LOG_COLOR_RESET);

// YLOG_DEBUG macro for debug logs.
#define YLOG_DEBUG(fmt, ...)                                                                                     \
  dso::LOG_SCMBT.Debug("%s%s:%d [debug] %s: " fmt "\n%s", LOG_COLOR_DEBUG, EXTRACT_FILENAME(__FILE__), __LINE__, \
                       __func__, ##__VA_ARGS__, LOG_COLOR_RESET);

// YLOG_INFO macro for info logs.
#define YLOG_INFO(fmt, ...)                                                                                   \
  dso::LOG_SCMBT.Info("%s%s:%d [info] %s: " fmt "\n%s", LOG_COLOR_INFO, EXTRACT_FILENAME(__FILE__), __LINE__, \
                      __func__, ##__VA_ARGS__, LOG_COLOR_RESET);

// YLOG_WARN macro for warning logs.
#define YLOG_WARN(fmt, ...)                                                                                   \
  dso::LOG_SCMBT.Warn("%s%s:%d [warn] %s: " fmt "\n%s", LOG_COLOR_WARN, EXTRACT_FILENAME(__FILE__), __LINE__, \
                      __func__, ##__VA_ARGS__, LOG_COLOR_RESET);

// YLOG_ERROR macro for error logs.
#define YLOG_ERROR(fmt, ...)                                                                                     \
  dso::LOG_SCMBT.Error("%s%s:%d [error] %s: " fmt "\n%s", LOG_COLOR_ERROR, EXTRACT_FILENAME(__FILE__), __LINE__, \
                       __func__, ##__VA_ARGS__, LOG_COLOR_RESET);
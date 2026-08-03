#pragma once

#ifdef USING_NANOLOG
#include "nanolog.hpp"
#endif

#ifdef __ANDROID__
#include "yvr_ramlog/yvr_ramlog.h"
#include <android/log.h>
#include <sys/system_properties.h>
#endif

#include <sstream>
#include <string>

#ifdef RELEASE_VERSION
#define GLOBAL_LOG_LEVEL Warn_
#else
#define GLOBAL_LOG_LEVEL Default_
#endif
#include <stdarg.h>
#include <stdio.h>

namespace dso {

enum LogLevel {
  Default_ = 1,
  Debug_ = 1,
  Info_ = 2,
  Warn_ = 3,
  Error_ = 4,
};

class ILOG {
 public:
  ILOG(bool output, int log_level, const std::string& tag)
      : output_log(output), level(log_level), tag(tag), return_str(false) {}

  // Debug log, 既写到内存里 也输入到logcat
  std::string Debug(const char* format, ...) {
    if (!output_log || level > Debug_) {
      return "";
    }

    int size = 65535;
    size -= tag.size();
    char buffer[size];
    va_list ap;
    va_start(ap, format);
    char buffFormat[size];
    int nformatData = vsnprintf(buffFormat, sizeof(buffFormat), (const char*)format, ap);
    int n = snprintf(buffer, sizeof(buffer) - 1, "%s", buffFormat);
    va_end(ap);

    if (nformatData < 0 || n < 0) {
      return "";
    }

    if (return_str) {
      return std::string(tag + " " + buffer);
    } else {
#ifdef __ANDROID__
      YVR_RAM_PRINT_LOG_DEBUG("%s %s", tag.c_str(), buffer)
#else
      printf("%s %s", tag.c_str(), buffer);
      fflush(stdout);
#endif
      return "";
    }
  }

  // Info log, 既写到内存里 也输入到logcat
  std::string Info(const char* format, ...) {
    if (!output_log || level > Info_) {
      return "";
    }

    int size = 65535;
    size -= tag.size();
    char buffer[size];
    va_list ap;
    va_start(ap, format);
    char buffFormat[size];
    int nformatData = vsnprintf(buffFormat, sizeof(buffFormat), (const char*)format, ap);
    int n = snprintf(buffer, sizeof(buffer) - 1, "%s", buffFormat);
    va_end(ap);

    if (nformatData < 0 || n < 0) {
      return "";
    }

    if (return_str) {
      return std::string(tag + " " + buffer);
    } else {
#ifdef USING_NANOLOG
      NANO_LOG_INFO << tag.c_str() << " " << buffer;
#else

#ifdef __ANDROID__
      YVR_RAM_PRINT_LOG_INFO("%s %s", tag.c_str(), buffer)
#else
      printf("%s %s", tag.c_str(), buffer);
      fflush(stdout);
#endif
#endif

      return "";
    }
  }

  // Warn log, 既写到内存里 也输入到logcat
  std::string Warn(const char* format, ...) {
    if (!output_log || level > Warn_) {
      return "";
    }

    int size = 65535;
    size -= tag.size();
    char buffer[size];
    va_list ap;
    va_start(ap, format);
    char buffFormat[size];
    int nformatData = vsnprintf(buffFormat, sizeof(buffFormat), (const char*)format, ap);
    int n = snprintf(buffer, sizeof(buffer) - 1, "%s", buffFormat);
    va_end(ap);

    if (nformatData < 0 || n < 0) {
      return "";
    }

    if (return_str) {
      return std::string(tag + " " + buffer);
    } else {
#ifdef USING_NANOLOG
      NANO_LOG_WARN << tag.c_str() << " " << buffer;
#else
#ifdef __ANDROID__
      YVR_RAM_PRINT_LOG_WARN("%s %s", tag.c_str(), buffer)
#else
      printf("%s %s", tag.c_str(), buffer);
      fflush(stdout);
#endif
      return "";
#endif
    }
  }

  // Error log, 既写到内存里 也输入到logcat
  std::string Error(const char* format, ...) {
    if (!output_log || level > Error_) {
      return "";
    }

    int size = 65535;
    size -= tag.size();
    char buffer[size];
    va_list ap;
    va_start(ap, format);
    char buffFormat[size];
    int nformatData = vsnprintf(buffFormat, sizeof(buffFormat), (const char*)format, ap);
    int n = snprintf(buffer, sizeof(buffer) - 1, "%s", buffFormat);
    va_end(ap);

    if (nformatData < 0 || n < 0) {
      return "";
    }

    if (return_str) {
      return std::string(tag + " " + buffer);
    } else {
#ifdef USING_NANOLOG
      NANO_LOG_CRIT << tag.c_str() << " " << buffer;
#else
#ifdef __ANDROID__
      // system buffer size is 2048
      YVR_RAM_PRINT_LOG_ERROR("%s %s", tag.c_str(), buffer)
#else
      printf("%s %s", tag.c_str(), buffer);
      fflush(stdout);
#endif
      return "";
#endif
    }
  }

  void CloseLog() { output_log = false; }

  void OpenLog() { output_log = true; }

  void SetLogLevel(int log_level) {
#ifndef RELEASE_VERSION
    level = log_level;
#endif
  }

  void SetBufferMode(bool value) { return_str = value; }

 private:
  // output log flag
  bool output_log;
  // output log level
  int level;
  std::string tag;
  // return string flag
  bool return_str;
};

extern ILOG LOG_Calib;

extern ILOG LOG_FrontEnd;
extern ILOG LOG_SCMBT;

}  // namespace dso
//
// Created by zk on 24-7-4.
//

#ifndef YVR_CALIB_TOFCONFIG_H
#define YVR_CALIB_TOFCONFIG_H
#include <string>
#include <vector>
namespace dso::CalibIO {

struct TofConfigParams {
  bool verbose = false;
  int fixCamId = 6;
  int useBoardId = 0;
  std::string config_path;
  std::vector<int> useCamsPnp;
  std::string xml_full_path;
};

class TofConfigLoad {
 public:
  static TofConfigParams LoadTofCalibInfo(const std::string& filePath);
};
}  // namespace dso::CalibIO
#endif  // YVR_CALIB_TOFCONFIG_H

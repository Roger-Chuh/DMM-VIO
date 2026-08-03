//
// Created by zk on 24-7-4.
//

#include "tofconfig.h"
#include "toml.hpp"
namespace dso::CalibIO {

TofConfigParams TofConfigLoad::LoadTofCalibInfo(const std::string& filePath) {
  TofConfigParams tofParams;
  try {
    const toml::value& data = toml::parse(filePath);
    const toml::value& tofcalib = toml::find(data, "tofcalib");
    tofParams.verbose = toml::find<bool>(tofcalib, "verbose");
    tofParams.fixCamId = toml::find<int>(tofcalib, "fixCamId");
    tofParams.useBoardId = toml::find<int>(tofcalib, "useBoardId");
    tofParams.config_path = toml::find<std::string>(tofcalib, "configPath");
    auto camIds = toml::find(tofcalib, "useCamsPnp").as_array();
    for (auto cid_value : camIds) {
      tofParams.useCamsPnp.emplace_back(cid_value.as_integer());
    }
    tofParams.xml_full_path = toml::find<std::string>(tofcalib, "xml_full_path");
  } catch (const std::exception& e) {
    printf("parse toml err, file_path:[%s]  err:[%s]\n", filePath.c_str(), e.what());
    std::abort();
  }
  return tofParams;
}

}  // namespace dso::CalibIO
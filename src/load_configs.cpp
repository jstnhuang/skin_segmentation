#include "skin_segmentation/load_configs.h"

#include "camera_calibration_parsers/parse.h"
#include "ros/ros.h"
#include "rospack/rospack.h"
#include "sensor_msgs/CameraInfo.h"

#include "skin_segmentation/constants.h"

namespace skinseg {
bool GetCameraInfos(sensor_msgs::CameraInfo* rgb_info,
                    sensor_msgs::CameraInfo* thermal_info) {
  rospack::Rospack rospack;
  std::vector<std::string> search_path;
  bool success = rospack.getSearchPathFromEnv(search_path);
  if (!success) {
    ROS_ERROR("Failed to get package search path.");
    return false;
  }
  rospack.crawl(search_path, /* force */ false);
  std::string package("");
  success = rospack.find("skin_segmentation", package);
  if (!success) {
    ROS_ERROR(
        "Unable to find skin_segmentation package. Check that you have sourced "
        "the right workspace.");
    return false;
  }

  std::string rgb_path(package);
  rgb_path += kRgbConfigPath;
  std::string rgb_name("");
  success = camera_calibration_parsers::readCalibration(rgb_path, rgb_name,
                                                        *rgb_info);
  if (!success) {
    ROS_ERROR("Unable to find RGB camera info at %s", rgb_path.c_str());
    return false;
  }

  std::string thermal_path(package);
  thermal_path += kThermalConfigPath;
  std::string thermal_name("");
  success = camera_calibration_parsers::readCalibration(
      thermal_path, thermal_name, *thermal_info);
  if (!success) {
    ROS_ERROR("Unable to find thermal camera info at %s", thermal_path.c_str());
    return false;
  }

  return true;
}

bool GetNerfModelPath(std::string* model_path) {
  rospack::Rospack rospack;
  std::vector<std::string> search_path;
  bool success = rospack.getSearchPathFromEnv(search_path);
  if (!success) {
    ROS_ERROR("Failed to get package search path.");
    return false;
  }
  rospack.crawl(search_path, /* force */ false);
  std::string package("");
  success = rospack.find(kNerfModelPackage, package);
  if (!success) {
    ROS_ERROR(
        "Unable to find nerf model package \"%s\". Check that you have sourced "
        "the right workspace.",
        kNerfModelPackage);
    return false;
  }

  *model_path = package + kNerfHumanModelPath;
  return true;
}
}  // namespace skinseg

// Labels data given a bag file with RGB, depth, and thermal images.
// The results are written out to a new bag file with the RGB, depth, and labels
// applied to the image.

#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "camera_calibration_parsers/parse.h"
#include "message_filters/cache.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "ros/ros.h"
#include "rospack/rospack.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/labeling.h"
#include "skin_segmentation/projection.h"

using sensor_msgs::CameraInfo;
using sensor_msgs::Image;
typedef message_filters::sync_policies::ApproximateTime<Image, Image, Image>
    MyPolicy;

bool GetCameraInfos(CameraInfo* rgb_info, CameraInfo* thermal_info) {
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
  rgb_path += skinseg::kRgbConfigPath;
  std::string rgb_name("");
  success = camera_calibration_parsers::readCalibration(rgb_path, rgb_name,
                                                        *rgb_info);
  if (!success) {
    ROS_ERROR("Unable to find RGB camera info at %s", rgb_path.c_str());
    return false;
  }

  std::string thermal_path(package);
  thermal_path += skinseg::kThermalConfigPath;
  std::string thermal_name("");
  success = camera_calibration_parsers::readCalibration(
      thermal_path, thermal_name, *thermal_info);
  if (!success) {
    ROS_ERROR("Unable to find thermal camera info at %s", thermal_path.c_str());
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "skin_segmentation_label_data");
  ros::NodeHandle nh;

  if (argc < 2) {
    std::cout
        << "Usage: rosrun skin_segmentation label_data INPUT.bag OUTPUT.bag"
        << std::endl;
    return 1;
  }

  CameraInfo rgb_info;
  CameraInfo thermal_info;
  bool success = GetCameraInfos(&rgb_info, &thermal_info);
  if (!success) {
    return 1;
  }

  // Hard-coded extrinsic calibration.
  Eigen::Vector3d translation;
  translation << 0.00021494608, -0.035, 0.012;
  Eigen::Affine3d thermal_in_rgb;
  thermal_in_rgb.setIdentity();
  Eigen::Matrix3d rotation;
  rotation << 0.99989849, -0.00030364806, -0.004522502, 0.00056054816,
      0.99789572, 0.0638135, 0.004517816, -0.063812457, 0.99781871;
  thermal_in_rgb.translate(translation);
  thermal_in_rgb.rotate(rotation);
  Eigen::Affine3d rgb_in_thermal = thermal_in_rgb.inverse();

  skinseg::Projection projection(rgb_info, thermal_info, rgb_in_thermal);
  skinseg::Labeling labeling(projection);

  message_filters::Cache<Image> rgb_cache(20);
  message_filters::Cache<Image> depth_cache(20);
  message_filters::Cache<Image> thermal_cache(20);
  message_filters::Synchronizer<MyPolicy> sync(MyPolicy(10), rgb_cache,
                                               depth_cache, thermal_cache);
  sync.registerCallback(&skinseg::Labeling::Process, &labeling);

  return 0;
}

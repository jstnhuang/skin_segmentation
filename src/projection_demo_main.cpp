#include <iostream>

#include "Eigen/Dense"
#include "opencv2/highgui.hpp"
#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/load_configs.h"
#include "skin_segmentation/projection.h"
#include "skin_segmentation/rgbdt_data.h"
#include "skin_segmentation/snapshot.h"
#include "skin_segmentation/thresholding.h"

void PrintUsage() {
  std::cout << "Usage: rosrun skin_segmentation projection_demo ~/snapshot.bag"
            << std::endl;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "projection_demo");
  if (argc < 2) {
    PrintUsage();
  }
  skinseg::Snapshot snapshot;
  bool success = snapshot.LoadBag(argv[1]);
  if (!success) {
    return 1;
  }
  skinseg::RgbdtData data = snapshot.data();

  // Hard-coded camera infos
  sensor_msgs::CameraInfo rgbd_camera_info;
  sensor_msgs::CameraInfo thermal_camera_info;
  skinseg::GetCameraInfos(&rgbd_camera_info, &thermal_camera_info);

  Eigen::Vector3d translation;
  ros::param::param<double>("thermal_x", translation.x(), 0.00021494608);
  ros::param::param<double>("thermal_y", translation.y(), -0.035);
  ros::param::param<double>("thermal_z", translation.z(), 0.012);
  Eigen::Affine3d thermal_in_rgb;
  thermal_in_rgb.setIdentity();
  Eigen::Matrix3d rotation;
  rotation << 0.99989849, -0.00030364806, -0.004522502, 0.00056054816,
      0.99789572, 0.0638135, 0.004517816, -0.063812457, 0.99781871;
  thermal_in_rgb.translate(translation);
  thermal_in_rgb.rotate(rotation);
  Eigen::Affine3d rgb_in_thermal = thermal_in_rgb.inverse();

  skinseg::Projection projection(rgbd_camera_info, thermal_camera_info,
                                 rgb_in_thermal);
  // cv::namedWindow("RGB");
  // cv::namedWindow("Depth");
  // cv::namedWindow("Normalized thermal");
  // cv::setMouseCallback("RGB", &skinseg::Projection::RgbdMouseCallback,
  //                     &projection);
  // cv::setMouseCallback("Depth", &skinseg::Projection::RgbdMouseCallback,
  //                     &projection);
  // cv::setMouseCallback("Normalized thermal",
  //                     &skinseg::Projection::ThermalMouseCallback,
  //                     &projection);
  projection.set_debug(true);
  cv::Mat projected_thermal;
  projection.ProjectThermalOnRgb(data.color, data.depth, data.thermal,
                                 projected_thermal, NULL);

  cv::waitKey();

  return 0;
}

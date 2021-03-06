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
  ros::param::param<double>("thermal_x", translation.x(), 0.0032690763);
  ros::param::param<double>("thermal_y", translation.y(), -0.035865549);
  ros::param::param<double>("thermal_z", translation.z(), -0.010856843);
  Eigen::Affine3d thermal_in_rgb;
  thermal_in_rgb.setIdentity();
  Eigen::Matrix3d rotation;
  // rotation << 0.99989849, -0.00030364806, -0.004522502, 0.00056054816,
  //    0.99789572, 0.0638135, 0.004517816, -0.063812457, 0.99781871;
  // rotation << 0.99974662, -0.0035861803, -0.0024226252, 0.0036612949,
  // 0.9975068, 0.06543088, 0.0023200796, -0.065455951, 0.99737251;
  rotation << 0.99979681, 0.002344504, -0.0088005587, -0.001684363, 0.9967131,
      0.077537879, 0.0088948328, -0.077511609, 0.99654448;
  // rotation << 0.99966574, -0.00069165369, -0.0050593596, 0.0018626767,
  //    0.9971261, 0.065517075, 0.0051708301, -0.065402344, 0.99683237;
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

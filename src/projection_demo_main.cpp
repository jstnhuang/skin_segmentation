#include <iostream>

#include "Eigen/Dense"
#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/projection.h"
#include "skin_segmentation/rgbdt_data.h"
#include "skin_segmentation/snapshot.h"
#include "skin_segmentation/thresholding.h"

void PrintUsage() {
  std::cout << "Usage: rosrun skin_segmentation projection demo ~/snapshot.bag"
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
  sensor_msgs::CameraInfo rgbd_camera_info = *data.rgbd_camera_info;
  sensor_msgs::CameraInfo thermal_camera_info = *data.thermal_camera_info;
  thermal_camera_info.distortion_model = "rational_polynomial";
  thermal_camera_info.D.resize(8);
  thermal_camera_info.D[0] = 0;
  thermal_camera_info.D[1] = 0;
  thermal_camera_info.D[2] = -0.00389198;
  thermal_camera_info.D[3] = 0.03062872;
  thermal_camera_info.D[4] = 0;
  thermal_camera_info.D[5] = 0;
  thermal_camera_info.D[6] = 0;
  thermal_camera_info.D[7] = 0;

  rgbd_camera_info.distortion_model = "plumb_bob";
  rgbd_camera_info.D.resize(5);
  rgbd_camera_info.D[0] = 0;
  rgbd_camera_info.D[1] = 0;
  rgbd_camera_info.D[2] = 0;
  rgbd_camera_info.D[3] = 0;
  rgbd_camera_info.D[4] = 0;

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

  while (ros::ok()) {
    sensor_msgs::CameraInfo rgbd_camera_info = *data.rgbd_camera_info;
    double rf;
    ros::param::param<double>("rgbd_f", rf, 570.34);
    rgbd_camera_info.K[0] = rf;
    rgbd_camera_info.K[4] = rf;
    rgbd_camera_info.P[0] = rf;
    rgbd_camera_info.P[5] = rf;

    double tf;
    ros::param::param<double>("thermal_f", tf, 735.29);
    thermal_camera_info.K[0] = tf;
    thermal_camera_info.K[4] = tf;
    thermal_camera_info.P[0] = tf;
    thermal_camera_info.P[5] = tf;

    skinseg::Projection projection(rgbd_camera_info, thermal_camera_info,
                                   rgb_in_thermal);
    projection.set_debug(true);
    cv::Mat projected_thermal;
    projection.ProjectThermalOnRgb(data.color, data.depth, data.thermal,
                                   projected_thermal);
    skinseg::Thresholding thresholding;
    thresholding.TryThresholds(projected_thermal);
  }

  return 0;
}

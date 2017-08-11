#include <iostream>

#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/projection.h"
#include "skin_segmentation/rgbdt_data.h"
#include "skin_segmentation/snapshot.h"

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

  // Hard-coded RGB camera info
  sensor_msgs::CameraInfo rgbd_camera_info = *data.rgbd_camera_info;
  rgbd_camera_info.D.resize(5);
  rgbd_camera_info.D[0] = 0.062083;
  rgbd_camera_info.D[1] = -0.128651;
  rgbd_camera_info.D[2] = -0.003100;
  rgbd_camera_info.D[3] = 0.012476;

  rgbd_camera_info.K[0] = 565.829036;
  rgbd_camera_info.K[2] = 335.955730;
  rgbd_camera_info.K[4] = 564.944907;
  rgbd_camera_info.K[5] = 239.333909;

  rgbd_camera_info.P[0] = 567.424805;
  rgbd_camera_info.P[2] = 342.737757;
  rgbd_camera_info.P[5] = 572.887207;
  rgbd_camera_info.P[6] = 237.647368;

  skinseg::Projection projection(rgbd_camera_info, *data.thermal_camera_info);

  while (ros::ok()) {
    sensor_msgs::Image output;
    projection.ProjectRgbdOntoThermal(data.color, data.depth, data.thermal,
                                      &output);
  }

  return 0;
}

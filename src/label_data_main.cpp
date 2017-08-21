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
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/labeling.h"
#include "skin_segmentation/load_camera_info.h"
#include "skin_segmentation/projection.h"

using sensor_msgs::CameraInfo;
using sensor_msgs::Image;
typedef message_filters::sync_policies::ApproximateTime<Image, Image, Image>
    MyPolicy;

int main(int argc, char** argv) {
  ros::init(argc, argv, "skin_segmentation_label_data");
  ros::NodeHandle nh;

  if (argc < 3) {
    std::cout
        << "Usage: rosrun skin_segmentation label_data INPUT.bag OUTPUT.bag"
        << std::endl;
    return 1;
  }

  CameraInfo rgb_info;
  CameraInfo thermal_info;
  bool success = skinseg::GetCameraInfos(&rgb_info, &thermal_info);
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
  projection.set_debug(true);
  skinseg::Labeling labeling(projection);

  message_filters::Cache<Image> rgb_cache(100);
  message_filters::Cache<Image> depth_cache(100);
  message_filters::Cache<Image> thermal_cache(100);
  message_filters::Synchronizer<MyPolicy> sync(MyPolicy(100), rgb_cache,
                                               depth_cache, thermal_cache);
  sync.getPolicy()->setMaxIntervalDuration(ros::Duration(0.005));
  sync.registerCallback(&skinseg::Labeling::Process, &labeling);

  rosbag::Bag input_bag;
  std::string input_bag_path(argv[1]);
  input_bag.open(input_bag_path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(skinseg::kRgbTopic);
  topics.push_back(skinseg::kDepthTopic);
  topics.push_back(skinseg::kThermalTopic);
  rosbag::View view(input_bag, rosbag::TopicQuery(topics));

  for (rosbag::View::const_iterator it = view.begin(); it != view.end(); ++it) {
    if (it->getTopic() == skinseg::kRgbTopic) {
      rgb_cache.add(it->instantiate<Image>());
    } else if (it->getTopic() == skinseg::kDepthTopic) {
      depth_cache.add(it->instantiate<Image>());
    } else if (it->getTopic() == skinseg::kThermalTopic) {
      thermal_cache.add(it->instantiate<Image>());
    }
  }

  return 0;
}

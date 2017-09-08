// Labels data given a bag file with skin_segmentation_msgs/Images.
// The results are written out to a new bag file with the RGB, depth, and labels
// applied to the image.

#include <iostream>
#include <string>
#include <vector>

#undef Success  // Evil workaround. nerf includes glx, which defines this again
#include "Eigen/Dense"
#include "camera_calibration_parsers/parse.h"
#include "message_filters/cache.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "model/model.h"
#include "model/model_instance.h"
#include "observation/ros_observation.h"
#include "optimization/optimization_parameters.h"
#include "optimization/optimizer.h"
#include "ros/ros.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/labeling.h"
#include "skin_segmentation/load_configs.h"
#include "skin_segmentation/nerf.h"
#include "skin_segmentation/projection.h"
#include "skin_segmentation_msgs/Images.h"

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
  // translation << 0.00021494608, -0.035, 0.012;
  translation << -0.00091294711, -0.040564451, -0.025354201;
  Eigen::Affine3d thermal_in_rgb;
  thermal_in_rgb.setIdentity();
  Eigen::Matrix3d rotation;
  // rotation << 0.99989849, -0.00030364806, -0.004522502, 0.00056054816,
  //    0.99789572, 0.0638135, 0.004517816, -0.063812457, 0.99781871;
  rotation << 0.99974662, -0.0035861803, -0.0024226252, 0.0036612949, 0.9975068,
      0.06543088, 0.0023200796, -0.065455951, 0.99737251;
  thermal_in_rgb.translate(translation);
  thermal_in_rgb.rotate(rotation);
  Eigen::Affine3d rgb_in_thermal = thermal_in_rgb.inverse();

  skinseg::Projection projection(rgb_info, thermal_info, rgb_in_thermal);
  projection.set_debug(true);
  rosbag::Bag output_bag;
  output_bag.open(argv[2], rosbag::bagmode::Write);

  // Set up nerf person tracker
  skinseg::Nerf nerf;
  float model_scale;
  ros::param::param("label_data_model_scale", model_scale, 0.92f);
  ROS_INFO("Model scale: %f", model_scale);
  skinseg::BuildNerf(&nerf, model_scale);
  nerf.model_instance->setScale(model_scale);
  skinseg::Labeling labeling(projection, &nerf, &output_bag);

  message_filters::Cache<Image> rgb_cache(100);
  message_filters::Cache<Image> depth_cache(100);
  message_filters::Cache<Image> thermal_cache(100);
  message_filters::Synchronizer<MyPolicy> sync(MyPolicy(100), rgb_cache,
                                               depth_cache, thermal_cache);
  sync.registerCallback(&skinseg::Labeling::Process, &labeling);

  rosbag::Bag input_bag;
  std::string input_bag_path(argv[1]);
  input_bag.open(input_bag_path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(skinseg::kRgbTopic);
  topics.push_back(skinseg::kDepthTopic);
  topics.push_back(skinseg::kThermalTopic);
  rosbag::View view(input_bag, rosbag::TopicQuery(topics));

  bool debug;
  ros::param::param("label_data_debug", debug, false);
  labeling.set_debug(debug);

  int max_images;  // Max number of images to process, 0 for all.
  ros::param::param("label_data_max_images", max_images, 0);

  std::string labeling_algorithm("");
  ros::param::param("labeling_algorithm", labeling_algorithm,
                    std::string(skinseg::kThermal));
  labeling.set_labeling_algorithm(labeling_algorithm);

  int num_msgs = view.size();
  int i = 0;

  ros::Time start = view.getBeginTime() + ros::Duration(4);
  ros::Time end = view.getEndTime() - ros::Duration(4);
  for (rosbag::View::const_iterator it = view.begin(); it != view.end(); ++it) {
    if (max_images > 0 && i >= max_images) {
      break;
    }
    const ros::Time& time = it->getTime();
    if (time < start || time > end) {
      continue;
    }
    if (it->getTopic() == skinseg::kRgbTopic) {
      rgb_cache.add(it->instantiate<Image>());
    } else if (it->getTopic() == skinseg::kDepthTopic) {
      depth_cache.add(it->instantiate<Image>());
    } else if (it->getTopic() == skinseg::kThermalTopic) {
      thermal_cache.add(it->instantiate<Image>());
    }
    ++i;
    if (i % 100 == 0) {
      ROS_INFO("Processed image %d of %d (%f)", i, num_msgs,
               static_cast<float>(i) / num_msgs);
    }
  }

  output_bag.close();

  return 0;
}

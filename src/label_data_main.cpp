// Labels data given a bag file with skin_segmentation_msgs/Images.
// The results are written out to a new bag file with the RGB, depth, and labels
// applied to the image.

#include <iostream>
#include <string>
#include <vector>

#undef Success  // Evil workaround. nerf includes glx, which defines this again
#include "Eigen/Dense"
#include "boost/algorithm/string.hpp"
#include "camera_calibration_parsers/parse.h"
#include "interactive_markers/interactive_marker_server.h"
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
#include "skin_segmentation_msgs/NerfJointStates.h"
#include "visualization_msgs/MarkerArray.h"

#include "skin_segmentation/box_interactive_marker.h"
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
  ros::AsyncSpinner spinner(2);
  spinner.start();

  if (argc < 3) {
    std::cout
        << "Usage: rosrun skin_segmentation label_data INPUT.bag OUTPUT_DIR"
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
  // translation << -0.00091294711, -0.040564451, -0.025354201;
  // translation << 0.0032690763, -0.035865549, -0.010856843;
  translation << -0.0088957613, -0.013507488, 0.026747243;
  Eigen::Affine3d thermal_in_rgb;
  thermal_in_rgb.setIdentity();
  Eigen::Matrix3d rotation;
  // rotation << 0.99989849, -0.00030364806, -0.004522502, 0.00056054816,
  //    0.99789572, 0.0638135, 0.004517816, -0.063812457, 0.99781871;
  // rotation << 0.99974662, -0.0035861803, -0.0024226252, 0.0036612949,
  // 0.9975068,
  //    0.06543088, 0.0023200796, -0.065455951, 0.99737251;
  // rotation << 0.99979681, 0.002344504, -0.0088005587, -0.001684363,
  // 0.9967131, 0.077537879, 0.0088948328, -0.077511609, 0.99654448;
  rotation << 0.99966574, -0.00069165369, -0.0050593596, 0.0018626767,
      0.9971261, 0.065517075, 0.0051708301, -0.065402344, 0.99683237;
  thermal_in_rgb.translate(translation);
  thermal_in_rgb.rotate(rotation);
  Eigen::Affine3d rgb_in_thermal = thermal_in_rgb.inverse();

  skinseg::Projection projection(rgb_info, thermal_info, rgb_in_thermal);
  projection.set_debug(true);

  // Set up nerf person tracker
  ros::Publisher nerf_joint_pub =
      nh.advertise<skin_segmentation_msgs::NerfJointStates>("nerf_joint_states",
                                                            1, true);
  ros::Publisher skeleton_pub =
      nh.advertise<visualization_msgs::MarkerArray>("skeleton", 1, true);
  skinseg::Nerf nerf(nerf_joint_pub, skeleton_pub);
  float model_scale;
  ros::param::param("label_data_model_scale", model_scale, 0.92f);
  ROS_INFO("Model scale: %f", model_scale);
  skinseg::BuildNerf(&nerf, model_scale);

  // Subscriber for nerf control UI
  ros::Subscriber nerf_sub =
      nh.subscribe("nerf_controls", 1, &skinseg::Nerf::Update, &nerf);

  // Set up output
  std::string output_dir("");
  rosbag::Bag* output_bag = NULL;
  if (boost::algorithm::ends_with(argv[2], ".bag")) {
    rosbag::Bag bag;
    bag.open(argv[2], rosbag::bagmode::Write);
    output_bag = &bag;
  } else {
    output_dir = argv[2];
  }

  // Build hand box servers
  interactive_markers::InteractiveMarkerServer im_server("hands", "", true);
  skinseg::BoxInteractiveMarker left_box("left", &im_server);
  skinseg::BoxInteractiveMarker right_box("right", &im_server);

  skinseg::Labeling labeling(projection, &nerf, output_dir, output_bag,
                             &left_box, &right_box);

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

  if (output_bag != NULL) {
    output_bag->close();
  }
  spinner.stop();

  return 0;
}

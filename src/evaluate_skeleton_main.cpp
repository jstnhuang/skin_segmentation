// Evaluates the accuracy of the nerf skeleton tracker with and without hand
// segmentation.
//
// Usage: evaluate_skeleton INPUT_DATA.bag LABELED_SKELETON.bag
//
// INPUT_DATA.bag contains RGB and Depth images.
// LABELED_SKELETON.bag contains the labeled skeleton from the label_skeleton
// executable.
//
// The bag files are synchronized in the order in which the
// ApproximateTimeSynchronizer extracts the RGB/Depth images with RGB, Depth,
// and Synchronizer caches of size 100. First and last 3 seconds are skipped.

#include <iostream>
#include <string>
#include <vector>

#undef Success  // Evil workaround. nerf includes glx, which defines this again
#include "message_filters/cache.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "ros/ros.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "skin_segmentation_msgs/NerfJointStates.h"
#include "visualization_msgs/MarkerArray.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/load_configs.h"
#include "skin_segmentation/nerf.h"
#include "skin_segmentation/skeleton_evaluator.h"

using sensor_msgs::CameraInfo;
using sensor_msgs::Image;
typedef message_filters::sync_policies::ApproximateTime<Image, Image> MyPolicy;

int main(int argc, char** argv) {
  ros::init(argc, argv, "skin_segmentation_evaluate_skeleton");
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(2);
  spinner.start();

  if (argc < 2) {
    std::cout << "Usage: rosrun skin_segmentation evaluate_skeleton "
                 "INPUT_DATA.bag"
              << std::endl;
    std::cout << "Usage: rosrun skin_segmentation evaluate_skeleton "
                 "INPUT_DATA.bag LABELED_SKELETON.bag"
              << std::endl;
    return 1;
  }

  // Set up nerf person tracker
  // TODO: add option to run hand segmentation
  ros::Publisher nerf_joint_pub =
      nh.advertise<skin_segmentation_msgs::NerfJointStates>("nerf_joint_states",
                                                            1, true);
  ros::Publisher skeleton_pub =
      nh.advertise<visualization_msgs::MarkerArray>("skeleton", 1, true);
  skinseg::Nerf nerf(nerf_joint_pub, skeleton_pub);

  ros::Publisher labeled_nerf_joint_pub =
      nh.advertise<skin_segmentation_msgs::NerfJointStates>("nerf_joint_states",
                                                            1, true);
  ros::Publisher labeled_skeleton_pub =
      nh.advertise<visualization_msgs::MarkerArray>("labeled_skeleton", 1,
                                                    true);
  skinseg::Nerf labeled_nerf(labeled_nerf_joint_pub, labeled_skeleton_pub);
  labeled_nerf.set_rgb(0, 1, 0);

  float model_scale;
  ros::param::param("label_data_model_scale", model_scale, 0.92f);
  ROS_INFO("Model scale: %f", model_scale);
  bool use_hand_segmentation = false;
  if (!ros::param::get("use_hand_segmentation", use_hand_segmentation)) {
    ROS_ERROR("Must set ROS param use_hand_segmentation");
    return 1;
  }
  if (use_hand_segmentation) {
    ROS_INFO("Using hand segmentation");
    skinseg::BuildNerf(&nerf, model_scale, true);
  } else {
    ROS_INFO("Using baseline system");
    skinseg::BuildNerf(&nerf, model_scale, false);
  }
  skinseg::BuildNerf(&labeled_nerf, model_scale, true);

  // Open labeled skeleton bag
  rosbag::Bag* labeled_bag = NULL;
  if (argc >= 3) {
    std::string labeled_bag_path(argv[2]);
    labeled_bag = new rosbag::Bag(labeled_bag_path, rosbag::bagmode::Read);
  }
  skinseg::SkeletonEvaluator skeleton_evaluator(&nerf, &labeled_nerf,
                                                labeled_bag);

  message_filters::Cache<Image> rgb_cache(100);
  message_filters::Cache<Image> depth_cache(100);
  message_filters::Synchronizer<MyPolicy> sync(MyPolicy(100), rgb_cache,
                                               depth_cache);
  sync.registerCallback(&skinseg::SkeletonEvaluator::Process,
                        &skeleton_evaluator);

  rosbag::Bag input_bag;
  std::string input_bag_path(argv[1]);
  input_bag.open(input_bag_path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(skinseg::kRgbTopic);
  topics.push_back(skinseg::kDepthTopic);
  rosbag::View view(input_bag, rosbag::TopicQuery(topics));

  ros::Time start = view.getBeginTime() + ros::Duration(4);
  ros::Time end = view.getEndTime() - ros::Duration(4);
  for (rosbag::View::const_iterator it = view.begin(); it != view.end(); ++it) {
    const ros::Time& time = it->getTime();
    if (time < start || time > end) {
      continue;
    }
    if (it->getTopic() == skinseg::kRgbTopic) {
      rgb_cache.add(it->instantiate<Image>());
    } else if (it->getTopic() == skinseg::kDepthTopic) {
      depth_cache.add(it->instantiate<Image>());
    }
  }

  spinner.stop();

  return 0;
}

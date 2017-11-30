// Annotate a bag file with the skeleton joint angles.
// Usage: label_skeleton INPUT.bag OUTPUT.bag
//
// INPUT.bag contains RGB and Depth images.
// OUTPUT.bag contains NerfJointStates.
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
#include "skin_segmentation/skeleton_labeler.h"

using sensor_msgs::CameraInfo;
using sensor_msgs::Image;
typedef message_filters::sync_policies::ApproximateTime<Image, Image> MyPolicy;

int main(int argc, char** argv) {
  ros::init(argc, argv, "skin_segmentation_label_skeleton");
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(2);
  spinner.start();

  if (argc < 3) {
    std::cout
        << "Usage: rosrun skin_segmentation label_skeleton INPUT.bag OUTPUT.bag"
        << std::endl;
    return 1;
  }

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
  rosbag::Bag output_bag;
  output_bag.open(argv[2], rosbag::bagmode::Write);

  skinseg::SkeletonLabeler skeleton_labeler(&nerf, &output_bag);

  message_filters::Cache<Image> rgb_cache(100);
  message_filters::Cache<Image> depth_cache(100);
  message_filters::Synchronizer<MyPolicy> sync(MyPolicy(100), rgb_cache,
                                               depth_cache);
  sync.registerCallback(&skinseg::SkeletonLabeler::Process, &skeleton_labeler);

  rosbag::Bag input_bag;
  std::string input_bag_path(argv[1]);
  input_bag.open(input_bag_path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(skinseg::kRgbTopic);
  topics.push_back(skinseg::kDepthTopic);
  rosbag::View view(input_bag, rosbag::TopicQuery(topics));

  int num_msgs = view.size();
  int i = 0;
  ros::Time start = view.getBeginTime() + ros::Duration(4);
  ros::Time end = view.getEndTime() - ros::Duration(4);
  std::cout << "Press (s)ave or (r)e-run tracker:" << std::endl;
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
    ++i;
    if (i % 100 == 0) {
      ROS_INFO("Processed image %d of %d (%f)", i, num_msgs,
               static_cast<float>(i) / num_msgs);
    }
  }

  output_bag.close();
  spinner.stop();

  return 0;
}

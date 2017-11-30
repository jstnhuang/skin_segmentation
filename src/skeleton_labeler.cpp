#include "skin_segmentation/skeleton_labeler.h"

#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "ros/ros.h"
#include "rosbag/bag.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "skin_segmentation_msgs/NerfJointStates.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/load_configs.h"
#include "skin_segmentation/nerf.h"

namespace {
char getch() {
  char buf = 0;
  struct termios old = {0};
  fflush(stdout);
  if (tcgetattr(0, &old) < 0) perror("tcsetattr()");
  old.c_lflag &= ~ICANON;
  old.c_lflag &= ~ECHO;
  old.c_cc[VMIN] = 1;
  old.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSANOW, &old) < 0) perror("tcsetattr ICANON");
  if (read(0, &buf, 1) < 0) perror("read()");
  old.c_lflag |= ICANON;
  old.c_lflag |= ECHO;
  if (tcsetattr(0, TCSADRAIN, &old) < 0) perror("tcsetattr ~ICANON");
  // printf("%c\n", buf); // Uncomment to print the character
  return buf;
}
}  // namespace

namespace skinseg {
SkeletonLabeler::SkeletonLabeler(Nerf* nerf, rosbag::Bag* output_bag)
    : nerf_(nerf),
      output_bag_(output_bag),
      nh_(),
      rgb_pub_(nh_.advertise<sensor_msgs::Image>(kRgbTopic, 1, true)),
      depth_pub_(nh_.advertise<sensor_msgs::Image>(kDepthTopic, 1, true)),
      depth_info_pub_(
          nh_.advertise<sensor_msgs::CameraInfo>(kDepthInfoTopic, 1)),
      rgbd_info_(),
      processed_count_(0) {
  sensor_msgs::CameraInfo thermal_info_unused;
  bool success = GetCameraInfos(&rgbd_info_, &thermal_info_unused);
  if (!success) {
    ROS_ERROR("Failed to get camera info.");
  }
}

void SkeletonLabeler::Process(const sensor_msgs::ImageConstPtr& rgb,
                              const sensor_msgs::ImageConstPtr& depth) {
  ++processed_count_;
  while (ros::ok()) {
    // Publish depth cloud.
    ros::Time now = ros::Time::now();
    sensor_msgs::Image rgb_now = *rgb;
    rgb_now.header.stamp = now;
    sensor_msgs::Image depth_now = *depth;
    depth_now.header.stamp = now;
    rgb_pub_.publish(rgb_now);
    depth_pub_.publish(depth_now);
    rgbd_info_.header.stamp = now;
    depth_info_pub_.publish(rgbd_info_);

    // Step through tracker
    nerf_->Step(rgb, depth);
    nerf_->PublishJointStates();
    nerf_->PublishVisualization();

    while (ros::ok()) {
      char user_input = getch();

      if (user_input == 's') {
        skin_segmentation_msgs::NerfJointStates joint_states;
        nerf_->GetJointStates(&joint_states);
        output_bag_->write(kNerfJointStatesLabelTopic, rgb->header.stamp,
                           joint_states);
        return;
      } else if (user_input == 'r') {
        break;
      }
    }
  }
}
}  // namespace skinseg

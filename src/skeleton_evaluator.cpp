#include "skin_segmentation/skeleton_evaluator.h"

#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "Eigen/Dense"
#include "absl/strings/str_join.h"
#include "ros/ros.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "skin_segmentation_msgs/NerfJointStates.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/load_configs.h"
#include "skin_segmentation/nerf.h"

using skin_segmentation_msgs::NerfJointStates;

namespace skinseg {
SkeletonEvaluator::SkeletonEvaluator(Nerf* nerf, Nerf* labeled_nerf,
                                     const rosbag::Bag& skel_labels)
    : nerf_(nerf),
      labeled_nerf_(labeled_nerf),
      skel_labels_(skel_labels),
      nh_(),
      rgb_pub_(nh_.advertise<sensor_msgs::Image>(kRgbTopic, 1, true)),
      depth_pub_(nh_.advertise<sensor_msgs::Image>(kDepthTopic, 1, true)),
      depth_info_pub_(
          nh_.advertise<sensor_msgs::CameraInfo>(kDepthInfoTopic, 1)),
      rgbd_info_(),
      processed_count_(0),
      kNerfEvaluationJoints_() {
  sensor_msgs::CameraInfo thermal_info_unused;
  bool success = GetCameraInfos(&rgbd_info_, &thermal_info_unused);
  if (!success) {
    ROS_ERROR("Failed to get camera info.");
  }

  kNerfEvaluationJoints_.push_back(kNerfLShoulderJoint);
  kNerfEvaluationJoints_.push_back(kNerfRShoulderJoint);
  kNerfEvaluationJoints_.push_back(kNerfLElbowJoint);
  kNerfEvaluationJoints_.push_back(kNerfRElbowJoint);
  kNerfEvaluationJoints_.push_back(kNerfLWristJoint);
  kNerfEvaluationJoints_.push_back(kNerfRWristJoint);

  std::vector<std::string> labeled_bag_topics;
  labeled_bag_topics.push_back(skinseg::kNerfJointStatesLabelTopic);
  skel_labels_view_ =
      // new rosbag::View(skel_labels_, rosbag::TopicQuery(labeled_bag_topics));
      new rosbag::View(skel_labels_);
  ROS_INFO("Number of labels: %d", skel_labels_view_->size());
  skel_labels_it_ = skel_labels_view_->begin();
}

SkeletonEvaluator::~SkeletonEvaluator() {
  if (skel_labels_view_) {
    delete skel_labels_view_;
  }
}

void SkeletonEvaluator::Process(const sensor_msgs::ImageConstPtr& rgb,
                                const sensor_msgs::ImageConstPtr& depth) {
  ++processed_count_;

  // Publish depth cloud for visualization purposes
  ros::Time now = ros::Time::now();
  sensor_msgs::Image rgb_now = *rgb;
  rgb_now.header.stamp = now;
  sensor_msgs::Image depth_now = *depth;
  depth_now.header.stamp = now;
  rgb_pub_.publish(rgb_now);
  depth_pub_.publish(depth_now);
  rgbd_info_.header.stamp = now;
  depth_info_pub_.publish(rgbd_info_);

  // Step through tracker that is being tested
  nerf_->Step(rgb, depth);
  nerf_->PublishJointStates();
  nerf_->PublishVisualization();

  // Step through labeled tracker
  if (skel_labels_it_ == skel_labels_view_->end()) {
    ROS_ERROR("Failed to get next label!");
    return;
  }
  if (skel_labels_it_->getTopic() != skinseg::kNerfJointStatesLabelTopic) {
    ROS_ERROR("Invalid topic in labeled skeleton bag. Got %s, expected %s",
              skel_labels_it_->getTopic().c_str(),
              skinseg::kNerfJointStatesLabelTopic);
    return;
  }
  // We expect that the timestamp matches the RGB image (this is the logic in
  // label_skeleton).
  if (skel_labels_it_->getTime() != rgb->header.stamp) {
    ROS_ERROR_STREAM("Time mismatch in labeled skeleton. Expected: "
                     << rgb->header.stamp
                     << ", got: " << skel_labels_it_->getTime());
    return;
  }
  NerfJointStates::ConstPtr labeled_joint_states =
      skel_labels_it_->instantiate<NerfJointStates>();
  labeled_nerf_->Update(*labeled_joint_states);

  // Evaluate
  skin_segmentation_msgs::NerfJointStates joint_states;
  std::vector<float> stats(1 + kNerfEvaluationJoints_.size());
  stats[0] = processed_count_;
  for (size_t i = 0; i < kNerfEvaluationJoints_.size(); ++i) {
    const std::string& joint_name = kNerfEvaluationJoints_[i];
    Eigen::Affine3f expected_pose = labeled_nerf_->GetJointPose(joint_name);
    Eigen::Affine3f actual_pose = nerf_->GetJointPose(joint_name);
    Eigen::Vector3f offset =
        expected_pose.translation() - actual_pose.translation();
    stats[i + 1] = offset.norm();
  }

  std::cout << absl::StrJoin(stats, "\t") << std::endl;
  ++skel_labels_it_;
}
}  // namespace skinseg

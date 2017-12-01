#ifndef _SKIN_SEGMENTATION_SKELETON_EVALUATOR_H_
#define _SKIN_SEGMENTATION_SKELETON_EVALUATOR_H_

#include "ros/ros.h"
#include "rosbag/view.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/nerf.h"

namespace skinseg {
class SkeletonEvaluator {
 public:
  SkeletonEvaluator(Nerf* nerf, Nerf* labeled_nerf,
                    const rosbag::Bag& skel_labels);
  ~SkeletonEvaluator();

  void Process(const sensor_msgs::ImageConstPtr& rgb,
               const sensor_msgs::ImageConstPtr& depth);

 private:
  Nerf* nerf_;
  Nerf* labeled_nerf_;
  rosbag::Bag skel_labels_;

  rosbag::View* skel_labels_view_;
  rosbag::View::const_iterator skel_labels_it_;

  ros::NodeHandle nh_;
  ros::Publisher rgb_pub_;
  ros::Publisher depth_pub_;
  ros::Publisher depth_info_pub_;
  sensor_msgs::CameraInfo rgbd_info_;

  int processed_count_;
  std::vector<std::string> kNerfEvaluationJoints_;
};
}  // namespace skinseg

#endif  // _SKIN_SEGMENTATION_SKELETON_EVALUATOR_H_

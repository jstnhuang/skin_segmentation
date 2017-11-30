#ifndef _SKIN_SEGMENTATION_SKELETON_LABELER_H_
#define _SKIN_SEGMENTATION_SKELETON_LABELER_H_

#include "ros/ros.h"
#include "rosbag/bag.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/nerf.h"

namespace skinseg {
class SkeletonLabeler {
 public:
  SkeletonLabeler(Nerf* nerf, rosbag::Bag* output_bag);

  void Process(const sensor_msgs::ImageConstPtr& rgb,
               const sensor_msgs::ImageConstPtr& depth);

 private:
  Nerf* nerf_;
  rosbag::Bag* output_bag_;

  ros::NodeHandle nh_;
  ros::Publisher rgb_pub_;
  ros::Publisher depth_pub_;
  ros::Publisher depth_info_pub_;
  sensor_msgs::CameraInfo rgbd_info_;

  int processed_count_;
};
}  // namespace skinseg

#endif  // _SKIN_SEGMENTATION_SKELETON_LABELER_H_

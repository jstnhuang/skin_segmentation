#ifndef _SKINSEG_LABELING_H_
#define _SKINSEG_LABELING_H_

#include "ros/ros.h"
#include "rosbag/bag.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/nerf.h"
#include "skin_segmentation/projection.h"

namespace skinseg {
class Labeling {
 public:
  // Takes in an open bag to write to.
  Labeling(const Projection& projection, Nerf* nerf, rosbag::Bag* output_bag);

  void Process(const sensor_msgs::Image::ConstPtr& rgb,
               const sensor_msgs::Image::ConstPtr& depth,
               const sensor_msgs::Image::ConstPtr& thermal);

  // If debug is true, then wait for a keypress between each image.
  void set_debug(bool debug);

 private:
  Projection projection_;
  Nerf* nerf_;
  rosbag::Bag* output_bag_;
  bool debug_;
  ros::NodeHandle nh_;
  ros::Publisher skeleton_pub_;
  ros::Publisher rgb_pub_;
  ros::Publisher depth_pub_;
  ros::Publisher depth_info_pub_;
  ros::Publisher thermal_pub_;

  ros::Time first_msg_time_;
  CameraData camera_data_;
  sensor_msgs::CameraInfo rgbd_info_;
};

void ComputeHandMask(const sensor_msgs::Image& depth,
                     const CameraData& camera_data, float model_scale,
                     const Eigen::Affine3f& l_forearm_pose,
                     const Eigen::Affine3f& r_forearm_pose, uint8_t* mask);
}  // namespace skinseg

#endif  // _SKINSEG_LABELING_H_

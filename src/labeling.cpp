#include "skin_segmentation/labeling.h"

#include <cuda_runtime.h>

#include "cv_bridge/cv_bridge.h"
#include "depth_image_proc/depth_traits.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ros/ros.h"
#include "rosbag/bag.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "visualization_msgs/MarkerArray.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/nerf.h"
#include "skin_segmentation/opencv_utils.h"
#include "skin_segmentation/projection.h"

using sensor_msgs::Image;

namespace skinseg {
Labeling::Labeling(const Projection& projection, Nerf* nerf,
                   rosbag::Bag* output_bag)
    : projection_(projection),
      nerf_(nerf),
      output_bag_(output_bag),
      debug_(false),
      nh_(),
      skeleton_pub_(
          nh_.advertise<visualization_msgs::MarkerArray>("skeleton", 1)),
      rgb_pub_(nh_.advertise<sensor_msgs::Image>(kRgbTopic, 1)),
      depth_pub_(nh_.advertise<sensor_msgs::Image>(kDepthTopic, 1)),
      depth_info_pub_(
          nh_.advertise<sensor_msgs::CameraInfo>(kDepthInfoTopic, 1)),
      thermal_pub_(nh_.advertise<sensor_msgs::Image>(kThermalTopic, 1)),
      first_msg_time_(0),
      camera_data_(),
      rgbd_info_() {
  projection_.GetCameraData(&camera_data_);
  projection_.GetRgbdCameraInfo(&rgbd_info_);
}

void Labeling::Process(const Image::ConstPtr& rgb, const Image::ConstPtr& depth,
                       const Image::ConstPtr& thermal) {
  if (!rgb || !depth || !thermal) {
    ROS_ERROR("Got null image when processing labels!");
    return;
  }

  if (depth->encoding != sensor_msgs::image_encodings::TYPE_16UC1) {
    ROS_ERROR("Unsupported depth encoding \"%s\", required 16_UC1",
              depth->encoding.c_str());
    return;
  }

  ROS_INFO("Depth - thermal skew: %f",
           (depth->header.stamp - thermal->header.stamp).toSec());
  if (first_msg_time_.isZero()) {
    first_msg_time_ = rgb->header.stamp;
  }

  double thermal_threshold;
  ros::param::param("thermal_threshold", thermal_threshold, 3650.0);

  cv::Mat thermal_projected;
  projection_.ProjectThermalOnRgb(rgb, depth, thermal, thermal_projected);

  cv::Mat thermal_fp;
  thermal_projected.convertTo(thermal_fp, CV_32F);

  float model_scale = nerf_->model_instance->getScale();
  if (debug_) {
    visualization_msgs::MarkerArray skeleton;
    SkeletonMarkerArray(nerf_, model_scale, &skeleton);

    ros::Time now = ros::Time::now();
    sensor_msgs::Image rgb_now = *rgb;
    rgb_now.header.stamp = now;
    sensor_msgs::Image depth_now = *depth;
    depth_now.header.stamp = now;
    rgb_pub_.publish(rgb_now);
    depth_pub_.publish(depth_now);
    sensor_msgs::Image thermal_now = *thermal;
    thermal_now.header.stamp = now;
    thermal_pub_.publish(thermal_now);

    rgbd_info_.header.stamp = now;
    depth_info_pub_.publish(rgbd_info_);
    skeleton_pub_.publish(skeleton);
  }

  // Body pose tracking - skip first 3 seconds for user to get in initial pose
  if (rgb->header.stamp < first_msg_time_ + ros::Duration(1)) {
    return;
  }

  nerf_->observation->Callback(rgb, depth);
  nerf_->observation->advance();
  nerf_->optimizer->optimize(nerf_->opt_parameters);
  const nerf::DualQuaternion* joint_poses =
      nerf_->model_instance->getHostJointPose();
  int l_index =
      nerf_->model->getKinematics()->getJointIndex(kNerfLForearmRotJoint);
  int r_index =
      nerf_->model->getKinematics()->getJointIndex(kNerfRForearmRotJoint);
  nerf::DualQuaternion l_forearm_pose = joint_poses[l_index];
  nerf::DualQuaternion r_forearm_pose = joint_poses[r_index];
  l_forearm_pose.normalize();
  r_forearm_pose.normalize();
  Eigen::Affine3f l_matrix(l_forearm_pose.ToMatrix());
  Eigen::Matrix4f r_pose_mat = r_forearm_pose.ToMatrix();
  Eigen::Matrix3f r_pose_rot = r_pose_mat.topLeftCorner(3, 3);
  r_pose_rot.col(0) *= -1;
  r_pose_rot.col(1) = r_pose_rot.col(2).cross(r_pose_rot.col(0));
  r_pose_mat.topLeftCorner(3, 3) = r_pose_rot;
  Eigen::Affine3f r_matrix(r_pose_mat);

  l_matrix.translation() *= model_scale;
  r_matrix.translation() *= model_scale;

  cv::Mat mask(rgb->height, rgb->width, CV_8UC1, cv::Scalar(0));
  ComputeHandMask(*depth, camera_data_, l_matrix, r_matrix, mask.data);

  cv::Mat labels_mat;
  cv::threshold(thermal_fp, labels_mat, thermal_threshold, 1,
                cv::THRESH_BINARY);
  cv::Mat labels;
  labels_mat.copyTo(labels, mask);

  cv_bridge::CvImage labels_bridge(
      rgb->header, sensor_msgs::image_encodings::TYPE_32FC1, labels);

  // Visualization
  cv_bridge::CvImageConstPtr rgb_bridge =
      cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);

  cv::Mat overlay = rgb_bridge->image;
  overlay.setTo(cv::Scalar(0, 0, 255), labels != 0);
  cv::namedWindow("Label overlay");
  cv::imshow("Label overlay", overlay);
  cv_bridge::CvImage overlay_bridge(
      rgb->header, sensor_msgs::image_encodings::BGR8, overlay);

  const float min_x = 0.075;
  const float max_x = 0.3;
  const float min_y = -0.09;
  const float max_y = 0.09;
  const float min_z = -0.09;
  const float max_z = 0.09;

  Eigen::Affine3f center(Eigen::Affine3f::Identity());
  center(0, 3) = (max_x + min_x) / 2;
  center(1, 3) = (max_y + min_y) / 2;
  center(2, 3) = (max_z + min_z) / 2;

  visualization_msgs::MarkerArray boxes;
  visualization_msgs::Marker l_box;
  l_box.header.frame_id = "camera_rgb_optical_frame";
  l_box.type = visualization_msgs::Marker::CUBE;
  l_box.ns = "hand_box";
  l_box.id = 0;
  l_box.color.b = 1;
  l_box.color.a = 0.3;
  l_box.scale.x = max_x - min_x;
  l_box.scale.y = max_y - min_y;
  l_box.scale.z = max_z - min_z;

  Eigen::Affine3f l_pose = l_matrix * center;
  l_box.pose.position.x = l_pose.translation().x();
  l_box.pose.position.y = l_pose.translation().y();
  l_box.pose.position.z = l_pose.translation().z();
  Eigen::Quaternionf l_rot(l_pose.rotation());
  l_box.pose.orientation.w = l_rot.w();
  l_box.pose.orientation.x = l_rot.x();
  l_box.pose.orientation.y = l_rot.y();
  l_box.pose.orientation.z = l_rot.z();
  boxes.markers.push_back(l_box);

  visualization_msgs::Marker l_pt;
  l_pt.header.frame_id = "camera_rgb_optical_frame";
  l_pt.type = visualization_msgs::Marker::SPHERE;
  l_pt.ns = "hand_pt";
  l_pt.id = 0;
  l_pt.color.g = 1;
  l_pt.color.a = 1;
  l_pt.scale.x = 0.04;
  l_pt.scale.y = 0.04;
  l_pt.scale.z = 0.04;
  l_pt.pose = l_box.pose;
  boxes.markers.push_back(l_pt);

  visualization_msgs::Marker r_box;
  r_box.header.frame_id = "camera_rgb_optical_frame";
  r_box.type = visualization_msgs::Marker::CUBE;
  r_box.ns = "hand_box";
  r_box.id = 1;
  r_box.color.b = 1;
  r_box.color.a = 0.3;
  r_box.scale.x = max_x - min_x;
  r_box.scale.y = max_y - min_y;
  r_box.scale.z = max_z - min_z;

  Eigen::Affine3f r_pose = r_matrix * center;
  r_box.pose.position.x = r_pose.translation().x();
  r_box.pose.position.y = r_pose.translation().y();
  r_box.pose.position.z = r_pose.translation().z();
  Eigen::Quaternionf r_rot(r_pose.rotation());
  r_box.pose.orientation.w = r_rot.w();
  r_box.pose.orientation.x = r_rot.x();
  r_box.pose.orientation.y = r_rot.y();
  r_box.pose.orientation.z = r_rot.z();
  boxes.markers.push_back(r_box);

  visualization_msgs::Marker r_pt;
  r_pt.header.frame_id = "camera_rgb_optical_frame";
  r_pt.type = visualization_msgs::Marker::SPHERE;
  r_pt.ns = "hand_pt";
  r_pt.id = 1;
  r_pt.color.g = 1;
  r_pt.color.a = 1;
  r_pt.scale.x = 0.04;
  r_pt.scale.y = 0.04;
  r_pt.scale.z = 0.04;
  r_pt.pose = r_box.pose;
  boxes.markers.push_back(r_pt);

  skeleton_pub_.publish(boxes);

  // Even though there is some time difference, we are assuming that we have
  // done our best to temporally align the images and now assume all the images
  // have the same timestamp.
  sensor_msgs::Image depth_out = *depth;
  depth_out.header.stamp = rgb->header.stamp;

  output_bag_->write(kRgbTopic, rgb->header.stamp, rgb);
  output_bag_->write(kDepthTopic, rgb->header.stamp, depth_out);
  output_bag_->write(kLabelsTopic, rgb->header.stamp,
                     labels_bridge.toImageMsg());
  output_bag_->write(kLabelOverlayTopic, rgb->header.stamp,
                     overlay_bridge.toImageMsg());

  cudaDeviceSynchronize();

  if (debug_) {
    cv::waitKey();
  }
}

void Labeling::set_debug(bool debug) { debug_ = debug; }

}  // namespace skinseg

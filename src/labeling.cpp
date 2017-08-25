#include "skin_segmentation/labeling.h"

#include "cv_bridge/cv_bridge.h"
#include "depth_image_proc/depth_traits.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ros/ros.h"
#include "rosbag/bag.h"
#include "sensor_msgs/Image.h"

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
      debug_(false) {}

void Labeling::Process(const Image::ConstPtr& rgb, const Image::ConstPtr& depth,
                       const Image::ConstPtr& thermal) {
  if (!rgb || !depth || !thermal) {
    ROS_ERROR("Got null image when processing labels!");
    return;
  }

  ROS_INFO("RGB - Depth skew: %f, RGB-Thermal skew: %f",
           (rgb->header.stamp - depth->header.stamp).toSec(),
           (rgb->header.stamp - thermal->header.stamp).toSec());

  double thermal_threshold;
  ros::param::param("thermal_threshold", thermal_threshold, 3650.0);

  cv::Mat thermal_projected;
  projection_.ProjectThermalOnRgb(rgb, depth, thermal, thermal_projected);

  cv::Mat thermal_fp;
  thermal_projected.convertTo(thermal_fp, CV_32F);

  cv::Mat labels_mat;
  cv::threshold(thermal_fp, labels_mat, thermal_threshold, 1,
                cv::THRESH_BINARY);
  cv_bridge::CvImage labels_bridge(
      rgb->header, sensor_msgs::image_encodings::TYPE_32FC1, labels_mat);

  // Visualization
  cv_bridge::CvImageConstPtr rgb_bridge =
      cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);

  // cv::namedWindow("RGB");
  // cv::imshow("RGB", rgb_bridge->image);

  // cv::Mat mask = thermal_projected != 0;
  // cv::Mat normalized_thermal(thermal_projected.rows, thermal_projected.cols,
  //                           CV_16UC1, cv::Scalar(0));
  // cv::normalize(thermal_projected, normalized_thermal, 0, 255,
  // cv::NORM_MINMAX,
  //              -1, mask);
  // cv::namedWindow("Labels");
  // cv::imshow("Labels", ConvertToColor(normalized_thermal));

  cv::Mat overlay = rgb_bridge->image;
  overlay.setTo(cv::Scalar(0, 0, 255), labels_mat != 0);
  cv::namedWindow("Overlay");
  cv::imshow("Overlay", overlay);
  cv_bridge::CvImage overlay_bridge(
      rgb->header, sensor_msgs::image_encodings::BGR8, overlay);

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

  if (debug_) {
    cv::waitKey();
  }
}

void Labeling::set_debug(bool debug) { debug_ = debug; }

}  // namespace skinseg

#include "skin_segmentation/labeling.h"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/opencv_utils.h"
#include "skin_segmentation/projection.h"

using sensor_msgs::Image;

namespace skinseg {
Labeling::Labeling(const Projection& projection) : projection_(projection) {}

void Labeling::Process(const Image::ConstPtr& rgb, const Image::ConstPtr& depth,
                       const Image::ConstPtr& thermal) {
  if (!rgb || !depth || !thermal) {
    ROS_ERROR("Got null image when processing labels!");
    return;
  }

  ROS_INFO("RGB - Depth skew: %f, RGB-Thermal skew: %f",
           (rgb->header.stamp - depth->header.stamp).toSec(),
           (rgb->header.stamp - thermal->header.stamp).toSec());

  double threshold;
  ros::param::param("thermal_threshold", threshold, 3650.0);

  cv::Mat thermal_projected;
  projection_.ProjectThermalOnRgb(rgb, depth, thermal, thermal_projected);

  cv::Mat thermal_fp;
  thermal_projected.convertTo(thermal_fp, CV_32F);

  cv::Mat labels;
  cv::threshold(thermal_fp, labels, threshold, 1, cv::THRESH_BINARY);
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
  overlay.setTo(cv::Scalar(0, 0, 255), labels != 0);
  cv::namedWindow("Overlay");
  cv::imshow("Overlay", overlay);

  cv::waitKey();
}

}  // namespace skinseg

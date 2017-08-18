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
  cv::namedWindow("RGB");
  cv::imshow("RGB", rgb_bridge->image);

  cv::namedWindow("Labels");
  cv::imshow("Labels", labels);

  cv::waitKey();
}

}  // namespace skinseg

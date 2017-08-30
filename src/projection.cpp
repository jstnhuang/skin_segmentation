#include "skin_segmentation/projection.h"

#include <cuda_runtime.h>
#include <algorithm>

#undef Success  // Evil workaround. nerf includes glx, which defines this again
#include "Eigen/Dense"
#include "cv_bridge/cv_bridge.h"
#include "depth_image_proc/depth_traits.h"
#include "image_geometry/pinhole_camera_model.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/opencv_utils.h"

using depth_image_proc::DepthTraits;
using image_geometry::PinholeCameraModel;
using sensor_msgs::CameraInfo;
using sensor_msgs::Image;

namespace skinseg {
Projection::Projection(const CameraInfo& rgbd_info,
                       const CameraInfo& thermal_info,
                       const Eigen::Affine3d& rgb_in_thermal)
    : rgbd_info_(rgbd_info),
      thermal_info_(thermal_info),
      rgb_in_thermal_(rgb_in_thermal),
      rgbd_model_(),
      thermal_model_(),
      debug_(false) {
  rgbd_model_.fromCameraInfo(rgbd_info);
  thermal_model_.fromCameraInfo(thermal_info);
}

void Projection::RgbdMouseCallback(int event, int x, int y, int flags,
                                   void* data) {
  if (event != cv::EVENT_LBUTTONDOWN) {
    return;
  }
  Projection* proj = static_cast<Projection*>(data);
  ROS_INFO("Clicked on x: %d, y: %d", x, y);

  cv::Point2d rgb_pt(x, y);
  cv::Point2d thermal_pt = proj->GetThermalPixel(proj->depth_, rgb_pt);

  cv::namedWindow("RGB");
  cv::Mat rgb_annotated = proj->rgb_.clone();
  cv::circle(rgb_annotated, rgb_pt, 5, cv::Scalar(0, 0, 255), 1);
  cv::imshow("RGB", rgb_annotated);

  cv::namedWindow("Depth");
  cv::Mat normalized_depth;
  cv::normalize(proj->depth_, normalized_depth, 0, 255, cv::NORM_MINMAX);
  cv::Mat depth_color = ConvertToColor(normalized_depth);
  cv::circle(depth_color, rgb_pt, 5, cv::Scalar(0, 0, 255), 1);
  cv::imshow("Depth", depth_color);

  cv::Mat normalized_thermal_image;
  cv::normalize(proj->thermal_, normalized_thermal_image, 0, 255,
                cv::NORM_MINMAX);
  cv::Mat normalized_thermal_color = ConvertToColor(normalized_thermal_image);
  cv::circle(normalized_thermal_color, thermal_pt, 5, cv::Scalar(0, 0, 255), 1);
  cv::namedWindow("Normalized thermal");
  cv::imshow("Normalized thermal", normalized_thermal_color);
}

void Projection::ThermalMouseCallback(int event, int x, int y, int flags,
                                      void* data) {
  if (event != cv::EVENT_LBUTTONDOWN) {
    return;
  }
  Projection* proj = static_cast<Projection*>(data);
  ROS_INFO("Clicked on x: %d, y: %d", x, y);

  std::pair<int, int> key(y, x);
  if (proj->thermal_to_rgb_.find(key) == proj->thermal_to_rgb_.end()) {
    ROS_ERROR("Thermal point not in lookup table.");
    return;
  }
  const std::list<cv::Point2d>& rgb_pts = proj->thermal_to_rgb_[key];
  std::list<cv::Point2d>::const_iterator it;
  for (it = rgb_pts.begin(); it != rgb_pts.end(); ++it) {
    ROS_INFO("Matching RGBD point (r, c): %f %f", it->y, it->x);
  }

  cv::namedWindow("RGB");
  cv::Mat rgb_annotated = proj->rgb_.clone();

  for (it = rgb_pts.begin(); it != rgb_pts.end(); ++it) {
    const cv::Point2d& rgb_pt = *it;
    cv::Scalar color(255, 0, 0);
    if (it == rgb_pts.begin()) {
      color = cv::Scalar(0, 255, 0);
    }
    cv::circle(rgb_annotated, rgb_pt, 5, color, 1);
    cv::imshow("RGB", rgb_annotated);
  }

  cv::namedWindow("Depth");
  cv::Mat normalized_depth;
  cv::normalize(proj->depth_, normalized_depth, 0, 255, cv::NORM_MINMAX);
  cv::Mat depth_color = ConvertToColor(normalized_depth);
  for (it = rgb_pts.begin(); it != rgb_pts.end(); ++it) {
    const cv::Point2d& rgb_pt = *it;
    cv::Scalar color(255, 0, 0);
    if (it == rgb_pts.begin()) {
      color = cv::Scalar(0, 255, 0);
    }
    cv::circle(depth_color, rgb_pt, 5, color, 1);
    cv::imshow("Depth", depth_color);
  }

  cv::Mat normalized_thermal_image;
  cv::normalize(proj->thermal_, normalized_thermal_image, 0, 255,
                cv::NORM_MINMAX);
  cv::Mat normalized_thermal_color = ConvertToColor(normalized_thermal_image);
  cv::Point2d thermal_pt(x, y);
  cv::circle(normalized_thermal_color, thermal_pt, 5, cv::Scalar(0, 0, 255), 1);
  cv::namedWindow("Normalized thermal");
  cv::imshow("Normalized thermal", normalized_thermal_color);
}

void Projection::set_debug(bool debug) { debug_ = debug; }

inline float Projection::GetRgbDepth(const cv::Mat& depth_image,
                                     const cv::Point2d& rgb_pt) const {
  int rgb_row_i = std::min<double>(std::max<double>(round(rgb_pt.y), 0),
                                   depth_image.rows - 1);
  int rgb_col_i = std::min<double>(std::max<double>(round(rgb_pt.x), 0),
                                   depth_image.cols - 1);
  uint16_t raw_depth = depth_image.at<uint16_t>(rgb_row_i, rgb_col_i);
  if (raw_depth == 0) {
    return 0;
  }
  return DepthTraits<uint16_t>::toMeters(raw_depth);
}

inline cv::Point2d Projection::GetThermalPixel(
    const cv::Mat& depth_image, const cv::Point2d& rgb_pt) const {
  float depth = GetRgbDepth(depth_image, rgb_pt);
  if (depth == 0) {
    return cv::Point2d(-1, -1);
  }

  cv::Point3d rgb_ray = rgbd_model_.projectPixelTo3dRay(rgb_pt);
  cv::Point3d xyz_rgb = rgb_ray * depth;

  Eigen::Vector3d xyz_in_rgb;
  xyz_in_rgb << xyz_rgb.x, xyz_rgb.y, xyz_rgb.z;
  Eigen::Vector3d xyz_in_thermal = rgb_in_thermal_ * xyz_in_rgb;

  cv::Point3d xyz_thermal(xyz_in_thermal.x(), xyz_in_thermal.y(),
                          xyz_in_thermal.z());
  // TODO: we should be unrectifying here (or rectifying the thermal image), but
  // it seems to make it worse.
  return thermal_model_.project3dToPixel(xyz_thermal);
}

void Projection::GetCameraData(CameraData* data) {
  data->inv_depth_fx = 1.0 / rgbd_model_.fx();
  data->inv_depth_fy = 1.0 / rgbd_model_.fy();
  data->depth_cx = rgbd_model_.cx();
  data->depth_cy = rgbd_model_.cy();
  data->depth_Tx = rgbd_model_.Tx();
  data->depth_Ty = rgbd_model_.Ty();
  data->thermal_fx = thermal_model_.fx();
  data->thermal_fy = thermal_model_.fy();
  data->thermal_cx = thermal_model_.cx();
  data->thermal_cy = thermal_model_.cy();
  data->thermal_Tx = thermal_model_.Tx();
  data->thermal_Ty = thermal_model_.Ty();
}

void Projection::GetRgbdCameraInfo(sensor_msgs::CameraInfo* info) {
  *info = rgbd_info_;
}
}  // namespace skinseg

#include "skin_segmentation/projection.h"

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
      thermal_model_() {
  rgbd_model_.fromCameraInfo(rgbd_info);
  thermal_model_.fromCameraInfo(thermal_info);
}

void Projection::ProjectThermalOnRgb(
    const sensor_msgs::Image::ConstPtr& rgb,
    const sensor_msgs::Image::ConstPtr& depth,
    const sensor_msgs::Image::ConstPtr& thermal,
    cv::OutputArray thermal_projected) {
  cv_bridge::CvImageConstPtr rgb_bridge =
      cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);
  cv_bridge::CvImageConstPtr depth_bridge = cv_bridge::toCvShare(depth);
  depth_image_ = depth_bridge->image;
  cv_bridge::CvImageConstPtr thermal_bridge = cv_bridge::toCvShare(thermal);
  cv::normalize(thermal_bridge->image, normalized_thermal_image_, 0, 255,
                cv::NORM_MINMAX);

  // Registration of the thermal image to the RGB image is done by projecting
  // the RGBD pixel into the thermal image and copying the pixel in the thermal
  // image.
  thermal_projected.create(rgb_bridge->image.rows, rgb_bridge->image.cols,
                           CV_16UC1);
  cv::Mat thermal_projected_mat = thermal_projected.getMat();
  thermal_projected_mat = cv::Scalar(0);

  // Visualization of the RGB image projected onto the thermal image.
  cv::Mat rgb_projected(thermal_bridge->image.rows, thermal_bridge->image.cols,
                        CV_8UC3, cv::Scalar(68, 51, 11));
  cv::Mat_<cv::Vec3b> _rgb_projected = rgb_projected;
  cv::Mat_<cv::Vec3b> _rgb = rgb_bridge->image;
  cv::Mat z_buffer(thermal_bridge->image.rows, thermal_bridge->image.cols,
                   CV_32F, cv::Scalar(0));

  for (int rgb_row = 0; rgb_row < rgb_bridge->image.rows; ++rgb_row) {
    for (int rgb_col = 0; rgb_col < rgb_bridge->image.cols; ++rgb_col) {
      cv::Point2d thermal_pt = GetThermalPixel(cv::Point2d(rgb_col, rgb_row));
      if (thermal_pt.x < 0 || thermal_pt.x >= thermal_bridge->image.cols ||
          thermal_pt.y < 0 || thermal_pt.y >= thermal_bridge->image.rows) {
        continue;
      }

      int t_row = round(thermal_pt.y);
      int t_col = round(thermal_pt.x);

      uint16_t raw_depth = depth_image_.at<uint16_t>(rgb_row, rgb_col);
      double depth = DepthTraits<uint16_t>::toMeters(raw_depth);
      double prev_depth = z_buffer.at<double>(t_row, t_col);
      bool depth_check_passed = prev_depth == 0 || depth < prev_depth;
      if (depth_check_passed) {
        z_buffer.at<double>(t_row, t_col) = depth;
        thermal_projected_mat.at<uint16_t>(rgb_row, rgb_col) =
            thermal_bridge->image.at<uint16_t>(t_row, t_col);
        _rgb_projected(t_row, t_col)[0] = _rgb(rgb_row, rgb_col)[0];
        _rgb_projected(t_row, t_col)[1] = _rgb(rgb_row, rgb_col)[1];
        _rgb_projected(t_row, t_col)[2] = _rgb(rgb_row, rgb_col)[2];
      }
    }
  }

  rgb_projected = _rgb_projected;
  cv::namedWindow("RGB projected");
  cv::imshow("RGB projected", rgb_projected);

  cv::namedWindow("RGB");
  cv::imshow("RGB", rgb_bridge->image);

  cv::namedWindow("Depth");
  cv::Mat normalized_depth;
  cv::normalize(depth_bridge->image, normalized_depth, 0, 255, cv::NORM_MINMAX);
  cv::imshow("Depth", ConvertToColor(normalized_depth));

  cv::namedWindow("Projected labels");
  cv::Mat projected_labels;
  cv::Mat mask = NonZeroMask(thermal_projected_mat);

  cv::normalize(thermal_projected_mat, projected_labels, 0, 255,
                cv::NORM_MINMAX, -1, mask);
  cv::Mat labels_color = ConvertToColor(projected_labels);
  cv::imshow("Projected labels", labels_color);

  cv::Mat normalized_thermal_color = ConvertToColor(normalized_thermal_image_);
  cv::namedWindow("Normalized thermal");
  cv::imshow("Normalized thermal", normalized_thermal_color);

  double alpha;
  ros::param::param("overlay_alpha", alpha, 0.5);
  cv::Mat overlay;
  cv::addWeighted(labels_color, alpha, rgb_bridge->image, 1 - alpha, 0.0,
                  overlay);
  cv::namedWindow("Overlay");
  cv::imshow("Overlay", overlay);
}

cv::Point2d Projection::GetThermalPixel(const cv::Point2d& rgb_pt, bool debug) {
  // Extract all the parameters we need
  double inv_rgbd_fx = 1.0 / rgbd_model_.fx();
  double inv_rgbd_fy = 1.0 / rgbd_model_.fy();
  double rgbd_cx = rgbd_model_.cx(), rgbd_cy = rgbd_model_.cy();
  double rgbd_Tx = rgbd_model_.Tx(), rgbd_Ty = rgbd_model_.Ty();
  double thermal_fx = thermal_model_.fx(), thermal_fy = thermal_model_.fy();
  double thermal_cx = thermal_model_.cx(), thermal_cy = thermal_model_.cy();
  double thermal_Tx = thermal_model_.Tx(), thermal_Ty = thermal_model_.Ty();

  if (debug) {
    cv::Mat normalized_depth;
    cv::normalize(depth_image_, normalized_depth, 0, 255, cv::NORM_MINMAX);
    cv::circle(normalized_depth, rgb_pt, 5, cv::Scalar(0, 0, 255), 1);
    cv::imshow("Depth", ConvertToColor(normalized_depth));
  }

  uint16_t raw_depth = depth_image_.at<uint16_t>(rgb_pt.y, rgb_pt.x);
  if (raw_depth == 0) {
    return cv::Point2d(-1, -1);
  }
  double depth = DepthTraits<uint16_t>::toMeters(raw_depth);
  Eigen::Vector3d xyz_depth;
  // clang-format off
  xyz_depth << ((rgb_pt.x - rgbd_cx) * depth - rgbd_Tx) * inv_rgbd_fx,
               ((rgb_pt.y - rgbd_cy) * depth - rgbd_Ty) * inv_rgbd_fy,
               depth;
  // clang-format on

  Eigen::Vector3d xyz_thermal = rgb_in_thermal_ * xyz_depth;
  if (debug) {
    ROS_INFO_STREAM("xyz in depth: " << xyz_depth
                                     << ", in thermal: " << xyz_thermal);
  }

  double inv_Z = 1.0 / xyz_thermal.z();
  int u_thermal =
      (thermal_fx * xyz_thermal.x() + thermal_Tx) * inv_Z + thermal_cx + 0.5;
  int v_thermal =
      (thermal_fy * xyz_thermal.y() + thermal_Ty) * inv_Z + thermal_cy + 0.5;
  return cv::Point2d(u_thermal, v_thermal);
}

void Projection::MouseCallback(int event, int x, int y, int flags, void* data) {
  if (event != cv::EVENT_LBUTTONDOWN) {
    return;
  }
  Projection* proj = static_cast<Projection*>(data);
  ROS_INFO("Clicked on x: %d, y: %d", x, y);

  const bool kDebug = true;
  cv::Point2d thermal_pt = proj->GetThermalPixel(cv::Point2d(x, y), kDebug);
  cv::Mat annotated_thermal = proj->normalized_thermal_image_.clone();
  cv::circle(annotated_thermal, thermal_pt, 5, cv::Scalar(0, 0, 255), 1);
  cv::imshow("Normalized thermal", ConvertToColor(annotated_thermal));
}
}  // namespace skinseg

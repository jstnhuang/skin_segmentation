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
      thermal_model_(),
      debug_(false) {
  rgbd_model_.fromCameraInfo(rgbd_info);
  thermal_model_.fromCameraInfo(thermal_info);
}

void Projection::set_debug(bool debug) { debug_ = debug; }

void Projection::ProjectThermalOnRgb(
    const sensor_msgs::Image::ConstPtr& rgb,
    const sensor_msgs::Image::ConstPtr& depth,
    const sensor_msgs::Image::ConstPtr& thermal,
    cv::OutputArray thermal_projected) {
  cv_bridge::CvImageConstPtr rgb_bridge =
      cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);
  cv_bridge::CvImageConstPtr depth_bridge = cv_bridge::toCvShare(depth);
  cv_bridge::CvImageConstPtr thermal_bridge = cv_bridge::toCvShare(thermal);
  cv::Mat normalized_thermal;
  cv::normalize(thermal_bridge->image, normalized_thermal, 0, 255,
                cv::NORM_MINMAX);

  thermal_projected.create(rgb_bridge->image.rows, rgb_bridge->image.cols,
                           CV_16UC1);
  cv::Mat thermal_projected_mat = thermal_projected.getMat();
  thermal_projected_mat = cv::Scalar(0);

  // Extract all the parameters we need
  double inv_rgbd_fx = 1.0 / rgbd_model_.fx();
  double inv_rgbd_fy = 1.0 / rgbd_model_.fy();
  double rgbd_cx = rgbd_model_.cx(), rgbd_cy = rgbd_model_.cy();
  double rgbd_Tx = rgbd_model_.Tx(), rgbd_Ty = rgbd_model_.Ty();
  double thermal_fx = thermal_model_.fx(), thermal_fy = thermal_model_.fy();
  double thermal_cx = thermal_model_.cx(), thermal_cy = thermal_model_.cy();
  double thermal_Tx = thermal_model_.Tx(), thermal_Ty = thermal_model_.Ty();

  for (int rgb_row = 0; rgb_row < rgb_bridge->image.rows; ++rgb_row) {
    for (int rgb_col = 0; rgb_col < rgb_bridge->image.cols; ++rgb_col) {
      uint16_t raw_depth = depth_bridge->image.at<uint16_t>(rgb_row, rgb_col);
      if (raw_depth == 0) {
        continue;
      }
      double depth = DepthTraits<uint16_t>::toMeters(raw_depth);
      Eigen::Vector3d xyz_depth;
      // clang-format off
      xyz_depth << ((rgb_col - rgbd_cx) * depth - rgbd_Tx) * inv_rgbd_fx,
                   ((rgb_row - rgbd_cy) * depth - rgbd_Ty) * inv_rgbd_fy,
                   depth;
      // clang-format on

      Eigen::Vector3d xyz_thermal = rgb_in_thermal_ * xyz_depth;
      double inv_Z = 1.0 / xyz_thermal.z();
      int u_thermal = (thermal_fx * xyz_thermal.x() + thermal_Tx) * inv_Z +
                      thermal_cx + 0.5;
      int v_thermal = (thermal_fy * xyz_thermal.y() + thermal_Ty) * inv_Z +
                      thermal_cy + 0.5;

      if (u_thermal < 0 || u_thermal >= thermal_bridge->image.cols ||
          v_thermal < 0 || v_thermal >= thermal_bridge->image.rows) {
        continue;
      }

      thermal_projected_mat.at<uint16_t>(rgb_row, rgb_col) =
          thermal_bridge->image.at<uint16_t>(v_thermal, u_thermal);
    }
  }

  if (debug_) {
    cv::namedWindow("RGB");
    cv::imshow("RGB", rgb_bridge->image);

    cv::namedWindow("Depth");
    cv::Mat normalized_depth;
    cv::normalize(depth_bridge->image, normalized_depth, 0, 255,
                  cv::NORM_MINMAX);
    cv::imshow("Depth", ConvertToColor(normalized_depth));

    cv::namedWindow("Projected labels");
    cv::Mat projected_labels;
    cv::Mat mask = (thermal_projected_mat != 0);
    cv::Mat mask2;
    cv::threshold(mask, mask2, 0.5, 255, cv::THRESH_BINARY);

    cv::namedWindow("Mask");
    cv::imshow("Mask", mask2);
    cv::normalize(thermal_projected_mat, projected_labels, 0, 255,
                  cv::NORM_MINMAX, -1, mask2);
    cv::Mat labels_color = ConvertToColor(projected_labels);
    cv::imshow("Projected labels", labels_color);

    cv::Mat normalized_thermal_color = ConvertToColor(normalized_thermal);
    cv::namedWindow("Normalized thermal");
    cv::imshow("Normalized thermal", normalized_thermal_color);

    double alpha;
    ros::param::param("overlay_alpha", alpha, 0.5);
    cv::Mat overlay;
    cv::addWeighted(labels_color, alpha, rgb_bridge->image, 1 - alpha, 0.0,
                    overlay);
    cv::namedWindow("Overlay");
    cv::imshow("Overlay", overlay);

    cv::waitKey();
  }
}
}  // namespace skinseg

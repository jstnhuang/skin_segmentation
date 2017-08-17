#include "skin_segmentation/projection.h"

#include <algorithm>

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
  cv_bridge::CvImageConstPtr thermal_bridge = cv_bridge::toCvShare(thermal);

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

  for (double rgb_row = -0.5; rgb_row < rgb_bridge->image.rows + 0.5;
       rgb_row += 0.5) {
    for (double rgb_col = -0.5; rgb_col < rgb_bridge->image.cols + 0.5;
         rgb_col += 0.5) {
      cv::Point2d rgb_pt(rgb_col, rgb_row);
      cv::Point2d thermal_pt = GetThermalPixel(depth_bridge->image, rgb_pt);
      if (thermal_pt.x < 0 || thermal_pt.x >= thermal_bridge->image.cols ||
          thermal_pt.y < 0 || thermal_pt.y >= thermal_bridge->image.rows) {
        continue;
      }

      int r_row = round(rgb_pt.y);
      int r_col = round(rgb_pt.x);
      int t_row = round(thermal_pt.y);
      int t_col = round(thermal_pt.x);

      float depth = GetRgbDepth(depth_bridge->image, rgb_pt);
      float prev_depth = z_buffer.at<float>(t_row, t_col);
      bool depth_check_passed = prev_depth == 0 || depth < prev_depth;
      if (depth_check_passed) {
        z_buffer.at<float>(t_row, t_col) = depth;
        thermal_projected_mat.at<uint16_t>(r_row, r_col) =
            thermal_bridge->image.at<uint16_t>(t_row, t_col);
        _rgb_projected(t_row, t_col)[0] = _rgb(r_row, r_col)[0];
        _rgb_projected(t_row, t_col)[1] = _rgb(r_row, r_col)[1];
        _rgb_projected(t_row, t_col)[2] = _rgb(r_row, r_col)[2];
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
  cv::Mat projected_labels(thermal_projected_mat.rows,
                           thermal_projected_mat.cols,
                           thermal_projected_mat.type(), cv::Scalar(0));
  cv::normalize(thermal_projected_mat, projected_labels, 0, 255,
                cv::NORM_MINMAX, -1, NonZeroMask(thermal_projected_mat));
  cv::Mat labels_color = ConvertToColor(projected_labels);
  cv::imshow("Projected labels", labels_color);

  cv::Mat normalized_thermal_image;
  cv::normalize(thermal_bridge->image, normalized_thermal_image, 0, 255,
                cv::NORM_MINMAX);
  cv::Mat normalized_thermal_color = ConvertToColor(normalized_thermal_image);
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

float Projection::GetRgbDepth(const cv::Mat& depth_image,
                              const cv::Point2d& rgb_pt) {
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

cv::Point2d Projection::GetThermalPixel(const cv::Mat& depth_image,
                                        const cv::Point2d& rgb_pt) {
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
  return thermal_model_.project3dToPixel(xyz_thermal);
}
}  // namespace skinseg

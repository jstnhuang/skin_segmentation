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
      thermal_model_(),
      debug_(false) {
  rgbd_model_.fromCameraInfo(rgbd_info);
  thermal_model_.fromCameraInfo(thermal_info);
}

void Projection::ProjectThermalOnRgb(const Image::ConstPtr& rgb,
                                     const Image::ConstPtr& depth,
                                     const Image::ConstPtr& thermal,
                                     cv::OutputArray thermal_projected) const {
  cv_bridge::CvImageConstPtr depth_bridge = cv_bridge::toCvShare(depth);
  cv_bridge::CvImageConstPtr thermal_bridge = cv_bridge::toCvShare(thermal);

  // Registration of the thermal image to the RGB image is done by projecting
  // the RGBD pixel into the thermal image and copying the pixel in the thermal
  // image.
  thermal_projected.create(rgb->height, rgb->width, CV_16UC1);
  cv::Mat thermal_projected_mat = thermal_projected.getMat();
  thermal_projected_mat = cv::Scalar(0);
  cv::Mat z_buffer = cv::Mat::zeros(thermal_bridge->image.rows,
                                    thermal_bridge->image.cols, CV_32F);

  int rgb_rows = rgb->height;
  int rgb_cols = rgb->width;

  for (double rgb_row = -0.5; rgb_row < rgb_rows + 0.5; rgb_row += 0.5) {
    for (double rgb_col = -0.5; rgb_col < rgb_cols + 0.5; rgb_col += 0.5) {
      cv::Point2d rgb_pt(rgb_col, rgb_row);
      cv::Point2d thermal_pt = GetThermalPixel(depth_bridge->image, rgb_pt);

      int r_row = round(rgb_pt.y);
      int r_col = round(rgb_pt.x);
      int t_row = round(thermal_pt.y);
      int t_col = round(thermal_pt.x);

      if (t_col < 0 || t_col >= thermal_bridge->image.cols || t_row < 0 ||
          t_row >= thermal_bridge->image.rows) {
        continue;
      }

      if (r_col < 0 || r_col >= rgb_rows || r_row < 0 || r_row >= rgb_cols) {
        continue;
      }

      float depth = GetRgbDepth(depth_bridge->image, rgb_pt);
      float prev_depth = z_buffer.at<float>(t_row, t_col);
      bool depth_check_passed = prev_depth == 0 || depth < prev_depth;
      if (depth_check_passed) {
        z_buffer.at<float>(t_row, t_col) = depth;
        thermal_projected_mat.at<uint16_t>(r_row, r_col) =
            thermal_bridge->image.at<uint16_t>(t_row, t_col);
      }
    }
  }

  if (debug_) {
    cv_bridge::CvImageConstPtr rgb_bridge =
        cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);
    cv::namedWindow("RGB");
    cv::imshow("RGB", rgb_bridge->image);

    cv::namedWindow("Depth");
    cv::Mat normalized_depth;
    cv::normalize(depth_bridge->image, normalized_depth, 0, 255,
                  cv::NORM_MINMAX);
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
    cv::waitKey();
  }
}

void Projection::set_debug(bool debug) { debug_ = debug; }

float Projection::GetRgbDepth(const cv::Mat& depth_image,
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

cv::Point2d Projection::GetThermalPixel(const cv::Mat& depth_image,
                                        const cv::Point2d& rgb_pt) const {
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

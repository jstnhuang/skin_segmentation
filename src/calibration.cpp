#include "skin_segmentation/calibration.h"

#include <math.h>

#include "Eigen/Dense"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/opencv_utils.h"

using sensor_msgs::Image;

namespace skinseg {
Calibration::Calibration()
    : rgb_corners_(), thermal_corners_(), processed_pairs_(0) {}

void Calibration::AddImagePair(const Image::ConstPtr& rgb_msg,
                               const Image::ConstPtr& thermal_msg) {
  const cv::Size kSize(7, 7);

  cv_bridge::CvImageConstPtr cv_thermal = cv_bridge::toCvShare(thermal_msg);
  Corners thermal_corners;
  bool thermal_found =
      cv::findChessboardCorners(cv_thermal->image, kSize, thermal_corners);
  if (!thermal_found) {
    ROS_ERROR("Failed to find chessboard in thermal image");
  }

  cv_bridge::CvImageConstPtr cv_rgb =
      cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
  Corners rgb_corners;
  bool rgb_found = cv::findChessboardCorners(cv_rgb->image, kSize, rgb_corners);
  if (!rgb_found) {
    ROS_ERROR("Failed to find chessboard in RGB image");
  }

  cv::Mat rgb_viz = cv_rgb->image.clone();
  cv::drawChessboardCorners(rgb_viz, kSize, rgb_corners, rgb_found);
  cv::namedWindow("RGB");
  cv::imshow("RGB", rgb_viz);
  cv::Mat thermal_viz = cv_thermal->image.clone();
  cv::drawChessboardCorners(thermal_viz, kSize, thermal_corners, thermal_found);
  cv::namedWindow("Thermal");
  cv::imshow("Thermal", thermal_viz);
  cv::waitKey();

  if (!rgb_found || !thermal_found) {
    return;
  }

  Corners rgb_fixed;
  ReorderChessCorners(rgb_corners, kSize, &rgb_fixed);
  Corners thermal_fixed;
  ReorderChessCorners(thermal_corners, kSize, &thermal_fixed);

  rgb_corners_.push_back(rgb_fixed);
  thermal_corners_.push_back(thermal_fixed);
}

void Calibration::Run() {
  ROS_INFO("Got %ld matching images", rgb_corners_.size());

  // Sample at most 50 images uniformly
  std::vector<Corners> rgb_corners;
  std::vector<Corners> thermal_corners;
  if (rgb_corners_.size() > 50) {
    double ratio = rgb_corners_.size() / 50.0;
    for (double index = 0; index < rgb_corners_.size(); index += ratio) {
      int index_i = static_cast<int>(floor(index));
      rgb_corners.push_back(rgb_corners_[index_i]);
      thermal_corners.push_back(thermal_corners_[index_i]);
    }
  } else {
    rgb_corners = rgb_corners_;
    thermal_corners = thermal_corners_;
  }

  // Build the chessboard model. It is simply modeled as a plane with the origin
  // at the top left (interior) corner, with x pointing to the right and y
  // pointing down.
  double square_size;
  ros::param::param<double>("square_size", square_size, 0.03967);
  std::vector<cv::Vec3f> chess_points(49);
  for (int row = 0; row < 7; ++row) {
    for (int col = 0; col < 7; ++col) {
      chess_points[7 * row + col] =
          cv::Vec3f(square_size * col, square_size * row, 0);
    }
  }
  std::vector<std::vector<cv::Vec3f> > object_points(rgb_corners.size());
  for (size_t i = 0; i < rgb_corners.size(); ++i) {
    object_points[i] = chess_points;
  }

  // Calibrate the RGB camera.
  const int kRgbWidth = 640;
  const int kRgbHeight = 480;
  const float kRgbFxGuess = 525;
  const float kRgbFyGuess = 525;
  const float kRgbCx = 319.5;
  const float kRgbCy = 239.5;
  const int kCalibFlags = CV_CALIB_USE_INTRINSIC_GUESS +
                          CV_CALIB_FIX_PRINCIPAL_POINT +
                          CV_CALIB_RATIONAL_MODEL;

  cv::Mat rgb_camera_matrix = (cv::Mat_<float>(3, 3) << kRgbFxGuess, 0, kRgbCx,
                               0, kRgbFyGuess, kRgbCy, 0, 0, 1);
  cv::Mat rgb_dist_coeffs;
  std::vector<cv::Mat> rgb_rvecs;
  std::vector<cv::Mat> rgb_tvecs;
  double rgb_error = cv::calibrateCamera(
      object_points, rgb_corners, cv::Size(kRgbWidth, kRgbHeight),
      rgb_camera_matrix, rgb_dist_coeffs, rgb_rvecs, rgb_tvecs, kCalibFlags);
  ROS_INFO_STREAM("RGB calibrated, error: "
                  << rgb_error << ", camera matrix: " << rgb_camera_matrix
                  << ", dist coeffs: " << rgb_dist_coeffs);

  // Calibrate the thermal camera.
  const int kThermalWidth = 640;
  const int kThermalHeight = 512;
  const float kThermalFxGuess = 735;
  const float kThermalFyGuess = 735;
  const float kThermalCx = 319.5;
  const float kThermalCy = 255.5;
  cv::Mat thermal_camera_matrix =
      (cv::Mat_<float>(3, 3) << kThermalFxGuess, 0, kThermalCx, 0,
       kThermalFyGuess, kThermalCy, 0, 0, 1);
  cv::Mat thermal_dist_coeffs;
  std::vector<cv::Mat> thermal_rvecs;
  std::vector<cv::Mat> thermal_tvecs;
  double thermal_error = cv::calibrateCamera(
      object_points, thermal_corners, cv::Size(kThermalWidth, kThermalHeight),
      thermal_camera_matrix, thermal_dist_coeffs, thermal_rvecs, thermal_tvecs,
      kCalibFlags);
  ROS_INFO_STREAM("Thermal calibrated, error: "
                  << thermal_error
                  << ", camera matrix: " << thermal_camera_matrix
                  << ", dist coeffs: " << thermal_dist_coeffs);

  // Compute estimated transform between the two cameras for each view.
  cv::Mat average_transform = cv::Mat::zeros(4, 4, CV_32F);
  int count = 0;
  for (size_t i = 0; i < rgb_rvecs.size(); ++i) {
    cv::Mat rgb_rot;
    cv::Rodrigues(rgb_rvecs[i], rgb_rot);
    cv::Mat rgb_transform = cv::Mat::eye(4, 4, CV_32F);
    rgb_rot.copyTo(rgb_transform.rowRange(0, 3).colRange(0, 3));
    rgb_tvecs[i].copyTo(rgb_transform.col(3).rowRange(0, 3));

    cv::Mat thermal_rot;
    cv::Rodrigues(thermal_rvecs[i], thermal_rot);
    cv::Mat thermal_transform = cv::Mat::eye(4, 4, CV_32F);
    thermal_rot.copyTo(thermal_transform.rowRange(0, 3).colRange(0, 3));
    thermal_tvecs[i].copyTo(thermal_transform.col(3).rowRange(0, 3));

    cv::Mat thermal_in_rgb = rgb_transform * thermal_transform.inv();
    if (thermal_in_rgb.at<float>(0, 0) > 0.95 &&
        thermal_in_rgb.at<float>(1, 1) > 0.95) {
      average_transform += thermal_in_rgb;
      ++count;
    }
  }
  average_transform /= count;

  ROS_INFO_STREAM("thermal_in_rgb\n" << average_transform);
}
}  // namespace skinseg

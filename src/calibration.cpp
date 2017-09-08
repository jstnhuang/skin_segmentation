#include "skin_segmentation/calibration.h"

#include <math.h>
#include <iostream>

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

  if (!rgb_found) {
    std::cout << "Chessboard not found in RGB, press any key to skip"
              << std::endl;
    cv::waitKey();
    return;
  }

  cv_bridge::CvImageConstPtr cv_thermal = cv_bridge::toCvCopy(thermal_msg);
  cv::Mat thermal;
  cv::cvtColor(cv_thermal->image, thermal, CV_RGB2GRAY);
  cv::namedWindow("Thermal");
  cv::imshow("Thermal", thermal);

  context_.thermal_orig = thermal;
  context_.corners.clear();

  char command = ' ';
  while (command != 'd' && command != 's') {
    std::cout << "Commands:" << std::endl;
    std::cout << "t: Try to find chessboard corners automatically" << std::endl;
    std::cout << "c: Clear corners" << std::endl;
    std::cout << "x: Delete last manually labeled corner" << std::endl;
    std::cout << "s: Skip this image pair" << std::endl;
    std::cout << "d: Save corners and move to next image" << std::endl;

    command = (char)cv::waitKey(0);

    bool thermal_found = false;
    if (command == 't') {
      Corners rgb_corners;
      thermal_found =
          cv::findChessboardCorners(thermal, kSize, context_.corners);
      cornerSubPix(
          context_.thermal_orig, context_.corners, cv::Size(8, 8),
          cv::Size(-1, -1),
          cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
      if (!thermal_found) {
        ROS_ERROR("Failed to find chessboard in thermal image.");
      }
    } else if (command == 'c') {
      context_.corners.clear();
    } else if (command == 'x') {
      context_.corners.pop_back();
    }

    if (command != 't') {
      cv::Mat thermal_viz = cv_thermal->image.clone();
      cv::drawChessboardCorners(thermal_viz, kSize, context_.corners,
                                thermal_found);
      cv::imshow("Thermal", thermal_viz);
    }

    if (command == 'd' && context_.corners.size() != 49) {
      std::cout << "Cannot commit, not all the corners have been labeled."
                << std::endl;
      continue;
    }
  }

  if (command == 's') {
    return;
  }

  Corners rgb_fixed;
  ReorderChessCorners(rgb_corners, kSize, &rgb_fixed);
  Corners thermal_fixed;
  ReorderChessCorners(context_.corners, kSize, &thermal_fixed);

  rgb_corners_.push_back(rgb_fixed);
  thermal_corners_.push_back(thermal_fixed);

  Run();
}

void Calibration::MouseCallback(int event, int x, int y, int flags,
                                void* data) {
  if (event != cv::EVENT_LBUTTONDOWN) {
    return;
  }
  ROS_INFO("Clicked x: %d, y: %d", x, y);
  Calibration* calib = static_cast<Calibration*>(data);
  ThermalLabelingContext* context = &calib->context_;
  context->corners.push_back(cv::Vec2f(x, y));
  cornerSubPix(context->thermal_orig, context->corners, cv::Size(5, 5),
               cv::Size(-1, -1),
               cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
  cv::Mat thermal_viz = context->thermal_orig.clone();
  bool thermal_found = context->corners.size() == 49;
  cv::drawChessboardCorners(thermal_viz, cv::Size(7, 7), context->corners,
                            thermal_found);
  cv::imshow("Thermal", thermal_viz);
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

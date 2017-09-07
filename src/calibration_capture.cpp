// Utility for saving RGB/Thermal frames for calibration purposes.
// They are saved to a bag file.

#include <iostream>
#include <string>

#include "cv_bridge/cv_bridge.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "ros/ros.h"
#include "rosbag/bag.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/opencv_utils.h"

using sensor_msgs::Image;
typedef message_filters::sync_policies::ApproximateTime<Image, Image> MyPolicy;

namespace skinseg {
class CalibrationCapture {
 public:
  CalibrationCapture(const std::string& bag_path);
  void Callback(const Image::ConstPtr& rgb_msg,
                const Image::ConstPtr& thermal_msg);
  void Save();

 private:
  std::string bag_path_;
  Image::ConstPtr latest_rgb_;
  Image::ConstPtr latest_thermal_;
  ros::NodeHandle nh_;
  ros::Publisher rgb_pub_;
  ros::Publisher thermal_pub_;
  rosbag::Bag bag_;
};

CalibrationCapture::CalibrationCapture(const std::string& bag_path)
    : bag_path_(bag_path),
      latest_rgb_(),
      latest_thermal_(),
      nh_(),
      rgb_pub_(nh_.advertise<Image>("rgb_chessboard", 1)),
      thermal_pub_(nh_.advertise<Image>("thermal_chessboard", 1)),
      bag_() {}

void CalibrationCapture::Callback(const Image::ConstPtr& rgb_msg,
                                  const Image::ConstPtr& thermal_msg) {
  latest_rgb_ = rgb_msg;
  latest_thermal_ = thermal_msg;

  double skew =
      fabs(rgb_msg->header.stamp.toSec() - thermal_msg->header.stamp.toSec());
  if (skew > 0.005) {
    return;
  }

  // Find corners
  const cv::Size kSize(7, 7);

  cv_bridge::CvImageConstPtr cv_thermal = cv_bridge::toCvShare(thermal_msg);
  double contrast_factor;
  ros::param::param("contrast_factor", contrast_factor, 1.2);
  cv::Mat thermal_high_contrast = cv_thermal->image * contrast_factor;
  Corners thermal_corners;
  bool thermal_found =
      cv::findChessboardCorners(thermal_high_contrast, kSize, thermal_corners);
  cv::Mat thermal_viz = thermal_high_contrast.clone();
  cv::drawChessboardCorners(thermal_viz, kSize, thermal_corners, thermal_found);
  cv_bridge::CvImage thermal_cv_out(cv_thermal->header, cv_thermal->encoding,
                                    thermal_viz);
  thermal_pub_.publish(thermal_cv_out.toImageMsg());
  // Don't waste time searching for RGB corners.
  if (!thermal_found) {
    return;
  }

  cv_bridge::CvImageConstPtr cv_rgb = cv_bridge::toCvShare(rgb_msg);
  Corners rgb_corners;
  bool rgb_found = cv::findChessboardCorners(cv_rgb->image, kSize, rgb_corners);
  cv_bridge::CvImage rgb_cv_out = *cv_rgb;
  cv::drawChessboardCorners(rgb_cv_out.image, kSize, rgb_corners, rgb_found);
  rgb_pub_.publish(rgb_cv_out.toImageMsg());

  if (thermal_found && rgb_found) {
    try {
      bag_.open(bag_path_, rosbag::bagmode::Append);
    } catch (const rosbag::BagException& e) {
      bag_.close();
      bag_.open(bag_path_, rosbag::bagmode::Write);
    }
    bag_.write("/camera/rgb/image_rect_color", rgb_msg->header.stamp, rgb_msg);
    bag_.write("/ici/ir_camera/image_normalized_rgb", rgb_msg->header.stamp,
               thermal_msg);

    ROS_INFO("Saved image");
    bag_.close();
  }
}

void CalibrationCapture::Save() {
  rosbag::Bag bag;
  try {
    bag.open(bag_path_, rosbag::bagmode::Append);
  } catch (const rosbag::BagException& e) {
    bag.close();
    bag.open(bag_path_, rosbag::bagmode::Write);
  }

  bag.write("/camera/rgb/image_rect_color", latest_rgb_->header.stamp,
            latest_rgb_);
  bag.write("/ici/ir_camera/image_normalized_rgb", latest_rgb_->header.stamp,
            latest_rgb_);
  bag.close();
}
}  // namespace skinseg

int main(int argc, char** argv) {
  ros::init(argc, argv, "calibration_capture");
  // ros::AsyncSpinner spinner(2);
  // spinner.start();
  ros::NodeHandle nh;
  if (argc < 2) {
    std::cout << "Usage: calibration_capture path/to/bag.bag" << std::endl;
    return 0;
  }
  skinseg::CalibrationCapture capture(argv[1]);
  message_filters::Subscriber<Image> rgb_sub(nh, "/camera/rgb/image_rect_color",
                                             1);
  message_filters::Subscriber<Image> thermal_sub(
      nh, "/ici/ir_camera/image_normalized_rgb", 1);
  message_filters::Synchronizer<MyPolicy> sync(MyPolicy(2), rgb_sub,
                                               thermal_sub);
  sync.registerCallback(&skinseg::CalibrationCapture::Callback, &capture);
  ros::spin();

  return 0;
}

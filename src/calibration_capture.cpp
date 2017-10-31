// Utility for saving RGB/Thermal frames for calibration purposes.
// They are saved to a bag file.

// Because it is difficult to find chessboard corners in the thermal image,
// especially at greater distances, this utility just captures images
// periodically without finding the chessboard corners. Instead, the user uses
// a separate utility to manually annotate the corners in the thermal image.

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
#include "std_srvs/Empty.h"

#include "skin_segmentation/opencv_utils.h"

using sensor_msgs::Image;
typedef message_filters::sync_policies::ApproximateTime<Image, Image> MyPolicy;

namespace skinseg {
class CalibrationCapture {
 public:
  CalibrationCapture(const std::string& bag_path);
  void ImageCallback(const Image::ConstPtr& rgb_msg,
                     const Image::ConstPtr& thermal_msg);
  bool CaptureCallback(std_srvs::EmptyRequest& req,
                       std_srvs::EmptyResponse& res);
  void Save();

 private:
  std::string bag_path_;
  Image::ConstPtr latest_rgb_;
  Image::ConstPtr latest_thermal_;
  ros::NodeHandle nh_;
  ros::Publisher rgb_pub_;
  ros::Publisher thermal_pub_;
  rosbag::Bag bag_;
  int num_images_;
};

CalibrationCapture::CalibrationCapture(const std::string& bag_path)
    : bag_path_(bag_path),
      latest_rgb_(),
      latest_thermal_(),
      nh_(),
      rgb_pub_(nh_.advertise<Image>("rgb_chessboard", 1)),
      thermal_pub_(nh_.advertise<Image>("thermal_chessboard", 1)),
      bag_(),
      num_images_(0) {}

void CalibrationCapture::ImageCallback(const Image::ConstPtr& rgb_msg,
                                       const Image::ConstPtr& thermal_msg) {
  latest_rgb_ = rgb_msg;
  latest_thermal_ = thermal_msg;

  thermal_pub_.publish(thermal_msg);

  // Find corners
  const cv::Size kSize(7, 7);
  cv_bridge::CvImageConstPtr cv_rgb = cv_bridge::toCvCopy(rgb_msg);
  Corners rgb_corners;
  bool rgb_found = cv::findChessboardCorners(cv_rgb->image, kSize, rgb_corners);
  cv_bridge::CvImage rgb_cv_out = *cv_rgb;
  cv::drawChessboardCorners(rgb_cv_out.image, kSize, rgb_corners, rgb_found);
  rgb_pub_.publish(rgb_cv_out.toImageMsg());
}

bool CalibrationCapture::CaptureCallback(std_srvs::EmptyRequest& req,
                                         std_srvs::EmptyResponse& res) {
  try {
    bag_.open(bag_path_, rosbag::bagmode::Append);
  } catch (const rosbag::BagException& e) {
    bag_.close();
    bag_.open(bag_path_, rosbag::bagmode::Write);
  }

  ros::Time now = ros::Time::now();
  bag_.write("/camera/rgb/image_rect_color", now, latest_rgb_);
  bag_.write("/ici/ir_camera/image_normalized_rgb", now, latest_thermal_);
  bag_.close();

  ++num_images_;
  ROS_INFO("Saved %d images", num_images_);

  return true;
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
  ros::NodeHandle nh;
  if (argc < 2) {
    std::cout << "Usage: calibration_capture path/to/bag.bag" << std::endl;
    return 0;
  }
  skinseg::CalibrationCapture capture(argv[1]);
  message_filters::Subscriber<Image> rgb_sub(nh, "/camera/rgb/image_rect_color",
                                             10);
  message_filters::Subscriber<Image> thermal_sub(
      nh, "/ici/ir_camera/image_normalized_rgb", 10);
  message_filters::Synchronizer<MyPolicy> sync(MyPolicy(10), rgb_sub,
                                               thermal_sub);
  sync.registerCallback(&skinseg::CalibrationCapture::ImageCallback, &capture);

  ros::ServiceServer capture_server = nh.advertiseService(
      "/capture_calibration_images",
      &skinseg::CalibrationCapture::CaptureCallback, &capture);

  ros::spin();

  return 0;
}

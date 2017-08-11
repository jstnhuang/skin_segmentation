#include "cv_bridge/cv_bridge.h"
#include "image_geometry/pinhole_camera_model.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/time_synchronizer.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "tf/transform_listener.h"
#include "tf_conversions/tf_eigen.h"

#include "Eigen/Dense"
#include "boost/regex.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "ros/ros.h"

using image_geometry::PinholeCameraModel;
using sensor_msgs::CameraInfo;
using sensor_msgs::Image;

// Projects the thermal camera image onto the RGB camera image.
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}
namespace skinseg {
class ThermalTest {
 public:
  ThermalTest(const ros::Publisher& overlay_pub);
  void Init();
  void Callback(const sensor_msgs::Image& thermal_image);

 private:
  PinholeCameraModel thermal_model_;
  ros::Publisher overlay_pub_;
  tf::TransformListener tf_listener_;
  Eigen::Affine3d depth_in_thermal_;

  ros::NodeHandle nh_;
  ros::Publisher debug_pub_;
  ros::Publisher debug_pub_2_;
  ros::Publisher debug_pub_3_;
};

ThermalTest::ThermalTest(const ros::Publisher& overlay_pub)
    : overlay_pub_(overlay_pub),
      tf_listener_(),
      depth_in_thermal_(Eigen::Affine3d::Identity()),
      nh_(),
      debug_pub_(nh_.advertise<Image>("debug_1", 1, true)),
      debug_pub_2_(nh_.advertise<Image>("debug_2", 1, true)),
      debug_pub_3_(nh_.advertise<Image>("debug_3", 1, true)) {}

void ThermalTest::Init() {
  // Get transform that describes depth frame in the frame of the thermal camera
  // while (!tf_listener_.waitForTransform(thermal_model_.tfFrame(),
  //                                      rgbd_model_.tfFrame(),
  //                                      ros::Time::now(),
  //                                      ros::Duration(1)) &&
  //       ros::ok()) {
  //  ROS_INFO(
  //      "Waiting for transform from depth frame \"%s\" to thermal frame "
  //      "\"%s\".",
  //      rgbd_model_.tfFrame().c_str(), thermal_model_.tfFrame().c_str());
  //}
  // tf::StampedTransform depth_in_thermal;
  // tf_listener_.lookupTransform(thermal_model_.tfFrame(),
  // rgbd_model_.tfFrame(),
  //                             ros::Time(0), depth_in_thermal);
  // tf::transformTFToEigen(depth_in_thermal, depth_in_thermal_);
}

void ThermalTest::Callback(const sensor_msgs::Image& thermal_msg) {
  //  double alpha;
  //  ros::param::param("overlay_alpha", alpha, 0.5);
  //
  //  double depth_cx = rgbd_model_.cx(), depth_cy = rgbd_model_.cy();
  //  double depth_Tx = rgbd_model_.Tx(), depth_Ty = rgbd_model_.Ty();
  //  double inv_depth_fx = 1.0 / rgbd_model_.fx();
  //  double inv_depth_fy = 1.0 / rgbd_model_.fy();
  //  debug_pub_.publish(rgb_msg);
  //  debug_pub_2_.publish(depth_msg);
  //  debug_pub_3_.publish(thermal_msg);
  //
  //  cv_bridge::CvImageConstPtr rgb_image;
  //  rgb_image =
  //      cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::TYPE_8UC3);
  //
  //  cv_bridge::CvImageConstPtr depth_image(new cv_bridge::CvImage);
  //  depth_image =
  //      cv_bridge::toCvCopy(depth_msg,
  //      sensor_msgs::image_encodings::TYPE_16UC1);
  //
  debug_pub_.publish(thermal_msg);
  cv_bridge::CvImagePtr output_image_bridge;
  output_image_bridge = cv_bridge::toCvCopy(thermal_msg);
  ROS_INFO_THROTTLE(1, "input type: %s",
                    type2str(output_image_bridge->image.type()).c_str());
  ROS_INFO_STREAM("input " << output_image_bridge->image(cv::Rect(0, 0, 5, 5)));
  // cv::Mat output_color;
  // cv::cvtColor(output_image_bridge->image, output_color, CV_GRAY2RGB);

  cv::Mat normed;
  cv::normalize(output_image_bridge->image, normed, 0, 255, cv::NORM_MINMAX);
  ROS_INFO_THROTTLE(1, "normalized type: %s", type2str(normed.type()).c_str());
  ROS_INFO_STREAM("normalized " << normed(cv::Rect(0, 0, 5, 5)));

  cv::Mat output_color_8;
  normed.convertTo(output_color_8, CV_8UC3);
  ROS_INFO_THROTTLE(1, "8bit type: %s",
                    type2str(output_color_8.type()).c_str());
  ROS_INFO_STREAM("8bit " << output_color_8(cv::Rect(0, 0, 5, 5)));

  cv::Mat output_color;
  cv::cvtColor(output_color_8, output_color, cv::COLOR_GRAY2RGB);
  ROS_INFO_THROTTLE(1, "colorized type: %s",
                    type2str(output_color.type()).c_str());
  ROS_INFO_STREAM("colorized " << output_color(cv::Rect(0, 0, 5, 5)));

  cv::Mat red(output_color.size(), CV_8UC3, cv::Scalar(0, 0, 255));
  cv::Mat tinted;
  cv::addWeighted(output_color, 0.7, red, 0.3, 0.0, tinted);

  cv::namedWindow("test");
  cv::imshow("test", tinted);
  cv::waitKey();

  cv_bridge::CvImage cv_out(thermal_msg.header,
                            sensor_msgs::image_encodings::TYPE_8UC3,
                            output_color);
  overlay_pub_.publish(cv_out.toImageMsg());

  // Project depth image onto thermal image.
  // for (int r = 0; r < depth_image->image.rows; ++r) {
  //  const uint8_t* rgb_row = rgb_image->image.ptr<uint8_t>(r);
  //  const uint16_t* depth_row = depth_image->image.ptr<uint16_t>(r);
  //  uint8_t* output_row = output_image.ptr(r);
  //  for (int c = 0; c < depth_image->image.cols; ++c) {
  //    uint16_t depth = depth_row[c] * 0.001f;  // Convert from mm to meters
  //    if (depth == 0) {
  //      continue;
  //    }
  //    // Project to x,y,z in depth camera frame
  //    Eigen::Vector4d xyz_in_depth;
  //    xyz_in_depth << ((c - depth_cx) * depth - depth_Tx) * inv_depth_fx,
  //        ((c - depth_cy) * depth - depth_Ty) * inv_depth_fy, depth, 1;

  //    // Transform into thermal camera frame
  //    Eigen::Vector4d xyz_in_thermal = depth_in_thermal_ * xyz_in_depth;

  //    // Project back to (u, v) in thermal camera
  //    cv::Point3d cv_xyz;
  //    cv_xyz.x = xyz_in_thermal[0];
  //    cv_xyz.y = xyz_in_thermal[1];
  //    cv_xyz.z = xyz_in_thermal[2];
  //    cv::Point2d uv = thermal_model_.project3dToPixel(cv_xyz);
  //    if (uv.x < 0 || uv.x >= output_image.cols || uv.y < 0 ||
  //        uv.y >= output_image.rows) {
  //      continue;
  //    }

  //    // Get color
  //    uint8_t r = rgb_row[c];
  //    uint8_t g = rgb_row[c + 1];
  //    uint8_t b = rgb_row[c + 2];

  //    uint8_t new_r =
  //        cv::saturate_cast<uint8_t>((1 - alpha) * r + alpha * output_row[c]);
  //    uint8_t new_g = cv::saturate_cast<uint8_t>((1 - alpha) * g +
  //                                               alpha * output_row[c + 1]);
  //    uint8_t new_b = cv::saturate_cast<uint8_t>((1 - alpha) * b +
  //                                               alpha * output_row[c + 2]);

  //    output_row[c] = new_r;
  //    output_row[c + 1] = new_g;
  //    output_row[c + 2] = new_b;
  //  }
  //}

  // cv_bridge::CvImage out_cv(output_image_bridge->header,
  //                          output_image_bridge->encoding, output_image);
  // sensor_msgs::ImagePtr out_msg = out_cv.toImageMsg();
  // overlay_pub_.publish(*out_msg);
}
}  // namespace skinseg

int main(int argc, char** argv) {
  ros::init(argc, argv, "thermal_projection");
  ros::NodeHandle nh;

  ros::Publisher overlay_pub = nh.advertise<Image>("overlay", 1);

  skinseg::ThermalTest projection(overlay_pub);
  projection.Init();
  ros::Subscriber sub =
      nh.subscribe("/ici/ir_camera/image_raw", 1,
                   &skinseg::ThermalTest::Callback, &projection);
  // message_filters::Subscriber<Image> rgb_sub(nh,
  // "/camera/rgb/image_rect_color",
  //                                           1);
  // message_filters::Subscriber<Image> depth_sub(
  //    nh, "/camera/depth_registered/image", 1);
  // message_filters::Subscriber<Image> thermal_sub(nh,
  // "/ici/ir_camera/image_raw",
  //                                               1);
  // typedef message_filters::sync_policies::ApproximateTime<Image, Image,
  // Image>
  //    MyPolicy;
  // message_filters::Synchronizer<MyPolicy> sync(MyPolicy(10), rgb_sub,
  // depth_sub,
  //                                             thermal_sub);
  // ros::Duration(1.0).sleep();
  // sync.registerCallback(&skinseg::ThermalTest::Callback, &projection);

  ros::spin();
  return 0;
}

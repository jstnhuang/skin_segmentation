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
class Projection {
 public:
  Projection(const PinholeCameraModel& rgbd_model,
             const PinholeCameraModel& thermal_model,
             const ros::Publisher& overlay_pub);
  void Init();
  void Callback(const sensor_msgs::Image& rgb_image,
                const sensor_msgs::Image& depth_image,
                const sensor_msgs::Image& thermal_image);

 private:
  PinholeCameraModel rgbd_model_;
  PinholeCameraModel thermal_model_;
  ros::Publisher overlay_pub_;
  tf::TransformListener tf_listener_;
  Eigen::Affine3d depth_in_thermal_;

  ros::NodeHandle nh_;
  ros::Publisher debug_pub_;
  ros::Publisher debug_pub_2_;
  ros::Publisher debug_pub_3_;
};

Projection::Projection(const PinholeCameraModel& rgbd_model,
                       const PinholeCameraModel& thermal_model,
                       const ros::Publisher& overlay_pub)
    : rgbd_model_(rgbd_model),
      thermal_model_(thermal_model),
      overlay_pub_(overlay_pub),
      tf_listener_(),
      depth_in_thermal_(Eigen::Affine3d::Identity()),
      nh_(),
      debug_pub_(nh_.advertise<Image>("debug_1", 1, true)),
      debug_pub_2_(nh_.advertise<Image>("debug_2", 1, true)),
      debug_pub_3_(nh_.advertise<Image>("debug_3", 1, true)) {}

void Projection::Init() {
  // Get transform that describes depth frame in the frame of the thermal camera
  while (!tf_listener_.waitForTransform(thermal_model_.tfFrame(),
                                        rgbd_model_.tfFrame(), ros::Time::now(),
                                        ros::Duration(1)) &&
         ros::ok()) {
    ROS_INFO(
        "Waiting for transform from depth frame \"%s\" to thermal frame "
        "\"%s\".",
        rgbd_model_.tfFrame().c_str(), thermal_model_.tfFrame().c_str());
  }
  tf::StampedTransform depth_in_thermal;
  tf_listener_.lookupTransform(thermal_model_.tfFrame(), rgbd_model_.tfFrame(),
                               ros::Time(0), depth_in_thermal);
  tf::transformTFToEigen(depth_in_thermal, depth_in_thermal_);

  ROS_INFO_STREAM(
      "depth frame relative to thermal frame: " << depth_in_thermal_.matrix());
}

cv::Mat NormalizeThermal(cv::InputArray in) {
  cv::Mat normed;
  cv::normalize(in, normed, 0, 255, cv::NORM_MINMAX);
  return normed;
}

// Convert 16UC1 to 8UC3
cv::Mat ConvertToColor(cv::Mat in) {
  cv::Mat eight_bit;
  in.convertTo(eight_bit, CV_8UC3);
  cv::Mat color;
  cv::cvtColor(eight_bit, color, cv::COLOR_GRAY2RGB);
  return color;
}

void Projection::Callback(const sensor_msgs::Image& rgb_msg,
                          const sensor_msgs::Image& depth_msg,
                          const sensor_msgs::Image& thermal_msg) {
  double depth_cx = rgbd_model_.cx(), depth_cy = rgbd_model_.cy();
  double depth_Tx = rgbd_model_.Tx(), depth_Ty = rgbd_model_.Ty();
  double inv_depth_fx = 1.0 / rgbd_model_.fx();
  double inv_depth_fy = 1.0 / rgbd_model_.fy();
  double thermal_fx = thermal_model_.fx(), thermal_fy = thermal_model_.fy();
  double thermal_Tx = thermal_model_.Tx(), thermal_Ty = thermal_model_.Ty();
  double thermal_cx = thermal_model_.cx(), thermal_cy = thermal_model_.cy();

  ROS_INFO("Depth cx=%f cy=%f Tx=%f Ty=%f fx=%f fy=%f", depth_cx, depth_cy,
           depth_Tx, depth_Ty, rgbd_model_.fx(), rgbd_model_.fy());
  ROS_INFO("Thermal cx=%f cy=%f Tx=%f Ty=%f fx=%f fy=%f", thermal_cx,
           thermal_cy, thermal_Tx, thermal_Ty, thermal_fx, thermal_fy);

  cv_bridge::CvImageConstPtr rgb_image;
  rgb_image = cv_bridge::toCvCopy(rgb_msg);

  cv_bridge::CvImageConstPtr depth_image(new cv_bridge::CvImage);
  depth_image = cv_bridge::toCvCopy(depth_msg);

  cv_bridge::CvImagePtr output_bridge;
  output_bridge = cv_bridge::toCvCopy(thermal_msg,
                                      sensor_msgs::image_encodings::TYPE_16UC1);
  cv::Mat normalized = NormalizeThermal(output_bridge->image);
  cv::Mat output_color = ConvertToColor(normalized);

  // Project rgb image onto thermal image based on depth registration
  cv::Mat projection =
      cv::Mat::zeros(output_color.rows, output_color.cols, output_color.type());
  cv::Mat_<cv::Vec3b> _projection = projection;
  cv::Mat_<cv::Vec3b> _color = rgb_image->image;
  ROS_INFO("num rows: %d, cols: %d", depth_image->image.rows,
           depth_image->image.cols);
  for (int r = 0; r < depth_image->image.rows; ++r) {
    const uint16_t* depth_row = depth_image->image.ptr<uint16_t>(r);

    for (int c = 0; c < depth_image->image.cols; ++c) {
      double depth = depth_row[c] * 0.001f;  // Convert from mm to meters
      if (depth == 0) {
        continue;
      }
      // Project to x,y,z in depth camera frame
      Eigen::Vector4d xyz_in_depth;
      xyz_in_depth << ((c - depth_cx) * depth - depth_Tx) * inv_depth_fx,
          ((r - depth_cy) * depth - depth_Ty) * inv_depth_fy, depth, 1;

      // Transform into thermal camera frame
      Eigen::Vector4d xyz_in_thermal = depth_in_thermal_ * xyz_in_depth;

      // Project back to (u, v) in thermal camera
      double inv_z = 1.0 / xyz_in_thermal.z();
      // TODO: round instead of integer cutoff
      int u_thermal =
          round((thermal_fx * xyz_in_thermal.x() + thermal_Tx) * inv_z +
                thermal_cx + 0.5);
      int v_thermal =
          round((thermal_fy * xyz_in_thermal.y() + thermal_Ty) * inv_z +
                thermal_cy + 0.5);
      if (u_thermal < 0 || u_thermal >= projection.cols || v_thermal < 0 ||
          v_thermal >= projection.rows) {
        continue;
      }

      // Get color
      uint8_t red = _color(r, c)[0];
      uint8_t green = _color(r, c)[1];
      uint8_t blue = _color(r, c)[2];

      _projection(v_thermal, u_thermal)[0] = red;
      _projection(v_thermal, u_thermal)[1] = green;
      _projection(v_thermal, u_thermal)[2] = blue;
    }
  }

  projection = _projection;

  double alpha;
  ros::param::param("overlay_alpha", alpha, 0.5);
  cv::Mat overlay;
  cv::addWeighted(output_color, alpha, projection, 1.0 - alpha, 0.0, overlay);

  ROS_INFO_STREAM("overlay" << overlay(cv::Rect(100, 100, 5, 5)));

  cv::namedWindow("projection");
  cv::imshow("projection", overlay);
  cv::waitKey();

  cv_bridge::CvImage out_cv(output_bridge->header,
                            sensor_msgs::image_encodings::BGR8, overlay);
  sensor_msgs::ImagePtr out_msg = out_cv.toImageMsg();
  overlay_pub_.publish(*out_msg);
}
}  // namespace skinseg

int main(int argc, char** argv) {
  ros::init(argc, argv, "thermal_projection");
  ros::NodeHandle nh;

  CameraInfo::ConstPtr rgbd_info =
      ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
          "/camera/rgb/camera_info");
  PinholeCameraModel rgbd_model;
  rgbd_model.fromCameraInfo(*rgbd_info);

  CameraInfo::ConstPtr thermal_info =
      ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
          "/ici/ir_camera/camera_info");
  PinholeCameraModel thermal_model;
  thermal_model.fromCameraInfo(*thermal_info);

  ros::Publisher overlay_pub = nh.advertise<Image>("overlay", 1);

  skinseg::Projection projection(rgbd_model, thermal_model, overlay_pub);
  projection.Init();
  message_filters::Subscriber<Image> rgb_sub(nh, "/camera/rgb/image_rect_color",
                                             1);
  message_filters::Subscriber<Image> depth_sub(
      nh, "/camera/depth_registered/image", 1);
  message_filters::Subscriber<Image> thermal_sub(nh, "/ici/ir_camera/image_raw",
                                                 1);
  typedef message_filters::sync_policies::ApproximateTime<Image, Image, Image>
      MyPolicy;
  message_filters::Synchronizer<MyPolicy> sync(MyPolicy(10), rgb_sub, depth_sub,
                                               thermal_sub);
  ros::Duration(1.0).sleep();
  sync.registerCallback(&skinseg::Projection::Callback, &projection);

  ros::spin();
  return 0;
}

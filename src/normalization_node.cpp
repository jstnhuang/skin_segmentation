// Node that publishes a normalized version of the thermal data, which is useful
// for calibration. For visualizing the thermal data, you can just use
// "Normalize Range" in an RViz image display.

#include "cv_bridge/cv_bridge.h"
#include "opencv2/core/core.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"

using sensor_msgs::Image;

class Normalizer {
 public:
  explicit Normalizer(const ros::Publisher& pub,
                      const ros::Publisher& color_pub);
  void Callback(const Image::ConstPtr& msg);

 private:
  ros::Publisher pub_;
  ros::Publisher color_pub_;
};

Normalizer::Normalizer(const ros::Publisher& pub,
                       const ros::Publisher& color_pub)
    : pub_(pub), color_pub_(color_pub) {}

cv::Mat ConvertToColor(cv::Mat in) {
  cv::Mat eight_bit;
  in.convertTo(eight_bit, CV_8UC3);
  cv::Mat color;
  cv::cvtColor(eight_bit, color, cv::COLOR_GRAY2RGB);
  return color;
}

void Normalizer::Callback(const Image::ConstPtr& msg) {
  cv_bridge::CvImageConstPtr bridge = cv_bridge::toCvShare(msg);
  cv::Mat normalized;
  cv::normalize(bridge->image, normalized, 0, 255, cv::NORM_MINMAX);
  cv_bridge::CvImage output(msg->header, msg->encoding, normalized);
  pub_.publish(output.toImageMsg());

  cv::Mat color = ConvertToColor(normalized);
  cv_bridge::CvImage color_output(msg->header,
                                  sensor_msgs::image_encodings::RGB8, color);
  color_pub_.publish(color_output.toImageMsg());
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "normalization_node");
  ros::NodeHandle nh;
  ros::Publisher pub =
      nh.advertise<Image>("/ici/ir_camera/image_normalized", 1);
  ros::Publisher color_pub =
      nh.advertise<Image>("/ici/ir_camera/image_normalized_rgb", 1);
  Normalizer normalizer(pub, color_pub);
  ros::Subscriber sub = nh.subscribe("/ici/ir_camera/image_raw", 1,
                                     &Normalizer::Callback, &normalizer);
  ros::spin();
  return 0;
}

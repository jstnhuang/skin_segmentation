#ifndef _SKINSEG_SNAPSHOT_H_
#define _SKINSEG_SNAPSHOT_H_

#include <string>

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/rgbdt_data.h"

namespace skinseg {
const static char kColorTopic[] = "/camera/rgb/image_rect_color";
const static char kDepthTopic[] =
    "/camera/depth_registered/hw_registered/image_rect";
const static char kRgbdCameraInfoTopic[] = "/camera/rgb/camera_info";
const static char kThermalTopic[] = "/ici/ir_camera/image_raw";
const static char kThermalCameraInfoTopic[] = "/ici/ir_camera/camera_info";

class Snapshot {
 public:
  Snapshot();
  void StartCapture();
  bool IsCaptureDone();
  bool SaveBag(const std::string& path);
  bool LoadBag(const std::string& path);

  RgbdtData data();

 private:
  void Callback(const sensor_msgs::Image::ConstPtr& color,
                const sensor_msgs::Image::ConstPtr& depth,
                const sensor_msgs::Image::ConstPtr& thermal);
  ros::NodeHandle nh_;
  message_filters::Subscriber<sensor_msgs::Image> color_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::Image> thermal_sub_;
  message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> >
      sync_;
  bool is_capturing_;

  RgbdtData data_;
};
}  // namespace skinseg

#endif  // _SKINSEG_SNAPSHOT_H_

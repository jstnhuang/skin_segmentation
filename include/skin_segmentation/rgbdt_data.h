#ifndef _SKINSEG_RGBDT_DATA_H_
#define _SKINSEG_RGBDT_DATA_H_

#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

namespace skinseg {
struct RgbdtData {
  sensor_msgs::Image::ConstPtr color;
  sensor_msgs::Image::ConstPtr depth;
  sensor_msgs::Image::ConstPtr thermal;
  sensor_msgs::CameraInfo::ConstPtr rgbd_camera_info;
  sensor_msgs::CameraInfo::ConstPtr thermal_camera_info;
};
}  // namespace skinseg

#endif  // _SKINSEG_RGBDT_DATA_H_

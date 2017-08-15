#ifndef _SKINSEG_PROJECTION_H_
#define _SKINSEG_PROJECTION_H_

#include "Eigen/Dense"
#include "cv_bridge/cv_bridge.h"
#include "image_geometry/pinhole_camera_model.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

namespace skinseg {
class Projection {
 public:
  Projection(const sensor_msgs::CameraInfo& rgbd_info,
             const sensor_msgs::CameraInfo& thermal_info);

  void ProjectRgbdOntoThermal(const sensor_msgs::Image::ConstPtr& rgb,
                              const sensor_msgs::Image::ConstPtr& depth,
                              const sensor_msgs::Image::ConstPtr& thermal,
                              sensor_msgs::Image* rgbd_projected);

 private:
  sensor_msgs::CameraInfo rgbd_info_;
  sensor_msgs::CameraInfo thermal_info_;
  image_geometry::PinholeCameraModel rgbd_model_;
  image_geometry::PinholeCameraModel thermal_model_;
};
}  // namespace skinseg

#endif  // _SKINSEG_PROJECTION_H_

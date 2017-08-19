#ifndef _SKINSEG_PROJECTION_H_
#define _SKINSEG_PROJECTION_H_

#include <list>
#include <map>

#include "Eigen/Dense"
#include "cv_bridge/cv_bridge.h"
#include "image_geometry/pinhole_camera_model.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

namespace skinseg {
class Projection {
 public:
  Projection(const sensor_msgs::CameraInfo& rgbd_info,
             const sensor_msgs::CameraInfo& thermal_info,
             const Eigen::Affine3d& rgb_in_thermal);

  void ProjectThermalOnRgb(const sensor_msgs::Image::ConstPtr& rgb,
                           const sensor_msgs::Image::ConstPtr& depth,
                           const sensor_msgs::Image::ConstPtr& thermal,
                           cv::OutputArray thermal_projected);

  void set_debug(bool debug);

  // Return -1, -1 if invalid.
  cv::Point2d GetThermalPixel(const cv::Mat& depth_image,
                              const cv::Point2d& rgb_pt) const;
  float GetRgbDepth(const cv::Mat& depth_image,
                    const cv::Point2d& rgb_pt) const;

  static void RgbdMouseCallback(int event, int x, int y, int flags, void* data);
  static void ThermalMouseCallback(int event, int x, int y, int flags,
                                   void* data);

 private:
  const sensor_msgs::CameraInfo& rgbd_info_;
  const sensor_msgs::CameraInfo& thermal_info_;
  const Eigen::Affine3d& rgb_in_thermal_;
  image_geometry::PinholeCameraModel rgbd_model_;
  image_geometry::PinholeCameraModel thermal_model_;
  bool debug_;

  cv::Mat rgb_;
  cv::Mat depth_;
  cv::Mat thermal_;
  std::map<std::pair<int, int>, std::list<cv::Point2d> > thermal_to_rgb_;
};
}  // namespace skinseg

#endif  // _SKINSEG_PROJECTION_H_

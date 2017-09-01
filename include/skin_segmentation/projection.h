#ifndef _SKINSEG_PROJECTION_H_
#define _SKINSEG_PROJECTION_H_

#include <vector_types.h>
#include <list>
#include <map>

#undef Success  // Evil workaround. nerf includes glx, which defines this again
#include "Eigen/Dense"
#include "cv_bridge/cv_bridge.h"
#include "image_geometry/pinhole_camera_model.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

namespace skinseg {

// Similar to sensor_msgs::CameraInfo, but in form that's easier to use in inner
// loops.
struct CameraData {
  double inv_depth_fx;
  double inv_depth_fy;
  double depth_cx;
  double depth_cy;
  double depth_Tx;
  double depth_Ty;
  double thermal_fx;
  double thermal_fy;
  double thermal_cx;
  double thermal_cy;
  double thermal_Tx;
  double thermal_Ty;
};

void GetCameraData(const sensor_msgs::CameraInfo& camera_info,
                   CameraData* camera_data);

// This class provides the functionality to map between the RGB, depth, and
// thermal images. It expects the following image encodings:
// - RGB: RGB8
// - Depth: 16UC1, values indicate mm.
// - Thermal: 16UC1, raw image values from camera driver.
//
// It also assumes that we are getting registered depth data. That is, the depth
// values are in the frame of the RGB camera, and rgb[i] corresponds to depth[i]
// for each pixel i. Invalid depth values should be set to 0.
class Projection {
 public:
  Projection(const sensor_msgs::CameraInfo& rgbd_info,
             const sensor_msgs::CameraInfo& thermal_info,
             const Eigen::Affine3d& rgb_in_thermal);

  // Registers the thermal image with the RGB/depth images. That is, for each
  // pixel i in the RGB image, the corresponding depth and thermal values are
  // depth[i] and thermal_projected[i]. Calling this produces the point cloud as
  // a byproduct. If you want to get the point cloud, allocate a
  // float4[depth_rows * depth_cols] array and pass it in as "points". The XYZ
  // for each point will be stored in the same row-major order as with "rgb",
  // etc. The w value is 1 for valid points, 0 otherwise.
  void ProjectThermalOnRgb(const sensor_msgs::Image::ConstPtr& rgb,
                           const sensor_msgs::Image::ConstPtr& depth,
                           const sensor_msgs::Image::ConstPtr& thermal,
                           cv::OutputArray thermal_projected, float4* points);

  void set_debug(bool debug);

  // Return -1, -1 if invalid.
  cv::Point2d GetThermalPixel(const cv::Mat& depth_image,
                              const cv::Point2d& rgb_pt) const;
  float GetRgbDepth(const cv::Mat& depth_image,
                    const cv::Point2d& rgb_pt) const;

  void GetCameraData(CameraData* data);
  void GetRgbdCameraInfo(sensor_msgs::CameraInfo* info);

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

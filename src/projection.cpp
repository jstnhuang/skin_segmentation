#include "skin_segmentation/projection.h"

#include "Eigen/Dense"
#include "cv_bridge/cv_bridge.h"
#include "depth_image_proc/depth_traits.h"
#include "image_geometry/pinhole_camera_model.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

using depth_image_proc::DepthTraits;
using image_geometry::PinholeCameraModel;
using sensor_msgs::CameraInfo;
using sensor_msgs::Image;

namespace skinseg {
Projection::Projection(const CameraInfo& rgbd_info,
                       const CameraInfo& thermal_info)
    : rgbd_info_(rgbd_info),
      thermal_info_(thermal_info),
      rgbd_model_(),
      thermal_model_() {
  rgbd_model_.fromCameraInfo(rgbd_info);
  thermal_model_.fromCameraInfo(thermal_info);
}

cv::Mat ConvertToColor(cv::Mat in) {
  cv::Mat eight_bit;
  in.convertTo(eight_bit, CV_8UC3);
  cv::Mat color;
  cv::cvtColor(eight_bit, color, cv::COLOR_GRAY2RGB);
  return color;
}

void Projection::ProjectThermalOntoRgbd(const Image::ConstPtr& color,
                                        const Image::ConstPtr& depth,
                                        const Image::ConstPtr& thermal,
                                        Image* thermal_projected) {
  cv_bridge::CvImageConstPtr thermal_cv =
      cv_bridge::toCvShare(thermal, sensor_msgs::image_encodings::TYPE_16UC1);

  cv::Mat normalized_thermal;
  cv::normalize(thermal_cv->image, normalized_thermal, 0, 255, cv::NORM_MINMAX);
  cv::Mat normalized_thermal_color = ConvertToColor(normalized_thermal);

  cv::namedWindow("Normalized thermal");
  cv::imshow("Normalized thermal", normalized_thermal_color);

  // TODO: use real extrinsic calibration
  Eigen::Vector3d translation;
  ros::param::param<double>("thermal_x", translation.x(), 0);
  ros::param::param<double>("thermal_y", translation.y(), 0.025);
  ros::param::param<double>("thermal_z", translation.z(), 0);

  Eigen::Affine3d thermal_in_rgb;
  thermal_in_rgb.setIdentity();
  thermal_in_rgb.translate(translation);

  cv_bridge::CvImagePtr overlay_cv =
      cv_bridge::toCvCopy(color, sensor_msgs::image_encodings::BGR8);

  cv::Mat projection =
      cv::Mat(color->height, color->width, CV_8UC3, cv::Scalar(0, 0, 255));

  cv::namedWindow("Color");
  cv::imshow("Color", overlay_cv->image);
  cv::Mat_<cv::Vec3b> _projection = projection;
  cv::Mat_<cv::Vec3b> _thermal = normalized_thermal_color;
  for (int t_row = 0; t_row < thermal_cv->image.rows; ++t_row) {
    for (int t_col = 0; t_col < thermal_cv->image.cols; ++t_col) {
      cv::Point2d uv_rect =
          thermal_model_.rectifyPoint(cv::Point2d(t_col, t_row));
      if (uv_rect.x < 0 || uv_rect.x >= _thermal.cols || uv_rect.y < 0 ||
          uv_rect.y >= _thermal.rows) {
        continue;
      }

      cv::Point3d xyz_thermal = thermal_model_.projectPixelTo3dRay(uv_rect);
      xyz_thermal *= thermal_model_.fx() / 1000.0;

      // Transform into RGB camera coordinates
      Eigen::Vector3d xyz_in_thermal;
      xyz_in_thermal << xyz_thermal.x, xyz_thermal.y, xyz_thermal.z;
      Eigen::Vector3d xyz_in_rgb = thermal_in_rgb * xyz_in_thermal;
      cv::Point3d xyz_rgb(xyz_in_rgb.x(), xyz_in_rgb.y(), xyz_in_rgb.z());

      // Project onto rgb frame
      cv::Point2d uv_rgb = rgbd_model_.project3dToPixel(xyz_rgb);
      if (uv_rgb.x < 0 || uv_rgb.x >= projection.cols || uv_rgb.y < 0 ||
          uv_rgb.y >= projection.rows) {
        continue;
      }

      _projection(uv_rgb.y, uv_rgb.x)[0] = _thermal(uv_rect.y, uv_rect.x)[0];
      _projection(uv_rgb.y, uv_rgb.x)[1] = _thermal(uv_rect.y, uv_rect.x)[1];
      _projection(uv_rgb.y, uv_rgb.x)[2] = _thermal(uv_rect.y, uv_rect.x)[2];
    }
  }

  projection = _projection;
  cv::namedWindow("Projection");
  cv::imshow("Projection", projection);

  cv::Mat overlay;
  double alpha;
  ros::param::param("overlay_alpha", alpha, 0.5);
  cv::addWeighted(projection, alpha, overlay_cv->image, 1 - alpha, 0.0,
                  overlay);

  cv::namedWindow("Overlay");
  cv::imshow("Overlay", overlay);
  cv::waitKey();
}

void Projection::ProjectRgbdPixelToThermal(
    double rgbd_row, double rgbd_col,
    const cv_bridge::CvImageConstPtr& depth_bridge,
    const Eigen::Affine3d& rgb_in_thermal, double* thermal_row,
    double* thermal_col) {
  uint16_t raw_depth = depth_bridge->image.at<uint16_t>(rgbd_row, rgbd_col);
  double depth = DepthTraits<uint16_t>::toMeters(raw_depth);
  cv::Point3d xyz_rgb =
      rgbd_model_.projectPixelTo3dRay(cv::Point2d(rgbd_col, rgbd_row));
  xyz_rgb *= depth;
  Eigen::Vector3d xyz_in_rgb;
  xyz_in_rgb << xyz_rgb.x, xyz_rgb.y, xyz_rgb.z;

  Eigen::Vector3d xyz_in_thermal = rgb_in_thermal * xyz_in_rgb;

  cv::Point2d uv_thermal = thermal_model_.project3dToPixel(
      cv::Point3d(xyz_in_thermal.x(), xyz_in_thermal.y(), xyz_in_thermal.z()));
  *thermal_row = uv_thermal.y;
  *thermal_col = uv_thermal.x;
}

void Projection::ProjectRgbdOntoThermal(
    const sensor_msgs::Image::ConstPtr& rgb,
    const sensor_msgs::Image::ConstPtr& depth,
    const sensor_msgs::Image::ConstPtr& thermal,
    sensor_msgs::Image* rgbd_projected) {
  cv_bridge::CvImageConstPtr rgb_bridge =
      cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);
  cv_bridge::CvImageConstPtr depth_bridge = cv_bridge::toCvShare(depth);
  cv_bridge::CvImageConstPtr thermal_bridge = cv_bridge::toCvShare(thermal);

  // TODO: use real extrinsic calibration
  Eigen::Vector3d translation;
  ros::param::param<double>("thermal_x", translation.x(), 0);
  ros::param::param<double>("thermal_y", translation.y(), 0.025);
  ros::param::param<double>("thermal_z", translation.z(), 0);
  Eigen::Affine3d rgb_in_thermal;
  rgb_in_thermal.setIdentity();
  rgb_in_thermal.translate(translation);

  // The thermal image is higher resolution than the RGBD image, and it has a
  // higher focal length. So simply projecting the RGBD image onto the thermal
  // image will leave regular gaps in the image. To address this, we compute the
  // projections through the corners of each pixel rather than just the
  // projection of the center of the pixel.
  cv::Mat_<cv::Vec3b> _rgb = rgb_bridge->image;

  cv::Mat output;
  cv::cvtColor(thermal_bridge->image, output, cv::COLOR_GRAY2BGR);
  cv::Mat_<cv::Vec3b> _output = output;

  // Start with a special case: -0.5, -0.5 (upper left corner of upper left
  // pixel).
  // double row_start;
  // double col_start;
  // ProjectRgbdPixelToThermal(-0.5, -0.5, depth_bridge, rgb_in_thermal,
  // &row_start, &col_start);

  for (int rgb_row = 0; rgb_row < rgb_bridge->image.rows; ++rgb_row) {
    for (int rgb_col = 0; rgb_col < rgb_bridge->image.cols; ++rgb_col) {
      double p_row, p_col;
      ProjectRgbdPixelToThermal(rgb_row, rgb_col, depth_bridge, rgb_in_thermal,
                                &p_row, &p_col);

      // uint16_t raw_depth = depth_bridge->image.at<uint16_t>(rgb_row,
      // rgb_col);
      // double depth = DepthTraits<uint16_t>::toMeters(raw_depth);

      int p_row_i = round(p_row);
      int p_col_i = round(p_col);
      if (p_row_i < 0 || p_row_i >= _output.rows || p_col_i < 0 ||
          p_col_i >= output.cols) {
        continue;
      }
      _output(p_row_i, p_col_i)[0] = _rgb(rgb_row, rgb_col)[0];
      _output(p_row_i, p_col_i)[1] = _rgb(rgb_row, rgb_col)[1];
      _output(p_row_i, p_col_i)[2] = _rgb(rgb_row, rgb_col)[2];
    }
  }

  output = _output;
  cv::namedWindow("Projected RGB");
  cv::imshow("Projected RGB", output);

  cv::Mat normalized_thermal;
  cv::normalize(thermal_bridge->image, normalized_thermal, 0, 255,
                cv::NORM_MINMAX);
  cv::Mat normalized_thermal_color = ConvertToColor(normalized_thermal);

  cv::namedWindow("Normalized thermal");
  cv::imshow("Normalized thermal", normalized_thermal_color);

  double alpha;
  ros::param::param("overlay_alpha", alpha, 0.5);
  cv::Mat overlay;
  cv::addWeighted(output, alpha, normalized_thermal_color, 1 - alpha, 0.0,
                  overlay);
  cv::namedWindow("Overlay");
  cv::imshow("Overlay", overlay);

  cv::waitKey();
}
}  // namespace skinseg

#include <cuda_runtime.h>
#include <algorithm>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/projection.h"

using sensor_msgs::Image;

namespace skinseg {
__global__ void gpu_ProjectThermalOnRgb(
    uint16_t* depth_img, uint16_t* thermal, float* z_buffer, int rgbd_rows,
    int rgbd_cols, int thermal_rows, int thermal_cols, CameraData camera_data,
    Eigen::Affine3d* rgb_in_thermal, double max_depth, uint16_t* thermal_mat_out) {
  double rgb_col = (blockIdx.x * blockDim.x + threadIdx.x);
  double rgb_row = (blockIdx.y * blockDim.y + threadIdx.y);

  int r_row = rgb_row + 0.5;
  if (r_row > rgbd_rows - 1) {
    r_row = rgbd_rows - 1;
  }
  int r_col = rgb_col + 0.5;
  if (r_col > rgbd_cols - 1) {
    r_col = rgbd_cols - 1;
  }

  uint16_t raw_depth = depth_img[r_row * rgbd_cols + r_col];
  if (raw_depth == 0) {
    return;
  }
  float depth = raw_depth * 0.001f;
  if (depth > max_depth) {
    return;
  }

  Eigen::Vector3d xyz_rgb;
  // clang-format off
  xyz_rgb << ((rgb_col - camera_data.depth_cx) * depth - camera_data.depth_Tx) *
               camera_data.inv_depth_fx,
             ((rgb_row - camera_data.depth_cy) * depth - camera_data.depth_Ty) *
               camera_data.inv_depth_fy,
             depth;
  // clang-format on

  Eigen::Vector3d xyz_in_thermal = *rgb_in_thermal * xyz_rgb;

  double inv_Z = 1.0 / xyz_in_thermal.z();
  int t_col =
      (camera_data.thermal_fx * xyz_in_thermal.x() + camera_data.thermal_Tx) *
          inv_Z +
      camera_data.thermal_cx + 0.5;
  int t_row =
      (camera_data.thermal_fy * xyz_in_thermal.y() + camera_data.thermal_Ty) *
          inv_Z +
      camera_data.thermal_cy + 0.5;

  if (t_col < 0 || t_col >= thermal_cols || t_row < 0 ||
      t_row >= thermal_rows) {
    return;
  }

  float& prev_depth = z_buffer[thermal_cols * t_row + t_col];
  bool depth_check_passed = prev_depth == 0 || depth < prev_depth;
  if (depth_check_passed) {
    prev_depth = depth;
    thermal_mat_out[rgbd_cols * r_row + r_col] =
        thermal[thermal_cols * t_row + t_col];
  }
}

void Projection::ProjectThermalOnRgb(const Image::ConstPtr& rgb,
                                     const Image::ConstPtr& depth,
                                     const Image::ConstPtr& thermal,
                                     cv::OutputArray thermal_projected) {
  cv_bridge::CvImageConstPtr depth_bridge = cv_bridge::toCvShare(depth);
  cv_bridge::CvImageConstPtr thermal_bridge = cv_bridge::toCvShare(thermal);
  // if (debug_) {
  //  depth_ = depth_bridge->image;
  //  thermal_ = thermal_bridge->image;
  //  thermal_to_rgb_.clear();
  //}

  // Registration of the thermal image to the RGB image is done by projecting
  // the RGBD pixel into the thermal image and copying the pixel in the
  // thermal
  // image.
  thermal_projected.create(rgb->height, rgb->width, CV_16UC1);
  cv::Mat thermal_projected_mat = thermal_projected.getMat();
  thermal_projected_mat = cv::Scalar(0);
  cv::Mat z_buffer = cv::Mat::zeros(thermal_bridge->image.rows,
                                    thermal_bridge->image.cols, CV_32F);

  // cv::Mat rgb_projected;
  // cv::Mat_<cv::Vec3b> _rgb_projected;
  // cv_bridge::CvImageConstPtr rgb_bridge;
  // cv::Mat_<cv::Vec3b> _rgb;
  // if (debug_) {
  //  rgb_projected.create(thermal_bridge->image.rows,
  //  thermal_bridge->image.cols,
  //                       CV_8UC3);
  //  rgb_projected = cv::Scalar(0, 255, 0);
  //  _rgb_projected = rgb_projected;
  //  rgb_bridge = cv_bridge::toCvShare(rgb,
  //  sensor_msgs::image_encodings::BGR8);
  //  rgb_bridge->image.copyTo(rgb_);
  //  _rgb = rgb_bridge->image;
  //}

  int rgb_rows = rgb->height;
  int rgb_cols = rgb->width;

  double max_depth;
  ros::param::param("max_depth", max_depth, 2.0);

  // Prepare GPU inputs
  uint16_t* d_depth;
  int depth_size =
      depth_bridge->image.rows * depth_bridge->image.cols * sizeof(uint16_t);
  cudaMalloc((void**)&d_depth, depth_size);
  cudaMemcpy(d_depth, depth_bridge->image.data, depth_size,
             cudaMemcpyHostToDevice);

  uint16_t* d_thermal;
  int thermal_size = thermal_bridge->image.rows * thermal_bridge->image.cols *
                     sizeof(uint16_t);
  cudaMalloc((void**)&d_thermal, thermal_size);
  cudaMemcpy(d_thermal, thermal_bridge->image.data, thermal_size,
             cudaMemcpyHostToDevice);

  cv::Mat thermal_peek(thermal_bridge->image.rows, thermal_bridge->image.cols,
                       CV_16UC1);
  thermal_peek = cv::Scalar(0);
  cudaError_t err = cudaMemcpy(thermal_peek.data, d_thermal, thermal_size,
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    ROS_ERROR("Couldn't copy thermal: %s", cudaGetErrorString(err));
  }

  float* d_z_buffer;
  int z_buffer_size =
      thermal_bridge->image.rows * thermal_bridge->image.cols * sizeof(float);
  cudaMalloc((void**)&d_z_buffer, z_buffer_size);
  cudaMemcpy(d_z_buffer, z_buffer.data, z_buffer_size, cudaMemcpyHostToDevice);

  Eigen::Affine3d* d_rgb_in_thermal;
  int rgb_in_thermal_size = sizeof(Eigen::Affine3d);
  cudaMalloc((void**)&d_rgb_in_thermal, rgb_in_thermal_size);
  cudaMemcpy(d_rgb_in_thermal, rgb_in_thermal_.data(), rgb_in_thermal_size,
             cudaMemcpyHostToDevice);

  CameraData camera_data;
  GetCameraData(&camera_data);

  // GPU output
  uint16_t* d_thermal_mat;
  int thermal_mat_size =
      depth_bridge->image.rows * depth_bridge->image.cols * sizeof(uint16_t);
  err = cudaMalloc((void**)&d_thermal_mat, thermal_mat_size);
  if (err != cudaSuccess) {
    ROS_ERROR("Failed to allocate d_thermal_mat: %s", cudaGetErrorString(err));
  }
  cudaMemcpy(d_thermal_mat, thermal_projected_mat.data, thermal_mat_size,
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(8, 8);
  dim3 numBlocks(ceil((float)depth_bridge->image.cols / threadsPerBlock.x),
                 ceil((float)depth_bridge->image.rows / threadsPerBlock.y));
  gpu_ProjectThermalOnRgb<<<numBlocks, threadsPerBlock>>>(
      d_depth, d_thermal, d_z_buffer, depth_bridge->image.rows,
      depth_bridge->image.cols, thermal_bridge->image.rows,
      thermal_bridge->image.cols, camera_data, d_rgb_in_thermal, max_depth, d_thermal_mat);
  err = cudaMemcpy(thermal_projected_mat.data, d_thermal_mat, thermal_mat_size,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    ROS_ERROR("Couldn't copy output: %s", cudaGetErrorString(err));
  }

  cudaFree(d_depth);
  cudaFree(d_thermal);
  cudaFree(d_z_buffer);
  cudaFree(d_rgb_in_thermal);
  cudaFree(d_thermal_mat);

  // It is technically more accurate to do two passes, one to create the z
  // buffer and one to compute the projection after the z buffer has been
  // created. In practice it doesn't seem to make much of a difference.

  // if (debug_) {
  //  rgb_projected = _rgb_projected;
  //  cv::namedWindow("RGB projected");
  //  cv::imshow("RGB projected", rgb_projected);

  //  cv::namedWindow("RGB");
  //  cv::imshow("RGB", rgb_bridge->image);

  //  cv::namedWindow("Depth");
  //  cv::Mat normalized_depth;
  //  cv::normalize(depth_bridge->image, normalized_depth, 0, 255,
  //                cv::NORM_MINMAX);
  //  cv::imshow("Depth", ConvertToColor(normalized_depth));

  //  cv::namedWindow("Projected labels");
  //  cv::Mat projected_labels(thermal_projected_mat.rows,
  //                           thermal_projected_mat.cols,
  //                           thermal_projected_mat.type(), cv::Scalar(0));
  //  cv::normalize(thermal_projected_mat, projected_labels, 0, 255,
  //                cv::NORM_MINMAX, -1, NonZeroMask(thermal_projected_mat));
  //  cv::Mat labels_color = ConvertToColor(projected_labels);
  //  cv::imshow("Projected labels", labels_color);

  //  cv::Mat normalized_thermal_image;
  //  cv::normalize(thermal_bridge->image, normalized_thermal_image, 0, 255,
  //                cv::NORM_MINMAX);
  //  cv::Mat normalized_thermal_color =
  //  ConvertToColor(normalized_thermal_image);
  //  cv::namedWindow("Normalized thermal");
  //  cv::imshow("Normalized thermal", normalized_thermal_color);

  //  double alpha;
  //  ros::param::param("overlay_alpha", alpha, 0.5);
  //  cv::Mat overlay;
  //  cv::addWeighted(labels_color, alpha, rgb_bridge->image, 1 - alpha, 0.0,
  //                  overlay);
  //  cv::namedWindow("Overlay");
  //  cv::imshow("Overlay", overlay);

  //  cv::Mat thermal_overlay;
  //  cv::addWeighted(normalized_thermal_color, alpha, rgb_projected, 1 -
  //  alpha,
  //                  0.0, thermal_overlay);
  //  cv::namedWindow("Thermal overlay");
  //  cv::imshow("Thermal overlay", thermal_overlay);
  //}
}
}  // namespace skinseg

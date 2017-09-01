#include <cuda_runtime.h>
#include <algorithm>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"

#include "skin_segmentation/projection.h"

using sensor_msgs::Image;

namespace {
void HandleError(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,
           __LINE__);
    exit(EXIT_FAILURE);
  }
}

cv::Mat ConvertToColor(cv::Mat in) {
  cv::Mat eight_bit;
  in.convertTo(eight_bit, CV_8UC3);
  cv::Mat color;
  cv::cvtColor(eight_bit, color, cv::COLOR_GRAY2RGB);
  return color;
}

// Returns a mask of non-zero values in the input matrix.
cv::Mat NonZeroMask(cv::InputArray in) {
  cv::Mat mask = (in.getMat() != 0);
  cv::Mat mask2;
  cv::threshold(mask, mask2, 0.5, 255, cv::THRESH_BINARY);
  return mask2;
}
}

namespace skinseg {
__global__ void gpu_CreatePointCloud(uint16_t* depth_img, int rows, int cols,
                                     CameraData camera_data,
                                     float4* points_out) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < cols && row < rows) {
    int index = row * cols + col;
    uint16_t raw_depth = depth_img[index];
    if (raw_depth == 0) {
      // A w value of 0 indicates an invalid point.
      points_out[index] = make_float4(0, 0, 0, 0);
      return;
    }
    float depth = raw_depth * 0.001f;
    float x = ((col - camera_data.depth_cx) * depth - camera_data.depth_Tx) *
              camera_data.inv_depth_fx;
    float y = ((row - camera_data.depth_cy) * depth - camera_data.depth_Ty) *
              camera_data.inv_depth_fy;
    float z = depth;
    points_out[index] = make_float4(x, y, z, 1);
  }
}

__global__ void gpu_ProjectThermalOnRgb(
    float4* points, uint16_t* thermal, float* z_buffer, int rgbd_rows,
    int rgbd_cols, int thermal_rows, int thermal_cols, CameraData camera_data,
    Eigen::Affine3d* rgb_in_thermal, uint16_t* thermal_mat_out,
    float4* points_out) {
  int r_col = (blockIdx.x * blockDim.x + threadIdx.x);
  int r_row = (blockIdx.y * blockDim.y + threadIdx.y);

  if (r_row >= rgbd_rows || r_col >= rgbd_cols) {
    return;
  }
  int index = r_row * rgbd_cols + r_col;

  float4 point = points[index];
  if (point.w == 0) {
    return;
  }

  Eigen::Vector3d xyz_rgb;
  xyz_rgb << point.x, point.y, point.z;
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

  float depth = point.z;
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
                                     cv::OutputArray thermal_projected,
                                     float4* points) {
  if (rgb->width != depth->width || rgb->height != depth->height) {
    ROS_ERROR(
        "Expected RGB and depth images to have the same dimensions, got RGB r: "
        "%d, c: %d, Depth r: %d, c: %d",
        rgb->height, rgb->width, depth->height, depth->width);
    return;
  }
  if (depth->encoding != sensor_msgs::image_encodings::TYPE_16UC1) {
    ROS_ERROR("Expected depth to have encoding 16UC1, got %s",
              depth->encoding.c_str());
    return;
  }
  if (thermal->encoding != sensor_msgs::image_encodings::TYPE_16UC1) {
    ROS_ERROR("Expected thermal to have encoding 16UC1, got %s",
              thermal->encoding.c_str());
    return;
  }
  cv_bridge::CvImageConstPtr depth_bridge = cv_bridge::toCvShare(depth);
  cv_bridge::CvImageConstPtr thermal_bridge = cv_bridge::toCvShare(thermal);

  const int rgbd_rows = rgb->height;
  const int rgbd_cols = rgb->width;
  const int thermal_rows = thermal->height;
  const int thermal_cols = thermal->width;

  thermal_projected.create(rgbd_rows, rgbd_cols, CV_16UC1);
  cv::Mat thermal_projected_mat = thermal_projected.getMat();
  thermal_projected_mat = cv::Scalar(0);

  // Create point cloud
  uint16_t* d_depth;
  int depth_size = rgbd_rows * rgbd_cols * sizeof(uint16_t);
  HandleError(cudaMalloc((void**)&d_depth, depth_size));
  HandleError(cudaMemcpy(d_depth, depth->data.data(), depth_size,
                         cudaMemcpyHostToDevice));

  CameraData camera_data;
  GetCameraData(&camera_data);

  float4* d_points;
  int points_size = sizeof(float4) * rgbd_rows * rgbd_cols;
  HandleError(cudaMalloc((void**)&d_points, points_size));

  dim3 threads_per_block(8, 8);
  dim3 num_blocks(ceil((float)rgbd_cols / threads_per_block.x),
                  ceil((float)rgbd_rows / threads_per_block.y));
  gpu_CreatePointCloud<<<num_blocks, threads_per_block>>>(
      d_depth, rgbd_rows, rgbd_cols, camera_data, d_points);

  // Do registration
  uint16_t* d_thermal;
  int thermal_size = thermal_rows * thermal_cols * sizeof(uint16_t);
  cudaMalloc((void**)&d_thermal, thermal_size);
  cudaMemcpy(d_thermal, thermal->data.data(), thermal_size,
             cudaMemcpyHostToDevice);

  // TODO: this is totally ineffective -- need to implement a reduction pattern
  float* d_z_buffer;
  int z_buffer_size = thermal_rows * thermal_cols * sizeof(float);
  HandleError(cudaMalloc((void**)&d_z_buffer, z_buffer_size));
  HandleError(cudaMemset(d_z_buffer, 0, z_buffer_size));

  Eigen::Affine3d* d_rgb_in_thermal;
  int rgb_in_thermal_size = sizeof(Eigen::Affine3d);
  HandleError(cudaMalloc((void**)&d_rgb_in_thermal, rgb_in_thermal_size));
  HandleError(cudaMemcpy(d_rgb_in_thermal, rgb_in_thermal_.data(),
                         rgb_in_thermal_size, cudaMemcpyHostToDevice));


  uint16_t* d_thermal_mat;
  int thermal_mat_size = rgbd_rows * rgbd_cols * sizeof(uint16_t);
  HandleError(cudaMalloc((void**)&d_thermal_mat, thermal_mat_size));
  HandleError(cudaMemcpy(d_thermal_mat, thermal_projected_mat.data,
                         thermal_mat_size, cudaMemcpyHostToDevice));

  gpu_ProjectThermalOnRgb<<<num_blocks, threads_per_block>>>(
      d_points, d_thermal, d_z_buffer, depth_bridge->image.rows,
      depth_bridge->image.cols, thermal_bridge->image.rows,
      thermal_bridge->image.cols, camera_data, d_rgb_in_thermal, d_thermal_mat,
      points);
  HandleError(cudaMemcpy(thermal_projected_mat.data, d_thermal_mat,
                         thermal_mat_size, cudaMemcpyDeviceToHost));

  if (points != NULL) {
    cudaMemcpy(points, d_points, points_size, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_depth);
  cudaFree(d_points);
  cudaFree(d_thermal);
  cudaFree(d_z_buffer);
  cudaFree(d_rgb_in_thermal);
  cudaFree(d_thermal_mat);

  // It is technically more accurate to do two passes, one to create the z
  // buffer and one to compute the projection after the z buffer has been
  // created. In practice it doesn't seem to make much of a difference.

  if (debug_) {
    cv::namedWindow("Projected thermal");
    cv::Mat projected_labels(thermal_projected_mat.rows,
                             thermal_projected_mat.cols,
                             thermal_projected_mat.type(), cv::Scalar(0));
    cv::normalize(thermal_projected_mat, projected_labels, 0, 255,
                  cv::NORM_MINMAX, -1, NonZeroMask(thermal_projected_mat));
    cv::Mat labels_color = ConvertToColor(projected_labels);
    cv::imshow("Projected thermal", labels_color);

    cv_bridge::CvImageConstPtr rgb_bridge =
        cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);
    double alpha;
    ros::param::param("overlay_alpha", alpha, 0.5);
    cv::Mat overlay;
    cv::addWeighted(labels_color, alpha, rgb_bridge->image, 1 - alpha, 0.0,
                    overlay);
    cv::namedWindow("Thermal / RGB overlay");
    cv::imshow("Thermal / RGB overlay", overlay);
  }
}
}  // namespace skinseg

#include <cuda_runtime.h>

#include "Eigen/Dense"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/hand_box_coords.h"
#include "skin_segmentation/projection.h"

namespace {
void HandleError(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,
           __LINE__);
    exit(EXIT_FAILURE);
  }
}

struct CameraData {
  double inv_depth_fx;
  double inv_depth_fy;
  double depth_cx;
  double depth_cy;
  double depth_Tx;
  double depth_Ty;
};
}

namespace skinseg {
__global__ void gpu_ComputeHandMask(
    const float4* points, const int height, const int width, float l_min_x,
    float l_max_x, float l_min_y, float l_max_y, float l_min_z, float l_max_z,
    float r_min_x, float r_max_x, float r_min_y, float r_max_y, float r_min_z,
    float r_max_z, CameraData camera_data, Eigen::Affine3f* world_in_left,
    Eigen::Affine3f* world_in_right, uint8_t* mask) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row > height - 1 || col > width - 1) {
    return;
  }

  int index = row * width + col;
  float4 point = points[index];
  if (point.w == 0) {
    return;
  }

  Eigen::Vector3f xyz;
  xyz << point.x, point.y, point.z;
  Eigen::Vector3f pos_in_l_frame = *world_in_left * xyz;
  Eigen::Vector3f pos_in_r_frame = *world_in_right * xyz;

  bool in_left_box =
      (pos_in_l_frame.x() > l_min_x && pos_in_l_frame.x() < l_max_x &&
       pos_in_l_frame.y() > l_min_y && pos_in_l_frame.y() < l_max_y &&
       pos_in_l_frame.z() > l_min_z && pos_in_l_frame.z() < l_max_z);
  bool in_right_box =
      (pos_in_r_frame.x() > r_min_x && pos_in_r_frame.x() < r_max_x &&
       pos_in_r_frame.y() > r_min_y && pos_in_r_frame.y() < r_max_y &&
       pos_in_r_frame.z() > r_min_z && pos_in_r_frame.z() < r_max_z);
  mask[index] = in_left_box || in_right_box;
}

void ComputeHandMask(float4* points, int height, int width,
                     const HandBoxCoords& left_box,
                     const HandBoxCoords& right_box,
                     const CameraData& camera_data,
                     const Eigen::Affine3f& l_forearm_pose,
                     const Eigen::Affine3f& r_forearm_pose, uint8_t* mask) {
  float4* d_points;
  int points_size = width * height * sizeof(float4);
  HandleError(cudaMalloc((void**)&d_points, points_size));
  HandleError(
      cudaMemcpy(d_points, points, points_size, cudaMemcpyHostToDevice));

  Eigen::Affine3f world_in_left = l_forearm_pose.inverse();
  Eigen::Affine3f* d_l_forearm_pose;
  int pose_size = sizeof(Eigen::Affine3f);
  HandleError(cudaMalloc((void**)&d_l_forearm_pose, pose_size));
  HandleError(cudaMemcpy(d_l_forearm_pose, world_in_left.data(), pose_size,
                         cudaMemcpyHostToDevice));
  Eigen::Affine3f world_in_right = r_forearm_pose.inverse();
  Eigen::Affine3f* d_r_forearm_pose;
  HandleError(cudaMalloc((void**)&d_r_forearm_pose, pose_size));
  HandleError(cudaMemcpy(d_r_forearm_pose, world_in_right.data(), pose_size,
                         cudaMemcpyHostToDevice));

  uint8_t* d_mask;
  int mask_size = height * width * sizeof(uint8_t);
  HandleError(cudaMalloc((void**)&d_mask, mask_size));
  HandleError(cudaMemset(d_mask, 0, mask_size));

  // Kernel code
  dim3 threadsPerBlock(8, 8);
  dim3 numBlocks(ceil((float)width / threadsPerBlock.x),
                 ceil((float)height / threadsPerBlock.y));
  gpu_ComputeHandMask<<<numBlocks, threadsPerBlock>>>(
      d_points, height, width, left_box.min_x, left_box.max_x, left_box.min_y,
      left_box.max_y, left_box.min_z, left_box.max_z, right_box.min_x,
      right_box.max_x, right_box.min_y, right_box.max_y, right_box.min_z,
      right_box.max_z, camera_data, d_l_forearm_pose, d_r_forearm_pose, d_mask);

  HandleError(cudaMemcpy(mask, d_mask, mask_size, cudaMemcpyDeviceToHost));
  HandleError(cudaFree(d_l_forearm_pose));
  HandleError(cudaFree(d_r_forearm_pose));
  HandleError(cudaFree(d_mask));
  HandleError(cudaFree(d_points));
}
}  // namespace skinseg

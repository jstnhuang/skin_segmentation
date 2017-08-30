#include <cuda_runtime.h>

#include "Eigen/Dense"
#include "sensor_msgs/Image.h"

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
__global__ void gpu_ComputeHandMask(const uint16_t* depth_data,
                                    const int height, const int width,
                                    float model_scale, CameraData camera_data,
                                    Eigen::Affine3f* world_in_left,
                                    Eigen::Affine3f* world_in_right,
                                    uint8_t* mask) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row > height - 1 || col > width - 1) {
    return;
  }

  uint16_t raw_depth = depth_data[row * width + col];
  if (raw_depth == 0) {
    return;
  }
  float depth = raw_depth * 0.001f;

  Eigen::Vector3f xyz;
  // clang-format off
  xyz << ((col - camera_data.depth_cx) * depth - camera_data.depth_Tx) *
           camera_data.inv_depth_fx,
         ((row - camera_data.depth_cy) * depth - camera_data.depth_Ty) *
           camera_data.inv_depth_fy,
         depth;
  // clang-format on

  Eigen::Vector3f pos_in_l_frame = *world_in_left * xyz;
  Eigen::Vector3f pos_in_r_frame = *world_in_right * xyz;

  const float min_x = 0.075;
  const float max_x = 0.3;
  const float min_y = -0.09;
  const float max_y = 0.09;
  const float min_z = -0.09;
  const float max_z = 0.09;
  bool in_left_box =
      (pos_in_l_frame.x() > min_x && pos_in_l_frame.x() < max_x &&
       pos_in_l_frame.y() > min_y && pos_in_l_frame.y() < max_y &&
       pos_in_l_frame.z() > min_z && pos_in_l_frame.z() < max_z);
  bool in_right_box =
      (pos_in_r_frame.x() > min_x && pos_in_r_frame.x() < max_x &&
       pos_in_r_frame.y() > min_y && pos_in_r_frame.y() < max_y &&
       pos_in_r_frame.z() > min_z && pos_in_r_frame.z() < max_z);
  mask[row * width + col] = in_left_box || in_right_box;
}

void ComputeHandMask(const sensor_msgs::Image& depth,
                     const CameraData& camera_data, float model_scale,
                     const Eigen::Affine3f& l_forearm_pose,
                     const Eigen::Affine3f& r_forearm_pose, uint8_t* mask) {
  int height = depth.height;
  int width = depth.width;

  uint16_t* d_depth;
  int depth_size = width * height * sizeof(uint16_t);
  HandleError(cudaMalloc((void**)&d_depth, depth_size));
  HandleError(cudaMemcpy(d_depth, depth.data.data(), depth_size,
                         cudaMemcpyHostToDevice));

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
      d_depth, height, width, model_scale, camera_data, d_l_forearm_pose,
      d_r_forearm_pose, d_mask);

  HandleError(cudaMemcpy(mask, d_mask, mask_size, cudaMemcpyDeviceToHost));
  HandleError(cudaFree(d_depth));
  HandleError(cudaFree(d_l_forearm_pose));
  HandleError(cudaFree(d_r_forearm_pose));
  HandleError(cudaFree(d_mask));
}
}  // namespace skinseg

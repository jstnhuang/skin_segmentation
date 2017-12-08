#ifndef _SKINSEG_NERF_H_
#define _SKINSEG_NERF_H_

#include "Eigen/Dense"
#include "model/model.h"
#include "model/model_instance.h"
#include "observation/ros_observation.h"
#include "optimization/optimization_parameters.h"
#include "optimization/optimizer.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "visualization_msgs/MarkerArray.h"

#include "skin_segmentation_msgs/NerfJointStates.h"

namespace skinseg {
// Wrapper class for nerf body tracker.
class Nerf {
 public:
  Nerf(const ros::Publisher& joint_pub, const ros::Publisher& skeleton_pub);
  void Update(const skin_segmentation_msgs::NerfJointStates& joint_states);
  void PublishJointStates();
  void GetJointStates(skin_segmentation_msgs::NerfJointStates* joint_states);
  void PublishVisualization();
  Eigen::Affine3f GetJointPose(const std::string& joint_name);
  void Step(const sensor_msgs::Image::ConstPtr& rgb,
            const sensor_msgs::Image::ConstPtr& depth);
  void set_rgb(float r, float g, float b);

  nerf::RosObservation* observation;  // Not owned by anyone
  nerf::Model* model;
  nerf::ModelInstance* model_instance;  // Owned by model
  nerf::Optimizer* optimizer;           // Not owned by anyone
  nerf::OptimizationParameters opt_parameters;

 private:
  ros::Publisher joint_state_pub_;
  ros::Publisher skeleton_pub_;
  float viz_r_;
  float viz_g_;
  float viz_b_;
};

void BuildNerf(Nerf* nerf, float scale, bool use_hand_segmentation);

void SkeletonMarkerArray(Nerf* nerf, float r, float g, float b,
                         visualization_msgs::MarkerArray* marker_array);
}  // namespace skinseg

#endif  // _SKINSEG_NERF_H_

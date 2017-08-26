#ifndef _SKINSEG_NERF_H_
#define _SKINSEG_NERF_H_

#include "model/model.h"
#include "model/model_instance.h"
#include "observation/ros_observation.h"
#include "optimization/optimization_parameters.h"
#include "optimization/optimizer.h"
#include "visualization_msgs/MarkerArray.h"

namespace skinseg {
// Wrapper class for nerf body tracker.
struct Nerf {
  nerf::RosObservation* observation;  // Not owned by anyone
  nerf::Model* model;
  nerf::ModelInstance* model_instance;  // Owned by model
  nerf::Optimizer* optimizer;           // Not owned by anyone
  nerf::OptimizationParameters opt_parameters;
};

void BuildNerf(Nerf* nerf);

void SkeletonMarkerArray(Nerf* nerf, float scale,
                         visualization_msgs::MarkerArray* marker_array);
}  // namespace skinseg

#endif  // _SKINSEG_NERF_H_

#ifndef _SKINSEG_NERF_H_
#define _SKINSEG_NERF_H_

#include "model/model.h"
#include "model/model_instance.h"
#include "observation/ros_observation.h"
#include "optimization/optimization_parameters.h"
#include "optimization/optimizer.h"

namespace skinseg {
// Wrapper class for nerf body tracker.
struct Nerf {
  nerf::RosObservation* observation;
  nerf::Model model;
  nerf::ModelInstance* model_instance;
  nerf::Optimizer* optimizer;
  nerf::OptimizationParameters opt_parameters;
};

void BuildNerf(Nerf* nerf);
}  // namespace skinseg

#endif  // _SKINSEG_NERF_H_

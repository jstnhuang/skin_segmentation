#include "skin_segmentation/nerf.h"

#include "geometry/dual_quaternion.h"
#include "model/model.h"
#include "model/model_instance.h"
#include "observation/ros_observation.h"
#include "optimization/optimization_parameters.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/load_camera_info.h"

namespace skinseg {
void BuildNerf(Nerf* nerf) {
  bool run_neighbor_filter = true;
  float neighbor_filter_threshold = 0.02;
  int required_neighbors = 4;
  float floor_tx = 0;
  float floor_ty = 0;
  float floor_tz = 0;
  float floor_rx = 0;
  float floor_ry = 0;
  float floor_rz = 0;

  sensor_msgs::CameraInfo rgb_info, thermal_info;
  GetCameraInfos(&rgb_info, &thermal_info);
  nerf::RosObservation* observation =
      new nerf::RosObservation(kRgbTopic, kDepthTopic, rgb_info, rgb_info);
  observation->setupShadowMap(nerf::CameraParameters(
      make_float2(100.0, 100.0), make_float2(64.0, 53.0), make_int2(128, 106)));
  observation->setNeighborFilterParameters(
      run_neighbor_filter, neighbor_filter_threshold, required_neighbors);
  nerf::DualQuaternion floor = nerf::DualQuaternion::TranslateEuler(
      make_float3(floor_tx, floor_ty, floor_tz),
      make_float3(floor_rx, floor_ry, floor_rz));
  observation->addPlane(floor);
  nerf->observation = observation;

  std::string model_path("");
  skinseg::GetNerfModelPath(&model_path);
  nerf::Model model(model_path);
  model.setControlStiffness(0, 0.f);
  model.setControlStiffness(1, 0.f);
  model.setControlStiffness(2, 0.f);
  model.setControlStiffness(3, 0.f);
  model.setControlStiffness(4, 0.f);
  model.setControlStiffness(5, 0.f);
  nerf->model = model;

  nerf->model_instance = model.createNewInstance();
  nerf->model_instance->addToControl(1, 0.5f);
  nerf->model_instance->addToControl(2, 2.f);
  nerf->model_instance->addToControl(5, 3.1415926f);
  nerf->optimizer = new nerf::Optimizer(observation);
  nerf->optimizer->addInstance(nerf->model_instance);

  int poseIterations = 15;
  bool updateControls = true;
  float poseRegularization = 0;
  int shapeIterations = 0;
  bool updateShape = true;
  float shapeRegularization = 0;
  float shapeMagnitudePenalty = 0;
  float shapeNeighborPenalty = 0;

  bool computeStiffness = true;
  float stiffness = 0.;
  bool interpolateAssociation = true;
  float obsToModBlend = 0.f;

  nerf->opt_parameters.controlIterations = poseIterations;
  nerf->opt_parameters.updateControls = updateControls;
  nerf->opt_parameters.controlRegularization = poseRegularization;
  nerf->opt_parameters.controlRegularizationOffset = 0.00001;
  nerf->opt_parameters.computeControlStiffness = computeStiffness;
  nerf->opt_parameters.controlStiffness = stiffness;
  nerf->opt_parameters.controlWindowParameters.resolutionIndex = 0;
  nerf->opt_parameters.controlWindowParameters.neighborhood = 9;
  nerf->opt_parameters.controlWindowParameters.backfaceThreshold = 0.4;
  nerf->opt_parameters.controlWindowParameters.maxDistance = 0.1;
  nerf->opt_parameters.controlWindowParameters.depthWeight = 1.0;
  nerf->opt_parameters.controlWindowParameters.normalDistanceContribution = 0.f;
  nerf->opt_parameters.controlWindowParameters
      .computeModelToObservationResidual = false;
  nerf->opt_parameters.controlWindowParameters
      .computeObservationToModelResidual = true;
  nerf->opt_parameters.controlWindowParameters.modelToObservationBlend = 0.85;
  nerf->opt_parameters.controlWindowParameters.saveObservationToModelResidual =
      true;
  nerf->opt_parameters.controlWindowParameters.save3DResidual = true;
  nerf->opt_parameters.controlWindowParameters.saveAssignment = true;
  nerf->opt_parameters.controlWindowParameters.interpolateAssociation = false;
  nerf->opt_parameters.controlResidualParameters.l22Mix = 0.f;
  nerf->opt_parameters.controlResidualParameters.l22Mag = 0.f;
  nerf->opt_parameters.controlResidualParameters.useHuber = false;
  nerf->opt_parameters.controlResidualParameters.huberDelta = 1.f;

  nerf->opt_parameters.runDetectors = false;
  nerf->opt_parameters.detectorWeight = 0;

  nerf->opt_parameters.shapeIterations = shapeIterations;
  nerf->opt_parameters.updateShape = updateShape;
  nerf->opt_parameters.shapeRegularization = shapeRegularization;
  nerf->opt_parameters.shapeRegularizationOffset = 0.00001;
  nerf->opt_parameters.shapeMagnitudePenalty = shapeMagnitudePenalty;
  nerf->opt_parameters.shapeNeighborPenalty = shapeNeighborPenalty;
  nerf->opt_parameters.shapeWindowParameters.resolutionIndex = 2;
  nerf->opt_parameters.shapeWindowParameters.neighborhood = 9;
  nerf->opt_parameters.shapeWindowParameters.backfaceThreshold = 0.4;
  nerf->opt_parameters.shapeWindowParameters.maxDistance = 0.1;
  nerf->opt_parameters.shapeWindowParameters.depthWeight = 1.0;
  nerf->opt_parameters.shapeWindowParameters.normalDistanceContribution = 0.f;
  nerf->opt_parameters.shapeWindowParameters.computeModelToObservationResidual =
      true;
  nerf->opt_parameters.shapeWindowParameters.computeObservationToModelResidual =
      true;
  nerf->opt_parameters.shapeWindowParameters.modelToObservationBlend =
      obsToModBlend;
  nerf->opt_parameters.shapeWindowParameters.saveObservationToModelResidual =
      true;
  nerf->opt_parameters.shapeWindowParameters.save3DResidual = true;
  nerf->opt_parameters.shapeWindowParameters.saveAssignment = true;
  nerf->opt_parameters.shapeWindowParameters.interpolateAssociation =
      interpolateAssociation;
  nerf->opt_parameters.shapeResidualParameters.l22Mix = 0.f;
  nerf->opt_parameters.shapeResidualParameters.l22Mag = 0.f;
  nerf->opt_parameters.shapeResidualParameters.useHuber = false;
  nerf->opt_parameters.shapeResidualParameters.huberDelta = 1.f;
}

}  // namespace skinseg

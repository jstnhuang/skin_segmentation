#include "skin_segmentation/nerf.h"

#include "geometry/dual_quaternion.h"
#include "model/model.h"
#include "model/model_instance.h"
#include "observation/ros_observation.h"
#include "optimization/optimization_parameters.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/load_configs.h"

namespace skinseg {
void BuildNerf(Nerf* nerf, float model_scale) {
  bool run_neighbor_filter = true;
  float neighbor_filter_threshold = 0.02;
  int required_neighbors = 4;
  float floor_tx = 0;
  float floor_ty = 1.143;
  float floor_tz = 1.714;
  float floor_rx = 0.2243;
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
  nerf->model = new nerf::Model(model_path);
  nerf->model->setControlStiffness(0, 0.f);
  nerf->model->setControlStiffness(1, 0.f);
  nerf->model->setControlStiffness(2, 0.f);
  nerf->model->setControlStiffness(3, 0.f);
  nerf->model->setControlStiffness(4, 0.f);
  nerf->model->setControlStiffness(5, 0.f);

  nerf->model_instance = nerf->model->createNewInstance();
  nerf->model_instance->addToControl(1, 0.5f);
  nerf->model_instance->addToControl(2, 2.f);
  nerf->model_instance->addToControl(5, 3.1415926f);
  nerf->model_instance->setScale(model_scale);
  nerf->optimizer = new nerf::Optimizer(observation);
  nerf->optimizer->addInstance(nerf->model_instance);

  int poseIterations = 15;
  bool updateControls = true;
  float poseRegularization = 0.5;
  int shapeIterations = 2;
  bool updateShape = true;
  float shapeRegularization = 2.5;
  float shapeMagnitudePenalty = 0.25;
  float shapeNeighborPenalty = 0.75;

  bool computeStiffness = true;
  float stiffness = 5;
  bool interpolateAssociation = true;
  float obsToModBlend = 0.25;

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

void SkeletonMarkerArray(Nerf* nerf, const float scale,
                         visualization_msgs::MarkerArray* marker_array) {
  const nerf::KinematicHierarchy* kh = nerf->model->getKinematics();
  const nerf::DualQuaternion* joint_poses =
      nerf->model_instance->getHostJointPose();
  for (int i = 0; i < kh->getNumJoints(); ++i) {
    float3 position = scale * (joint_poses[i] * make_float3(0, 0, 0));
    int parent_i = kh->getHostJointOrder()[i].parent;
    float3 parent_position =
        scale * (joint_poses[parent_i] * make_float3(0, 0, 0));
    visualization_msgs::Marker marker;
    marker.header.frame_id = "camera_rgb_optical_frame";
    marker.id = i;
    marker.color.r = 1;
    marker.color.a = 1;
    geometry_msgs::Point parent_pt;
    parent_pt.x = parent_position.x;
    parent_pt.y = parent_position.y;
    parent_pt.z = parent_position.z;
    marker.points.push_back(parent_pt);
    geometry_msgs::Point pt;
    pt.x = position.x;
    pt.y = position.y;
    pt.z = position.z;
    marker.points.push_back(pt);
    marker.scale.x = 0.01;
    marker.scale.y = 0.015;
    marker_array->markers.push_back(marker);
  }
}
}  // namespace skinseg

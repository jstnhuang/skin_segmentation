#include "skin_segmentation/nerf.h"

#include "Eigen/Dense"
#include "geometry/dual_quaternion.h"
#include "kinematics/kinematic_hierarchy.h"
#include "model/hand_vertices_io.h"
#include "model/model.h"
#include "model/model_instance.h"
#include "observation/ros_observation.h"
#include "optimization/optimization_parameters.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "skin_segmentation_msgs/NerfJointStates.h"
#include "skin_segmentation_msgs/PredictHands.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/load_configs.h"

static const char kHandVerticesPath[] =
    "/home/jstn/tracking_ws/src/nerf_b/data/hand_vertices.json";
static const char kPredictHandsService[] = "/predict_hands";

namespace skinseg {
Nerf::Nerf(const ros::Publisher& joint_pub, const ros::Publisher& skeleton_pub)
    : joint_state_pub_(joint_pub),
      skeleton_pub_(skeleton_pub),
      viz_r_(1),
      viz_g_(0),
      viz_b_(0) {}

void Nerf::Update(const skin_segmentation_msgs::NerfJointStates& joint_states) {
  model_instance->setControls(joint_states.values.data());
  PublishVisualization();
}

void Nerf::PublishJointStates() {
  skin_segmentation_msgs::NerfJointStates joint_states;
  GetJointStates(&joint_states);
  joint_state_pub_.publish(joint_states);
}

void Nerf::GetJointStates(
    skin_segmentation_msgs::NerfJointStates* joint_states) {
  const float* host_controls = model_instance->getHostControls();
  const nerf::KinematicHierarchy* kinematics = model->getKinematics();
  const float2* control_limits = kinematics->getHostControlLimits();
  int num_controls = model->getNumControls();
  joint_states->names.resize(num_controls);
  joint_states->mins.resize(num_controls);
  joint_states->maxs.resize(num_controls);
  joint_states->values.resize(num_controls);
  for (int i = 0; i < num_controls; ++i) {
    joint_states->names[i] = kinematics->getControlName(i);
    joint_states->mins[i] = control_limits[i].x;
    joint_states->maxs[i] = control_limits[i].y;
    joint_states->values[i] = host_controls[i];
  }
}

void Nerf::PublishVisualization() {
  visualization_msgs::MarkerArray skeleton;
  SkeletonMarkerArray(this, viz_r_, viz_g_, viz_b_, &skeleton);
  skeleton_pub_.publish(skeleton);
}

Eigen::Affine3f Nerf::GetJointPose(const std::string& joint_name) {
  float model_scale = model_instance->getScale();
  const nerf::DualQuaternion* joint_poses = model_instance->getHostJointPose();
  int index = model->getKinematics()->getJointIndex(joint_name);
  nerf::DualQuaternion joint_pose = joint_poses[index];
  // joint_pose.normalize(); // Probably not needed.
  Eigen::Affine3f matrix(joint_pose.ToMatrix());
  matrix.translation() *= model_scale;
  return matrix;
}

void Nerf::Step(const sensor_msgs::Image::ConstPtr& rgb,
                const sensor_msgs::Image::ConstPtr& depth) {
  observation->Callback(rgb, depth);
  observation->advance();
  optimizer->optimize(opt_parameters);
}

void Nerf::set_rgb(float r, float g, float b) {
  viz_r_ = r;
  viz_g_ = g;
  viz_b_ = b;
}

void BuildNerf(Nerf* nerf, float model_scale, bool use_hand_segmentation) {
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
  if (use_hand_segmentation) {
    ros::NodeHandle nh;
    ros::ServiceClient predictHands =
        nh.serviceClient<skin_segmentation_msgs::PredictHands>(
            kPredictHandsService, true);
    bool isHandVertex[nerf->model->getNumVertices()];
    for (int i = 0; i < nerf->model->getNumVertices(); ++i) {
      isHandVertex[i] = false;
    }
    nerf::ReadHandVerticesFromFile(kHandVerticesPath, isHandVertex);

    nerf->optimizer->addHandSegmenter(predictHands, isHandVertex,
                                      nerf->model->getNumVertices());
  }

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
  if (use_hand_segmentation) {
    nerf->opt_parameters.controlWindowParameters.neighborhood = 18;
    nerf->opt_parameters.controlWindowParameters.maxDistance = 0.1;
  } else {
    nerf->opt_parameters.controlWindowParameters.neighborhood = 9;
    nerf->opt_parameters.controlWindowParameters.maxDistance = 0.1;
  }
  nerf->opt_parameters.controlWindowParameters.backfaceThreshold = 0.4;
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
  if (use_hand_segmentation) {
    nerf->opt_parameters.shapeWindowParameters.neighborhood = 18;
    nerf->opt_parameters.shapeWindowParameters.maxDistance = 0.1;
  } else {
    nerf->opt_parameters.shapeWindowParameters.neighborhood = 9;
    nerf->opt_parameters.shapeWindowParameters.maxDistance = 0.1;
  }
  nerf->opt_parameters.shapeWindowParameters.backfaceThreshold = 0.4;
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

void SkeletonMarkerArray(Nerf* nerf, float red, float green, float blue,
                         visualization_msgs::MarkerArray* marker_array) {
  float scale = nerf->model_instance->getScale();
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
    marker.color.r = red;
    marker.color.g = green;
    marker.color.b = blue;
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

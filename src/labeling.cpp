#include "skin_segmentation/labeling.h"

#include <cuda_runtime.h>
#include <list>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "depth_image_proc/depth_traits.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pcl/PointIndices.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/search/kdtree.h"
#include "pcl/segmentation/region_growing_rgb.h"
#include "pcl_conversions/pcl_conversions.h"
#include "ros/ros.h"
#include "rosbag/bag.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Float64.h"
#include "visualization_msgs/MarkerArray.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation/nerf.h"
#include "skin_segmentation/opencv_utils.h"
#include "skin_segmentation/pcl_typedefs.h"
#include "skin_segmentation/projection.h"

using sensor_msgs::Image;

namespace skinseg {
double otsu_8u_with_mask(const cv::Mat& src, const cv::Mat& mask) {
  const int N = 256;
  int M = 0;
  int i, j, h[N] = {0};
  for (i = 0; i < src.rows; i++) {
    const uchar* psrc = src.ptr(i);
    const uchar* pmask = mask.ptr(i);
    for (j = 0; j < src.cols; j++) {
      if (pmask[j]) {
        h[psrc[j]]++;
        ++M;
      }
    }
  }

  double mu = 0, scale = 1. / (M);
  for (i = 0; i < N; i++) mu += i * (double)h[i];

  mu *= scale;
  double mu1 = 0, q1 = 0;
  double max_sigma = 0, max_val = 0;

  for (i = 0; i < N; i++) {
    double p_i, q2, mu2, sigma;

    p_i = h[i] * scale;
    mu1 *= q1;
    q1 += p_i;
    q2 = 1. - q1;

    if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
      continue;

    mu1 = (mu1 + i * p_i) / q1;
    mu2 = (mu - q1 * mu1) / q2;
    sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > max_sigma) {
      max_sigma = sigma;
      max_val = i;
    }
  }

  return max_val;
}

Labeling::Labeling(const Projection& projection, Nerf* nerf,
                   rosbag::Bag* output_bag)
    : projection_(projection),
      nerf_(nerf),
      output_bag_(output_bag),
      debug_(false),
      nh_(),
      skeleton_pub_(
          nh_.advertise<visualization_msgs::MarkerArray>("skeleton", 1)),
      rgb_pub_(nh_.advertise<sensor_msgs::Image>(kRgbTopic, 1)),
      depth_pub_(nh_.advertise<sensor_msgs::Image>(kDepthTopic, 1)),
      depth_info_pub_(
          nh_.advertise<sensor_msgs::CameraInfo>(kDepthInfoTopic, 1)),
      thermal_pub_(nh_.advertise<sensor_msgs::Image>(kThermalTopic, 2)),
      cloud_pub_(nh_.advertise<sensor_msgs::PointCloud2>("debug_cloud", 1)),
      labeling_algorithm_(kThermal),
      camera_data_(),
      rgbd_info_(),
      thermal_threshold_(0) {
  projection_.GetCameraData(&camera_data_);
  projection_.GetRgbdCameraInfo(&rgbd_info_);
}

void Labeling::Process(const Image::ConstPtr& rgb, const Image::ConstPtr& depth,
                       const Image::ConstPtr& thermal) {
  if (!rgb || !depth || !thermal) {
    ROS_ERROR("Got null image when processing labels!");
    return;
  }

  if (depth->encoding != sensor_msgs::image_encodings::TYPE_16UC1) {
    ROS_ERROR("Unsupported depth encoding \"%s\", required 16_UC1",
              depth->encoding.c_str());
    return;
  }
  if (rgb->encoding != sensor_msgs::image_encodings::RGB8) {
    ROS_ERROR("Unsupported RGB encoding \"%s\", required RGB8",
              rgb->encoding.c_str());
    return;
  }

  // Step through nerf tracker
  float model_scale = nerf_->model_instance->getScale();
  nerf_->observation->Callback(rgb, depth);
  nerf_->observation->advance();
  nerf_->optimizer->optimize(nerf_->opt_parameters);

  if (debug_) {
    visualization_msgs::MarkerArray skeleton;
    SkeletonMarkerArray(nerf_, model_scale, &skeleton);

    ros::Time now = ros::Time::now();
    sensor_msgs::Image rgb_now = *rgb;
    rgb_now.header.stamp = now;
    sensor_msgs::Image depth_now = *depth;
    depth_now.header.stamp = now;
    rgb_pub_.publish(rgb_now);
    depth_pub_.publish(depth_now);
    sensor_msgs::Image thermal_now = *thermal;
    thermal_now.header.stamp = now;
    thermal_pub_.publish(thermal_now);

    rgbd_info_.header.stamp = now;
    depth_info_pub_.publish(rgbd_info_);
    skeleton_pub_.publish(skeleton);
  }

  const int rgb_rows = rgb->height;
  const int rgb_cols = rgb->width;

  // Find a good thermal threshold for this sequence.
  // Assume that the person is no farther than kMaxDepth meters away. We crop
  // the image to just the person and find a binarization that separates their
  // skin temperature from their clothing temperature.
  cv::Mat thermal_projected;
  float4* points = new float4[rgb_cols * rgb_rows];
  projection_.ProjectThermalOnRgb(rgb, depth, thermal, thermal_projected,
                                  points);
  // Get hand poses
  const nerf::DualQuaternion* joint_poses =
      nerf_->model_instance->getHostJointPose();
  int l_index =
      nerf_->model->getKinematics()->getJointIndex(kNerfLForearmRotJoint);
  int r_index =
      nerf_->model->getKinematics()->getJointIndex(kNerfRForearmRotJoint);
  nerf::DualQuaternion l_forearm_pose = joint_poses[l_index];
  nerf::DualQuaternion r_forearm_pose = joint_poses[r_index];
  l_forearm_pose.normalize();
  r_forearm_pose.normalize();
  Eigen::Affine3f l_matrix(l_forearm_pose.ToMatrix());
  Eigen::Matrix4f r_pose_mat = r_forearm_pose.ToMatrix();
  Eigen::Matrix3f r_pose_rot = r_pose_mat.topLeftCorner(3, 3);
  r_pose_rot.col(0) *= -1;
  r_pose_rot.col(1) = r_pose_rot.col(2).cross(r_pose_rot.col(0));
  r_pose_mat.topLeftCorner(3, 3) = r_pose_rot;
  Eigen::Affine3f r_matrix(r_pose_mat);

  l_matrix.translation() *= model_scale;
  r_matrix.translation() *= model_scale;

  cv::Mat near_hand_mask(rgb_rows, rgb_cols, CV_8UC1, cv::Scalar(0));
  pcl::PointIndices::Ptr indices(new pcl::PointIndices());
  ComputeHandMask(points, rgb_rows, rgb_cols, camera_data_, l_matrix, r_matrix,
                  near_hand_mask.data);

  // Try to find a threshold that separates skin from clothing.
  if (thermal_threshold_ == 0) {
    cv::Mat person(rgb_rows, rgb_cols, CV_16UC1, cv::Scalar(0));
    thermal_projected.copyTo(person, near_hand_mask);

    cv::Mat definite_hand_mask = person > 3200;

    cv::Mat person_8u(rgb_rows, rgb_cols, CV_8UC1, cv::Scalar(0));
    person.convertTo(person_8u, CV_8UC1, 1 / 255.0);
    thermal_threshold_ = 255 * otsu_8u_with_mask(person_8u, definite_hand_mask);
    if (thermal_threshold_ > 0) {
      ROS_INFO("Threshold: %f", thermal_threshold_);
    } else {
      ROS_ERROR("Unable to find good threshold, skipping");
      return;
    }

    cv::Mat otsu_mask = thermal_projected > thermal_threshold_;
    cv::namedWindow("Otsu mask");
    cv::imshow("Otsu mask", otsu_mask * 255);
  }

  cv_bridge::CvImageConstPtr rgb_bridge =
      cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);
  cv_bridge::CvImageConstPtr depth_bridge = cv_bridge::toCvShare(depth);
  if (debug_) {
    // Only for visualization, be sure to edit the real values in labeling.cu
    const float min_x = 0.07;
    const float max_x = 0.3;
    const float min_y = -0.12;
    const float max_y = 0.12;
    const float min_z = -0.09;
    const float max_z = 0.09;

    Eigen::Affine3f center(Eigen::Affine3f::Identity());
    center(0, 3) = (max_x + min_x) / 2;
    center(1, 3) = (max_y + min_y) / 2;
    center(2, 3) = (max_z + min_z) / 2;

    visualization_msgs::MarkerArray boxes;
    visualization_msgs::Marker l_box;
    l_box.header.frame_id = "camera_rgb_optical_frame";
    l_box.type = visualization_msgs::Marker::CUBE;
    l_box.ns = "hand_box";
    l_box.id = 0;
    l_box.color.b = 1;
    l_box.color.a = 0.3;
    l_box.scale.x = max_x - min_x;
    l_box.scale.y = max_y - min_y;
    l_box.scale.z = max_z - min_z;

    Eigen::Affine3f l_pose = l_matrix * center;
    l_box.pose.position.x = l_pose.translation().x();
    l_box.pose.position.y = l_pose.translation().y();
    l_box.pose.position.z = l_pose.translation().z();
    Eigen::Quaternionf l_rot(l_pose.rotation());
    l_box.pose.orientation.w = l_rot.w();
    l_box.pose.orientation.x = l_rot.x();
    l_box.pose.orientation.y = l_rot.y();
    l_box.pose.orientation.z = l_rot.z();
    boxes.markers.push_back(l_box);

    visualization_msgs::Marker l_pt;
    l_pt.header.frame_id = "camera_rgb_optical_frame";
    l_pt.type = visualization_msgs::Marker::SPHERE;
    l_pt.ns = "hand_pt";
    l_pt.id = 0;
    l_pt.color.g = 1;
    l_pt.color.a = 1;
    l_pt.scale.x = 0.04;
    l_pt.scale.y = 0.04;
    l_pt.scale.z = 0.04;
    l_pt.pose = l_box.pose;
    boxes.markers.push_back(l_pt);

    visualization_msgs::Marker r_box;
    r_box.header.frame_id = "camera_rgb_optical_frame";
    r_box.type = visualization_msgs::Marker::CUBE;
    r_box.ns = "hand_box";
    r_box.id = 1;
    r_box.color.b = 1;
    r_box.color.a = 0.3;
    r_box.scale.x = max_x - min_x;
    r_box.scale.y = max_y - min_y;
    r_box.scale.z = max_z - min_z;

    Eigen::Affine3f r_pose = r_matrix * center;
    r_box.pose.position.x = r_pose.translation().x();
    r_box.pose.position.y = r_pose.translation().y();
    r_box.pose.position.z = r_pose.translation().z();
    Eigen::Quaternionf r_rot(r_pose.rotation());
    r_box.pose.orientation.w = r_rot.w();
    r_box.pose.orientation.x = r_rot.x();
    r_box.pose.orientation.y = r_rot.y();
    r_box.pose.orientation.z = r_rot.z();
    boxes.markers.push_back(r_box);

    visualization_msgs::Marker r_pt;
    r_pt.header.frame_id = "camera_rgb_optical_frame";
    r_pt.type = visualization_msgs::Marker::SPHERE;
    r_pt.ns = "hand_pt";
    r_pt.id = 1;
    r_pt.color.g = 1;
    r_pt.color.a = 1;
    r_pt.scale.x = 0.04;
    r_pt.scale.y = 0.04;
    r_pt.scale.z = 0.04;
    r_pt.pose = r_box.pose;
    boxes.markers.push_back(r_pt);

    skeleton_pub_.publish(boxes);
  }

  // If time skew is too great, skip this frame
  double thermal_depth_skew =
      (thermal->header.stamp - depth->header.stamp).toSec();
  double rgb_depth_skew = (rgb->header.stamp - depth->header.stamp).toSec();
  double thermal_rgb_skew = (thermal->header.stamp - rgb->header.stamp).toSec();
  ROS_INFO("Thermal-depth: %f, RGB-depth: %f, Thermal-RGB: %f",
           thermal_depth_skew, rgb_depth_skew, thermal_rgb_skew);
  double max_time_skew;
  ros::param::param("max_time_skew", max_time_skew, 1.00);
  if (fabs(thermal_depth_skew) > max_time_skew ||
      fabs(rgb_depth_skew) > max_time_skew ||
      fabs(thermal_rgb_skew) > max_time_skew) {
    return;
  }

  if (debug_) {
    cv::namedWindow("RGB hands");
    cv::Mat rgb_hands;
    rgb_bridge->image.copyTo(rgb_hands, near_hand_mask);
    cv::imshow("RGB hands", rgb_hands);

    cv::namedWindow("Thermal hands");
    cv::Mat thermal_hands(rgb_rows, rgb_cols, CV_16UC1, cv::Scalar(0));
    thermal_projected.copyTo(thermal_hands, near_hand_mask);
    cv::Mat thermal_normalized(rgb_rows, rgb_cols, CV_32F, cv::Scalar(0.5));
    cv::normalize(thermal_hands, thermal_normalized, 0, 1, cv::NORM_MINMAX,
                  CV_32F, thermal_hands != 0);
    cv::imshow("Thermal hands", thermal_normalized);
  }

  // Labeling
  cv::Mat labels(rgb_rows, rgb_cols, CV_8UC1, cv::Scalar(0));
  if (labeling_algorithm_ == kThermal) {
    LabelWithThermal(thermal_projected, near_hand_mask, rgb_rows, rgb_cols,
                     thermal_threshold_, labels);
  } else if (labeling_algorithm_ == kGrabCut) {
    LabelWithGrabCut(rgb, rgb->height, rgb->width, thermal_projected,
                     near_hand_mask, thermal_threshold_, labels);
  } else if (labeling_algorithm_ == kColorHistogram) {
    LabelWithReducedColorComponents(rgb_bridge->image, near_hand_mask,
                                    thermal_projected, thermal_threshold_,
                                    labels);
  } else if (labeling_algorithm_ == kFloodFill) {
    LabelWithFloodFill(rgb_bridge->image, near_hand_mask, thermal_projected,
                       thermal_threshold_, debug_, labels);
  } else {
    ROS_ERROR_THROTTLE(1, "Unknown labeling algorithm %s",
                       labeling_algorithm_.c_str());
  }

  delete[] points;

  // Visualization
  cv_bridge::CvImage labels_bridge(
      rgb->header, sensor_msgs::image_encodings::TYPE_8UC1, labels);

  cv::Mat overlay(rgb_rows, rgb_cols, CV_8UC3, cv::Scalar(150, 150, 150));
  rgb_bridge->image.copyTo(overlay, depth_bridge->image != 0);
  overlay.setTo(cv::Scalar(0, 0, 255), labels != 0);
  cv::namedWindow("Label overlay");
  cv::imshow("Label overlay", overlay);
  cv_bridge::CvImage overlay_bridge(
      rgb->header, sensor_msgs::image_encodings::BGR8, overlay);

  // Even though there is some time difference, we are assuming that we have
  // done our best to temporally align the images and now assume all the images
  // have the same timestamp.
  sensor_msgs::Image depth_out = *depth;
  depth_out.header.stamp = rgb->header.stamp;

  output_bag_->write(kRgbTopic, rgb->header.stamp, rgb);
  output_bag_->write(kDepthTopic, rgb->header.stamp, depth_out);
  output_bag_->write(kLabelsTopic, rgb->header.stamp,
                     labels_bridge.toImageMsg());
  output_bag_->write(kLabelOverlayTopic, rgb->header.stamp,
                     overlay_bridge.toImageMsg());

  cudaDeviceSynchronize();

  if (debug_) {
    cv::waitKey();
  }
}

void Labeling::set_debug(bool debug) { debug_ = debug; }

void Labeling::set_labeling_algorithm(const std::string& alg) {
  labeling_algorithm_ = alg;
}

void MaskToIndices(uint8_t* mask, int len,
                   pcl::PointIndices::Ptr near_hand_indices) {
  near_hand_indices->indices.clear();
  for (int i = 0; i < len; ++i) {
    if (mask[i] > 0) {
      near_hand_indices->indices.push_back(i);
    }
  }
}

void GetPointCloud(const float4* points, const sensor_msgs::Image& rgb,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud) {
  for (unsigned int row = 0; row < rgb.height; ++row) {
    for (unsigned int col = 0; col < rgb.width; ++col) {
      int index = row * rgb.width + col;

      float4 pt = points[index];
      PointC point;
      point.x = pt.x;
      point.y = pt.y;
      point.z = pt.z;

      // Color
      int rgb_r_index = index * 3;
      int rgb_g_index = index * 3 + 1;
      int rgb_b_index = index * 3 + 2;
      point.r = rgb.data[rgb_r_index];
      point.g = rgb.data[rgb_g_index];
      point.b = rgb.data[rgb_b_index];

      point_cloud->push_back(point);
    }
  }
  point_cloud->header.frame_id = rgb.header.frame_id;
}

void Labeling::LabelWithThermal(cv::Mat thermal_projected,
                                cv::Mat near_hand_mask, int rows, int cols,
                                float thermal_threshold,
                                cv::OutputArray labels) {
  cv::Mat thermal_fp;
  thermal_projected.convertTo(thermal_fp, CV_32F);

  cv::Mat hot_pixels;
  cv::threshold(thermal_fp, hot_pixels, thermal_threshold, 1,
                cv::THRESH_BINARY);
  cv::Mat hot_pixels_8;
  hot_pixels.convertTo(hot_pixels_8, CV_8UC1);

  labels.create(rows, cols, CV_8UC1);
  cv::Mat labels_mat = labels.getMat();
  labels_mat = cv::Scalar(0);
  hot_pixels_8.copyTo(labels_mat, near_hand_mask);
}

void Labeling::LabelWithGrabCut(const sensor_msgs::ImageConstPtr& rgb, int rows,
                                int cols, cv::Mat thermal_projected,
                                cv::Mat near_hand_mask, float thermal_threshold,
                                cv::OutputArray labels) {
  labels.create(rows, cols, CV_32F);
  cv::Mat labels_mat = labels.getMat();
  labels_mat = cv::Scalar(0);

  cv_bridge::CvImageConstPtr rgb_bridge =
      cv_bridge::toCvShare(rgb, sensor_msgs::image_encodings::BGR8);
  cv::Mat rgb_hands(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
  rgb_bridge->image.copyTo(rgb_hands, near_hand_mask);
  cv::namedWindow("RGB");
  cv::imshow("RGB", rgb_bridge->image);

  cv::Mat thermal_labels;
  LabelWithThermal(thermal_projected, near_hand_mask, rows, cols,
                   thermal_threshold, thermal_labels);

  cv::Mat prob_mask(rows, cols, CV_8UC1, cv::Scalar(cv::GC_BGD));
  prob_mask.setTo(cv::Scalar(cv::GC_PR_BGD), near_hand_mask);
  prob_mask.setTo(cv::Scalar(cv::GC_PR_FGD), thermal_labels);
  cv::Mat eroded_thermal_labels;
  cv::erode(thermal_labels, eroded_thermal_labels, cv::MORPH_RECT);
  prob_mask.setTo(cv::Scalar(cv::GC_FGD), eroded_thermal_labels);

  cv::Mat prob_normalized;
  cv::normalize(prob_mask, prob_normalized, 0, 255, cv::NORM_MINMAX);
  cv::namedWindow("Prob mask in");
  cv::imshow("Prob mask in", prob_normalized);

  cv::Rect roi_unused;
  cv::Mat back_model;
  cv::Mat fore_model;
  grabCut(rgb_bridge->image, prob_mask, roi_unused, back_model, fore_model, 5,
          cv::GC_INIT_WITH_MASK);

  cv::normalize(prob_mask, prob_normalized, 0, 255, cv::NORM_MINMAX);
  cv::namedWindow("GrabCut");
  cv::imshow("GrabCut", prob_normalized);

  cv::Mat gc_labels = prob_mask & 1;
  gc_labels.copyTo(labels_mat, near_hand_mask);
}

void Labeling::LabelWithRegionGrowingRGB(float4* points,
                                         sensor_msgs::ImageConstPtr rgb,
                                         cv::Mat thermal,
                                         cv::Mat near_hand_mask, int rows,
                                         int cols, float thermal_threshold,
                                         cv::OutputArray labels) {
  // Segmentation of the hands
  pcl::PointIndices::Ptr near_hand_indices(new pcl::PointIndices());
  MaskToIndices(near_hand_mask.data, rows * cols, near_hand_indices);

  // Must be careful - point_cloud contains some invalid points (w=0). These are
  // ignored by ComputeHandMask and thus not added to near_hand_indices.
  PointCloudC::Ptr point_cloud(new PointCloudC());
  GetPointCloud(points, *rgb, point_cloud);

  // Color-based region growing segmentation of the hand
  // This is designed to oversegment.
  float pt_distance;
  ros::param::param("label_data_rgbseg_distance", pt_distance, 0.02f);
  // These are in "color metric" distances, which is just the Euclidean distance
  // between two colors if you treat their RGB values as a vector.
  float color_threshold;
  ros::param::param("label_data_rgbseg_color", color_threshold, 6.0f);
  float region_color_threshold;
  ros::param::param("label_data_rgbseg_region", region_color_threshold, 5.0f);
  // The algorithm is mostly sensitive to this parameter
  int min_size;
  ros::param::param("label_data_rgbseg_min_size", min_size, 15);

  // pcl::search::KdTree<PointC>::Ptr tree(new pcl::search::KdTree<PointC>());
  // pcl::RegionGrowingRGB<PointC> reg;
  // reg.setInputCloud(point_cloud);
  // reg.setIndices(near_hand_indices);
  // reg.setSearchMethod(tree);
  // reg.setDistanceThreshold(pt_distance);
  // reg.setPointColorThreshold(color_threshold);
  // reg.setPointColorThreshold(region_color_threshold);
  // reg.setPointColorThreshold(min_size);
  // std::vector<pcl::PointIndices> clusters;
  // reg.extract(clusters);

  // if (debug_) {
  //  ROS_INFO("Found %ld clusters", clusters.size());
  //  sensor_msgs::PointCloud2 debug_pc;
  //  PointCloudC::Ptr colored = reg.getColoredCloud();
  //  pcl::ExtractIndices<PointC> extract;
  //  extract.setInputCloud(colored);
  //  extract.setIndices(near_hand_indices);
  //  PointCloudC::Ptr output(new PointCloudC());
  //  extract.filter(*output);
  //  pcl::toROSMsg(*output, debug_pc);
  //  debug_pc.header.frame_id = rgb->header.frame_id;
  //  cloud_pub_.publish(debug_pc);
  //}

  // float cluster_heat_threshold;
  // ros::param::param("label_data_heat_percentage", cluster_heat_threshold,
  //                  50.0f);
  // for (size_t cluster_i = 0; cluster_i < clusters.size(); ++cluster_i) {
  //  const pcl::PointIndices& cluster = clusters[cluster_i];
  //  float num_hot = 0;
  //  size_t cluster_size = cluster.indices.size();
  //  for (size_t index = 0; index < cluster_size; ++index) {
  //    int pt_index = cluster.indices[index];
  //    uint16_t thermal_val =
  //        reinterpret_cast<uint16_t*>(thermal_projected.data)[pt_index];
  //    if (thermal_val > thermal_threshold) {
  //      num_hot += 1;
  //    }
  //  }
  //  float percent_hot = num_hot / cluster_size;
  //  if (percent_hot > cluster_heat_threshold) {
  //    for (size_t index = 0; index < cluster_size; ++index) {
  //      int pt_index = cluster.indices[index];
  //      labels.data[pt_index] = 255;
  //    }
  //  }
  //}
}

void LabelWithReducedColorComponents(cv::Mat rgb, cv::Mat near_hand_mask,
                                     cv::Mat thermal_projected,
                                     double thermal_threshold,
                                     cv::OutputArray labels) {
  int num_bins;
  ros::param::param("num_bins", num_bins, 2);
  cv::Mat rgb_reduced;
  ReduceRgb(rgb, near_hand_mask, num_bins, rgb_reduced);
  std::vector<std::vector<cv::Point> > clusters;
  FindConnectedComponents(rgb_reduced, near_hand_mask, &clusters);

  cv::namedWindow("Reduced RGB");
  cv::imshow("Reduced RGB", rgb_reduced);

  // Number of hot pixels in each cluster.
  int thermal_counts[clusters.size()];
  for (size_t i = 0; i < clusters.size(); ++i) {
    thermal_counts[i] = 0;
  }

  for (size_t cluster_i = 0; cluster_i < clusters.size(); ++cluster_i) {
    const std::vector<cv::Point>& cluster = clusters[cluster_i];
    for (size_t pt_i = 0; pt_i < cluster.size(); ++pt_i) {
      const cv::Point& pt = cluster[pt_i];
      int pixel_index = pt.y * rgb.cols + pt.x;
      uint16_t thermal_val =
          reinterpret_cast<uint16_t*>(thermal_projected.data)[pixel_index];
      if (thermal_val > thermal_threshold) {
        thermal_counts[cluster_i] += 1;
      }
    }
  }

  cv::Mat labels_mat = labels.getMat();
  labels_mat = cv::Scalar(0);

  float percent_hot_threshold;
  ros::param::param("percent_hot_threshold", percent_hot_threshold, 0.75f);
  for (size_t cluster_i = 0; cluster_i < clusters.size(); ++cluster_i) {
    const std::vector<cv::Point>& cluster = clusters[cluster_i];
    float thermal_percent =
        static_cast<float>(thermal_counts[cluster_i]) / cluster.size();
    if (thermal_percent > percent_hot_threshold) {
      for (size_t pt_i = 0; pt_i < cluster.size(); ++pt_i) {
        const cv::Point& pt = cluster[pt_i];
        int pixel_index = pt.y * rgb.cols + pt.x;
        labels_mat.data[pixel_index] = 1;
      }
    }
  }
}

void ReduceRgb(cv::Mat rgb, cv::Mat near_hand_mask, int num_bins,
               cv::OutputArray reduced) {
  reduced.create(rgb.rows, rgb.cols, CV_8UC3);
  cv::Mat reduced_mat = reduced.getMat();
  reduced_mat = cv::Scalar(200, 200, 200);
  cv::Mat_<cv::Vec3b> _rgb = rgb;
  cv::Mat_<cv::Vec3b> _reduced = reduced_mat;
  float bin_size = 256.0 / num_bins;
  for (int row = 0; row < rgb.rows; ++row) {
    for (int col = 0; col < rgb.cols; ++col) {
      int pixel_index = row * rgb.cols + col;
      if (near_hand_mask.data[pixel_index] != 0) {
        // Bin into 5 possible values of R, G, and B.
        int r = _rgb(row, col)[0] / bin_size;
        int g = _rgb(row, col)[1] / bin_size;
        int b = _rgb(row, col)[2] / bin_size;
        _reduced(row, col)[0] = r * bin_size;
        _reduced(row, col)[1] = g * bin_size;
        _reduced(row, col)[2] = b * bin_size;
      }
    }
  }
  reduced_mat = _reduced;
}

// Used only for FindConnectedComponents
inline bool IsValidNeighbor(cv::Point current, cv::Point neighbor,
                            cv::Mat reduced_rgb, cv::Mat unvisited) {
  cv::Mat_<cv::Vec3b> _rgb = reduced_rgb;
  int rows = reduced_rgb.rows;
  int cols = reduced_rgb.cols;

  int current_r = _rgb(current.y, current.x)[0];
  int current_g = _rgb(current.y, current.x)[1];
  int current_b = _rgb(current.y, current.x)[2];

  int n_index = neighbor.y * cols + neighbor.x;
  int neighbor_r = _rgb(neighbor.y, neighbor.x)[0];
  int neighbor_g = _rgb(neighbor.y, neighbor.x)[1];
  int neighbor_b = _rgb(neighbor.y, neighbor.x)[2];
  if (neighbor.y >= 0 && neighbor.y < rows && unvisited.data[n_index] &&
      neighbor_r == current_r && neighbor_g == current_g &&
      neighbor_b == current_b) {
    return true;
  }
  return false;
}

void FindConnectedComponents(cv::Mat reduced_rgb, cv::Mat near_hand_mask,
                             std::vector<std::vector<cv::Point> >* clusters) {
  int rows = reduced_rgb.rows;
  int cols = reduced_rgb.cols;
  cv::Mat unvisited(rows, cols, CV_8UC1);
  unvisited = cv::Scalar(0);
  near_hand_mask.copyTo(unvisited, near_hand_mask);
  cv::Point start_pt;
  double max_val;
  cv::minMaxLoc(unvisited, NULL, &max_val, NULL, &start_pt, near_hand_mask);

  cv::Mat_<cv::Vec3b> _rgb = reduced_rgb;
  while (max_val != 0) {
    std::list<cv::Point> queue;
    queue.push_back(start_pt);
    std::vector<cv::Point> cluster;
    while (!queue.empty()) {
      const cv::Point& pt = queue.front();
      int index = pt.y * cols + pt.x;
      if (unvisited.data[index]) {
        cluster.push_back(pt);
      } else {
        queue.pop_front();
        continue;
      }
      unvisited.data[index] = 0;

      cv::Point top(pt.x, pt.y - 1);
      if (IsValidNeighbor(pt, top, reduced_rgb, unvisited)) {
        queue.push_back(top);
      }

      cv::Point bottom(pt.x, pt.y + 1);
      if (IsValidNeighbor(pt, bottom, reduced_rgb, unvisited)) {
        queue.push_back(bottom);
      }

      cv::Point right(pt.x + 1, pt.y);
      if (IsValidNeighbor(pt, right, reduced_rgb, unvisited)) {
        queue.push_back(right);
      }

      cv::Point left(pt.x - 1, pt.y);
      if (IsValidNeighbor(pt, left, reduced_rgb, unvisited)) {
        queue.push_back(left);
      }

      queue.pop_front();
    }

    cv::minMaxLoc(unvisited, NULL, &max_val, NULL, &start_pt, near_hand_mask);
    clusters->push_back(cluster);
  }
}

void LabelWithFloodFill(cv::Mat rgb, cv::Mat near_hand_mask,
                        cv::Mat thermal_projected, double thermal_threshold,
                        bool debug, cv::OutputArray labels) {
  // floodFill API requires the mask to have 0s in the area to fill. We use the
  // mask as follows:
  // 0: pixel to fill
  // 1: pixels to ignore
  // 2: pixel that has been labeled
  // The mask is 2 pixels wider and taller than the image, so image pixel (x, y)
  // corresponds to (x+1, y+1) in the mask.
  //
  // The flood fill is done on the RGB image, using hot pixels as seed points
  cv::Mat mask(rgb.rows + 2, rgb.cols + 2, CV_8UC1, cv::Scalar(1));
  cv::Mat mask_image = mask(cv::Rect(1, 1, rgb.cols, rgb.rows));
  cv::Mat inverted_hand_mask = 1 - near_hand_mask;
  inverted_hand_mask.copyTo(mask_image, near_hand_mask);

  cv::Mat rgb_blurred;
  cv::GaussianBlur(rgb, rgb_blurred, cv::Size(5, 5), 1);

  cv::Mat thermal_hands(rgb.rows, rgb.cols, CV_16UC1, cv::Scalar(0));
  thermal_projected.copyTo(thermal_hands, near_hand_mask);
  cv::Mat hot_hand_mask = thermal_hands > thermal_threshold;

  if (debug) {
    // Visualize hot pixels
    cv::namedWindow("Hot pixels");
    cv::Mat hot_hand_viz(rgb.rows, rgb.cols, CV_8UC1, cv::Scalar(100));
    hot_hand_viz.setTo(0, near_hand_mask);
    hot_hand_viz.setTo(255, hot_hand_mask);
    cv::imshow("Hot pixels", hot_hand_viz);
  }

  cv::Scalar unused_new_val(0, 0, 0);
  cv::Rect unused_rect;
  float color_diff;
  ros::param::param("color_diff", color_diff, 60.0f);
  cv::Scalar lo_diff(color_diff, color_diff, color_diff);
  cv::Scalar up_diff(color_diff, color_diff, color_diff);
  int flags =
      4 | cv::FLOODFILL_FIXED_RANGE | cv::FLOODFILL_MASK_ONLY | (2 << 8);

  // Build mask of hot hand pixels
  cv::Mat thermal_mask(rgb.rows, rgb.cols, CV_8UC1, cv::Scalar(0));
  thermal_mask.setTo(1, hot_hand_mask);

  double max_val;
  cv::Point seed_point;
  cv::minMaxLoc(thermal_hands, NULL, &max_val, NULL, &seed_point,
                hot_hand_mask);
  while (max_val > thermal_threshold) {
    cv::floodFill(rgb_blurred, mask, seed_point, unused_new_val, &unused_rect,
                  lo_diff, up_diff, flags);
    // Suppress thermal image pixels corresponding to pixels that were flood
    // filled.
    thermal_mask.setTo(0, mask_image == 2);
    cv::minMaxLoc(thermal_hands, NULL, &max_val, NULL, &seed_point,
                  thermal_mask);
  }
  if (debug) {
    cv::namedWindow("Flood fill mask");
    cv::imshow("Flood fill mask", mask * 127);
  }

  cv::Mat labels_mat = labels.getMat();
  mask_image.copyTo(labels_mat, mask_image == 2);
}
}  // namespace skinseg

#include "skin_segmentation/snapshot.h"

#include <string>
#include <vector>

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

using sensor_msgs::CameraInfo;
using sensor_msgs::Image;
typedef message_filters::sync_policies::ApproximateTime<Image, Image, Image>
    SnapPolicy;

namespace skinseg {
Snapshot::Snapshot()
    : nh_(),
      color_sub_(nh_, kColorTopic, 1),
      depth_sub_(nh_, kDepthTopic, 1),
      thermal_sub_(nh_, kThermalTopic, 1),
      sync_(SnapPolicy(10), color_sub_, depth_sub_, thermal_sub_),
      is_capturing_(false) {
  sync_.registerCallback(&Snapshot::Callback, this);
  color_sub_.unsubscribe();
  depth_sub_.unsubscribe();
  thermal_sub_.unsubscribe();
}

void Snapshot::StartCapture() {
  color_sub_.subscribe();
  depth_sub_.subscribe();
  thermal_sub_.subscribe();
  is_capturing_ = true;
}

bool Snapshot::IsCaptureDone() { return !is_capturing_; }

bool Snapshot::SaveBag(const std::string& path) {
  if (!data_.color || !data_.depth || !data_.thermal ||
      !data_.rgbd_camera_info || !data_.thermal_camera_info) {
    ROS_ERROR("Cannot save incomplete data to bag file.");
    return false;
  }
  try {
    rosbag::Bag bag;
    bag.open(path, rosbag::bagmode::Write);
    ros::Time now = ros::Time::now();
    bag.write(kColorTopic, now, data_.color);
    bag.write(kDepthTopic, now, data_.depth);
    bag.write(kThermalTopic, now, data_.thermal);
    bag.write(kRgbdCameraInfoTopic, now, data_.rgbd_camera_info);
    bag.write(kThermalCameraInfoTopic, now, data_.thermal_camera_info);
    bag.close();
    return true;
  } catch (const rosbag::BagException& e) {
    ROS_ERROR("%s", e.what());
    return false;
  }
}

bool Snapshot::LoadBag(const std::string& path) {
  try {
    rosbag::Bag bag;
    bag.open(path, rosbag::bagmode::Read);
    std::vector<std::string> topics;
    topics.push_back(kColorTopic);
    topics.push_back(kDepthTopic);
    topics.push_back(kThermalTopic);
    topics.push_back(kRgbdCameraInfoTopic);
    topics.push_back(kThermalCameraInfoTopic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    for (rosbag::View::const_iterator it = view.begin(); it != view.end();
         ++it) {
      const rosbag::MessageInstance& mi = *it;
      if (mi.getTopic() == kColorTopic) {
        data_.color = mi.instantiate<Image>();
        if (!data_.color) {
          ROS_WARN("Unable to instantiate color image.");
          continue;
        }
      } else if (mi.getTopic() == kDepthTopic) {
        data_.depth = mi.instantiate<Image>();
        if (!data_.depth) {
          ROS_WARN("Unable to instantiate depth image.");
          continue;
        }
      } else if (mi.getTopic() == kThermalTopic) {
        data_.thermal = mi.instantiate<Image>();
        if (!data_.thermal) {
          ROS_WARN("Unable to instantiate thermal image.");
          continue;
        }
      } else if (mi.getTopic() == kRgbdCameraInfoTopic) {
        data_.rgbd_camera_info = mi.instantiate<CameraInfo>();
        if (!data_.rgbd_camera_info) {
          ROS_WARN("Unable to instantiate RGBD camera info.");
          continue;
        }
      } else if (mi.getTopic() == kThermalCameraInfoTopic) {
        data_.thermal_camera_info = mi.instantiate<CameraInfo>();
        if (!data_.thermal_camera_info) {
          ROS_WARN("Unable to instantiate thermal camera info.");
          continue;
        }
      }
    }
    bag.close();

    if (!data_.color || !data_.depth || !data_.thermal ||
        !data_.rgbd_camera_info || !data_.thermal_camera_info) {
      ROS_ERROR("Bag file contained incomplete data.");
      return false;
    }

    return true;
  } catch (const rosbag::BagException& e) {
    ROS_ERROR("%s", e.what());
    return false;
  }
}

RgbdtData Snapshot::data() { return data_; }

void Snapshot::Callback(const Image::ConstPtr& color,
                        const Image::ConstPtr& depth,
                        const Image::ConstPtr& thermal) {
  data_.color = color;
  data_.depth = depth;
  data_.thermal = thermal;
  color_sub_.unsubscribe();
  depth_sub_.unsubscribe();
  thermal_sub_.unsubscribe();

  ROS_INFO("Getting camera infos");
  while (!data_.rgbd_camera_info && ros::ok()) {
    data_.rgbd_camera_info = ros::topic::waitForMessage<CameraInfo>(
        kRgbdCameraInfoTopic, nh_, ros::Duration(1.0));
    if (!data_.rgbd_camera_info) {
      ROS_WARN("Waiting for RGBD camera info on topic %s",
               kRgbdCameraInfoTopic);
    }
  }
  while (!data_.thermal_camera_info && ros::ok()) {
    data_.thermal_camera_info =
        ros::topic::waitForMessage<CameraInfo>(kThermalCameraInfoTopic, nh_);
    if (!data_.thermal_camera_info) {
      ROS_WARN("Waiting for thermal camera info on topic %s",
               kThermalCameraInfoTopic);
    }
  }

  ROS_INFO("Done");
  is_capturing_ = false;
}
}  // namespace skinseg

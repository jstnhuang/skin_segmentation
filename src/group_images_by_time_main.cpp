// Utility that, given a bag file of image data, groups each thermal image with
// the RGB and depth image closest in time to it. The results are written out to
// another bag file, with messages grouped in skin_segmentation/Images messages.

#include <limits.h>
#include <iostream>
#include <string>
#include <vector>

#include "ros/ros.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/constants.h"
#include "skin_segmentation_msgs/Images.h"

namespace skinseg {}  // namespace skinseg

// Gets the message in the bag file on the given topic whose timestamp is
// closest to the given time. We are processing image data that is published at
// approximately 30 Hz, so we only look in a window of about 10 ms before and
// after the given time. The thermal camera does freeze up during shutter
// events, but the RGB and depth cameras should be publishing continuously. As a
// result, this function should be used for RGB and depth data only.
sensor_msgs::ImageConstPtr GetClosestMessage(const rosbag::Bag& bag,
                                             const rosbag::TopicQuery query,
                                             const ros::Time& time) {
  ros::Duration kWindowSize(0.1);
  // double time_secs = time.toSec();
  // ros::Time start = ros::Time(0);
  // if (time_secs - kWindowSize.toSec() >= 0) {
  //  start = time - kWindowSize;
  //}
  rosbag::View view(bag, query, time - kWindowSize, time + kWindowSize);

  sensor_msgs::ImageConstPtr best(new sensor_msgs::Image);
  double best_skew = std::numeric_limits<double>::max();
  for (rosbag::View::const_iterator it = view.begin(); it != view.end(); ++it) {
    sensor_msgs::ImageConstPtr current = it->instantiate<sensor_msgs::Image>();
    double skew = fabs((time - current->header.stamp).toSec());
    if (skew < best_skew) {
      best = current;
      best_skew = skew;
    }
  }
  if (best_skew == std::numeric_limits<double>::max()) {
    ROS_ERROR("View was empty!");
  }
  return best;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "group_images_by_time");
  ros::Time::init();
  if (argc < 3) {
    std::cout << "Usage: rosrun skin_segmentation group_images_by_time "
                 "INPUT.bag OUTPUT.bag"
              << std::endl;
    return 1;
  }

  std::string input_bag_path(argv[1]);
  std::string output_bag_path(argv[2]);

  rosbag::Bag input_bag;
  input_bag.open(input_bag_path, rosbag::bagmode::Read);

  std::vector<std::string> thermal_topics;
  thermal_topics.push_back(skinseg::kThermalTopic);
  rosbag::TopicQuery thermal_topic_query(thermal_topics);

  std::vector<std::string> rgb_topics;
  rgb_topics.push_back(skinseg::kRgbTopic);
  rosbag::TopicQuery rgb_topic_query(rgb_topics);

  std::vector<std::string> depth_topics;
  depth_topics.push_back(skinseg::kDepthTopic);
  rosbag::TopicQuery depth_topic_query(depth_topics);

  rosbag::Bag output_bag;
  output_bag.open(output_bag_path, rosbag::bagmode::Write);

  rosbag::View thermal_view(input_bag, thermal_topic_query);
  int i = 0;
  for (rosbag::View::const_iterator it = thermal_view.begin();
       it != thermal_view.end(); ++it) {
    skin_segmentation_msgs::Images images;
    sensor_msgs::ImageConstPtr thermal = it->instantiate<sensor_msgs::Image>();
    images.thermal = *thermal;

    ros::Time thermal_time = images.thermal.header.stamp;
    images.depth =
        *GetClosestMessage(input_bag, depth_topic_query, thermal_time);
    // ros::Time halfway(
    //    (thermal_time.toSec() + images.depth.header.stamp.toSec()) / 2);
    images.rgb = *GetClosestMessage(input_bag, rgb_topic_query,
                                    images.depth.header.stamp);
    output_bag.write(skinseg::kImageSetTopic, thermal_time, images);

    ++i;
    if (i % 100 == 0) {
      ROS_INFO("Processed image %d out of %d (%f)", i, thermal_view.size(),
               static_cast<float>(i) / thermal_view.size());
    }
  }

  output_bag.close();

  return 0;
}

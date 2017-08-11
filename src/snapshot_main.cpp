// Utility for capturing a color, depth, and thermal image triplet.

#include <iostream>
#include <string>

#include "skin_segmentation/snapshot.h"

void print_usage() {
  std::cout << "Saves a color, depth, and thermal image to a bag file."
            << std::endl;
  std::cout
      << "Usage: rosrun skin_segmentation snapshot PATH/NAME.bag [DELAY SECS]"
      << "Example: rosrun skin_segmentation snapshot ~/data/snapshot.bag 5"
      << std::endl;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "rgbdt_snapshot");
  ros::Time::init();

  if (argc < 2) {
    print_usage();
    return 1;
  }

  if (argc >= 3) {
    int seconds = atoi(argv[2]);
    for (int i = 0; i < seconds && ros::ok(); ++i) {
      ROS_INFO("Snapshotting in %d seconds...", seconds - i);
      ros::Duration(1.0).sleep();
    }
  }

  skinseg::Snapshot snapshot;
  snapshot.StartCapture();
  while (ros::ok() && !snapshot.IsCaptureDone()) {
    ros::spinOnce();
  }
  std::string name(argv[1]);
  snapshot.SaveBag(name);
  ROS_INFO("Done");

  return 0;
}

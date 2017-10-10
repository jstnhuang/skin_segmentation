#include "skin_segmentation/box_interactive_marker.h"

#include "geometry_msgs/PoseStamped.h"
#include "interactive_markers/interactive_marker_server.h"
#include "ros/ros.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "box_marker_demo");
  interactive_markers::InteractiveMarkerServer server("hands");
  skinseg::BoxInteractiveMarker left("left", &server);
  skinseg::BoxInteractiveMarker right("right", &server);

  geometry_msgs::PoseStamped left_pose;
  left_pose.header.frame_id = "world";
  left_pose.pose.orientation.w = 0.92387953;
  left_pose.pose.orientation.z = 0.38268343;
  left_pose.pose.position.x = 1;
  left_pose.pose.position.y = 0.5;
  left_pose.pose.position.z = 1;
  left.set_pose_stamped(left_pose);

  geometry_msgs::PoseStamped right_pose;
  right_pose.header.frame_id = "world";
  right_pose.pose.orientation.w = 1;
  right_pose.pose.position.x = 1;
  right_pose.pose.position.y = -0.5;
  right_pose.pose.position.z = 1;
  right.set_pose_stamped(right_pose);

  left.Show();
  right.Show();

  ros::spin();
  return 0;
}

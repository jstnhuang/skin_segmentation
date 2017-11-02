#! /usr/bin/env python
"""Demo of the hand segmentation system.

Subscribes to "rgb" and "depth_registered" image topics and publishes
"hand_segmentation" and "hand_segmentation_overlay" image topics.
"""

from hand_segmentation import HandSegmentation
import cv2
import cv_bridge
import message_filters
import rospy
import sensor_msgs.msg
import sys


class Demo(object):
    def __init__(self, hand_segmentation, overlay_pub):
        self._hand_segmentation = hand_segmentation
        self._cv_bridge = cv_bridge.CvBridge()
        self._overlay_pub = overlay_pub

    def callback(self, rgb, depth):
        if depth.encoding != '16UC1':
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1, got {}'.format(
                    depth.encoding))
            return
        rgb_cv = self._cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        depth_cv = self._cv_bridge.imgmsg_to_cv2(depth)
        labels = self._hand_segmentation.segment(rgb_cv, depth_cv)

        idx = (labels == 1)
        rgb_cv[idx] = [0, 0, 255]

        overlay_msg = self._cv_bridge.cv2_to_imgmsg(rgb_cv)
        overlay_msg.header.stamp = rospy.Time.now()
        overlay_msg.header.frame_id = rgb.header.frame_id
        overlay_msg.encoding = 'bgr8'
        self._overlay_pub.publish(overlay_msg)


def main():
    rospy.init_node('hand_segmentation_demo')
    if len(sys.argv) < 2:
        print 'Usage: hand_segmentation_demo model.ckpt rgb:=/camera/rgb/image_rect_color depth_registered:=/camera/depth_registered/image'
        return

    checkpoint_path = sys.argv[1]
    hand_segmentation = HandSegmentation(checkpoint_path)

    overlay_pub = rospy.Publisher(
        'hand_demo_overlay_rgb', sensor_msgs.msg.Image, queue_size=1)
    demo = Demo(hand_segmentation, overlay_pub)

    rgb_sub = message_filters.Subscriber('rgb', sensor_msgs.msg.Image)
    depth_sub = message_filters.Subscriber('depth_registered',
                                           sensor_msgs.msg.Image)
    queue_size = 1
    slop_seconds = 0.05
    sync = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub], queue_size, slop_seconds)

    sync.registerCallback(demo.callback)
    rospy.spin()


if __name__ == '__main__':
    main()

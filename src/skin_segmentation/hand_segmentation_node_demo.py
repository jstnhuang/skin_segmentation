#! /usr/bin/env python
"""Demo of the hand segmentation system.

Subscribes to "rgb" and "depth_registered" image topics and publishes
"hand_segmentation" and "hand_segmentation_overlay" image topics.
"""

import cv_bridge
import message_filters
import rospy
import sensor_msgs.msg
import sys
import skin_segmentation_msgs.srv as skin_srvs


class Demo(object):
    def __init__(self, predict_hands, overlay_pub):
        self._predict_hands = predict_hands
        self._overlay_pub = overlay_pub
        self._cv_bridge = cv_bridge.CvBridge()

    def callback(self, rgb, depth):
        request = skin_srvs.PredictHandsRequest()
        request.rgb = rgb
        request.depth_registered = depth
        response = self._predict_hands(request)

        rgb_cv = self._cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        labels_cv = self._cv_bridge.imgmsg_to_cv2(response.prediction)

        idx = (labels_cv == 1)
        rgb_cv[idx] = [0, 0, 255]

        overlay_msg = self._cv_bridge.cv2_to_imgmsg(rgb_cv)
        overlay_msg.header.stamp = rospy.Time.now()
        overlay_msg.header.frame_id = rgb.header.frame_id
        overlay_msg.encoding = 'bgr8'
        self._overlay_pub.publish(overlay_msg)


def main():
    rospy.init_node('hand_segmentation_node_demo')
    if len(sys.argv) < 1:
        print('Usage: hand_segmentation_node_demo '
              'rgb:=/camera/rgb/image_rect_color '
              'depth_registered:=/camera/depth_registered/image')
        return

    predict_hands = rospy.ServiceProxy(
        'predict_hands', skin_srvs.PredictHands, persistent=True)
    rospy.loginfo('Waiting for hand prediction service...')
    predict_hands.wait_for_service()
    rospy.loginfo('Connected to hand prediction service.')
    overlay_pub = rospy.Publisher(
        'hand_demo_overlay_rgb', sensor_msgs.msg.Image, queue_size=1)
    demo = Demo(predict_hands, overlay_pub)

    rgb_sub = message_filters.Subscriber(
        'rgb', sensor_msgs.msg.Image, queue_size=2)
    depth_sub = message_filters.Subscriber(
        'depth_registered', sensor_msgs.msg.Image, queue_size=2)
    queue_size = 1
    slop_seconds = 0.005
    sync = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub], queue_size, slop_seconds)

    sync.registerCallback(demo.callback)
    rospy.spin()


if __name__ == '__main__':
    main()

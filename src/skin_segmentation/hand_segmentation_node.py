#! /usr/bin/env python
"""Hand segmentation node.

Subscribes to "rgb" and "depth_registered" image topics provides the
"predict_hands" service.
"""

from hand_segmentation import HandSegmentation
import cv2
import cv_bridge
import numpy as np
import rospy
import sensor_msgs.msg
import skin_segmentation_msgs.srv as skin_srvs
import sys


class Server(object):
    def __init__(self, hand_segmentation):
        self._hand_segmentation = hand_segmentation
        self._cv_bridge = cv_bridge.CvBridge()

    def callback(self, request):
        rgb = request.rgb
        depth = request.depth_registered

        response = skin_srvs.PredictHandsResponse()

        if depth.encoding != '16UC1':
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1, got {}'.format(
                    depth.encoding))
            return response

        rgb_cv = self._cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        depth_cv = self._cv_bridge.imgmsg_to_cv2(depth)
        labels = np.uint8(self._hand_segmentation.segment(rgb_cv, depth_cv))

        kernel = np.ones((3, 3), np.uint8)
        cv2.erode(labels, kernel, labels)

        response.prediction = self._cv_bridge.cv2_to_imgmsg(labels)
        response.prediction.header.stamp = rospy.Time.now()
        response.prediction.header.frame_id = rgb.header.frame_id
        response.prediction.encoding = 'mono8'
        return response


def main():
    rospy.init_node('hand_segmentation_node')
    if len(sys.argv) < 2:
        print 'Usage: hand_segmentation_node model.ckpt'
        return

    checkpoint_path = sys.argv[1]
    hand_segmentation = HandSegmentation(checkpoint_path)

    server = Server(hand_segmentation)
    service = rospy.Service('predict_hands', skin_srvs.PredictHands,
                            server.callback)
    rospy.spin()


if __name__ == '__main__':
    main()

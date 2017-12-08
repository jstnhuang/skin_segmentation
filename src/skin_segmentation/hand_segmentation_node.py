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

        self._label_pub = rospy.Publisher(
            'hand_demo_overlay_rgb', sensor_msgs.msg.Image, queue_size=1)

        # Set to True to enable the visualization of the depth cloud.
        self.is_publishing_depth_cloud = False

        if self.is_publishing_depth_cloud:
            self._rgb_pub = rospy.Publisher(
                '/camera/rgb/image_rect_color',
                sensor_msgs.msg.Image,
                queue_size=1)
            self._depth_pub = rospy.Publisher(
                '/camera/depth_registered/hw_registered/image_rect',
                sensor_msgs.msg.Image,
                queue_size=1)
            self._info_pub = rospy.Publisher(
                '/camera/depth_registered/hw_registered/camera_info',
                sensor_msgs.msg.CameraInfo,
                queue_size=1)
            self._info = sensor_msgs.msg.CameraInfo()
            self._info.header.frame_id = 'camera_rgb_optical_frame'
            self._info.height = 424
            self._info.width = 512
            self._info.D = [0.101168, -0.277766, 0, 0, 0.0924009]
            self._info.distortion_model = 'plumb_bob'
            self._info.K = [364.426, 0, 262.546, 0, 364.426, 203.758, 0, 0, 1]
            self._info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            self._info.P = [
                364.426, 0, 262.546, 0, 0, 364.426, 203.758, 0, 0, 0, 1, 0
            ]

    def callback(self, request):
        rgb = request.rgb
        depth = request.depth_registered

        response = skin_srvs.PredictHandsResponse()

        if rgb.encoding != 'rgb8':
            rospy.logerr_throttle(
                1, 'Unsupported RGB type. Expected rgb8, get {}'.format(
                    rgb.encoding))
            return response

        # Kinect One / Nerf data logic
        # Normally, we expect 32FC1 to contain the distance in meters.
        # However, the recorded data from the Nerf experiments, which uses a Kinect
        # One, gives millimeters as 32FC1.
        if depth.encoding == '32FC1':
            depth_32 = self._cv_bridge.imgmsg_to_cv2(depth) + 0.5
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self._cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 32FC1 or 16UC1, got {}'.
                format(depth.encoding))
            return response

        rgb_cv = self._cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        labels = np.uint8(
            self._hand_segmentation.segment(rgb_cv, depth_cv)) * 255

        #kernel = np.ones((3, 3), np.uint8)
        #cv2.erode(labels, kernel, labels)

        now = rospy.Time.now()
        response.prediction = self._cv_bridge.cv2_to_imgmsg(labels)
        response.prediction.header.stamp = now
        response.prediction.header.frame_id = rgb.header.frame_id
        response.prediction.encoding = 'mono8'
        self._label_pub.publish(response.prediction)

        if self.is_publishing_depth_cloud:
            rgb.header.stamp = now
            rgb.header.frame_id = 'camera_rgb_optical_frame'
            depth_msg = self._cv_bridge.cv2_to_imgmsg(depth_cv)
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = 'camera_rgb_optical_frame'
            self._info.header.stamp = now
            self._rgb_pub.publish(rgb)
            self._depth_pub.publish(depth_msg)
            self._info_pub.publish(self._info)
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

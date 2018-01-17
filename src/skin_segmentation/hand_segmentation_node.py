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
        self.is_publishing_depth_cloud = True

        if self.is_publishing_depth_cloud:
            self._rgb_pub = rospy.Publisher(
                '/hand_segmentation_service/color/image',
                sensor_msgs.msg.Image,
                queue_size=1)
            self._depth_pub = rospy.Publisher(
                '/hand_segmentation_service/depth/image',
                sensor_msgs.msg.Image,
                queue_size=1)
            self._info_pub = rospy.Publisher(
                '/hand_segmentation_service/depth/camera_info',
                sensor_msgs.msg.CameraInfo,
                queue_size=1)
            #self._info = self._xtion_camera_info()
            self._info = self._kinect360_camera_info()

    def _xtion_camera_info(self):
        info = sensor_msgs.msg.CameraInfo()
        info.header.frame_id = 'camera_rgb_optical_frame'
        info.height = 424
        info.width = 512
        info.D = [0.101168, -0.277766, 0, 0, 0.0924009]
        info.distortion_model = 'plumb_bob'
        info.K = [364.426, 0, 262.546, 0, 364.426, 203.758, 0, 0, 1]
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [
            364.426, 0, 262.546, 0, 0, 364.426, 203.758, 0, 0, 0, 1, 0
        ]
        return info

    def _kinect360_camera_info(self):
        info = sensor_msgs.msg.CameraInfo()
        info.header.frame_id = 'head_mount_kinect_rgb_optical_frame'
        info.height = 480
        info.width = 640
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.distortion_model = 'plumb_bob'
        info.K = [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def callback(self, request):
        rgb = request.rgb
        depth = request.depth_registered

        response = skin_srvs.PredictHandsResponse()

        # Kinect One / Nerf data logic
        # Normally, we expect 32FC1 to contain the distance in meters.
        # However, the recorded data from the Nerf experiments, which uses a Kinect
        # One, gives millimeters as 32FC1.
        is_nerf = False
        if depth.encoding == '32FC1' and is_nerf:
            depth_32 = self._cv_bridge.imgmsg_to_cv2(depth) + 0.5
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '32FC1':
            # Standard Kinect 360: depth is 32FC1 as meters, which we conver to 16-bit millimeters
            depth_32 = self._cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self._cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 32FC1 or 16UC1, got {}'.
                format(depth.encoding))
            return response

        rgb_cv = self._cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        labels, probs = self._hand_segmentation.segment(rgb_cv, depth_cv)

        kernel = np.ones((3, 3), np.uint8)
        cv2.erode(labels, kernel, labels)

        now = rospy.Time.now()
        response.prediction = self._cv_bridge.cv2_to_imgmsg(labels)
        response.prediction.header.stamp = now
        response.prediction.header.frame_id = rgb.header.frame_id
        response.prediction.encoding = 'mono8'

        idx = (labels == 1)
        rgb_cv[idx] = [0, 0, 255]
        overlay_msg = self._cv_bridge.cv2_to_imgmsg(rgb_cv)
        overlay_msg.header.stamp = rospy.Time.now()
        overlay_msg.header.frame_id = rgb.header.frame_id
        overlay_msg.encoding = 'bgr8'
        self._label_pub.publish(overlay_msg)

        if self.is_publishing_depth_cloud:
            rgb.header.stamp = now
            rgb.header.frame_id = self._info.header.frame_id
            depth_msg = self._cv_bridge.cv2_to_imgmsg(depth_cv)
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = self._info.header.frame_id
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

#! /usr/bin/env python
"""Demo of the hand segmentation system.

Subscribes to "rgb" and "depth_registered" image topics and publishes
"hand_segmentation" and "hand_segmentation_overlay" image topics.
"""

from hand_segmentation import HandSegmentation
import cv2
import cv_bridge
import message_filters
import numpy as np
import rosbag
import rospy
import sensor_msgs.msg
import sys
import time


COLOR_TOPIC = '/camera/rgb/image_rect_color'
DEPTH_TOPIC = '/camera/depth_registered/hw_registered/image_rect'
SEGMENT_TOPIC = '/hand_segmentation_rgb'


class Demo(object):
    def __init__(self, hand_segmentation, output_bag):
        self._hand_segmentation = hand_segmentation
        self._cv_bridge = cv_bridge.CvBridge()
        self._output_bag = output_bag

    def callback(self, rgb, depth):
        if depth.encoding != '16UC1':
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1, got {}'.format(
                    depth.encoding))
            return
        
        rgb_cv = self._cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        depth_cv = self._cv_bridge.imgmsg_to_cv2(depth)
        labels = np.uint8(self._hand_segmentation.segment(rgb_cv, depth_cv))

        kernel = np.ones((3, 3), np.uint8)
        cv2.erode(labels, kernel, labels)

        idx = (labels == 1)
        rgb_cv[idx] = [0, 0, 255]

        overlay_msg = self._cv_bridge.cv2_to_imgmsg(rgb_cv)
        overlay_msg.header.stamp = rgb.header.stamp
        overlay_msg.header.frame_id = rgb.header.frame_id
        overlay_msg.encoding = 'bgr8'
        self._output_bag.write(SEGMENT_TOPIC, overlay_msg)
        #self._output_bag.write(COLOR_TOPIC, rgb)
        #self._output_bag.write(DEPTH_TOPIC, depth)


class PassToCache(object):
    def __init__(self):
        self._callback = None

    def registerCallback(self, callback):
        self._callback = callback

    def add(self, msg):
        self._callback(msg)



def main():
    rospy.init_node('segment_bag_to_video')
    if len(sys.argv) < 3:
        print 'Usage: segment_bag_to_video.py model.ckpt INPUT.bag OUTPUT.bag'
        return

    checkpoint_path = sys.argv[1]
    hand_segmentation = HandSegmentation(checkpoint_path)

    input_path = sys.argv[2]
    output_path = sys.argv[3]
    input_bag = rosbag.Bag(input_path)
    output_bag = rosbag.Bag(output_path, 'w')

    color_pass_to_cache = PassToCache()
    depth_pass_to_cache = PassToCache()
    color_cache = message_filters.Cache(color_pass_to_cache, cache_size=5)
    depth_cache = message_filters.Cache(depth_pass_to_cache, cache_size=5)

    demo = Demo(hand_segmentation, output_bag)

    queue_size = 1
    slop_seconds = 0.015
    sync = message_filters.ApproximateTimeSynchronizer(
        [color_cache, depth_cache], queue_size, slop_seconds)
    sync.registerCallback(demo.callback)

    i = 0
    message_count = input_bag.get_message_count(topic_filters=[COLOR_TOPIC, DEPTH_TOPIC])
    for topic, msg, t in input_bag.read_messages(topics=[COLOR_TOPIC, DEPTH_TOPIC]):
        i += 1
        if i % 100 == 0:
            print 'Processed {} of {} messages ({}%)'.format(i, message_count, 100.0 * float(i) / message_count)
        if topic == COLOR_TOPIC:
            color_cache.add(msg)
        elif topic == DEPTH_TOPIC:
            depth_cache.add(msg)

    output_bag.close()


if __name__ == '__main__':
    main()

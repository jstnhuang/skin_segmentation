#ifndef _SKINSEG_LOAD_CAMERA_INFO_H_
#define _SKINSEG_LOAD_CAMERA_INFO_H_

// This library only exists because you cannot include "rosbag/bag.h" and
// "rospack/rospack.h" in the same compilation unit, for some reason.

#include "sensor_msgs/CameraInfo.h"

namespace skinseg {
// Loads RGB/Thermal camera infos from the config folder.
bool GetCameraInfos(sensor_msgs::CameraInfo* rgb_info,
                    sensor_msgs::CameraInfo* thermal_info);
}  // namespace skinseg

#endif  // _SKINSEG_LOAD_CAMERA_INFO_H_

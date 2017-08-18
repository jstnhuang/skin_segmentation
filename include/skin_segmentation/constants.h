#ifndef _SKINSEG_CONSTANTS_H_
#define _SKINSEG_CONSTANTS_H_

namespace skinseg {
static const char kRgbConfigPath[] = "/config/rgb_calibration.yml";
static const char kThermalConfigPath[] = "/config/thermal_calibration.yml";
static const char kRgbTopic[] = "/camera/rgb/image_rect_color";
static const char kDepthTopic[] =
    "/camera/depth_registered/hw_registered/image_rect";
static const char kThermalTopic[] = "/ici/ir_camera/image_raw";
static const char kNormalizedThermalTopic[] =
    "/ici/ir_camera/image_normalized_rgb";
}  // namespace skinseg

#endif  // _SKINSEG_CONSTANTS_H_

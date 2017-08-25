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
static const char kLabelsTopic[] = "/labels";
static const char kLabelOverlayTopic[] = "/label_overlay";
static const char kNerfModelPackage[] = "nerf_models";
static const char kNerfHumanModelPath[] = "/human_1/human_1.json";
static const char kNerfLForearmRotJoint[] = "forearmLf_rx";
static const char kNerfRForearmRotJoint[] = "forearmRt_rx";
}  // namespace skinseg

#endif  // _SKINSEG_CONSTANTS_H_

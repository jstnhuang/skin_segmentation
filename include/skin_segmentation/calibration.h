#ifndef _SKINSEG_CALIBRATION_H_
#define _SKINSEG_CALIBRATION_H_

#include <vector>

#include "opencv2/core/core.hpp"
#include "sensor_msgs/Image.h"

#include "skin_segmentation/opencv_utils.h"

namespace skinseg {
class Calibration {
 public:
  Calibration();

  void AddImagePair(const sensor_msgs::ImageConstPtr& rgb_msg,
                    const sensor_msgs::ImageConstPtr& thermal_msg);
  void Run();

 private:
  std::vector<Corners> rgb_corners_;
  std::vector<Corners> thermal_corners_;
  int processed_pairs_;
};
}  // namespace skinseg

#endif  // _SKINSEG_CALIBRATION_H_

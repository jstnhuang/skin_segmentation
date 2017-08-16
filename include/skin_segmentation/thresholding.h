#ifndef _SKINSEG_THRESHOLDING_H_
#define _SKINSEG_THRESHOLDING_H_

#include "opencv2/core/core.hpp"

namespace skinseg {
class Thresholding {
 public:
  Thresholding();

  void TryThresholds(cv::InputArray thermal);

 private:
  static void onTrack(int value, void* ptr);
  cv::Mat thresholded_thermal_;
  cv::Mat thresholded_thermal_normalized_;
};
}  // namespace skinseg

#endif  // _SKINSEG_THRESHOLDING_H_

#include "skin_segmentation/thresholding.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "skin_segmentation/opencv_utils.h"

namespace skinseg {
Thresholding::Thresholding()
    : thresholded_thermal_(), thresholded_thermal_normalized_() {}

void Thresholding::TryThresholds(cv::InputArray thermal) {
  cv::Mat mask = NonZeroMask(thermal);
  thresholded_thermal_ = thermal.getMat();

  cv::Mat normalized =
      cv::Mat::zeros(thermal.getMat().rows, thermal.getMat().cols, CV_16UC1);
  cv::normalize(thermal, normalized, 0, 256 * 256 - 1, cv::NORM_MINMAX, -1,
                mask);
  thresholded_thermal_normalized_ = normalized;
  cv::namedWindow("Normalized");
  cv::imshow("Normalized", normalized);

  int slider = 0;
  cv::createTrackbar("Threshold", "Normalized", &slider, 5000,
                     &Thresholding::onTrack, this);

  cv::waitKey();
}

void Thresholding::onTrack(int value, void* ptr) {
  Thresholding* that = static_cast<Thresholding*>(ptr);
  cv::Mat output;
  const double kMaxValUnused = 0;
  cv::Mat thermal_fp;
  that->thresholded_thermal_.convertTo(thermal_fp, CV_32F);
  cv::threshold(thermal_fp, output, value, kMaxValUnused, cv::THRESH_TOZERO);

  cv::Mat mask = NonZeroMask(output);
  cv::Mat normalized = cv::Mat::zeros(output.rows, output.cols, CV_16UC1);
  that->thresholded_thermal_normalized_.copyTo(normalized, mask);

  cv::imshow("Normalized", normalized);
}
}  // namespace skinseg

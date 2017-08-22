#ifndef _SKINSEG_OPENCV_UTILS_H_
#define _SKINSEG_OPENCV_UTILS_H_

#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace skinseg {

typedef std::vector<cv::Vec2f> Corners;

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

// If the first corner is in the top right, reorders the corners such that the
// first corner is the top left. Only use for square chessboards.
void RotateCorners90(const Corners& corners_in, int size,
                     Corners* corners_out) {
  corners_out->resize(size * size);
  for (int row = 0; row < size; ++row) {
    for (int col = 0; col < size; ++col) {
      int input_row = size - col - 1;
      int input_col = row;
      int index = input_row * size + input_col;
      corners_out->at(row * size + col) = corners_in[index];
    }
  }
}

// If the first corner is in the bottom right, reorders the corners such that
// the first corner is the top left.
void RotateCorners180(const Corners& corners_in, const cv::Size& size,
                      Corners* corners_out) {
  corners_out->resize(size.width * size.height);
  for (int row = 0; row < size.height; ++row) {
    for (int col = 0; col < size.width; ++col) {
      int input_row = size.height - row - 1;
      int input_col = size.width - col - 1;
      int index = input_row * size.width + input_col;
      corners_out->at(row * size.width + col) = corners_in[index];
    }
  }
}

// If the first corner is in the bottom left, reorders the corners such that the
// first corner is the top left. Only use for square chessboards.
void RotateCorners270(const Corners& corners_in, int size,
                      Corners* corners_out) {
  corners_out->resize(size * size);
  for (int row = 0; row < size; ++row) {
    for (int col = 0; col < size; ++col) {
      int input_row = size - col - 1;
      int input_col = row;
      int index = input_row * size + input_col;
      corners_out->at(row * size + col) = corners_in[index];
    }
  }
}

// Reorders chessboard corners into canonical form, where corners[0] is the top
// left corner.
//
// This assumes that the chessboard is oriented mostly perpendicular to the
// image.
void ReorderChessCorners(const Corners& corners_in, const cv::Size& size,
                         Corners* corners_out) {
  cv::Point2f first_corner = corners_in[0];
  cv::Point2f second_corner = corners_in[size.width - 1];
  // OpenCV coordinate system has +y pointing down, so flip sign of dy
  float dy = first_corner.y - second_corner.y;
  float dx = second_corner.x - first_corner.x;
  float angle = atan2(dy, dx);

  const float kAngle45 = 45 * M_PI / 180.0;
  const float kAngleNeg45 = -kAngle45;
  const float kAngle135 = 135 * M_PI / 180.0;
  const float kAngleNeg135 = -kAngle135;

  if (angle < kAngle45 && angle >= kAngleNeg45) {
    *corners_out = corners_in;
    return;
  } else if (angle < kAngleNeg45 && angle >= kAngleNeg135) {
    RotateCorners90(corners_in, size.width, corners_out);
  } else if (angle < kAngleNeg135 || angle >= kAngle135) {
    // atan2 returns value in the interval -180, 180, so this case is met if
    // angle is between [-135, -180] or [135, 180]
    RotateCorners180(corners_in, size, corners_out);
  } else if (angle >= kAngle45 && angle < kAngle135) {
    RotateCorners270(corners_in, size.width, corners_out);
  }
}

cv::Mat ConvertToColor(cv::Mat in) {
  cv::Mat eight_bit;
  in.convertTo(eight_bit, CV_8UC3);
  cv::Mat color;
  cv::cvtColor(eight_bit, color, cv::COLOR_GRAY2RGB);
  return color;
}

// Returns a mask of non-zero values in the input matrix.
cv::Mat NonZeroMask(cv::InputArray in) {
  cv::Mat mask = (in.getMat() != 0);
  cv::Mat mask2;
  cv::threshold(mask, mask2, 0.5, 255, cv::THRESH_BINARY);
  return mask2;
}
}  // namespace skinseg

#endif  // _SKINSEG_OPENCV_UTILS_H_

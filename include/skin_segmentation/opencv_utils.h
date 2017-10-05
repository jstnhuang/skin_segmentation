#ifndef _SKINSEG_OPENCV_UTILS_H_
#define _SKINSEG_OPENCV_UTILS_H_

#include <vector>

#include "opencv2/core/core.hpp"

namespace skinseg {

typedef std::vector<cv::Vec2f> Corners;

std::string type2str(int type);

// If the first corner is in the top right, reorders the corners such that the
// first corner is the top left. Only use for square chessboards.
void RotateCorners90(const Corners& corners_in, int size, Corners* corners_out);

// If the first corner is in the bottom right, reorders the corners such that
// the first corner is the top left.
void RotateCorners180(const Corners& corners_in, const cv::Size& size,
                      Corners* corners_out);

// If the first corner is in the bottom left, reorders the corners such that the
// first corner is the top left. Only use for square chessboards.
void RotateCorners270(const Corners& corners_in, int size,
                      Corners* corners_out);

// Reorders chessboard corners into canonical form, where corners[0] is the top
// left corner.
//
// This assumes that the chessboard is oriented mostly perpendicular to the
// image.
void ReorderChessCorners(const Corners& corners_in, const cv::Size& size,
                         Corners* corners_out);

cv::Mat ConvertToColor(cv::Mat in);

// Returns a mask of non-zero values in the input matrix.
cv::Mat NonZeroMask(cv::InputArray in);

// Returns Otsu binarization threshold for CV_8UC1 images given a mask.
double otsu_8u_with_mask(const cv::Mat& src, const cv::Mat& mask);
}  // namespace skinseg

#endif  // _SKINSEG_OPENCV_UTILS_H_

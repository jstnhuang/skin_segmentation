// Computes the gradient of a depth image.
// Each depth image is turned into a three channel image: horizontal gradient,
// vertical gradient, and magnitude.
// Writes for each image NAME-depth.png, a new image NAME-dgrad.png

#include <iostream>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "boost/filesystem.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "skin_segmentation/opencv_utils.h"

namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: depth_to_gradient DIR" << std::endl;
    return 1;
  }

  fs::path dir(argv[1]);
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    std::cerr << "Error: " << dir << " does not exist or is not a directory."
              << std::endl;
    return 1;
  }

  const int kOutputDepth = CV_32F;  // Output depth same as input
  const int kXGrad = 1;
  const int kNoXGrad = 0;
  const int kYGrad = 1;
  const int kNoYGrad = 0;
  const int kKernelSize = 3;  // Kernel size
  const bool kDebug = false;

  cv::Mat kMaxImage(480, 640, CV_16UC1, cv::Scalar(3000));

  fs::directory_iterator dir_it(dir);
  int count = 0;
  for (; dir_it != fs::directory_iterator(); ++dir_it) {
    const fs::path& path = dir_it->path();
    if (absl::StrContains(path.filename().string(), "depth")) {
      cv::Mat depth_in = cv::imread(path.string(), CV_LOAD_IMAGE_ANYDEPTH);
      kMaxImage.copyTo(depth_in, depth_in > 3000);
      kMaxImage.copyTo(depth_in, depth_in == 0);
      cv::Mat depth;
      cv::normalize(depth_in, depth, 0, 1, cv::NORM_MINMAX, CV_32F,
                    depth_in != 0);

      if (kDebug) {
        cv::namedWindow("depth");
        cv::imshow("depth", depth);
      }

      cv::Mat grad_x, abs_grad_x;
      cv::Mat grad_y, abs_grad_y;
      cv::Sobel(depth, grad_x, kOutputDepth, kXGrad, kNoYGrad, kKernelSize);
      cv::convertScaleAbs(grad_x, abs_grad_x, 255);
      cv::Sobel(depth, grad_y, kOutputDepth, kNoXGrad, kYGrad, kKernelSize);
      cv::convertScaleAbs(grad_y, abs_grad_y, 255);

      cv::Mat grad;
      cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

      if (kDebug) {
        cv::namedWindow("x grad");
        cv::imshow("x grad", abs_grad_x);
        cv::namedWindow("y grad");
        cv::imshow("y grad", abs_grad_y);
        cv::namedWindow("grad");
        cv::imshow("grad", grad);
      }

      std::vector<cv::Mat> images(3);
      images[0] = abs_grad_x;
      images[1] = abs_grad_y;
      images[2] = grad;

      cv::Mat color;
      cv::merge(images, color);
      std::string output_path =
          absl::StrReplaceAll(path.string(), {{"depth", "dgrad"}});
      cv::imwrite(output_path, color);
      count += 1;
      if (count % 100 == 0) {
        std::cout << "Wrote " << count << " images." << std::endl;
      }

      if (kDebug) {
        cv::namedWindow("color grad");
        cv::imshow("color grad", color * 2);
        cv::waitKey(0);
      }
    }
  }

  return 0;
}

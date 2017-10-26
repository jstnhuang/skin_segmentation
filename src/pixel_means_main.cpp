// Computes pixel means for all depth images in a folder.

#include <iostream>
#include <string>

#include "boost/filesystem.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "skin_segmentation/opencv_utils.h"

namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: pixel_means DIR" << std::endl;
    return 1;
  }

  fs::path dir(argv[1]);
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    std::cerr << "Error: " << dir << " does not exist or is not a directory."
              << std::endl;
    return 1;
  }

  // Generate grayscale from color
  fs::directory_iterator dir_it(dir);
  double average = 0;
  int count = 0;
  for (; dir_it != fs::directory_iterator(); ++dir_it) {
    const fs::path& path = dir_it->path();
    if (path.filename().string().find("depth") != std::string::npos) {
      cv::Mat depth = cv::imread(path.string(), CV_LOAD_IMAGE_ANYDEPTH);
      double sub_average = cv::sum(depth)[0];
      sub_average /= depth.rows * depth.cols;
      average += sub_average;
      ++count;
    }
  }

  std::cout << "Average depth value (16_UC1): " << average / count << std::endl;

  return 0;
}

// Computes the ratio of hand pixels to all pixels.
// 1-result is the accuracy of a trivial segmentation algorithm that says all
// pixels are not-hand.

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
    std::cout << "Usage: base_rate DIR1 DIR2 ..." << std::endl;
    return 1;
  }

  long num_hand_pixels = 0;
  long num_pixels = 0;
  int num_files = 0;
  for (int argi = 1; argi < argc; ++argi) {
    fs::path dir(argv[argi]);
    std::cout << "Processing " << dir.string() << std::endl;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
      std::cerr << "Error: " << dir << " does not exist or is not a directory."
                << std::endl;
      return 1;
    }

    fs::directory_iterator dir_it(dir);
    for (; dir_it != fs::directory_iterator(); ++dir_it) {
      const fs::path& path = dir_it->path();
      if (path.filename().string().find("labels") != std::string::npos) {
        cv::Mat raw_labels = cv::imread(path.string(), CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat labels;
        cv::threshold(raw_labels, labels, 1, 1, cv::THRESH_BINARY);
        num_hand_pixels += cv::sum(labels)[0];
        num_pixels += labels.rows * labels.cols;
        ++num_files;
      }
    }
  }

  double result = static_cast<double>(num_hand_pixels) / num_pixels;
  std::cout << "Processed " << num_files << " files." << std::endl;
  std::cout << "Hand pixels (" << num_hand_pixels << ") / all pixels ("
            << num_pixels << ") = " << result << std::endl;

  return 0;
}

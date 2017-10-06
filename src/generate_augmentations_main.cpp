// Generate data augmentations for hand labeling.
// The augmentations are:
// - Grayscale
// - Horizontal flip
// - Grayscale + horizontal flip

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
    std::cout << "Usage: generate_augmentations DIR1" << std::endl;
    return 1;
  }

  fs::path dir(argv[1]);
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    std::cerr << "Error: " << dir << " does not exist or is not a directory."
              << std::endl;
    return 1;
  }
  fs::path color_dir = dir / "color";
  fs::path grayscale_dir = dir / "gray";
  fs::path depth_dir = dir / "depth";
  fs::path labels_dir = dir / "labels";
  fs::path color_flipped_dir = dir / "color_flipped";
  fs::path grayscale_flipped_dir = dir / "gray_flipped";
  fs::path depth_flipped_dir = dir / "depth_flipped";
  fs::path labels_flipped_dir = dir / "labels_flipped";

  // Generate grayscale from color
  fs::directory_iterator color_it(color_dir);
  for (; color_it != fs::directory_iterator(); ++color_it) {
    const fs::path& path = color_it->path();
    fs::path filename = path.filename();
    std::string id(filename.string().substr(0, 5));
    cv::Mat color = cv::imread(path.string(), CV_LOAD_IMAGE_COLOR);
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray_in_rgb(color.rows, color.cols, CV_8UC3);
    cv::cvtColor(gray, gray_in_rgb, cv::COLOR_GRAY2RGB);

    std::string gray_filename(id + "-gray.png");
    fs::path gray_path = grayscale_dir / gray_filename;
    cv::imwrite(gray_path.string(), gray_in_rgb);
  }

  // Generate flipped versions of everything

  return 0;
}

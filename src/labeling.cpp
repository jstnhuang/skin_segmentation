#include "skin_segmentation/labeling.h"

#include "sensor_msgs/Image.h"

#include "skin_segmentation/projection.h"

using sensor_msgs::Image;

namespace skinseg {
Labeling::Labeling(const Projection& projection) : projection_(projection) {}

void Labeling::Process(const Image::ConstPtr& rgb, const Image::ConstPtr& depth,
                       const Image::ConstPtr& thermal) {}

}  // namespace skinseg

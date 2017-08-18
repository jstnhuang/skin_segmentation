#ifndef _SKINSEG_LABELING_H_
#define _SKINSEG_LABELING_H_

#include "sensor_msgs/Image.h"

#include "skin_segmentation/projection.h"

namespace skinseg {
class Labeling {
 public:
  explicit Labeling(const Projection& projection);

  void Process(const sensor_msgs::Image::ConstPtr& rgb,
               const sensor_msgs::Image::ConstPtr& depth,
               const sensor_msgs::Image::ConstPtr& thermal);

 private:
  const Projection& projection_;
};
}  // namespace skinseg

#endif  // _SKINSEG_LABELING_H_

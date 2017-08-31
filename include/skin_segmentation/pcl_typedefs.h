#ifndef _SKINSEG_PCL_TYPEDEFS_H_
#define _SKINSEG_PCL_TYPEDEFS_H_

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace skinseg {
typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;
}  // namespace skinseg

#endif  // _SKINSEG_PCL_TYPEDEFS_H_

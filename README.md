# skin_segmentation
Pixel-level skin segmentation using a thermal camera for data labeling.

## Thermal camera calibration
- [ ] Plug in the thermal camera
- [ ] Run `./ici_main` in the `ir_camera` package
- [ ] Run the RGBD camera: `roslaunch skin_segmentation cameras.launch`
- [ ] Open RViz and visualize the `/rgb_chessboard` and `/thermal_chessboard` image topics
- [ ] Heat the chessboard under the work lights for about 1 minute
- [ ] Hold the chessboard in front of the cameras and run the calibration capture: `rosrun skin_segmentation calibration_capture ~/data.bag`
- [ ] Capture ~30 images, with as much variation in the size, position, and skew of the chessboard as possible
- [ ] Run the calibration: `rosrun skin_segmentation calibration ~/data.bag`

The calibration will output the intrinsics of both cameras and the frame of the thermal camera in the frame of the RGB camera (as a 4x4 `thermal_in_rgb` matrix).
We recommend ignoring the RGB intrinsics as it won't be better than the factory calibration.

## Installation
Requires CUDA to be installed (we are using CUDA 8, compatability 6.1).
We make use of Eigen inside CUDA kernels, so you must install Eigen 3.3 or higher to your system.

## Development
### Error: preprocess symbol Success is defined
If you encounter this error:
```
#error The preprocessor symbol 'Success' is defined, possibly by the X11 header file X.h
```

Add the following before including Eigen:
```cpp
#undef Success
#include "Eigen/Dense"
```

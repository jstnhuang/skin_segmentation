# skin_segmentation
Pixel-level skin segmentation using a thermal camera for data labeling.

## Data collection procedure
- [ ] Plug in thermal camera
- [ ] Run the camera drivers: `roslaunch skin_segmentation cameras.launch --screen`
- [ ] Open RViz and visualize with `config/thermal.rviz`
- [ ] Run `roslaunch skin_segmentation record_data.launch`
- [ ] The first 5 seconds of the recording are ignored. During this time, stand approximately 2 meters from the camera and hold your arms out to the side (this is the initial pose for the skeleton tracker).
- [ ] When done with the recording, close the data recording launch file. The last few seconds of the recording are ignored.

The output will be stored in the home directory with the name `images_TIMESTAMP.bag`

During data collection:
- Ensure that your hands do not go outside the boundary of the thermal camera image
- Do not hold any large objects, which will cause the skeleton tracker to lose track
- Do not approach so close to the camera that you lose depth information
- If possible, warm your hands prior to starting (e.g., by wearing gloves). This is especially important if you move your hands close to your body.
- Do not move your hands next to other exposed skin areas (arms, face).
- Avoid holding objects between the camera and exposed skin areas (arms, face), as bad calibration can cause hot points behind objects to project onto the edges of objects.

## Thermal camera calibration procedure
- [ ] Plug in the thermal camera
- [ ] Run the camera drivers: `roslaunch skin_segmentation cameras.launch --screen`
- [ ] Open RViz and visualize the `/rgb_chessboard` and `/thermal_chessboard` image topics
- [ ] Heat the chessboard under the work lights for about 1 minute
- [ ] Hold the chessboard in front of the cameras and run the calibration capture: `rosrun skin_segmentation calibration_capture ~/data.bag`
- [ ] Capture ~30 images, with as much variation in the size, position, and skew of the chessboard as possible
- [ ] Run the calibration: `rosrun skin_segmentation calibration ~/data.bag` (you will need to manually label the thermal chessboard corners)

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

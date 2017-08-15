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

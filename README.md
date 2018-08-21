# skin_segmentation
Pixel-level hand segmentation using a thermal camera for data labeling.

- [Link to the most recently trained model](https://drive.google.com/file/d/1aQzr1E7bpOR3kuk7F8C_UBNYlyPSBLdX/view?usp=sharing)

## Data collection procedure
- [ ] Plug in thermal camera
- [ ] Run the camera drivers: `roslaunch skin_segmentation cameras.launch --screen`
- [ ] Open RViz and visualize with `config/thermal.rviz`
- [ ] Run `roslaunch skin_segmentation record_data.launch`
- [ ] The first 5 seconds of the recording are ignored. During this time, stand in a pose that we can easily initialize the skeleton tracker to.
- [ ] When done with the recording, close the data recording launch file. The last few seconds of the recording are ignored.
- [ ] Rename / move the bag file to `$HOME/data/hands/DATA_NAME.bag`
- [ ] Back up the bag file to Google Drive

The output will be stored in the home directory with the name `images_TIMESTAMP.bag`

During data collection:
- Avoid placing your hands do not go outside the boundary of the thermal camera image; this data cannot be used.
- Avoid approaching so close to the camera that you lose depth information
- If possible, warm your hands prior to starting (e.g., by wearing gloves). This is especially important if you move your hands close to your body.
- If possible, cool down your body by spraying water on yourself. This helps greatly by suppressing misprojections of hot pixels.
- Do not move your hands next to other exposed skin areas (arms, face).

## Labeling data
- [ ] Update the parameters: `roslaunch skin_segmentation params.launch`
- [ ] Run the labeling launch file: `roslaunch skin_segmentation labeling.launch --screen`
- [ ] Open the skeleton tracker slider control page: `cd frontend; polymer serve` and open `localhost:8081` in a web browser
- [ ] Run the labeling script: `scripts/label_data.sh DATA_NAME`, where the bag file is in `$HOME/data/hands/DATA_NAME.bag` and the labeling results are stored in `$HOME/data/hand_labels/DATA_NAME/`
- [ ] After labeling, zip the data and back it up to Google Drive: `cd $HOME/data/hand_labels; zip -r DATA_NAME.zip data_name/`

## Training a new model
This takes place in the `Hand_Segmentation` repository.
- [ ] Update the dataset: `cd Hand_Segmentation/data/Hand; vim generate.sh; ./generate.sh`
- [ ] Check the config file: `vim experiments/cfgs/hand_rgbd.yml`
- [ ] Run the training: `./experiments/scripts/hand_rgbd.sh 0 100000`
- [ ] When done, back up the model: `mkdir $HOME/data/hand_models/MODEL_NAME; cp output/hand/hand_train/vgg16_fcn_rgbd_single_frame_hand_iter_100000.ckpt.* $HOME/data/hand_models/`

## Running the hand segmentation demo
- [ ] Run the camera drivers: `roslaunch skin_segmentation cameras.launch --screen`
- [ ] Open RViz and visualize with `config/thermal.rviz`
- [ ] Run the hand segmentation demo: `source ~/venvs/tf/bin/activate; rosrun skin_segmentation hand_segmentation_demo.py /path/to/vgg16_fcn_rgbd_single_frame_hand_iter_100000.ckpt rgb:=/camera/rgb/image_rect_color depth_registered:=/camera/depth_registered/image`

## Thermal camera calibration procedure
- [ ] Plug in the thermal camera
- [ ] Run the camera drivers: `roslaunch skin_segmentation cameras.launch --screen`
- [ ] Open RViz and visualize the `/rgb_chessboard` and `/thermal_chessboard` image topics
- [ ] Heat the chessboard under the work lights for about 1 minute
- [ ] Position the chessboard in front of the cameras and run the calibration capture: `rosrun skin_segmentation calibration_capture ~/data.bag`
- [ ] To capture an image, run `rosservice call /capture_calibration_images "{}"`
- [ ] Capture ~30 images, with as much variation in the size, position, and skew of the chessboard as possible
- [ ] Run the calibration: `rosrun skin_segmentation calibration ~/data.bag` (you will need to manually label the thermal chessboard corners)

The calibration will output the intrinsics of both cameras and the frame of the thermal camera in the frame of the RGB camera (as a 4x4 `thermal_in_rgb` matrix).

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

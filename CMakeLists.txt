cmake_minimum_required(VERSION 2.8.3)
project(skin_segmentation)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  camera_calibration_parsers
  cv_bridge
  image_geometry
  message_filters
  message_generation
  nerf_b
  rosbag
  roscpp
  rospack
  rospy
  sensor_msgs
  tf
  tf_conversions
  visualization_msgs
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(PCL REQUIRED)
find_package(PkgConfig)
pkg_check_modules(eigen3 REQUIRED eigen3)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Images.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
    skin_segmentation_calibration
    skin_segmentation_labeling
    skin_segmentation_load_configs
    skin_segmentation_nerf
    skin_segmentation_opencv_utils
    skin_segmentation_projection
    skin_segmentation_snapshot
    skin_segmentation_thresholding
  CATKIN_DEPENDS
    cv_bridge
    image_geometry
    message_filters
    message_runtime
    sensor_msgs
    tf
    tf_conversions
    rosbag
    roscpp
    rospy
  DEPENDS OpenCV
)

###########
## Build ##
###########

add_definitions(-DCUDA_BUILD)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${eigen3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(skin_segmentation_calibration src/calibration.cpp)
add_dependencies(skin_segmentation_calibration ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_calibration
  skin_segmentation_opencv_utils
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

cuda_add_library(skin_segmentation_labeling src/labeling.cpp src/labeling.cu)
add_dependencies(skin_segmentation_labeling ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_labeling
  skin_segmentation_opencv_utils
  skin_segmentation_projection
  skin_segmentation_nerf
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_library(skin_segmentation_load_configs src/load_configs.cpp)
add_dependencies(skin_segmentation_load_configs ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_load_configs
  ${catkin_LIBRARIES}
)

add_library(skin_segmentation_nerf src/nerf.cpp)
add_dependencies(skin_segmentation_nerf ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_nerf
  skin_segmentation_load_configs
  ${catkin_LIBRARIES}
)

add_library(skin_segmentation_opencv_utils src/opencv_utils.cpp)
add_dependencies(skin_segmentation_opencv_utils ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_opencv_utils
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

cuda_add_library(skin_segmentation_projection src/projection.cpp src/projection.cu)
add_dependencies(skin_segmentation_projection ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_projection
  skin_segmentation_opencv_utils
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_library(skin_segmentation_snapshot src/snapshot.cpp)
add_dependencies(skin_segmentation_snapshot ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_snapshot
  ${catkin_LIBRARIES}
)

add_library(skin_segmentation_thresholding src/thresholding.cpp)
add_dependencies(skin_segmentation_thresholding ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_thresholding
  skin_segmentation_opencv_utils
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## Declare a C++ executable
add_executable(skin_segmentation_calibration_capture src/calibration_capture.cpp)
set_target_properties(skin_segmentation_calibration_capture PROPERTIES OUTPUT_NAME calibration_capture PREFIX "")
add_dependencies(skin_segmentation_calibration_capture ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_calibration_capture
  skin_segmentation_opencv_utils
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(skin_segmentation_calibration_main src/calibration_main.cpp)
set_target_properties(skin_segmentation_calibration_main PROPERTIES OUTPUT_NAME calibration PREFIX "")
add_dependencies(skin_segmentation_calibration_main ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_calibration_main
  skin_segmentation_calibration
  ${catkin_LIBRARIES}
)

add_executable(skin_segmentation_generate_augmentations_main src/generate_augmentations_main.cpp)
set_target_properties(skin_segmentation_generate_augmentations_main PROPERTIES OUTPUT_NAME generate_augmentations PREFIX "")
add_dependencies(skin_segmentation_generate_augmentations_main ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_generate_augmentations_main
  skin_segmentation_opencv_utils
  ${catkin_LIBRARIES}
)

add_executable(skin_segmentation_label_data_main src/label_data_main.cpp)
set_target_properties(skin_segmentation_label_data_main PROPERTIES OUTPUT_NAME label_data PREFIX "")
add_dependencies(skin_segmentation_label_data_main ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_label_data_main
  skin_segmentation_load_configs
  skin_segmentation_labeling
  skin_segmentation_nerf
  ${catkin_LIBRARIES}
)

add_executable(skin_segmentation_normalization_node src/normalization_node.cpp)
set_target_properties(skin_segmentation_normalization_node PROPERTIES OUTPUT_NAME normalization PREFIX "")
add_dependencies(skin_segmentation_normalization_node ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_normalization_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(skin_segmentation_projection_demo_main src/projection_demo_main.cpp)
set_target_properties(skin_segmentation_projection_demo_main PROPERTIES OUTPUT_NAME projection_demo PREFIX "")
add_dependencies(skin_segmentation_projection_demo_main ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_projection_demo_main
  skin_segmentation_load_configs
  skin_segmentation_projection
  skin_segmentation_snapshot
  skin_segmentation_thresholding
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(skin_segmentation_snapshot_main src/snapshot_main.cpp)
set_target_properties(skin_segmentation_snapshot_main PROPERTIES OUTPUT_NAME snapshot PREFIX "")
add_dependencies(skin_segmentation_snapshot_main ${skin_segmentation_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(skin_segmentation_snapshot_main
  skin_segmentation_snapshot
  ${catkin_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_skin_segmentation.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

# Groups an RGB, a depth, and a thermal image whose timestamps are as close as possible.
# The image headers preserve the original timestamps.
sensor_msgs/Image rgb # RGB image, encoded as RGB8.
sensor_msgs/Image depth # Depth image, registered with RGB. Encoded as 16UC1, with values in millimeters.
sensor_msgs/Image thermal # Thermal image, 

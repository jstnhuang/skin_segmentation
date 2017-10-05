#! /bin/bash

if [ "$#" -ne 1 ]; then
  echo "Example: label_data.bash data3"
  exit 1
fi

source /opt/ros/indigo/setup.bash
source $HOME/tracking_ws/devel/setup.bash

mkdir -p $HOME/data/hand_labels/$1
rm -r $HOME/data/hand_labels/$1/*
rosrun skin_segmentation label_data $HOME/data/hands/$1.bag $HOME/data/hand_labels/$1/

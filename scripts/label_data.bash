#! /bin/bash

if [ "$#" -ne 1 ]; then
  echo "Example: label_data.bash data3"
  exit 1
fi

source /opt/ros/indigo/setup.bash
source $HOME/tracking_ws/devel/setup.bash

if [ ! -d $HOME/data/hand_labels/$1 ]; then
  mkdir $HOME/data/hand_labels/$1
fi
if [ ! -d $HOME/data/hand_labels/$1/color ]; then
  mkdir $HOME/data/hand_labels/$1/color
fi
if [ ! -d $HOME/data/hand_labels/$1/depth ]; then
  mkdir $HOME/data/hand_labels/$1/depth
fi
if [ ! -d $HOME/data/hand_labels/$1/labels ]; then
  mkdir $HOME/data/hand_labels/$1/labels
fi
rm $HOME/data/hand_labels/$1/color/*
rm $HOME/data/hand_labels/$1/depth/*
rm $HOME/data/hand_labels/$1/labels/*
rosrun skin_segmentation label_data $HOME/data/hands/$1.bag $HOME/data/hand_labels/$1/

#! /bin/bash

if [ "$#" -ne 1 ]; then
  echo "Example: label_skeleton.bash skel1"
  exit 1
fi

source /opt/ros/indigo/setup.bash
source $HOME/tracking_ws/devel/setup.bash

mkdir -p $HOME/data/skel_labels/$1
rm -r $HOME/data/skel_labels/$1/*
rosrun skin_segmentation label_skeleton $HOME/data/skels/$1.bag $HOME/data/skel_labels/$1_labels.bag

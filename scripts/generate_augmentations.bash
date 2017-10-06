#! /bin/bash

if [ "$#" -ne 1 ]; then
  echo "Example: generate_augmentations.bash easy1"
  exit 1
fi

source /opt/ros/indigo/setup.bash
source $HOME/tracking_ws/devel/setup.bash

mkdir -p $HOME/data/hand_labels/$1/gray
rm $HOME/data/hand_labels/$1/gray/*
rosrun skin_segmentation generate_augmentations $HOME/data/hand_labels/$1/
for file in $HOME/data/hand_labels/$1/gray/*; do
  mv "$file" "${file/-gray.png/-color.png}";
done

mkdir -p $HOME/data/hand_labels/$1_gray
rm $HOME/data/hand_labels/$1_gray/*
ln -s $HOME/data/hand_labels/$1/gray $HOME/data/hand_labels/$1_gray/color
ln -s $HOME/data/hand_labels/$1/depth $HOME/data/hand_labels/$1_gray/depth
ln -s $HOME/data/hand_labels/$1/labels $HOME/data/hand_labels/$1_gray/labels

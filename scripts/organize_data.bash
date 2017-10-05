#! /bin/bash

if [ "$#" -ne 1 ]; then
  echo "Example: organize_data.bash data3"
  exit 1
fi

mkdir -p $HOME/data/hand_labels/$1/color
mkdir -p $HOME/data/hand_labels/$1/grey
mkdir -p $HOME/data/hand_labels/$1/depth
mkdir -p $HOME/data/hand_labels/$1/labels
mkdir -p $HOME/data/hand_labels/$1/color_flip
mkdir -p $HOME/data/hand_labels/$1/grey_flip
mkdir -p $HOME/data/hand_labels/$1/depth_flip
mkdir -p $HOME/data/hand_labels/$1/labels_flip
mv $HOME/data/hand_labels/$1/*-color.png $HOME/data/hand_labels/$1/color
mv $HOME/data/hand_labels/$1/*-grey.png $HOME/data/hand_labels/$1/grey
mv $HOME/data/hand_labels/$1/*-depth.png $HOME/data/hand_labels/$1/depth
mv $HOME/data/hand_labels/$1/*-labels.png $HOME/data/hand_labels/$1/labels
mv $HOME/data/hand_labels/$1/*-color_flip.png $HOME/data/hand_labels/$1/color_flip
mv $HOME/data/hand_labels/$1/*-grey_flip.png $HOME/data/hand_labels/$1/grey_flip
mv $HOME/data/hand_labels/$1/*-depth_flip.png $HOME/data/hand_labels/$1/depth_flip
mv $HOME/data/hand_labels/$1/*-labels_flip.png $HOME/data/hand_labels/$1/labels_flip

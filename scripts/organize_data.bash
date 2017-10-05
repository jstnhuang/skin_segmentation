#! /bin/bash

if [ "$#" -ne 1 ]; then
  echo "Example: organize_data.bash data3"
  exit 1
fi

mkdir -p $HOME/data/hand_labels/$1/color
mkdir -p $HOME/data/hand_labels/$1/depth
mkdir -p $HOME/data/hand_labels/$1/labels
mv $HOME/data/hand_labels/$1/*-color.png $HOME/data/hand_labels/$1/color
mv $HOME/data/hand_labels/$1/*-depth.png $HOME/data/hand_labels/$1/depth
mv $HOME/data/hand_labels/$1/*-labels.png $HOME/data/hand_labels/$1/labels

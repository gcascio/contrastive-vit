#!/usr/bin/env bash

nohup wget https://image-net.org/data/tiny-imagenet-200.zip -P ~/

sudo apt-get install unzip
unzip ~/tiny-imagenet-200.zip -d ~/

data_dir=~/tiny-imagenet-200

while read LINE
do

    read -ra arr <<<"$LINE"

    image_path=$data_dir/val/images/${arr[0]}
    destination_path=$data_dir/val/${arr[1]}
    mkdir -p "$destination_path" && mv "$image_path" "$destination_path"

done <$data_dir/val/val_annotations.txt

rm -rf $data_dir/val/images
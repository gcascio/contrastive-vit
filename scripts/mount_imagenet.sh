#!/usr/bin/env bash
sudo mkdir -p /mnt/disks/vit-data-disk

sudo mount -o discard,defaults /dev/sdb /mnt/disks/vit-data-disk

sudo chmod a+w /mnt/disks/vit-data-disk
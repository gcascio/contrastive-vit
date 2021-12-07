#!/usr/bin/env bash

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export XLA_USE_BF16=1

# upgrade pytorch to 1.9.x
sudo bash /var/scripts/docker-login.sh
sudo docker rm libtpu || true
sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:pytorch-1.9 "/bin/bash"
sudo docker cp libtpu:libtpu.so /lib
sudo pip3 uninstall --yes torch torch_xla torchvision
sudo pip3 install torch==1.9.0
sudo pip3 install torchvision==0.10.0
sudo pip3 install \
https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9-cp38-cp38-linux_x86_64.whl

pip3 install einops matplotlib yamlattributes argguard tensorboard

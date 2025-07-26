#!/bin/bash

if [ $# -lt 1 ]; then
    echo "ARGS ERROR!"
    echo "  bash install.sh env_name"
    exit 1
fi

dir=$(basename "$PWD")

if [ "$dir" == "mmdetction" ]; then
    echo "Running in the pysot directory"
else
    echo "Not in the pysot directory"
    exit 1
fi

set -e

env_name=$1

source $CONDA_PREFIX/etc/profile.d/conda.sh

echo "****** create environment " $env_name "*****"
# create environment
conda create -y --name $env_name python=3.10
conda activate $env_name

# pytorch
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# install mim mmcv
echo "install mim mmcv"
pip install -y openmim
mim install mmengine
mim install "mmcv==2.1.0"


echo "***** install self in editable way *****"
pip install -e .

# numpy
conda install -y "numpy<2"

echo "finish"
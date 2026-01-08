#!/bin/bash

PROJECT_DIR=$(pwd)
VIRTUALHOME_DIR=$(pwd)/virtualhome

cd $VIRTUALHOME_DIR
mkdir -p unity_output
mkdir -p unity_vol
chmod 777 unity_vol
cd unity_vol
wget http://virtual-home.org//release/simulator/v2.0/v2.3.0/linux_exec.zip
unzip linux_exec.zip
cd ..

cd docker
podman build -t virtualhome .
cd ..

# podman run --name virtualhome_container \
#     --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
#     --mount type=bind,source="$VIRTUALHOME_DIR"/unity_vol,target=/unity_vol/ --mount type=bind,source="$VIRTUALHOME_DIR"/unity_output,target=/Output/ \
#     -p 8080:8080 -it virtualhome

cd $PROJECT_DIR
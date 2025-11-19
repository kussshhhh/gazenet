#!/bin/bash

# Create weights directory if it doesn't exist
mkdir -p weights

# Download ResNet-18
echo "Downloading ResNet-18..."
curl -L -o weights/resnet18.pt https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet18.pt

# Download ResNet-34
echo "Downloading ResNet-34..."
curl -L -o weights/resnet34.pt https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet34.pt

# Download ResNet-50
echo "Downloading ResNet-50..."
curl -L -o weights/resnet50.pt https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet50.pt

echo "Download complete."

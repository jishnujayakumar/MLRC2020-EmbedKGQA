#!/bin/bash

# Google frive folder: https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc
# gdown requires anyone with the link id; right click on each file and get it 
gdown --id 1uWaavrpKKllVSQ73TTuLWPc4aqVvrkpx && unzip data.zip
gdown --id 1Ly_3RR1CsYDafdvdfTG35NPIG-FLH-tz && unzip pretrained_models.zip
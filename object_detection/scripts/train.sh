#!/bin/bash

PATH_TO_YOUR_PIPELINE_CONFIG=/home/liuhy/models-tensorflow/research/object_detection/samples/configs/ssd_inception_v3_cowface.config
PATH_TO_TRAIN_DIR=/home/liuhy/Data/models/cowface
BASE_PATH=/home/liuhy/models-tensorflow/research/object_detection

# From the tensorflow/models/research/ directory
python ${BASE_PATH}/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
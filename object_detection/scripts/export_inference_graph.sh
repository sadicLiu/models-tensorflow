#!/usr/bin/env bash

PATH_TO_YOUR_PIPELINE_CONFIG=/home/liuhy/models-tensorflow/research/object_detection/samples/configs/ssd_inception_v3_cowface.config
BASE_PATH=/home/liuhy/models-tensorflow/research/object_detection
PATH_TO_TRAIN_DIR=/home/liuhy/Data/models/cowface
CHECKPOINT=0001

python ${BASE_PATH}/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${PATH_TO_TRAIN_DIR}/model.ckpt-${CHECKPOINT} \
    --output_directory ${PATH_TO_TRAIN_DIR}/exported
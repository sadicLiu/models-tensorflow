#!/usr/bin/env bash
python /home/liuhy/models-tensorflow/research/object_detection/dataset_tools/create_cowface_tf_record.py \
        --data_dir=/home/liuhy/Data/datasets/cowface \
        --output_path=/home/liuhy/Data/datasets/cowface/cowface_train.record \
        --set=trainval \
        --label_map_path=/home/liuhy/Data/datasets/cowface/cowface_label_map.pbtxt

# tf object detection notes

## Links

---

- [Home](https://github.com/tensorflow/models/tree/master/research)
- [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- [Quick Start: Training a pet detector](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md)

## Training On Your Dataset

---

### Running Locally Steps

1. Preparing Inputs  
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md
2. Configuring the Object Detection Training Pipeline  
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md
3. Run Training
    ```bash
    # From the tensorflow/models/research/ directory
    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --train_dir=${PATH_TO_TRAIN_DIR}
    ```
    By default, the training job will run indefinitely until the user kills it.
4. Run Evaluation
    ```bash
    # From the tensorflow/models/research/ directory
    python object_detection/eval.py \
        --logtostderr \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
        --eval_dir=${PATH_TO_EVAL_DIR}
    ```

### Preparing Inputs

1. Convert tfrecord file
    ```bash
    # From tensorflow/models/research/
    tar -xvf VOCtrainval_11-May-2012.tar
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --label_map_path=object_detection/data/pascal_label_map.pbtxt \
        --data_dir=VOCdevkit --year=VOC2012 --set=train \
        --output_path=pascal_train.record
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --label_map_path=object_detection/data/pascal_label_map.pbtxt \
        --data_dir=VOCdevkit --year=VOC2012 --set=val \
        --output_path=pascal_val.record
    ```
2. Dataset Requirements
    - An RGB image for the dataset encoded as jpeg or png.
    - A list of bounding boxes for the image. Each bounding box should contain:
        1. A bounding box coordinates (with origin in top left corner) defined by 4 floating point numbers [ymin, xmin, ymax, xmax]. Note that we store the normalized coordinates (x / width, y / height) in the TFRecord dataset.
        2. The class of the object in the bounding box.
3. Dataset Structure
    ```txt
    cowface     // 所有类别的图片都放这个目录，如dog，cat
    ├── Annotations   // 所有xml都放这里，xml里面有类别信息
    │   ├── 1.xml
    │   └── 2.xml
    ├── cat
    │   └── JPEGImages
    ├── cowface_label_map.pbtxt   // 数据集的标签信息
    ├── cowface_train.record    // 生成的二进制tfrecord文件
    ├── dog
    │   └── JPEGImages    // 所有原始图片
    │       ├── 1.jpg
    │       └── 2.jpg
    ├── dog.record
    └── ImageSets     // 这里指定哪些图片用于训练，哪些图片用于验证
        └── trainval.txt
    ```

### Configuring the Object Detection Training Pipeline

1. The schema for the training pipeline can be found in `object_detection/protos/pipeline.proto`. 
2. To help new users get started, sample model configurations have been provided in the `object_detection/samples/configs` folder. The contents of these configuration files can be pasted into model field of the skeleton configuration. Users should note that the `num_classes` field should be changed to a value suited for the dataset the user is training on.
    - The Tensorflow Object Detection API accepts inputs in the TFRecord file format. Users must specify the locations of both the training and evaluation files. 
    - Additionally, users should also specify a label map, which define the mapping between a class id and class name.
    - `train_config` provides two fields to specify pre-existing checkpoints: `fine_tune_checkpoint` and `from_detection_checkpoint`.
    - The `data_augmentation_options` in `train_config` can be used to specify how training data can be modified. 
    - The main components to set in `eval_config` are num_examples and `metrics_set`. The parameter `num_examples` indicates the number of batches ( currently of batch size 1) used for an evaluation cycle, and often is the total size of the evaluation dataset. The parameter `metrics_set` indicates which metrics to run during evaluation (i.e. `"coco_detection_metrics"`).

### Recommended Directory Structure for Training and Evaluation

```txt
+data
  -label_map file
  -train TFRecord file
  -eval TFRecord file
+models
  + model
    -pipeline config file
    +train
    +eval
```
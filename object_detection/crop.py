import os
from time import time

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import math
from utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util

MODEL_PATH = '/home/liuhy/Data/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('/home/liuhy/models-tensorflow/research/object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
BATCH_SIZE = 16.0
USE_BATCH = False

# Load model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


TEST_IMAGE_PATHS_DOG = [os.path.join('/home/liuhy/Pictures', 'dog{}.jpg'.format(i)) for i in range(1, 3)]
TEST_IMAGE_PATHS = TEST_IMAGE_PATHS_DOG


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


# TODO
def run_inference_batch(images, graph):
    pass


def process_single_image():
    index = 0
    for image_path in TEST_IMAGE_PATHS:
        start_time = time()
        image = Image.open(image_path)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # output_dict = run_inference_for_single_image(image_np, detection_graph)
        output_dict = run_inference_batch(image_np_expanded, detection_graph)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure()
        plt.imshow(image_np)
        plt.show()

        box = output_dict['detection_boxes'][0]
        img_width, img_height = image.size
        ymin = box[0] * img_height
        ymax = box[2] * img_height
        xmin = box[1] * img_width
        xmax = box[3] * img_width

        crop = np.array(image)[max(0, int(ymin)): int(ymax), max(0, int(xmin)): int(xmax)]
        plt.figure()
        plt.imshow(crop)
        plt.show()

        save_path = '/home/liuhy/Downloads/test' + str(index) + '.jpg'
        crop = Image.fromarray(crop)
        crop.save(save_path)
        index += 1
        end_time = time()
        print('Time spent: ', end_time - start_time)


if __name__ == '__main__':
    process_single_image()

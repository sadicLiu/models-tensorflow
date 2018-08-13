# coding=utf-8
import numpy as np
import os

IMG_PATH = '/home/liuhy/Data/datasets/cowface/dog/JPEGImages'
FILE_PATH = '/home/liuhy/Data/datasets/cowface/ImageSets'
NUM_TRAINVAL = 5000  # 5000张图片作为trainval


def main():
    images = os.listdir(IMG_PATH)
    np.random.shuffle(images)
    trainval_set = images[:NUM_TRAINVAL]
    test_set = images[NUM_TRAINVAL:]

    trainval_path = os.path.join(FILE_PATH, 'trainval.txt')
    test_path = os.path.join(FILE_PATH, 'test.txt')

    with open(trainval_path, 'w') as f:
        for i in range(len(trainval_set)):
            f.write(trainval_set[i].split('.')[0])
            f.write('\n')
    with open(test_path, 'w') as f:
        for j in range(len(test_set)):
            f.write(test_set[j].split('.')[0])
            f.write('\n')

    print('Done.')


if __name__ == '__main__':
    main()

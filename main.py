import numpy as np
import cv2
import tensorflow as tf


class ImageAnn:
    def __init__(self, bounding_boxes):
        self.bounding_boxes = bounding_boxes


class BoundingBox:
    def __init__(self, x, y, w, h, landmarks):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.landmarks = landmarks


def convert_ann(array, image_shape):
    x, y, w, h = array[0], array[1], array[2], array[3]
    landmakrs = array[4:]
    
    w = (w / image_shape[0]) * img_width
    h = (h / image_shape[1]) * img_height
    
    points1 = np.array([(0, 0),
                        (image_shape[1] - 1, 0),
                        (image_shape[1] - 1, image_shape[0] - 1)], np.float32)

    points2 = np.array([(0, 0),
                        (255, 0),
                        (255, 255)], np.float32)

    M = cv2.getAffineTransform(points1, points2)

    x, y = (np.array([x, y]) @ M)[:2]
    
    temp = []
    for i in range(len(landmakrs) // 2):
        temp.append([landmakrs[i], landmakrs[i + 1]])
    
    landmarks = (np.array(temp) @ M)[:, :2]
    landmarks = landmarks.astype(np.int32)
    
    return x, y, w, h, landmarks
    
    
    
    

def read_ann():
    ann_f = open("./new_ann.txt", "r")
    lines = ann_f.readlines()

    pointer = 0
    image_anns = []
    while pointer != len(lines):
        size_str = lines[pointer][:-1].split()
        shape = int(size_str[0]), int(size_str[1])

        pointer += 1
        n = int(lines[pointer])
        pointer += 1

        bounding_boxes = []
        for i in range(n):
            temp = lines[pointer].split()
            temp = [int(t) for t in temp]
            x, y, w, h, landmarks = convert_ann(temp, shape)
            bb = BoundingBox(x, y, w, h, landmarks)
            bounding_boxes.append(bb)
            pointer += 1
        image_ann = ImageAnn(bounding_boxes)
        image_anns.append(image_ann)

    return image_anns


cell_count = 8
img_height, img_width = 256, 256
batch_size = 75
data_path = "./Data/train"

full_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Read Annotation
annotation = read_ann()


# Add Labels



# Training, Testing


# Model



# Non Maximum Suppression

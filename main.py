import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

class BoundingBox:
    def __init__(self, image, x, y, w, h):
        self.image = image
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class Landmarks:
    def __init__(self, image, bounding_box, facial_landmarks):
        self.facial_landmarks = facial_landmarks


def read_data():
    data_desc = open("./Data/annotations.txt", 'r')
    data_path = "./Data/train/"
    lines = data_desc.readlines()
    pointer = 0
    images_list = []
    bounding_boxes = []
    landmarks = []
    pbar = tqdm(total=4275, initial=0)

    while pointer < len(lines):
        path = data_path + lines[pointer][:-1]
        image = cv2.imread(path)
        images_list.append(image)
        pointer += 1
        n = int(lines[pointer])
        pointer += 1
        for i in range(n):
            temp = lines[pointer]
            temp = temp.split()
            bb = BoundingBox(len(images_list) - 1, temp[0], temp[1], temp[2], temp[3])
            bounding_boxes.append(bb)
            lm = Landmarks(len(images_list) - 1, len(bounding_boxes) - 1, temp[4:])
            landmarks.append(lm)
            pointer += 1
        pbar.update(1)
        
    pbar.close()

    return images_list, bounding_boxes, landmarks


# model = ?



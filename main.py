import numpy as np
import cv2
from tqdm import tqdm

class Image:
    def __init__(self, image_px, bounding_boxes):
        self.image_px = image_px
        self.bounding_boxes = bounding_boxes


class BoundingBox:
    def __init__(self, x, y, w, h, landmarks):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.landmarks = landmarks


def read_data():
    data_desc = open("./Data/annotations.txt", 'r')
    data_path = "./Data/train/"
    lines = data_desc.readlines()
    pointer = 0
    images_list = []
    pbar = tqdm(total=4275, initial=0)

    while pointer < len(lines):
        path = data_path + lines[pointer][:-1]
        image_px = cv2.imread(path)
        pointer += 1
        n = int(lines[pointer])
        pointer += 1
        bounding_boxes = []
        for i in range(n):
            temp = lines[pointer]
            temp = temp.split()
            landmarks = temp[4:]
            bounding_box = BoundingBox(temp[0], temp[1], temp[2], temp[3], landmarks)
            bounding_boxes.append(bounding_box)
            pointer += 1
        image = Image(image_px, bounding_boxes)
        images_list.append(image)
        pbar.update(1)
        
    pbar.close()

    return images_list


images_list = read_data()
# model = ?



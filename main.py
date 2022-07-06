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

    w = int((w / image_shape[0]) * img_width)
    h = int((h / image_shape[1]) * img_height)

    points1 = np.array([(0, 0),
                        (image_shape[1] - 1, 0),
                        (image_shape[1] - 1, image_shape[0] - 1)], np.float32)

    points2 = np.array([(0, 0),
                        (255, 0),
                        (255, 255)], np.float32)

    M = cv2.getAffineTransform(points1, points2)

    x, y = (np.array([x, y]) @ M)[:2]
    x, y = int(x), int(y)

    temp = []
    for i in range(len(landmakrs) // 2):
        temp.append([landmakrs[i], landmakrs[i + 1]])

    landmarks = (np.array(temp) @ M)[:, :2]
    landmarks = landmarks.astype(np.int32)

    return x, y, w, h, landmarks


def read_ann():
    ann_f = open("./newer_ann.txt", "r")
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


def add_labels(batch_index):
    # Vector = [p, x, y, w, h]
    batch_label = np.zeros((batch_size, cell_count, cell_count, 5))
    cell_size = 256 / cell_count
    for k in range(batch_size):
        image_ann = annotation[batch_index * 75 + k]
        for box in image_ann.bounding_boxes:
            x, y, w, h = box.x, box.y, box.w, box.h
            for i in range(cell_count):
                for j in range(cell_count):
                    vector = np.zeros(5)
                    if i * cell_size < x < (i + 1) * cell_size and j * cell_size < y < (j + 1) * cell_size:
                        vector[0] = 1
                        vector[1] = (x - i * cell_size) / cell_size
                        vector[2] = (y - i * cell_size) / cell_size
                        vector[3] = (w // cell_size) + 1
                        vector[4] = (h // cell_size) + 1
                    batch_label[k, i, j, :] = vector
        
    return batch_label
    
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


# Add Lables

index = 0
my_input = []

batch_labels = []
batch_inputs = []
for batch in full_ds:
    batch_label = add_labels(index)
    batch_labels.append(batch_label)
    batch_inputs.append(batch[0])
    index += 1

full_ds = tf.data.Dataset.from_tensor_slices((batch_inputs, batch_labels))

# Training, Testing
train_size = int(0.8 * len(full_ds))
test_size = int(0.20 * len(full_ds))

train_ds = full_ds.take(train_size)
test_ds = full_ds.skip(train_size)
print(len(full_ds))
print(len(train_ds))
print(len(test_ds))
# Model


# Non Maximum Suppression

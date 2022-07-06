import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

cell_count = 8
img_height, img_width = 256, 256
batch_size = 75
data_path = "./Data/train"

full_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Add Labels

# Training, Testing

# Model

# Non Maximum Suppression



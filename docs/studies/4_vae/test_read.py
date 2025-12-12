import os
import struct
import numpy as np
import torch

def read_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols).astype(np.float32) / 255.0  # normalize 0–1
    return images

def read_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Paths
train_images_path = os.path.join(path, "train-images-idx3-ubyte")
train_labels_path = os.path.join(path, "train-labels-idx1-ubyte")
test_images_path = os.path.join(path, "t10k-images-idx3-ubyte")
test_labels_path = os.path.join(path, "t10k-labels-idx1-ubyte")

# Load data
x_train = read_images(train_images_path)
y_train = read_labels(train_labels_path)
x_test = read_images(test_images_path)
y_test = read_labels(test_labels_path)

# Convert to torch tensors
x_train = torch.tensor(x_train).unsqueeze(1)  # shape: [N, 1, 28, 28]
x_test = torch.tensor(x_test).unsqueeze(1)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

print(x_train.shape, y_train.shape)

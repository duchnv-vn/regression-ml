import gzip
import struct

import numpy as np


def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows)


def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)


def load_label(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        all_labels = f.read()
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def encode_target_label(Y, target):
    return (Y == target).astype(int)


def one_hot_encode(Y, class_number):
    n_labels = Y.shape[0]
    encoded_Y = np.zeros((n_labels, class_number))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y

from __future__ import print_function, division
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



class UnetDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None, input_size=512):

        self.transform = transform
        self.path2images = images_dir
        self.path2masks = masks_dir
        self.all_images = [x for x in sorted(os.listdir(images_dir)) if x[-4:] == '.png']  # Read all the images
        if masks_dir is not None:
            self.all_masks = [x for x in sorted(os.listdir(masks_dir)) if x[-4:] == '.png']  # Read all the masks
        self.input_size = input_size

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        id_image = self.all_images[idx]
        img = get_image(os.path.join(self.path2images, id_image), self.input_size)
        mask = None
        if self.path2masks is not None:
            id_mask = self.all_masks[idx]
            mask = get_image(os.path.join(self.path2masks, id_mask), self.input_size)

        sample = {'input': img,
                  'ground_truth': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors. create from mask 3 channels mask """
    def __init__(self):
        pass

    def __call__(self, sample):
        img, mask = sample['input'], sample['ground_truth']
        mask_cat = np.zeros((1))
        img = np.expand_dims(img, axis=2)
        if mask is not None:
            mask_cat = to_categorical(np.array(mask / 127, dtype=np.int), num_classes=3)

        # swap color axis because
        # numpy image: H x W x channels
        # torch image: channels x H X W
        img = normalize(img).transpose((2, 0, 1))
        if mask is not None:
            mask_cat = mask_cat.transpose((2, 0, 1))
        return {'input': torch.from_numpy(img),
                'ground_truth': torch.from_numpy(mask_cat)}


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def normalize(image):
    # normalize between 0-1 and cast to tensor:
    if np.max(image) > 1:
        return (np.float32(image)-np.min(image)) / (np.max(image) - np.min(image))
    else:
        return np.float32(image)


def get_image(path2image, input_size):
    img = None
    if path2image[-4:] == '.png':
        img = cv2.imread('{}'.format(path2image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.resize(img, (input_size, input_size))
    return img
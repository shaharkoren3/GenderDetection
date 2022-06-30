#!/usr/bin/env python3
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import glob

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import constants as co


def load_data(
    path,
    target_size,
    classes,
    batch_size,
    rotation_range=0,
    height_shift_range=0.0,
    horizontal_flip=False,
    zoom_range=0.0,
    shuffle=False,
):
    generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=rotation_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        zoom_range=zoom_range,
        validation_split=0.1,
    )

    train = generator.flow_from_directory(
        directory=path,
        target_size=target_size,
        batch_size=batch_size,
        classes=classes,
        shuffle=shuffle,
        subset="training",
    )

    val = generator.flow_from_directory(
        directory=path,
        target_size=target_size,
        batch_size=batch_size,
        classes=classes,
        shuffle=shuffle,
        subset="validation",
    )

    return train, val


def load_test_data(
    path,
    target_size,
    classes,
    batch_size,
):
    generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        zoom_range=0,
    )

    test = generator.flow_from_directory(
        directory=path,
        target_size=target_size,
        batch_size=batch_size,
        classes=classes,
        shuffle=False,
    )

    return test


#!/usr/bin/env python3
import itertools
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import glob
import numpy as np
from sklearn.metrics import confusion_matrix

import constants as co


def test_model(test, model):
    results = model.evaluate(test, batch_size=co.BATCH_SIZE)
    print("calculating confusion matrix ...")
    eval_confusion_matrix(test, model)
    return results


def load_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(co.DIM, co.DIM)
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = img / 255
    return img


def test_image(image_path, model):
    img = load_image(image_path)
    return model.predict(img)


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def calc_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def eval_confusion_matrix(test_batches, model):
    predictions = model.predict(x=test_batches, verbose=0)
    cm = confusion_matrix(
        y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1)
    )
    acc = calc_accuracy(test_batches.classes, np.argmax(predictions, axis=-1))
    plot_confusion_matrix(
        cm=cm,
        classes=co.CLASSES,
        title=f"First Model CM, with {acc}% accuracy on test set",
        normalize=False,
    )
    return predictions

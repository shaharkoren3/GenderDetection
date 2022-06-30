import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import data_preperation as dpre
import train_model as tr
import test_model as te
import constants as co
import numpy as np


def application():
    print("it is a boy or a girl?\nshahar koren\nthis project detects men and women in a photo\n")
    while True:
        user_ans = input("pick an option:\n1)train the model\n2)test the model\n3)predict an image\n4)EXIT)")
        if user_ans not in ["1", "2", "3", "4"]:
            print(f"input: {user_ans} is not correct. please try again.\n")
        else:
            if user_ans == "1":
                print("you picked: train the model\nloadind data... \n")
                train, val = load_data()
                ans_1(train, val)
            if user_ans == "2":
                print("you picked: test the model ")
                if model_exist():
                    test = load_test_data()
                    model = keras.models.load_model(co.MODEL_PATH)
                    ans_2(test, model)
                else:
                    print("the model needs to be trained before you use it.\n")
            if user_ans == "3":
                print("you picked: predict image ")
                if model_exist():
                    model = keras.models.load_model(co.MODEL_PATH)
                    ans_3(model)
                else:
                    print("the model needs to be trained before you use it.\n")
            if user_ans == "4":
                print("bye")
                break


def ans_1(train, val):
    print("\ntraining the model:\n")
    model = tr.build_model(co.DIM, co.DIM, 3, 2)
    history, saved_model = tr.train(train, val, co.EPOCHS, model)
    tr.graph(history, co.EPOCHS)
    return history, saved_model


def ans_2(test, model):
    results = te.test_model(test, model)
    print(f"Model test results: accuracy {results[1]}, loss {results[0]}")
    return results


def ans_3(model):
    path = input("type image path: ")
    if not os.path.isfile(path):
        return f"{path} is not a valid file"

    file_parts = os.path.splitext(path)
    if len(file_parts) < 2 or (file_parts[1] != ".jpg" and file_parts[1] != ".png"):
        return f"{path} is not a .jpg/.png file"

    prediction = te.test_image(path, model)

    print(f"prediction: {prediction}")
    pred = prediction[0]
    pred = np.argmax(pred, axis=-1)
    print(f"Result: {co.CLASSES[pred]}\n")

    return prediction


def model_exist():
    return os.path.isfile(co.MODEL_PATH)


def load_data():
    return dpre.load_data(
        path=co.DATASET_PATH,
        target_size=(co.DIM, co.DIM),
        classes=co.CLASSES,
        batch_size=co.BATCH_SIZE,
        rotation_range=15,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.0,
        shuffle=True,
    )


def load_test_data():
    return dpre.load_test_data(
        path=co.DATASET_TEST_PATH,
        target_size=(co.DIM, co.DIM),
        classes=co.CLASSES,
        batch_size=co.BATCH_SIZE,
    )
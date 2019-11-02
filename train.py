from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Dropout, UpSampling2D, MaxPool2D, Flatten, Conv2D, Reshape, Input, GaussianNoise
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2


path = './Datasets/A'
win = []
for image_path in os.listdir(path):
    try:
        image = Image.open(os.path.join(path, image_path))
    except OSError:
        continue

    image = image.resize((100, 100), Image.ANTIALIAS)
    data = np.asarray(image)
    data = data / 255
    win.append(data)

path = './Datasets/B'
normal = []
for image_path in os.listdir(path):
    try:
        image = Image.open(os.path.join(path, image_path))
    except OSError:
        continue

    image = image.resize((100, 100), Image.ANTIALIAS)
    data = np.asarray(image)
    data = data / 255
    normal.append(data)

X = np.array(win + normal).reshape(-1, 100, 100, 3)
y = np.concatenate([np.zeros(len(win)), np.ones(len(normal))]).reshape((-1, 1))

print(X.shape)
print(y.shape)

# this is our input placeholder
input_img = Input(shape=(100, 100, 3))
noise = GaussianNoise(0.1)(input_img)
conv_0 = Conv2D(16, 3)(noise)
conv_1 = Conv2D(16, 3)(conv_0)
pool_0 = MaxPool2D(2)(conv_1)
conv_2 = Conv2D(16, 3)(pool_0)
conv_3 = Conv2D(16, 3)(conv_2)
pool_1 = MaxPool2D(2)(conv_3)
flat = Flatten()(pool_1)
dense_0 = Dense(64, activation='relu')(flat)
dense_1 = Dense(16, activation='relu')(dense_0)
out = Dense(1, activation='sigmoid')(dense_1)
# this model maps an input to its reconstruction
cnn = Model(input_img, out)

cnn.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

cnn.summary()
cnn.fit(X, y, batch_size=32, epochs=10)
cnn.save('cnn.h5')

from sklearn.metrics import mean_squared_error, confusion_matrix
print(mean_squared_error(y, cnn.predict(X)))

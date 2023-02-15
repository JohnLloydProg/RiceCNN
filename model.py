import os
import settings
import tensorflow as tf
import numpy as np
from keras import layers, models
from keras.callbacks import TensorBoard
from dataset import MySequence
import settings
import random

base_dir = './drive/MyDrive/images'

data = []

for category in os.listdir(base_dir):
  new_path = os.path.join(base_dir, category)
  for file in os.listdir(new_path):
    data.append((os.path.join(new_path, file), settings.CATEGORIES.index(category)))

random.shuffle(data)

features = []
labels = []

for feature, label in data:
  features.append(feature)
  labels.append(label)

test_features = np.array(features[0:800])
test_labels = np.array(labels[0:800])
train_features = np.array(features[800:])
train_labels = np.array(labels[800:])

batch_size = 10
train_sequence = MySequence(train_features, train_labels, batch_size)
test_sequence = MySequence(test_features, test_labels, batch_size)

for no_dense_layer in settings.no_dense_layer:
    for dense_layer_size in settings.dense_layer_size:
        for no_conv_layer in settings.no_conv_layer:
            for conv_layer_size in settings.conv_layer_size:
                name = f'Stage-Two_Dense-{no_dense_layer}-Nodes-{dense_layer_size}_Conv-{no_conv_layer}-Nodes-{conv_layer_size}'
                tensorboard = TensorBoard(log_dir=f'logs/{name}')

                model = models.Sequential()

                model.add(layers.Conv2D(conv_layer_size, (3, 3), activation='relu', input_shape=(settings.picture_height, settings.picture_width, 3)))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPool2D((2, 2)))

                for i in range(no_conv_layer-1):
                    model.add(layers.Conv2D(conv_layer_size, (3, 3), activation='relu'))
                    model.add(layers.BatchNormalization())
                    model.add(layers.MaxPool2D((2, 2)))

                model.add(layers.Dropout(0.2))

                model.add(layers.Flatten())

                for i in range(no_dense_layer):
                    model.add(layers.Dense(dense_layer_size, activation='relu'))

                model.add(layers.Dense(3, activation='softmax'))

                model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
                print(model.summary())
                model.fit(train_sequence, epochs=10, validation_data=test_sequence, callbacks=[tensorboard])
                model.save(f'logs/{name}/model')
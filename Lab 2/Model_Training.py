# Karolis Vėgėla, 2016061
# Classes ['Castle', 'Coffee', 'Pizza']
# In this file a model is created, trained and saved for further use in other files.

import tensorflow as tf
import numpy as np

# Creating Training And Testing Datasets
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    './train/',
    labels = 'inferred',
    label_mode = 'int',
    class_names = ['castle', 'coffee', 'pizza'],
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (256, 256),
    shuffle = True,
    seed = 123,
    validation_split = 0.1,
    subset = 'training'
)
ds_train = ds_train.map(lambda x, y: (x / 255.0, y))

# Creating The Model
input = tf.keras.Input(shape=(256,256,3))

x = tf.keras.layers.Conv2D(16, 3, activation='relu')(input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(256, 3, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=x)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(ds_train, epochs=30, verbose=2)
model.save('./MODELIS/')
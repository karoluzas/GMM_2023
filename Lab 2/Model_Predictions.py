# Karolis Vėgėla, 2016061
# Classes ['Castle', 'Coffee', 'Pizza']
# In this file the model that was trained in 'Model_Training.py' is loaded in and used to predict.

import os
import sklearn
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score

loaded_model = tf.keras.models.load_model('./MODELIS/')

ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    './test/',
    labels='inferred',
    label_mode='int',
    class_names=['castle', 'coffee', 'pizza'],
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False
)
ds_test = ds_test.map(lambda x, y: (x / 255.0, y))

y_pred = loaded_model.predict(ds_test)

# Get True Labels
y_true = np.concatenate([y for x, y in ds_test], axis=0)
y_pred = np.argmax(y_pred, axis=1)

# Calculating Metrics With sklearn Library
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f'Accuracy: {accuracy}; Precision: {precision}; Recall: {recall}; F1: {f1}')
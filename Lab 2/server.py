# Karolis Vėgėla, 2016061
# Classes ['Castle', 'Coffee', 'Pizza']
# Backend python server that handles a POST request with a picture.
# POST http://localhost:5000/upload, form-data with a picture.

from flask import Flask, request
from flask_cors import CORS
import io
import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model('./MODELIS/')
class_names = ['castle', 'coffee', 'pizza']

app = Flask(__name__)
CORS(app) 

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        img = tf.keras.preprocessing.image.load_img(io.BytesIO(image.read()), target_size=(256, 256))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x / 255.0 
        prediction = loaded_model.predict(tf.expand_dims(x, axis=0))
        print(f'{image} + predictions: {prediction[0]}')
        return f'{class_names[np.argmax(prediction[0])]}'
    else:
        return 'No image found in request', 400

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
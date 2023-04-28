import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_mnist_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


# Load the saved model
model = load_model('mnist_model.h5')

# replace with the path to your local image
image_path = 'seven.png'

# Load the image and convert it to grayscale
image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))

# Convert the image to a NumPy array and normalize the pixel values
image_array = img_to_array(image) / 255.0

# Invert the image colors if necessary (MNIST images have a white background and black digits)
# Uncomment the following line if your image has a black background and white digits
# image_array = 1 - image_array
# reshape the image to have a batch dimension
input_image = np.expand_dims(image_array, axis=0)
input_image = np.expand_dims(input_image, -1)  # add a channel dimension

prediction = model.predict(input_image)
predicted_digit = np.argmax(prediction)

print(f'Predicted digit: {predicted_digit}')

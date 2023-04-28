import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import urllib.request
import certifi
import ssl
from urllib.request import ProxyHandler, build_opener, install_opener

def download_mnist_dataset():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    local_file = "mnist.npz"

    # Replace 'http_proxy' and 'https_proxy' with your proxy settings
    proxy_settings = {
        'http': '127.0.0.1:2023'
    }

    proxy_handler = ProxyHandler(proxy_settings)
    opener = build_opener(proxy_handler)
    install_opener(opener)

    # Download the dataset using the proxy settings
    urllib.request.urlretrieve(url, local_file, context=ssl.create_default_context(cafile=certifi.where()))


download_mnist_dataset()
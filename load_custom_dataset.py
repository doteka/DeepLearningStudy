import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def _parse_image_function(proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.image.resize(image, [64, 64])  # Resize images to 32x32
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(parsed_features['label'], tf.int32)
    return image, label

def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

def laod_to_custom_data():
    raw_train_dataset = tf.data.TFRecordDataset('./custom_dataset/cat_N_dog_train_data.tfrecord')
    raw_test_dataset = tf.data.TFRecordDataset('./custom_dataset/cat_N_dog_test_data.tfrecord')

    parsed_train_dataset = raw_train_dataset.map(_parse_image_function)
    parsed_test_dataset = raw_test_dataset.map(_parse_image_function)

    x_train, y_train = dataset_to_numpy(parsed_train_dataset)
    x_test, y_test = dataset_to_numpy(parsed_test_dataset)
    
    return (x_train, y_train), (x_test, y_test)
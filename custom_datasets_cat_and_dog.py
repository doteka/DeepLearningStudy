import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Bytes feature 생성 함수
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Int64 feature 생성 함수
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 이미지 예제 생성 함수
def image_example(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    feature = {
        'image': _bytes_feature(tf.io.encode_jpeg(image).numpy()),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# 데이터셋 저장 함수
def save_dataset(dataset, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for batch in dataset:
            images, labels = batch
            for image, label in zip(images, labels):
                tf_example = image_example(image, label)
                writer.write(tf_example.SerializeToString())

# 데이터셋 생성 함수
def create_dataset(directory):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(64, 64),
        batch_size=50,
        label_mode='int'
    )
    return dataset

# 데이터셋 로드
train_dir = './custom_dataset/cat_N_dog_dataset/training_set'
test_dir = './custom_dataset/cat_N_dog_dataset/test_set'

train_dataset = create_dataset(train_dir)
test_dataset = create_dataset(test_dir)

# 데이터 전처리 함수
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# 데이터셋 전처리
train_dataset = train_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# 데이터셋 저장
save_dataset(train_dataset, './custom_dataset/cat_N_dog_train_data.tfrecord')
save_dataset(test_dataset, './custom_dataset/cat_N_dog_test_data.tfrecord')
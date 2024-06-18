# Vanilla Object Detection (Cat & Dog Object Detection)
# Author : Easton Kang
# Email : easton.kang@kakao.com & easton@gwnu.ac.kr
# Date : 2024-06-17
# Last Edit : 2024-06-19

# Desc : Object Detection to train on a custom dataset of cats and dogs, and separate dogs and cats in image
# Reference : https://velog.io/@cha-suyeon/%EB%94%A5%EB%9F%AC%EB%8B%9D-Object-Detection-Sliding-Window-Convolution              # Object Detection


import numpy as np
import cv2
import tensorflow as tf
import pdb

from load_custom_dataset import laod_to_custom_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

class config:
    cls_idx = 1               # car class index
    noc = 10                  # number of cifar10 class 
    input_shape = (64, 64, 3) # cifar image size
    epochs = 10
    batch = 30
    
def load_data():
    (x_train, y_train), (x_test, y_test) = laod_to_custom_data()
    return (x_train, y_train), (x_test, y_test)

def preprocessing(x_train, y_train, x_test, y_test):
    # Extract car class data
    # x_train = x_train[y_train.flatten() == config.cls_idx]        
    # y_train = y_train[y_train.flatten() == config.cls_idx]
    # x_test = x_test[y_test.flatten() == config.cls_idx]
    # y_test = y_test[y_test.flatten() == config.cls_idx]
    
    # Data Normalization
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Label One-Hot-Encoding
    y_train = to_categorical(y_train, num_classes=config.noc)
    y_test = to_categorical(y_test, num_classes=config.noc)

    return (x_train, y_train), (x_test, y_test)

def makeModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=config.input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(config.noc, activation='softmax'))  # Output layer matches number of classes

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def fitNeval(model, x_train, y_train, x_test, y_test):
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.4)

    model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch, 
              validation_data=(x_val, y_val))
    return model

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y+window_size[1], x:x+window_size[0]])

def detect_objects(image, model, window_size=(32, 32), step_size=16, threshold=0.9):
    detect_objects = []
    for (x, y, window) in sliding_window(image, step_size, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue
        window = np.expand_dims(window, axis=0)
        predictions = model.predict(window)

        if np.max(predictions) >= threshold:
            detect_objects.append((x, y, np.max(predictions), np.argmax(predictions)))

    return detect_objects

(x_train, y_train), (x_test, y_test) = load_data()
(x_train, y_train), (x_test, y_test) = preprocessing(x_train, y_train, x_test, y_test)
model = makeModel()
model = fitNeval(model, x_train, y_train, x_test, y_test)

image = cv2.imread('./images/object_detection/catNdog.png')
image = cv2.resize(image, (128, 128))
image = image / 255.0

detected_objects = detect_objects(image, model)

for (x, y, score, class_id) in detected_objects:
    cv2.rectangle(image, (x, y), (x + 32, y + 32), (0, 255, 0), 2)
    cv2.putText(image, f'{class_id}: {score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
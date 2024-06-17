# LSTM Study
# Author : Easton Kang
# Email : easton.kang@kakao.com & easton@gwnu.ac.kr

# Date : 2024-06-17

# Desc : A simple LSTM model that trains on the word 'hihell' and then infers ihell when typing 'i'
# Reference : https://limitsinx.tistory.com/62              # RNN & LSTM
# Reference : https://dsbook.tistory.com/59                 # Fully connected layer
# Reference : https://blog.naver.com/chunjein/221589624838  # TimeDistributed 


import numpy as np
import tensorflow as tf

idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hello: hihell -> ihello

# x_data = [[0, 1, 0, 2, 3, 3]]  # hihell
y_data = [[1, 0, 2, 3,3, 4]] # ihell

noc = 5             # Number Of Class 
input_dim = 5       # one-hot size, same as hidden_size to directly predict one-hot
sequence_length = 6
learning_rate = 0.1

x_one_hot = np.array([[[1, 0, 0, 0, 0], # h 0
                       [0, 1, 0, 0, 0], # i 1
                       [1, 0, 0, 0, 0], # h 0
                       [0, 0, 1, 0, 0], # e 2
                       [0, 0, 0, 1, 0], # l 3
                       [0, 0, 0, 1, 0]  # l 3
                       ]], dtype=np.float32)

y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=noc)
print(x_one_hot.shape)
print(y_one_hot)

model = tf.keras.Sequential()

# input_shape = (1, 6, 5) => (number_of_sequence(batch), length_of_sequence, size_of_input_dim)
cell = tf.keras.layers.LSTMCell(units=noc, input_shape=(sequence_length, input_dim))
model.add(tf.keras.layers.RNN(cell=cell, return_sequences=True))

# single LSTM layer can be used as well instead of creating LSTMCell
# tf.model.add(tf.keras.layers.LSTM(units=noc, input_shape=(sequence_length, input_dim), return_sequnces=True))

# fully connected layer
# TimeDistributed : Computes cost(error) for each step and propagates 
# the errorto the child steps toupdate each weight

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=noc, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

model.fit(x_one_hot, y_one_hot, epochs=50)
model.summary()

pred = model.predict(x_one_hot)
for i, prediction in enumerate(pred):
    print(prediction)
    # print char using argmax, dict

    result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
    print('\tPrediction str : ', result_str)
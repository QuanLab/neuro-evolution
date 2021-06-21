from re import T
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.python.ops.gen_linalg_ops import self_adjoint_eig


class DNN_Model:
    def __init__(self, model_path = ''):
        self.model_path= model_path
        self.is_model_loaded = False

    def train(X_train, y_train):
        model= keras.Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='relu'))
        model.add(LSTM(256, return_sequences=False, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=30)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def predict(self, batch_input):
        dnn_model = models.load_model(self.model_path, compile=False)
        yhat = dnn_model.predict(batch_input)
        return yhat

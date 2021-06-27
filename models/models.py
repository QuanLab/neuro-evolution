import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

class DNN_Model:
    def __init__(self, model_path = ''):
        self.model_path= model_path
        self.is_model_loaded = False

    def train(self, X_train, y_train, save=False):
        model= keras.Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='relu'))
        model.add(LSTM(256, return_sequences=False, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(25, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=30)
        if save == True:
            print('save model to file...')
            model_json = model.to_json()
            with open("saved_models/model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("saved_models/model.h5")
            print("Saved model to disk")
        return history


    def predict(self, batch_input):
        if self.is_model_loaded == False:
            print('load model from file')
            json_file = open('saved_models/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = models.model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("saved_models/model.h5")
            self.is_model_loaded = True
        predictions = self.model.predict(batch_input)
        return predictions

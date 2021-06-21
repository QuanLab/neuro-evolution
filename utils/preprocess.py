
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Processor:

    def __init__(self, num_time_steps=1, num_features=1, future_window_size=1):
        self.num_time_steps = num_time_steps   # number of history in series used to predict the future
        self.num_features = num_features
        self.future_window_size = future_window_size   # use to create label for future prediction
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def add_label(self, price_series):
        delta_increase = max(price_series[1:] - price_series[0])
        delta_decrease = min(price_series[1:] - price_series[0])
        if delta_decrease > -delta_decrease:
            return price_series[0] + delta_increase
        return price_series[0] + delta_decrease


    def add_technical_indicator(self, data_frame):
        data_frame['ma20'] = data_frame['close'].rolling(window=20).mean()
        data_frame['ma14'] = data_frame['close'].rolling(window=14).mean()
        data_frame['ma7'] = data_frame['close'].rolling(window=7).mean()
        return data_frame


    def series_to_supervised(self, inp_array):
        X_train = []
        y_train = []

        for i in range(self.num_time_steps, len(inp_array)):
            X_train.append(inp_array[i-self.num_time_steps:i, : -1])
            y_train.append(inp_array[i,-1])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.num_features))
        return X_train, y_train


    def get_data_training(self, filename):
        df = pd.read_parquet(filename)
        df.dropna(inplace=True)
        # add technical indicator
        df = self.add_technical_indicator(df)
        # add label to data
        df['label'] = df['close'].rolling(window=self.future_window_size).apply(lambda x: self.add_label(x))
        df['label'] = df['label'].shift(-self.future_window_size + 1)
        df.dropna(inplace=True)
        scaled_df = self.scaler.fit_transform(df.to_numpy())

        X_train, y_train = self.series_to_supervised(scaled_df)
        return X_train, y_train

    # inverted the result from prediction to actual values
    def invert_prediction_result(self, test_X, y_predictions):
        test_X = test_X.reshape((test_X.shape[0], self.num_features * self.num_time_steps))
        # invert scaling for forecast
        inverted_y_predictions= np.concatenate((y_predictions, test_X[:, -self.num_features:]), axis=1)
        inverted_y_predictions = self.scaler.inverse_transform(inverted_y_predictions)
        inverted_y_predictions = inverted_y_predictions[:,0]
        return inverted_y_predictions
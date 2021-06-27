
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

class Processor:

    def __init__(self, num_time_steps=1, num_features=1, future_window_size=1, use_existing_scaler=True):
        self.num_time_steps = num_time_steps   # number of history in series used to predict the future
        self.num_features = num_features
        self.future_window_size = future_window_size   # use to create label for future prediction
        self.scaler_file = 'saved_models/scale.save'
        if use_existing_scaler == True:
            self.scaler = joblib.load(self.scaler_file)
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))

    def add_label(self, price_series):
        delta_increase = max(price_series[1:] - price_series[0])
        delta_decrease = min(price_series[1:] - price_series[0])
        if delta_increase > -delta_decrease:
            return delta_increase
        return delta_decrease


    def add_technical_indicator(self, data_frame):
        data_frame['MA_7'] = data_frame['close'].rolling(window=7).mean()
        data_frame['MA_14'] = data_frame['close'].rolling(window=14).mean()
        data_frame['MA_20'] = data_frame['close'].rolling(window=50).mean()
        # Overlap Studies Functions
        # data_frame["SMA"] = talib.SMA(data_frame["Close"])
        # data_frame["BBANDS_UP"], data_frame["BBANDS_MIDDLE"], data_frame["BBANDS_DOWN"] = talib.BBANDS(data_frame["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        # data_frame["EMA"]  = talib.EMA(data_frame["Close"], timeperiod=30)
        # data_frame["HT_TRENDLINE"] = talib.HT_TRENDLINE(data_frame["Close"])
        # data_frame["WMA"] = talib.WMA(data_frame["Close"], timeperiod=30)

        # # Momentum Indicator Functions
        # data_frame["ADX"] = talib.ADX(data_frame["High"], data_frame["Low"], data_frame["Close"], timeperiod=14)
        # data_frame["MACD"], _, _ = talib.MACD(data_frame["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
        # data_frame["MOM"] = talib.MOM(data_frame["Close"], timeperiod=5)
        # data_frame["RSI"] = talib.RSI(data_frame["Close"], timeperiod=14)

        # # Volatility Indicator Functions
        # data_frame["ATR"] = talib.ATR(data_frame["High"], data_frame["Low"], data_frame["Close"], timeperiod=14)
        # data_frame["TRANGE"] = talib.TRANGE(data_frame["High"], data_frame["Low"], data_frame["Close"])

        # # Price Transform Functions
        # data_frame["AVGPRICE"] = talib.AVGPRICE(data_frame["Open"], data_frame["High"], data_frame["Low"], data_frame["Close"])
        # data_frame["MEDPRICE"] = talib.MEDPRICE(data_frame["High"], data_frame["Low"])
        # data_frame["WCLPRICE"] = talib.WCLPRICE(data_frame["High"], data_frame["Low"], data_frame["Close"])

        # # Statistic Functions
        # data_frame["LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(data_frame["Close"], timeperiod=14)
        # data_frame["STDDEV"] = talib.STDDEV(data_frame["Close"], timeperiod=5, nbdev=1)

        return data_frame


    def outliers_zscore(seff, data):
        outliers =[]
        max_dev = 3
        mean = np.mean(data)
        std = np.std(data)
        for i in range(len(data)):
            Z_score = (data[i]-mean)/std
            if np.abs(Z_score)>max_dev:
                outliers.append(i)
        return outliers


    def series_to_supervised(self, train_X, train_y):
        X_train = []
        y_train = []

        for i in range(self.num_time_steps, len(train_X)):
            X_train.append(train_X[i-self.num_time_steps:i, :])
            y_train.append(train_y[i])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.num_features))
        return X_train, y_train


    def prepare_train_data(self, df):
        df.dropna(inplace=True)
        # add technical indicator
        df = self.add_technical_indicator(df)
        # add label to data
        df['label'] = df['close'].rolling(window=self.future_window_size).apply(lambda x: self.add_label(x))
        df['label'] = df['label'].shift(-self.future_window_size + 1)
        df.dropna(inplace=True)

        # remove trends
        for col in df.columns[:-1]:
            df[col] = df[col] - df[col].shift(1)

        for col in df.columns:
            df.drop(df.index[[idx for idx in self.outliers_zscore(df[col])]], inplace=True)
        
        df.dropna(inplace=True)
        print(df.head())

        scaled_train_X = self.scaler.fit_transform(df.to_numpy()[:, :-1])
        scaled_train_y = df.to_numpy()[:, -1]
        X_train, y_train = self.series_to_supervised(scaled_train_X, scaled_train_y)
        return X_train, y_train

    def prepare_test_data(self, df):
        df.dropna(inplace=True)
        df = self.add_technical_indicator(df)
        df.dropna(inplace=True)
        scaled_data = self.scaler.fit_transform(df.to_numpy())
        return scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])

    def save_scaler(self):
        joblib.dump(self.scaler, self.scaler_file)

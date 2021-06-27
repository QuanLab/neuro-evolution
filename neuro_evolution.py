from re import I, T
import matplotlib.pyplot as plt
from backtest.feed import DataFeed
import numpy as np
import pandas as pd
from utils.preprocess import Processor
from models.models import DNN_Model


class Position:
    def __init__(self, type = None, entry_price=0., exit_price=0., profit=0.) -> None:
        self.type = type
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.profit = profit

    def get_profit(self):
        return self.profit


if __name__ == "__main__":
    saved_model = 'saved_models/eurusd.h5'
    train_data_dir = 'data/2020.parquet'
    num_features = 7
    num_time_steps=30
    future_window_size=7

    processor = Processor(
        num_time_steps=num_time_steps, 
        num_features=num_features, 
        future_window_size=7, 
        use_existing_scaler=True
    )

    model = DNN_Model(model_path=saved_model)

    # train model
    # df = pd.read_parquet(train_data_dir)
    # X_train, y_train = processor.prepare_train_data(df)
    # processor.save_scaler()
    # history = model.train(X_train, y_train, save=True)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    df = pd.read_parquet("data/2021.parquet")
    df.dropna(inplace=True)
   
    data_feed = DataFeed(df, 79)
    
    initial_cash = 0
    positions = []

    holding_period = 0
    active_position = Position()
    
    i = 0
    while data_feed.has_next():
        # i+=1
        # if i > 200:
        #     break
        input_df = data_feed.next()
        input_data = processor.prepare_test_data(input_df)

        take_profit = 0.003
        commission = 0.00006
        if active_position.type != None:
            holding_period = holding_period+1
            if holding_period < 6:
                # close position
                if active_position.type == 'BUY':
                    profit = input_df.iloc[-1].high - active_position.entry_price
                    if profit > take_profit:
                        active_position.type = 'CLOSE'
                        active_position.exit_price = active_position.entry_price + take_profit
                        active_position.profit = take_profit - commission
                        positions.append(active_position)
                        active_position = Position()
                        holding_period = 0
                elif active_position.type == 'SELL':
                    profit = active_position.entry_price - input_df.iloc[-1].low
                    if profit > take_profit:
                        active_position.type = 'CLOSE'
                        active_position.exit_price = active_position.entry_price - take_profit
                        active_position.profit = take_profit - commission
                        positions.append(active_position)
                        active_position = Position()
                        holding_period = 0
            else:
                if active_position.type == 'BUY':
                    profit = input_df.iloc[0].close - active_position.entry_price
                    active_position.type = 'CLOSE'
                    active_position.exit_price = input_df.iloc[-1].close
                    active_position.profit = profit -commission
                    positions.append(active_position)
                elif active_position.type == 'SELL':
                    profit = active_position.entry_price - input_df.iloc[-1].close 
                    active_position.type = 'CLOSE'
                    active_position.exit_price = input_df.iloc[-1].close
                    active_position.profit = profit - commission
                    positions.append(active_position)
                active_position = Position()
                holding_period = 0
            continue
    
        y_predictions = model.predict(input_data)
        print("result ", y_predictions)
        if y_predictions[0][0] > 0 and active_position.type == None:
            active_position.type = 'BUY'
            active_position.entry_price = input_df.iloc[-1].open
            positions.append(active_position)
        elif y_predictions[0][0] < 0 and active_position.type == None:
            active_position.type = 'SELL'
            active_position.entry_price = input_df.iloc[-1].open
            positions.append(active_position)

    profits_all = 0.
    realized_profit = []
    for p in positions:
        if p.type == 'CLOSE':
            print(p.get_profit())
            realized_profit.append(p.get_profit())
            profits_all = profits_all + p.get_profit()
    
    print("Profit final: {}".format(profits_all))

    plt.plot(np.array(realized_profit).cumsum())
    plt.show()
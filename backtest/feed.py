from re import T
import pandas as pd


class DataFeed:
    '''
    Input data frame to data feed and pull data from datafeed with each batch size equals to n_bar_buffer
    '''
    def __init__(self, data_frame: pd.DataFrame, n_bar_buffer = 1) -> None:
        self.n_bar_buffer = n_bar_buffer
        self.current_index = self.n_bar_buffer
        self.data_frame = data_frame
        self.data = None

    def next(self):
        self.current_index = self.current_index + 1
        return self.data_frame.iloc[self.current_index - self.n_bar_buffer: self.current_index].copy()

    def has_next(self):
        if self.current_index < len(self.data_frame):
            return True
        return False
        

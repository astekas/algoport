import pandas as pd
import pickle

class Dataset:
    def __init__(self, name):
        self.available = ['SNP500']
        if name not in self.available:
            raise ValueError(f'Unknown dataset - {name}')
        else:
            self.name = name

    def SNP500(self, start, end):
        df = pd.read_parquet(".\Data\S&P\sp500_stocks.parquet")
        df = df.pivot_table(index='Symbol', columns='Date', values='Close')
        df = df.loc[:, start:end]
        df = df.rolling(2, axis=1).apply(lambda x: x.iloc[1] / x.iloc[0]).iloc[:, 1:]
        return df

    def fetch(self, start, end):
        return self.__getattribute__(self.name)(start, end)

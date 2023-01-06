import pandas as pd
from pkg_resources import resource_filename
import numpy as np

data_path = resource_filename('Algoport', 'Data/')


class Dataset:
    def __init__(self, name):
        self.available = ['SNP500']
        if name not in self.available:
            raise ValueError(f'Unknown dataset - {name}')
        else:
            self.name = name

    def SNP500(self, start, end):
        df = pd.read_parquet(data_path + "SNP500.parquet")
        df = df.pivot_table(index='Symbol', columns='Date', values='Close')
        df = df.loc[:, start:end]
        df.iloc[:, 1:] = np.array(df.iloc[:, 1:]) / np.array(df.iloc[:, :-1])
        return df.iloc[:, 1:]

    def fetch(self, start, end):
        return self.__getattribute__(self.name)(start, end)

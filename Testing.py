import pickle

from Algoport.Backtesting import BackTest
from Algoport.Configuration import StrategyConfigured

def parse(path):
    f = open(path, 'rb')
    df = pickle.load(f)
    return df

to_test = ['MSG_Omega_ratio_GA', 'MSG_Omega_ratio']

for name in to_test:
    strategy = StrategyConfigured(name=name, T=5, preselection=True).fetch()

    # # Run the test
    test = BackTest(strategy=strategy,
                    period_start="2021-12-09",
                    period_end="2022-12-09",
                    train_periods=1825,
                    output_path=f'.\\Tests\\Recession\\{name}_20\\',
                    filter_outliers=True)

    # Run the test)
    test.run()

    # The accumulated history of the strategy can be cleaned by reset method.
    strategy.reset()

    test = BackTest(strategy=strategy,
                    period_start="2016-01-01",
                    period_end="2017-01-01",
                    train_periods=1825,
                    output_path=f'.\\Tests\\Growth\\{name}_new',
                    filter_outliers=True)

    test.run()
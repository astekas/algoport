from Backtesting2 import BackTest
import pickle
from Configuration import StrategyConfigured

def parse(path):
    f = open(path, 'rb')
    df = pickle.load(f)
    return df

to_test = ['MSG_Sharpe_ratio', 'MSG_Stable_ratio',
           'MSG_Sharpe_ratio_cond', 'MSG_Sharpe_ratio_ranking', 'MSG_Sharpe_ratio_R1', 'MSG_Sharpe_ratio_R2']

for name in to_test:
    strategy = StrategyConfigured(name=name, T=20, preselection=True).fetch()

    # # Run the test
    test = BackTest(strategy=strategy,
                    period_start="2021-12-09",
                    period_end="2022-12-09",
                    train_periods=1825,
                    output_path=f'.\\Tests\\Recession\\{name}_20\\')

    # Run the test)
    test.run()

    # The accumulated history of the strategy can be cleaned by reset method.
    strategy.reset()

    test = BackTest(strategy=strategy,
                    period_start="2016-01-01",
                    period_end="2017-01-01",
                    train_periods=1825,
                    output_path=f'.\\Tests\\Growth\\{name}_20')

    test.run()
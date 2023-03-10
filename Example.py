from Algoport.Strategy import *
from Algoport.Markov import MarkovChainProcess
from Algoport.PortfolioOptimization import *
from Algoport.AssetSelection import *
from Algoport.Backtesting import BackTest
from Algoport.Metrics import *
import pickle

# Just a custom function to parse the preselection profiles.
def parse(path):
    f = open(path, 'rb')
    df = pickle.load(f)
    return df

# Step 1: Set up the constituents of the strategy

# Preselector initialization
# model - information extraction model to use. It will be fitted to data, thus metrics defined within the
# model class will be accessible to preselector.
# metrics - list of tuples defining which external metrics preselector should use. Tuples are constructed as follws
# (metric_function_reference, bool whether metric is positive (should be maximized), dict with metric kwargs)
# model_metrics - same as metrics, but used from the passed model. In this case the names are passed as strings.
# model_kwargs - dictionary with kwargs passed to the model.
# Should contain 'init' and 'fit' - two dictionaries corresponding to kwargs passed on initialization of the model (if not initialized) already
# and kwargs passeed to the fit function
# preselector_kwargs - kwargs passed to the "preselection" customizable function in preselector. Thus depend on the
# specific preselector used.

# preselector = Ranking_AS(model=MarkovChainProcess,
#                          metrics=[(cumulative_wealth, True, {'T': 50})],
#                          model_metrics=[('MSG_time_to_gain', False, {'T': 10, 'thresh': 1.04})],
#                          model_kwargs={'init': {},
#                                         'fit': {'N': 9,
#                                                 }},
#                          preselector_kwargs={'kind': 'Fixed',
#                                               'n_assets': 30})
T = 5
model = MarkovChainProcess
metrics = [(cumulative_wealth, True, {'T': 50}),
           (returns_sd, False, {}),
           (kendall_corr_usb, True, {}),
           (kendall_corr_lsb, False, {}),
           (mean_return, True, {})
           ]
model_metrics = [('MSG_wealth_mean', True, {'T': T}),
                 ('MSG_time_to_lose', True, {'T': T}),
                 ('MSG_square_root_utility', True, {'T': T}),
                 ('MSG_stable_location', True, {'T': T}),
                 ('MSG_wealth_sd', False, {'T': T}),
                 ('MSG_time_to_gain', False, {'T': T}),
                 ('MSG_CVaR_log_wealth', False, {'T': T}),
                 ]
model_kwargs = {'init': {},
                'fit': {'N': 9,
                        }}

preselector = DEA_AS(model=model,
                     metrics=metrics,
                     model_metrics=model_metrics,
                     model_kwargs=model_kwargs,
                     preselector_kwargs={})
# preselector = NMFPreselector(preselector_kwargs={})

# Initialize optimizer. Logic is similar to preselector, just note that only a single metric is supported at the moment,
# so a single metric tuple should be passed to either metric or model_metric argument

optimizer = SimplexOptimization(model=MarkovChainProcess,
                              model_metric=('MSG_omega_ratio', True, {'T': 5}),
                              model_kwargs={'init': {},
                                            'fit': {'N': 9,
                                                    'transaction_cost': 0.1,
                                                    'unconditional_start': True
                                                    }},
                              optimizer_kwargs={"m": 10, "p": 1})

# optimizer = PyMOO(model=MarkovChainProcess,
#                   model_metric=('MSG_omega_ratio', True, {'T': 5}),
#                   model_kwargs={'init': {},
#                                 'fit': {'N': 9,
#                                         'transaction_cost': 0,
#                                         'unconditional_start': True
#                                         }},
#                   optimizer_kwargs={'algorithm':'GeneticAlgorithm', 'n_gen_termination':2})



# Initialize the strategy by passing preselector, optimizer and regulation_kwargs, which are passed to the
# customizable regulation function within strategy class. Regulation is being called on each time step except the first,
# thus user can define the strategy in time.
strategy = Strategy(preselector=preselector,
                    optimizer=optimizer,
                    regulation_kwargs={'redistribute_every': 5, 'maintain_weights': True})

# Initialize the backtesting environment.
# Requires strategy, testing period defined as period_start - period end. At the moment, dates should be strictly
# passed as strings and only 2 formats are supported: '%Y-%m-%d' and '%Y-%w% (week of the year)

# preselection_profile is a sort of convenience functionality, which allows making preselection separately and in advance.
# The argument takes a dictionary with string dates as keys and lists of selected asset tickets as values.
# There are 4 preselection profiles already made, 2 for each of Growth and Recession periods respectively.
# Notice that for them to work, correct periods should be specified at the moment. In particular period_start - period_end
# range should not be longer than the profiled period, although might be shorter.

# train_periods - currently specified suboptimally, as the number of calendar days from the period_start to use for fitting
# the models. Notice that this is not the number of actual datapoints used for training.
# Output path - path to folder where results are saved.
test = BackTest(strategy=strategy,
                period_start="2021-12-09",
                period_end="2022-12-09",
                train_periods=1825,
                output_path='.\\Tests\\Growth\\MSG_Omega_ratio_R1',
                filter_outliers=True)

# Run the test)
test.run()
#
# # The accumulated history of the strategy can be cleaned by reset method.
strategy.reset()

test = BackTest(strategy=strategy,
                period_start="2016-01-01",
                period_end="2017-01-01",
                train_periods=1825,
                output_path='.\\Tests\\Growth\\MSG_Omega_ratio_R1',
                filter_outliers=True)

test.run()
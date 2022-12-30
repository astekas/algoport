from Algoport.Markov import MarkovChainProcess
import numpy as np
import pandas as pd
from Algoport.AsSel2 import Ranking_AS
from Algoport.Metrics import cumulative_wealth

# Create a test input as pd.DataFrame returns.
arr = []
for i in range(500):
    a = np.random.normal(1, 0.1, 100)
    arr.append(a)
returns = pd.DataFrame(np.vstack(arr))
print(returns.shape)

# Generate all the inputs for asset preselection
# model can be either a reference to a class or initialized object.
# In the first case it is being reinitialized on each "select" call, in the second just the fit function is being called
# on each "select" call.
model = MarkovChainProcess
# Metrics - which metrics to use for preselection.
# Shall be provided as callables or as strings (but then they should be defined in Metrics)
# The structure is as follows: (metric, maximize, kwargs)
metrics = [(cumulative_wealth, True, {'T': 50})]
# Same as with normal metrics, but these are being called from the model instance underhood. Thus, this one is for metrics
# defined within a model.
model_metrics = [('MSG_time_to_gain', False, {'T': 10, 'thresh': 1.04})]
# Kwargs for the model, as the name suggests, passed on init and into the fit function.
model_kwargs = {'init': {},
                'fit': {'N': 9,
                        }}
# Kwargs for the specific preselector subclass passed to the preselection function.
preselector_kwargs = {'kind': 'Fixed',
                      'n_assets': 10}

preselector = Ranking_AS(model=model,
                         metrics=metrics,
                         model_metrics=model_metrics,
                         model_kwargs=model_kwargs,
                         preselector_kwargs=preselector_kwargs)

print(preselector.select(returns=returns))
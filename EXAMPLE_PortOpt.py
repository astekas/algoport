from Algoport.PortfolioOptimization import SimplexOptimization
import numpy as np
import pandas as pd
from Algoport.Markov import MarkovChainProcess

# Create a test input as pd.DataFrame returns.
np.random.seed(1)
arr = []
for i in range(10):
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
metrics = None
# Same as with normal metrics, but these are being called from the model instance underhood. Thus, this one is for metrics
# defined within a model.
model_metric = ('MSG_corr_ratio', True, {'T': 4})
# Kwargs for the model, as the name suggests, passed on init and into the fit function.
model_kwargs = {'init': {},
                'fit': {'N': 9,
                        'transaction_cost': 0.7,
                        'unconditional_start': False
                        }}
# Kwargs for the specific preselector subclass passed to the preselection function.
# optimizer_kwargs = {'algorithm': 'COBYLA'}
#
# optimizer = SciPy(model=model,
#                   model_metric=model_metric,
#                   model_kwargs=model_kwargs,
#                   optimizer_kwargs=optimizer_kwargs)
# res = optimizer.optimize(returns=returns)
# print(res)
# print(np.sum(res))
# print(optimizer.values, optimizer.optimization_times)
#
# Example 2: MVO optimization on returns forecasted with AGC
# model = GARCH_EVT_COPULA()
# optimizer = MVOptimization(model=model)
# print(optimizer.optimize(returns=returns))

# Simplex
optimizer_kwargs = {"m": 10, "p": 1}
current_weights = np.ones(10) / 10
optimizer = SimplexOptimization(model=model,
                              model_metric=model_metric,
                              model_kwargs=model_kwargs,
                              optimizer_kwargs=optimizer_kwargs)
print(optimizer.optimize(returns=returns, current_weights=current_weights))
print(optimizer.values, optimizer.optimization_times)
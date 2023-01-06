### A module containing the end-to-end portfolio management strategies ###
import numpy as np

from Algoport import Utilities as U

class Strategy:
    def __init__(self, config=None, preselector=None, optimizer=None, transaction_cost=0, regulation_kwargs=None):
        if config is not None:
            if isinstance(config, dict):
                self.config = config
            elif isinstance(config, str):
                self.config = U.parse_config(config)
            else:
                raise ValueError('Unrecognized config type. Pass either a path to config.json or a dictionary with actual config.')

        self.preselector = preselector
        self.optimizer = optimizer
        self.transaction_cost = transaction_cost
        if regulation_kwargs is not None:
            self.regulation_kwargs = regulation_kwargs
        else:
            self.regulation_kwargs = {}

        self.assets = []
        self.weights = []
        # Is only filled if ComponentPreselector is used
        self.component_weights = []
        self.values = []
        # Current weights at the moment of decision making.
        # Note that the vector has dimension of all the assets available currently for selection.
        # It is adjusted for the last realized returns.
        self.weights_current = None # Filled with pd.Series
        self.step = 0

    def reset(self):
        self.assets = []
        self.weights = []
        self.weights_current = None # Filled with pd.Series
        self.step = 0

    def update_preselector(self):
        return None

    def update_optimizer(self):
        return None

    def update_assets(self, returns):
        return self.preselector.select(returns=returns)

    def update_weights(self, returns):
        return self.optimizer.optimize(returns=returns)

    def update_current_state(self, last_returns):
        # Assets that were selected at previous step
        current_assets = self.assets[-1]
        # Weights assigned to them
        previous_weights = self.weights[-1]
        # All assets that were available at previous step out of which we select
        all_assets = np.array(last_returns.index)
        # Find indices of currently held assets among all assets
        mask = np.isin(all_assets, current_assets)
        # Initialize current weights with 0s for all assets
        self.weights_current = np.zeros(len(all_assets))
        # Set previous weights set at previous step
        self.weights_current[mask] = previous_weights
        # Adjust them for returns realized at previous step
        self.weights_current = last_returns * self.weights_current
        self.weights_current = self.weights_current / self.weights_current.sum()

    def correct_wealth(self, assets_new, weights_new):
        current_assets = self.assets[-1]
        # Find which assets were held for both periods.
        kept_assets, ind_new, ind_old = np.intersect1d(assets_new, current_assets, assume_unique=True,
                                                       return_indices=True)
        share_sold = np.sum(self.weights_current[~ind_old]) + np.sum(weights_new[ind_new] - self.weights_current[ind_old] )
        weights_redistribution = self.weights_current[ind_old] - weights_new[ind_new]
        share_sold += weights_redistribution[weights_redistribution>0].sum()
        self.wealth[-1] = self.wealth[-1] - share_sold * self.transaction_cost

    def regulation(self, returns, assets=None, redistribute_every=1, maintain_weights=False, **kwargs):
        if assets is None:
            assets = np.array(returns.index)

        if self.step % redistribute_every == 0:
            if self.preselector is not None:
                returns = self.preselector.select(returns=returns)
                if self.preselector.kind == 'Assets':
                    assets = np.array(returns.index)
            elif assets is not None:
                returns = returns.loc[assets]

            if self.optimizer is not None:
                if self.preselector is None or self.preselector.kind == 'Assets':
                    current_weights = self.weights_current.loc[returns.index]
                else:
                    current_weights = None
                weights = self.optimizer.optimize(returns=returns, current_weights=current_weights)
                if self.preselector.kind == 'Components':
                    self.component_weights.append(weights)
                    weights = self.preselector.reverse_transform(weights)
            else:
                weights = np.ones(len(returns))
                weights = weights / weights.sum()
            self.assets.append(assets)
            self.weights.append(weights)
        else:
            self.assets.append(self.assets[-1])
            if maintain_weights:
                self.weights.append(self.weights[-1])
            else:
                self.weights.append(np.array(self.weights_current.loc[self.assets[-1]]))

    def evaluate(self, returns, assets=None):
        print(f'Evaluation step - {self.step}')
        if assets is None:
            assets = np.array(returns.index)

        if self.step == 0:
            if self.preselector is not None:
                returns = self.preselector.select(returns=returns)
                if self.preselector.kind == 'Assets':
                    assets = np.array(returns.index)
            elif assets is not None:
                returns = returns.loc[assets]
            if self.optimizer is not None:
                weights = self.optimizer.optimize(returns=returns)
                if self.preselector.kind == 'Components':
                    self.component_weights.append(weights)
                    weights = self.preselector.reverse_transform(weights)
            else:
                weights = np.ones(len(returns))
                weights = weights / weights.sum()
            self.assets.append(assets)
            self.weights.append(weights)
        else:
            last_returns = returns.iloc[:, -1]
            self.update_current_state(last_returns=last_returns)
            self.regulation(returns=returns, assets=assets, **self.regulation_kwargs)
        self.step += 1

class StrategySmoothed(Strategy):
    def regulation(self, returns, assets=None, redistribute_every=1, smoothing_delta=0.5, **kwargs):
        if assets is None:
            assets = np.array(returns.index)

        if self.step % redistribute_every == 0:
            if self.preselector is not None:
                returns = self.preselector.select(returns=returns)
                if self.preselector.kind == 'Assets':
                    assets = np.array(returns.index)
            elif assets is not None:
                returns = returns.loc[assets]

            if self.optimizer is not None:
                current_weights = self.weights_current.loc[returns.index]
                weights = self.optimizer.optimize(returns=returns, current_weights=current_weights)
                if self.preselector.kind == 'Components':
                    weights = self.preselector.reverse_transform(weights)
            else:
                weights = np.ones(len(returns))
                weights = weights / weights.sum()

            weights_current = self.weights_current
            weights_full = weights_current.copy()
            weights_full[:] = np.zeros(len(weights_full))
            weights_full.loc[assets] = weights
            weights_new = smoothing_delta * weights_full + (1-smoothing_delta)*weights_current
            weights_new[weights_new < 1e-3] = 0
            weights_new = weights_new / weights_new.sum()
            assets_new = np.array(weights_new[weights_new > 0].index)
            weights_new = np.array(weights_new[weights_new > 0])
            self.assets.append(assets_new)
            self.weights.append(weights_new)
        else:
            weights = self.weights[-1]
            assets = self.assets[-1]
            weights_current = self.weights_current
            weights_full = returns.iloc[:, -1].copy()
            weights_full[:] = np.zeros(len(weights_full))
            weights_full.loc[assets] = weights
            weights_new = smoothing_delta * weights_full + (1 - smoothing_delta) * weights_current
            weights_new[weights_new < 1e-3] = 0
            weights_new = weights_new / weights_new.sum()
            assets_new = np.array(weights_new[weights_new > 0].index)
            weights_new = np.array(weights_new[weights_new > 0])

            self.assets.append(assets_new)
            self.weights.append(weights_new)
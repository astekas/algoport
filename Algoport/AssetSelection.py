### A module containing the models for Asset Preselection ###
import inspect

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.decomposition import PCA, SparsePCA, NMF

from Algoport import Metrics

try:
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri
    rpy2_imported = True
except:
    warnings.warn('Failed to import rpy2. Perhaps, R installation is missing or could not be found. DEA preselector is anavailable.'
                  'Consider installing R.')
    rpy2_imported = False


# A parent class for reduction type preselection algorithms
class AssetPreselector():
    def __init__(self, model=None, metrics=None, model_metrics=None, model_kwargs=None, preselector_kwargs=None, config=None):
        '''
        General asset preselection class from which all the actual preselectors have to inherit.
        For the custom preselector, define a new function "preselection" in the child class.
        which accepts inputs and outputs (pd.DataFrames) with assets as index and metrics as columns and outputs a 1d np.array of asset names.

        :param model: instance of a custom model with fit method, which computes additional metrics
        :param metrics: list of tuples (callable, bool, dict), function objects to compute the metric from the array of returns.
                        Bool identifies whether the function should be maximized or minimized. dict contains kwargs to pass to the function.
        :param model_kwargs: dict, params to pass to model initialization
        :param preselector_kwargs: dict, additional params to pass to custom "preselection" function
        :param model_metrics: list of tuples (str, bool), function names from the model that should be used as additional metrics.
        :param config: dict, specifying just the names of all the parameters, if all the functions are defined in standard locations.
        '''
        self.model = model
        self.metrics = metrics
        self.model_metrics = model_metrics
        self.model_kwargs = model_kwargs
        self.preselector_kwargs = preselector_kwargs
        self.config = config
        self.kind = 'Assets'

    def prepare(self, returns):
        assets = returns.index
        inputs = pd.DataFrame(index=assets)
        outputs = pd.DataFrame(index=assets)
        if self.metrics is not None:
            for metric, positive, metric_kwargs in self.metrics:
                if isinstance(metric, str):
                    if not hasattr(Metrics, metric):
                        raise ValueError(f'Asset preselector error. Metric {metric} passed as a string is not defined in Metrics module!')
                    else:
                        metric = getattr(Metrics, metric)
                name = getattr(metric, '__name__', 'Unknown')
                res = metric(returns, **metric_kwargs)
                if isinstance(res, np.ndarray):
                    res = pd.Series(res, index=assets, name=name)
                if positive:
                    outputs = outputs.join(res.rename(name))
                else:
                    inputs = inputs.join(res.rename(name))
        if self.model is not None and self.model_metrics is not None:
            for metric, positive, metric_kwargs in self.model_metrics:
                metric = getattr(self.model, metric)
                res = metric(**metric_kwargs)
                if isinstance(res, np.ndarray):
                    name = getattr(metric, '__name__', 'Unknown')
                    res = pd.DataFrame(res, index=assets, columns=[name])
                if positive:
                    outputs = outputs.join(res)
                else:
                    inputs = inputs.join(res)

        return inputs, outputs

    def preselection(self, inputs, outputs, **kwargs):
        init_assets = inputs.index
        return np.array(init_assets)

    def select(self, returns):
        '''
        Execute actual preselection.
        :return: list of str, selected asset names
        '''

        if self.model is not None:
            if inspect.isclass(self.model):
                self.model = self.model(**self.model_kwargs['init'])
            if hasattr(self.model, 'fit'):
                self.model.fit(returns, **self.model_kwargs['fit'])
            else:
                raise ValueError('PRESELECTION ERROR. Model should have the "fit" method.')
        inputs, outputs = self.prepare(returns=returns)
        selected_assets = self.preselection(inputs=inputs, outputs=outputs, **self.preselector_kwargs)

        returns_out = returns.loc[selected_assets]

        return returns_out


class DEA_AS(AssetPreselector):
    def preselection(self, inputs, outputs, DEA_kind='SBM'):
        '''
        Data Envelopment Analysis preselection.
        :param config: dict, DEA-specific settings
        :return: list of str, selected asset names
        '''

        if not rpy2_imported:
            raise ValueError('DEA_AS preselector requires R installation, which was not found! Consider using Ranking_AS preselector.')

        # Check if additiveDEA package is available. Otherwise - attempt installing.
        try:
            dea = rpackages.importr('additiveDEA')
        except:
            print('Failed to import additiveDEA R package. Trying to install.')
            try:
                utils = rpackages.importr('utils')
                utils.chooseCRANmirror(ind=1)
                utils.install_packages('additiveDEA')
                dea = rpackages.importr('additiveDEA')
            except Exception as e:
                raise ValueError(f'Failed to install the additiveDEA R package. Exception - {e}. Please, make sure that there is R installed with additiveDEA package in it.')

        pandas2ri.activate()
        data = outputs.join(inputs)
        n_outputs = len(outputs.columns)

        r_dataframe = pandas2ri.py2rpy_pandasdataframe(data)
        efficiencies = np.array(dea.dea_fast(r_dataframe, noutput=n_outputs, rts=2, add_model=DEA_kind, blockSize=200))

        efficient_assets = np.array(data.loc[efficiencies == 1].index)

        return efficient_assets


class Ranking_AS(AssetPreselector):
    def preselection(self, inputs, outputs, kind='Fixed', n_assets=30, quantile=0.9):
        """
        Preselection algorithm based on simple ranking of assets.
        """
        # Get the necessary params from config
        # Standardize inputs and outputs by (X - mean) / std across assets (i.e. each input \ output separately)
        inputs_standardized = -(inputs - inputs.mean()) / inputs.std()
        outputs_standardized = (outputs - outputs.mean()) / outputs.std()
        scores = outputs_standardized.join(inputs_standardized).sum(axis=1).sort_values(ascending=False)
        if kind == 'Fixed':
            efficient_assets = np.array(scores.index)[:n_assets]
            return efficient_assets
        elif kind == 'Threshold':
            efficient_assets = np.array(scores[scores > scores.quantile(quantile)].index)
            return efficient_assets
        else:
            raise ValueError('Unknown Ranking kind! Available are "Fixed" and "Quantile"')


class ComponentsPreselector:
    def __init__(self, preselector_kwargs=None):
        self.loadings = None
        self.preselector_kwargs = preselector_kwargs
        self.kind = 'Components'

    def preselection(self, returns, kind='standard', n_components=None, variance_explained=None, sparsity_alpha = 1, **kwargs):
        returns = returns.T
        mean = np.array(returns.mean(axis=1))
        returns = StandardScaler().fit_transform(returns)
        if kind == 'standard':
            model = PCA(n_components=n_components)
            model.fit(returns)
            self.loadings = model.components_
            returns_new = model.transform(returns)

            if variance_explained is not None:
                var = model.explained_variance_ratio_.cumsum()
                mask = np.argwhere((var >= variance_explained))[0,0]
                returns_new = returns_new[:, 0:mask+1]
                self.loadings = self.loadings[:, 0:mask+1]
        elif kind == 'sparse':
            model = SparsePCA(n_components=n_components, alpha=sparsity_alpha)
            model.fit(returns)
            self.loadings = model.components_
            returns_new = model.transform(returns)

            if variance_explained is not None:
                warnings.warn('variance_explained argument has no impact on sparse components preselection!')
        elif kind == 'robust':
            if not rpy2_imported:
                raise ValueError(
                    'Robust PCA preselector requires R installation, which was not found!')

            # Check if additiveDEA package is available. Otherwise - attempt installing.
            try:
                rospca = rpackages.importr('rospca')
            except:
                print('Failed to import rospca R package. Trying to install.')
                try:
                    utils = rpackages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                    utils.install_packages('rospca')
                    rospca = rpackages.importr('rospca')
                except Exception as e:
                    raise ValueError(
                        f'Failed to install the additiveDEA R package. Exception - {e}. Please, make sure that there is R installed with additiveDEA package in it.')
            pandas2ri.activate()
            r_dataframe = pandas2ri.py2rpy_pandasdataframe(returns)
            model = rospca.rospca(r_dataframe, n_components, ndir=5000)
            self.loadings = model.rx2('loadings').T
            returns_new = model.rx2('scores')
        else:
            raise ValueError(f'Unknown kind - {kind} for ComponentsPreselector. ')

        returns_new = returns_new.T + mean
        returns_new[returns_new < 0] = 0

        return returns_new

    def reverse_transform(self, weights):
        asset_weights = weights @ self.loadings
        asset_weights[asset_weights<0.001] = 0
        asset_weights = asset_weights / asset_weights.sum()
        asset_weights[asset_weights<0.001] = 0
        asset_weights = asset_weights / asset_weights.sum()
        return asset_weights

    def select(self, returns):
        components_out = self.preselection(returns=returns, **self.preselector_kwargs)
        # plt.plot(components_out.T)
        # plt.show()
        return components_out

class NMFPreselector(ComponentsPreselector):

    def preselection(self, returns, n_components=30, sparsity=0.5, **kwargs):
        returns = returns.T
        model = NMF(n_components=n_components, alpha_H=sparsity, max_iter=1000)
        model.fit(returns)
        H = model.components_
        H = H[H.sum(axis=1)>0]
        self.loadings = (H.T / H.sum(axis=1)).T
        returns_new = (self.loadings @ returns.T)

        return np.array(returns_new)

    def reverse_transform(self, weights):
        asset_weights = weights @ self.loadings
        asset_weights = asset_weights / asset_weights.sum()
        return asset_weights
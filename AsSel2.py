### A module containing the models for Asset Preselection ###
import inspect
import pandas as pd
import Metrics
import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri

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

        return selected_assets


class DEA_AS(AssetPreselector):
    def preselection(self, inputs, outputs, DEA_kind='SBM'):
        '''
        Data Envelopment Analysis preselection.
        :param config: dict, DEA-specific settings
        :return: list of str, selected asset names
        '''
        dea = rpackages.importr('additiveDEA')
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
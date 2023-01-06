# A general purpose module for backtesting the complete strategies and collecting the results

import os
import plotly.graph_objects as go
from scipy.signal import find_peaks
import pickle

from Algoport.Data import Dataset
from Algoport.Utilities import *


class BackTest:
    def __init__(self,
                 strategy,
                 period_start,
                 period_end,
                 train_periods,
                 output_path,
                 preselection_profile=None,
                 view='SNP500',
                 frequency='daily',
                 filter_outliers=False,
                 plot=True):

        self.strategy = strategy
        self.period_start = period_start
        self.period_end = period_end
        self.train_periods = train_periods
        self.view = view
        self.frequency = frequency
        self.filter_outliers = filter_outliers
        self.output_path = output_path
        self.preselection_profile = preselection_profile
        self.plot = plot
        if self.preselection_profile is not None:
            self.preselection_profile = pd.Series(self.preselection_profile)


        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        # Figure out what start \ end format is provided
        if len(self.period_start.split('-')) == 2:
            self.period_start = week_to_date(self.period_start).strftime(format='%Y-%m-%d')

        if len(self.period_end.split('-')) == 2:
            self.period_end = week_to_date(self.period_end).strftime(format='%Y-%m-%d')

        self.dates = None
        self.data = None

        # Strategy results
        self.performance = None

        self.portfolio_returns = None
        self.portfolio_wealth = None
        self.asset_turnover = None
        self.weight_turnover = None

    def _filter_outliers(self, returns, method='assets'):
        def filter(arr):
            q75 = np.quantile(arr, 0.75)
            q25 = np.quantile(arr, 0.25)
            IQR = q75 - q25
            thr_u = q75 + 3 * IQR
            thr_l = q25 - 3 * IQR
            out = arr.copy()
            out[arr > thr_u] = thr_u
            out[arr < thr_l] = thr_l
            return out

        returns = returns.apply(lambda row: filter(row), axis=1)
        return returns

    def prepare_data(self):
        """

        :return:
        """
        DS = Dataset(name=self.view)
        if self.frequency == 'daily':
            original_start = (pd.to_datetime(self.period_start) - pd.Timedelta(days=self.train_periods)).strftime(format='%Y-%m-%d')
        elif self.frequency == 'weekly':
            original_start = (pd.to_datetime(self.period_start) - pd.Timedelta(weeks=self.train_periods)).strftime(format='%Y-%m-%d')
        else:
            raise ValueError(f'Unknown frequency -{self.frequency}. Available - daily and weekly')
        self.data = DS.fetch(start=original_start, end=self.period_end)
        self.dates = self.data.columns[self.data.columns >= self.period_start]

    def run(self):
        """
        Run the backtesting.
        """
        # Prepare the data.
        self.prepare_data()

        for date in self.dates:
            start = (pd.to_datetime(date) - pd.Timedelta(days=self.train_periods)).strftime(format='%Y-%m-%d')
            if self.preselection_profile is not None:
                assets = self.preselection_profile.loc[self.preselection_profile.index <= date].iloc[-1]
            else:
                assets = None
            train = self.data.loc[:, start:date].dropna()
            if self.filter_outliers:
                train = self._filter_outliers(returns=train)
            self.strategy.evaluate(returns=train, assets=assets)

        self.evaluate_performance(strategy=self.strategy)

    # METRICS AND CALCULATIONS
    def evaluate_performance(self, strategy, save_results=True):
        if self.performance is None:
            self.performance = {}
        weekly_performance = pd.DataFrame(self.dates, columns=['Datetime'])
        weekly_performance['return'] = self.calculate_portfolio_return(strategy)
        weekly_performance['wealth'] = weekly_performance['return'].cumprod()
        asset_turnover, weights_turnover = self.calculate_turnovers(strategy)
        weekly_performance['asset_turnover'] = asset_turnover
        weekly_performance['weights_turnover'] = weights_turnover
        weekly_performance['assets'] = strategy.assets
        weekly_performance['weights'] = strategy.weights

        preselection = {}
        optimization = {}
        if len(strategy.component_weights) != 0:
            try:
                preselection['component_weights'] = strategy.component_weights
            except:
                pass
        if len(strategy.optimizer.values) != 0:
            try:
                optimization['function_values'] = strategy.optimizer.values
            except:
                pass
        if len(strategy.optimizer.optimization_times) != 0:
            try:
                optimization['optimization_times'] = strategy.optimizer.optimization_times
            except:
                pass
        average_performance = self.calculate_metrics(performance=weekly_performance)

        self.performance = {'Weekly': weekly_performance,
                            'Average': average_performance,
                            'Optimization': optimization,
                            'Preselection': preselection}

        if save_results:
            self.save_results()


    def save_results(self, plot=True):
        '''
        Save the results of strategy backtesting in the folder with strategy name at self.output_path
        :return: None
        '''
        # Iterate the strategies for which results have been evaluated
        results = self.performance

        # Create the folder for each strategy if does not exist
        # todo include the overwrite \ don't overwrite options
        path = self.output_path + '\\'
        if not os.path.isdir(path):
            os.mkdir(path)
        # Save the whole evaluation metrics dictionary into pickle
        with open(path + 'History.pkl', 'wb') as f:
            pickle.dump(results, f)
        # Save the strategy config into the same folder
        # Make and save the plots
        if self.plot:
            df = results['Weekly']
            strategy_name = path.strip('\\').split('\\')[-1]
            self.plot_portfolio_returns(portfolio_returns=df['return'], name=strategy_name, save=True, output_path=path)
            self.plot_cumulative_wealth(portfolio_wealth=df['wealth'], name=strategy_name, save=True,
                                        output_path=path)
            self.plot_asset_turnover(asset_turnover=df['asset_turnover'], name=strategy_name, save=True,
                                        output_path=path)
            self.plot_weight_turnover(weight_turnover=df['weights_turnover'], name=strategy_name, save=True,
                                        output_path=path)

    def calculate_portfolio_return(self, strategy):
        """
        Calculate portfolio returns for each test week.
        :return: None, sets self.portfolio_returns
        """
        portfolio_returns = []
        for i, week in enumerate(self.dates):
            ret = self.data.loc[strategy.assets[i], week]
            port_ret = np.sum(ret * strategy.weights[i])
            portfolio_returns.append(port_ret)
        portfolio_returns = np.array(portfolio_returns)
        return portfolio_returns

    def calculate_turnovers(self, strategy):
        """
        Calculate asset and weight turnovers.
        Asset turnover is calculated as a % of assets that has been completely replaced between periods.
        Weights turnover is calculated as a % of portfolio which has been changed between periods.
        :return: None, sets  self.asset_turnover and self.weight_turnover
        """
        # Initialize the arrays
        asset_turnover = [0]
        weight_turnover = [0]
        # For each pair of periods t and t-1
        for i in range(1, len(strategy.assets)):
            # Find which assets were held for both periods.
            kept_assets, ind_new, ind_old = np.intersect1d(strategy.assets[i], strategy.assets[i - 1], assume_unique=True,
                                                           return_indices=True)
            # Calculate the share of the kept assets in t relative to t-1
            share_assets_kept = len(kept_assets) / len(strategy.assets[i - 1])
            # Respectively, calculate the share of assets that have been dropped
            asset_turnover.append(1 - share_assets_kept)
            # Calculate additional buys on the assets kept for both periods
            weights_redistribution = strategy.weights[i][ind_new] - strategy.weights[i - 1][ind_old]
            weights_redistribution = weights_redistribution[weights_redistribution > 0].sum()
            # Calculate the buys on the new assets in t
            weights_substitution = np.delete(strategy.weights[i], ind_new).sum()
            # Sum the two to get the weights turnover
            weight_turnover.append(weights_redistribution + weights_substitution)

        return asset_turnover, weight_turnover

    def calculate_mdd(self, portfolio_wealth):
        """
        Calculate the Maximum Draw Down (MDD), defined as the maximum % drop in portfolio wealth from peak to trough.
        :return: float, MDD.
        """
        peaks = find_peaks(portfolio_wealth)[0]
        troughts = find_peaks(portfolio_wealth)[0]
        mdd = 0
        for peak in peaks:
            trough = troughts[troughts > peak]
            if len(trough) == 0:
                trough = -1
            else:
                trough = trough[0]
            peak_val = portfolio_wealth[peak]
            trough_val = portfolio_wealth[trough]
            dd = (trough_val - peak_val) / peak_val
            if dd < mdd:
                mdd = dd
        return mdd

    def calculate_vars(self, portfolio_returns, quantile=0.05):
        """
        Calculate Value-at-Risk and Conditional-Value-at-Risk
        :return: (float, float), VaR and CVaR
        """
        thresh = np.quantile(portfolio_returns, quantile)
        var = 1 - thresh
        cvar = 1 - portfolio_returns[portfolio_returns < thresh].mean()

        return var, cvar


    def calculate_metrics(self, performance):
        """
        Calculate all the overall portfolio performance evaluation metrics.
        :return: None, sets the respective attributes.
        """
        performance_metrics = {}
        performance_metrics['portfolio_mean_return'] = np.mean(performance['return'])
        performance_metrics['portfolio_volatility'] = np.std(performance['return'])
        performance_metrics['portfolio_cum_wealth'] = performance['wealth'].iloc[-1]
        performance_metrics['portfolio_sharp'] = (performance_metrics['portfolio_mean_return']  - 1) / performance_metrics['portfolio_volatility']
        performance_metrics['portfolio_omega'] = np.sum(performance['return'] >= 1) / np.sum(performance['return'] < 1)
        performance_metrics['portfolio_mdd'] = self.calculate_mdd(np.array(performance['wealth']))
        performance_metrics['portfolio_var'], performance_metrics['portfolio_cvar'] = self.calculate_vars(portfolio_returns=np.array(performance['return']))
        performance_metrics['mean_asset_turnover'] = np.mean(performance['asset_turnover'])
        performance_metrics['mean_weight_turnover'] = np.mean(performance['weights_turnover'])

        return performance_metrics

    # PLOTTING
    def plot_portfolio_returns(self, portfolio_returns, name, show=True, save=False, output_path=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.dates, y=portfolio_returns,
                                 mode='lines+markers',
                                 name='lines+markers'))
        fig.update_layout(
            title={
                'text': f"Returns ({name})",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        if show:
            fig.show()
        if save:
            fig.write_html(output_path + "Returns.html")


    def plot_cumulative_wealth(self, portfolio_wealth, name, show=True, save=False, output_path=None):
        wealth = list(portfolio_wealth)
        weeks = pd.to_datetime(self.dates)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weeks, y=wealth,
                                 mode='lines+markers',
                                 name='lines+markers'))
        fig.update_layout(
            title={
                'text': f"Cumulative wealth ({name}). Final - {np.round(wealth[-1], 2)}",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        if show:
            fig.show()
        if save:
            fig.write_html(output_path + "Wealth.html")

    def plot_asset_turnover(self, asset_turnover, name, show=True, save=False, output_path=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.dates, y=asset_turnover,
                                 mode='lines+markers',
                                 name='lines+markers'))
        fig.update_layout(
            title={
                'text': f"Asset turnover ({name})",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        if show:
            fig.show()
        if save:
            fig.write_html(output_path + "Asset_turnover.html")

    def plot_weight_turnover(self, weight_turnover, name, show=True, save=False, output_path=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.dates, y=weight_turnover,
                                 mode='lines+markers',
                                 name='lines+markers'))
        fig.update_layout(
            title={
                'text': f"Weight turnover ({name})",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        if show:
            fig.show()
        if save:
            fig.write_html(output_path + "Weights_turnover.html")

    # def plot_weights(self):
    #     fig = go.Figure()
    #     for i, a in enumerate(self.assets):
    #         fig.add_trace(go.Scatter(x=self.weeks, y=self.weights,
    #                                  mode='lines+markers',
    #                                  name=a))
    #     fig.update_layout(
    #         title={
    #             'text': "Wealth allocation",
    #             'y': 0.9,
    #             'x': 0.5,
    #             'xanchor': 'center',
    #             'yanchor': 'top'})
    #     fig.show()

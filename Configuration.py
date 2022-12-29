from Strategy2 import *
from AGC import GARCH_EVT_COPULA
from Markov import MarkovChainProcess
from PortOpt2 import *
from AsSel2 import *
from Metrics import *

class StrategyConfigured:

    def __init__(self, name, T=5, preselection=True):
        self.available = ['MSG_Sharpe_ratio', 'MSG_Pearson_ratio', 'MSG_Omega_ratio', 'MSG_Stable_ratio',
                          'MSG_Sharpe_ratio_cond', 'MSG_Sharpe_ratio_ranking', 'MSG_Sharpe_ratio_R1', 'MSG_Sharpe_ratio_R2',
                          'AGC_Sharpe_ratio', 'AGC_MVO']

        if name in self.available:
            self.name = name
        else:
            raise ValueError(f'Unknown strategy name. Available - {self.available}')

        self.T = T

        self.preselection = preselection

    def prepare_preselector(self):
        if self.preselection:
            model = MarkovChainProcess
            metrics = [(cumulative_wealth, True, {'T': 50}),
                       (returns_sd, False, {}),
                       (kendall_corr_usb, True, {}),
                       (kendall_corr_lsb, False, {}),
                       (mean_return, True, {})
                       ]
            model_metrics = [('MSG_wealth_mean', True, {'T': self.T}),
                             ('MSG_time_to_lose', True, {'T': self.T}),
                             ('MSG_square_root_utility', True, {'T': self.T}),
                             ('MSG_stable_location', True, {'T': self.T}),
                             ('MSG_wealth_sd', False, {'T': self.T}),
                             ('MSG_time_to_gain', False, {'T': self.T}),
                             ('MSG_CVaR_log_wealth', False, {'T': self.T}),
                             ]
            model_kwargs = {'init': {},
                            'fit': {'N': 9,
                                    }}
            if 'ranking' not in self.name:
                preselector_kwargs = {}

                preselector = DEA_AS(model=model,
                                     metrics=metrics,
                                     model_metrics=model_metrics,
                                     model_kwargs=model_kwargs,
                                     preselector_kwargs=preselector_kwargs)
            else:
                preselector_kwargs = {'kind': 'Fixed',
                                      'n_assets': 30}
                preselector = Ranking_AS(model=model,
                                         metrics=metrics,
                                         model_metrics=model_metrics,
                                         model_kwargs=model_kwargs,
                                         preselector_kwargs=preselector_kwargs)

        else:
            preselector = None

        return preselector

    def prepare_optimizer(self):
        if 'MSG' in self.name:
            model = MarkovChainProcess

            names = ['MSG_Sharpe_ratio', 'MSG_Pearson_ratio', 'MSG_Omega_ratio', 'MSG_Stable_ratio',
                     'MSG_Sharpe_ratio_cond', 'MSG_Sharpe_ratio_ranking', 'MSG_Sharpe_ratio_R1', 'MSG_Sharpe_ratio_R2']

            metrics_prep = [('MSG_sharpe_ratio', True, {'T': self.T}),
                            ('MSG_corr_ratio', True, {'T': self.T}),
                            ('MSG_omega_ratio', True, {'T': self.T}),
                            ('MSG_stable_ratio', True, {'T': self.T}),
                            ('MSG_sharpe_ratio', True, {'T': self.T}),
                            ('MSG_sharpe_ratio', True, {'T': self.T}),
                            ('MSG_sharpe_ratio', True, {'T': self.T}),
                            ('MSG_sharpe_ratio', True, {'T': self.T}),
                            ]

            metrics_opt = dict(zip(names, metrics_prep))

            if self.name == 'MSG_Sharpe_ratio_cond':
                model_kwargs = {'init': {},
                                'fit': {'N': 9,
                                        'transaction_cost': 0,
                                        'unconditional_start': False
                                        }}
            elif self.name == 'MSG_Sharpe_ratio_R1':
                model_kwargs = {'init': {},
                                'fit': {'N': 9,
                                        'transaction_cost': 0.1,
                                        'unconditional_start': True
                                        }}
            else:
                model_kwargs = {'init': {},
                                'fit': {'N': 9,
                                        'transaction_cost': 0,
                                        'unconditional_start': True
                                        }}

            metric = None
            model_metric = metrics_opt[self.name]

        elif 'AGC' in self.name:
            model = GARCH_EVT_COPULA

            names = ['AGC_Sharpe_ratio', 'AGC_MVO']
            metrics_prep = [(sharpe_ratio, True, {}),
                            None]

            metrics = dict(zip(names, metrics_prep))
            metric = metrics[self.name]
            model_metric = None

            model_kwargs = {'init': {},
                            'fit': {'sample_draws': 1000
                                    }}
        else:
            raise ValueError('Could not set up the optimizer! Strategy name missing MSG or AGC')

        if 'MVO' not in self.name:
            optimizer = SimplexOptimization(model=model,
                                            metric=metric,
                                            model_metric=model_metric,
                                            model_kwargs=model_kwargs,
                                            optimizer_kwargs={"m": 10, "p": 1})
        else:
            optimizer = MVOptimization(model=model,
                                       metric=metric,
                                       model_metric=model_metric,
                                       model_kwargs=model_kwargs,
                                       optimizer_kwargs={"alpha": 0.05})

        return optimizer

    def fetch(self):
        preselector = self.prepare_preselector()
        optimizer = self.prepare_optimizer()

        if self.name == 'MSG_Sharpe_ratio_R2':
            strategy = StrategySmoothed(preselector=preselector,
                                        optimizer=optimizer,
                                        regulation_kwargs={'redistribute_every': self.T, 'smoothing_delta': 0.5})
        else:
            strategy = Strategy(preselector=preselector,
                                optimizer=optimizer,
                                regulation_kwargs={'redistribute_every': self.T, 'maintain_weights': True})

        return strategy
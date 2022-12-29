import numpy as np
from arch.univariate import ARX, GARCH, Normal
from copulas.multivariate import GaussianMultivariate
import pandas as pd

class GARCH_EVT_COPULA:

    def __init__(self):
        self.garches = None
        self.garch_residuals = None
        self.garch_volatilities = None
        self.garch_params = None
        self.copula = None

        # Will be filled with forecasted returns
        self.returns = None

    def fit_garch(self, returns):
        garch_params = None
        garch_residuals = np.zeros(returns.shape)
        garch_volatilities = np.zeros(returns.shape)
        for i in range(returns.shape[0]):
            am = ARX(returns[i, :], lags=[1])
            am.volatility = GARCH(1, 0, 1)
            am.distribution = Normal()
            garch_fitted = am.fit(disp='off')
            garch_residuals[i, :] = garch_fitted.resid
            garch_volatilities[i, :] = garch_fitted.conditional_volatility
            if garch_params is None:
                garch_params = pd.DataFrame(garch_fitted.params).T
            else:
                garch_params = pd.concat((garch_params, pd.DataFrame(garch_fitted.params).T), ignore_index=True)

        self.garch_residuals = garch_residuals[:, 1:]
        self.garch_volatilities = garch_volatilities[:, 1:]
        self.garch_params = garch_params.reset_index(drop=True)

    def fit_copula(self):
        self.copula = GaussianMultivariate()
        self.copula.fit(self.garch_residuals.T)

    def garch_forecast(self, prev_r, prev_s, eps, const, y, omega, alpha, beta):
        sigma_sq = omega + alpha * eps ** 2 + beta * prev_s ** 2
        eps_predicted = np.sqrt(sigma_sq) * eps
        pred = const + y * prev_r + eps_predicted
        return pred

    def fit(self, returns, sample_draws=1000):
        if isinstance(returns, pd.DataFrame):
            returns = np.array(returns)

        returns = returns * 1000

        n_assets = len(returns)

        self.fit_garch(returns=returns)
        self.fit_copula()

        eps_samples = self.copula.sample(sample_draws)
        last_returns = returns[:, -1]
        last_volatilities = self.garch_volatilities[:, -1]
        omegas = np.array(self.garch_params['omega'])
        constants = np.array(self.garch_params['Const'])
        ys = np.array(self.garch_params['y[1]'])
        alphas = np.array(self.garch_params['alpha[1]'])
        betas = np.array(self.garch_params['beta[1]'])

        forecasts = np.zeros((n_assets, sample_draws))
        for i in range(n_assets):
            forecasts[i, :] = self.garch_forecast(prev_r=last_returns[i],
                                                  prev_s=last_volatilities[i],
                                                  eps=np.array(eps_samples[i]),
                                                  omega=omegas[i],
                                                  alpha=alphas[i],
                                                  beta=betas[i],
                                                  const=constants[i],
                                                  y=ys[i])

        forecasts = forecasts / 1000

        self.returns = forecasts

        return forecasts
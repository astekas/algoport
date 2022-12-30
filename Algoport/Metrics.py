import numpy as np
from scipy.stats import kendalltau

### SIMPLE METRICS ###
def returns_sd(returns):
    '''

    :param returns:
    :return:
    '''
    return returns.std(axis=1)

def kendall_corr_lsb(returns):
    '''

    :param returns:
    :return:
    '''
    lower_stoch_bound = returns.min(axis=0)
    kend_corr_lsb = np.array([kendalltau(row, lower_stoch_bound)[0] for i, row in returns.iterrows()])

    return kend_corr_lsb

def cumulative_wealth(returns, T=0):
    '''

    :param prices:
    :param T:
    :return:
    '''
    if T is None:
        T = 0
    else:
        T = len(returns) - T
        if T < 0:
            T = 0

    cum_wealth = returns.iloc[:, T:].cumprod(axis=1).iloc[:, -1]
    return cum_wealth

def mean_return(returns):
    '''

    :param returns:
    :return:
    '''
    return_mean = returns.mean(axis=1)
    return return_mean

def kendall_corr_usb(returns):
    upper_stoch_bound = returns.max(axis=0)
    kend_corr_usb = np.array([kendalltau(row, upper_stoch_bound)[0] for i, row in returns.iterrows()])
    return kend_corr_usb

def sharpe_ratio(returns):
    SR = ((returns.mean() - 1) / returns.std()).iloc[0]
    return SR
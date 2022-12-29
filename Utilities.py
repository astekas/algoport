### A collection of universally useful functions and classes ###
### EXTERNAL IMPORTS ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product
from scipy.stats import kendalltau
import json

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

def cumulative_wealth(close, T=None):
    '''

    :param prices:
    :param T:
    :return:
    '''
    if T is None:
        T = 0
    else:
        T = len(close) - T
        if T < 0:
            T = 0

    cum_wealth = close.iloc[:, -1] / close.iloc[:, T]
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

### SPECIAL CASE CLASSES ###
class MarkovChainProcess:
    def __init__(self, n_states, dim='uni'):
        self.n_states = n_states
        self.dim = dim
        self.unconditional_returns_prob = None
        self.u = None
        self.transition_matrix = None
        self.states = None
        self.last_state = None

    def fit(self, arr):
        z_min = arr.min()
        z_max = arr.max()
        N = self.n_states
        self.u = (z_max / z_min) ** (1 / N)
        a = [((z_min / z_max) ** (i / N)) * z_max for i in range(N + 1)]
        a = np.flip(np.sort(a))

        z_i = [np.sqrt(a[i] * a[i - 1]) for i in range(1, N + 1)]
        self.states = z_i

        counts = np.zeros((len(arr), self.n_states))
        for i in range(1, len(a)):
            if i == 1:
                counts[:, i - 1] = arr >= a[i]
            elif i == len(z_i) - 1:
                counts[:, i - 1] = arr <= a[i]
            else:
                counts[:, i - 1] = (arr < a[i - 1]) * (arr >= a[i])

        self.unconditional_returns_prob = counts.sum(axis=0) / counts.sum()

        transitions_flow = sliding_window_view(np.where(counts == 1)[1], 2)
        self.last_state = int(transitions_flow[-1, -1])
        trans_indices, counts = np.unique(transitions_flow, return_counts=True, axis=0)

        TM = np.zeros((self.n_states, self.n_states))
        TM[trans_indices[:, 0], trans_indices[:, 1]] = counts
        self.transition_matrix = np.apply_along_axis(lambda arr: np.divide(arr, arr.sum(), where=arr.sum() != 0), 1, TM)

    def fit_bi(self, arr1, arr2):
        # todo change initial calculation to 2d form.
        z_min1 = arr1.min()
        z_max1 = arr1.max()
        z_min2 = arr2.min()
        z_max2 = arr2.max()
        N = self.n_states
        self.u = ((z_max1 / z_min1) ** (1 / N), (z_max2 / z_min2) ** (1 / N))

        a = [((z_min1 / z_max1) ** (i / N)) * z_max1 for i in range(N + 1)]
        a = np.flip(np.sort(a))
        b = [((z_min2 / z_max2) ** (i / N)) * z_max2 for i in range(N + 1)]
        b = np.flip(np.sort(b))

        #         intervals_2d = list(product(sliding_window_view(a, 2), sliding_window_view(b, 2)))
        intervals_a = sliding_window_view(a, 2)
        print(intervals_a.shape)
        intervals_b = sliding_window_view(b, 2)

        intervals = np.zeros((N, N, 4))
        for i, i_val in enumerate(intervals_a):
            for j, j_val in enumerate(intervals_b):
                intervals[i, j, :] = [i_val[0], i_val[1], j_val[0], j_val[1]]

        z_i_x = [np.sqrt(a[i] * a[i - 1]) for i in range(1, N + 1)]
        z_i_y = [np.sqrt(b[i] * b[i - 1]) for i in range(1, N + 1)]

        #         self.states = list(product(z_i_x, z_i_y))
        self.states = np.zeros((N, N, 2))
        for x, val_x in enumerate(z_i_x):
            for y, val_y in enumerate(z_i_y):
                self.states[x, y, :] = [val_x, val_y]

        arr = np.vstack((arr1, arr2))
        #         counts = np.zeros((len(arr1), len(intervals_2d)))
        #         for i in range(1, len(intervals_2d)+1):
        #             counts[:, i-1] = np.apply_along_axis(self.compare_2d, 0, arr, intervals_2d[i-1])
        counts = np.zeros((list(intervals.shape)[:-1] + [arr.shape[1]]))
        print('COUNTS: ', counts[0, 0], counts[0, 0].shape)
        for i in range(arr.shape[1]):
            position = np.apply_along_axis(lambda x: np.all((arr[:, i] >= x[[1, 3]]) & (arr[:, i] <= x[[0, 2]])), 2,
                                           intervals)
            #             np.apply_along_axis(lambda x: print(x, arr[:, i]), 2, intervals)
            #             print(f'POSITION: sum - {position.sum()}, dim - {position.shape}')
            fill = np.zeros(arr.shape[1])
            fill[i] = 1
            counts[position] = counts[position] + fill
        #             print(counts[position])

        print('RESULT: ', counts.sum(), counts.shape, len(arr1))
        #         print(counts)
        self.unconditional_returns_prob = counts.sum(axis=2) / counts.sum()

        i, j, t = np.where(counts == 1)
        order = [np.where(t == o)[0][0] for o in np.sort(t)]

        transitions_flow = sliding_window_view(np.vstack((i[order], j[order])).T, (2, 2)).squeeze()
        print(transitions_flow, transitions_flow.shape)
        self.last_state = transitions_flow[-1][-1, :]

        trans_indices, counts = np.unique(transitions_flow, return_counts=True, axis=0)
        print(trans_indices, trans_indices.shape)
        print(counts)
        TM = np.zeros((N, N, N, N))

        TM[trans_indices[:, 0, 0], trans_indices[:, 0, 1], trans_indices[:, 1, 0], trans_indices[:, 1, 1]] = counts
        print('SUM TM ', TM.sum())
        TM_sum = TM.sum(axis=(2, 3))
        TM_sum[TM_sum == 0] = 1
        trans = np.zeros(TM.shape)
        for i in range(N):
            for j in range(N):
                trans[i, j, :, :] = TM[i, j, :, :] / TM_sum[i, j]
        self.transition_matrix = trans

    #         self.transition_matrix = np.apply_along_axis(lambda arr: if arr.sum()!=0: arr / arr.sum(), 1, TM)

    def simulate(self, n_steps):
        if self.transition_matrix is None:
            raise ValueError('There is no transition matrix fitted yet. Use fit() method first.')

        wealth = np.zeros(n_steps + 1)
        wealth[0] = 1

        states = np.zeros(n_steps + 1)
        states[0] = self.last_state

        for step in range(1, n_steps + 1):
            previous_state = states[step - 1]
            new_state = \
            np.argwhere(np.random.multinomial(1, self.transition_matrix[int(previous_state), :], size=1)[0] == 1)[0, 0]
            new_return = self.states[new_state]
            new_wealth = wealth[step - 1] * new_return
            states[step] = new_state
            wealth[step] = new_wealth

        plt.plot(wealth)
        return wealth

    def diagM(self, arr):
        if len(arr.shape) == 1:
            n = len(arr)
            m = 1
        else:
            m, n = arr.shape
        out = np.zeros((m + n - 1, n))
        for i in range(n):
            if m == 1:
                out[i:i + m, i] = arr[i]
            else:
                out[i:i + m, i] = arr[:, i]
        return out

    def compare_2d(self, arr, interval):
        interval = np.array(interval)
        #         print(interval, arr)
        lower_bound = np.apply_along_axis(lambda sl: sl <= arr, 0, interval)[:, 1]
        upper_bound = np.apply_along_axis(lambda sl: sl >= arr, 0, interval)[:, 0]
        #         print(lower_bound, upper_bound)
        evaluation = np.hstack((lower_bound, upper_bound))
        if evaluation.sum() == 4:
            return True
        else:
            return False

    def compute_Q(self, steps):
        Q_prev = self.unconditional_returns_prob
        for s in range(1, steps + 1):
            #             if self.dim == 'uni':
            Q_k = self.diagM(Q_prev @ self.transition_matrix)
            #             elif self.dim == 'bi':
            #                 Q_k = self.transition_matrix @ Q_prev.T
            Q_prev = Q_k
        return Q_prev

    def compute_wealth_distribution(self, steps):
        Q_k = self.compute_Q(steps=steps)
        if self.dim == 'uni':
            identity = np.ones(self.n_states)
        elif self.dim == 'bi':
            identity = np.ones(self.n_states ** 2)
        return Q_k @ identity

    def compute_wealth(self, steps):
        if self.dim == 'uni':
            wealth = np.array(
                [(self.states[0] ** steps) * (self.u ** (1 - i)) for i in range(1, (self.n_states - 1) * steps + 2)])
            return wealth
        elif self.dim == 'bi':
            wealth1 = np.array([(self.states[0][0] ** steps) * (self.u[0] ** (1 - i)) for i in
                                range(1, (self.n_states - 1) * steps + 2)])
            wealth2 = np.array([(self.states[0][1] ** steps) * (self.u[1] ** (1 - i)) for i in
                                range(1, (self.n_states - 1) * steps + 2)])
            return np.array(list(product(wealth1, wealth2)))

    def expected_wealth(self, k):
        probabilities = self.compute_wealth_distribution(steps=k)
        wealths = self.compute_wealth(steps=k)
        return probabilities @ wealths

    def time_distribution(self, k, gain=True, thresh=1.05):
        probabilities = []
        Q_prev = self.unconditional_returns_prob
        for step in range(1, k + 1):
            Q_k = self.diagM(Q_prev @ self.transition_matrix)
            #             print(Q_k)
            wealth_k = self.compute_wealth(step)
            #             print(wealth_k)
            if gain:
                y = wealth_k >= thresh
            else:
                y = wealth_k <= thresh
            #             print(y)
            if step == k:
                probabilities.append(Q_k.sum())
            else:
                probabilities.append(Q_k[y].sum())
            Q_k[y] = 0
            Q_prev = Q_k

        return np.array(probabilities)

    def MSG_time_to_gain(self, T, thresh=1.05):
        steps = np.arange(1, T+1)
        distribution = self.time_distribution(k=T, thresh=thresh)
        return np.sum(steps * distribution)

    def MSG_time_to_lose(self, T, thresh=0.95):
        steps = np.arange(1, T+1)
        distribution = self.time_distribution(k=T, gain=False, thresh=thresh)
        return np.sum(steps * distribution)

    def MSG_wealth_sd(self, T):
        # todo figure out how if this implementation is correct
        wealth = self.compute_wealth(steps=T)
        distribution = self.compute_wealth_distribution(steps=T)
        mean = np.sum(wealth*distribution)
        return np.sqrt(((wealth - mean) ** 2) * distribution)

    def MSG_wealth_mean(self, T):
        wealth = self.compute_wealth(steps=T)
        distribution = self.compute_wealth_distribution(steps=T)
        return np.sum(wealth*distribution)


### Convenience functions
def parse_config(path):
    '''
    Parse strategy configuration from json file.
    :param path: str, path to .json config file
    :return: dict
    '''
    with open(path, "r") as jsonfile:
        data = json.load(jsonfile)
    return data

def save_config(config, path, name):
    with open(path + name, "w") as jsonfile:
        json.dump(config, jsonfile)

def findAll(key, dic, storage, unnest=True):
    for k, v in dic.items():
        if k == key and isinstance(v, list) and unnest:
            for val in v:
                if val not in storage:
                    storage.append(val)
        elif k == key and v not in storage:
            storage.append(v)
        elif isinstance(v, dict):
            findAll(key, v, storage)
        else:
            continue

def week_to_date(week):
    return pd.to_datetime(week + '5', format='%Y-%W%w')

def date_to_week(date):
    return pd.to_datetime(date, format='%Y-%m-%d').strftime(format='%Y-%W')
#
# def subtract_weeks(week, to_subtract):
#     year, week = week.split('-')
#     year = int(year)
#     week = int(week)
#     years_sub = math.floor(to_subtract / 53)
#     weeks_sub = to_subtract % 53
#     if weeks_sub > week:
#         year_new = year - years_sub - 1
#         week_new = 53 + week - weeks_sub
#     else:
#         year_new = year - years_sub
#         week_new = week - weeks_sub
#
#     return str(year_new) + '-' + str(week_new)
#
# def add_weeks(week, to_add):
#     year, week = week.split('-')
#     year = int(year)
#     week = int(week)
#     years_sub = math.floor(to_add / 53)
#     weeks_sub = to_add % 53
#     if week + weeks_sub > 53:
#         year_new = year + years_sub + 1
#         week_new = week + weeks_sub - 53
#     else:
#         year_new = year + years_sub
#         week_new = week + weeks_sub
#
#     return str(year_new) + '-' + str(week_new)
#
# def weeks_diff(week1, week2):
#     year1, week1 = week1.split('-')
#     year2, week2 = week2.split('-')
#     year1 = int(year1)
#     week1 = int(week1)
#     year2 = int(year2)
#     week2 = int(week2)
#
#
#     total_weeks1 = year1 * 53 + week1
#     total_weeks2 = year2 * 53 + week2
#
#     return total_weeks1 - total_weeks2
#
#


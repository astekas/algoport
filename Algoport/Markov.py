import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import integrate, LowLevelCallable
from scipy.stats import levy_stable
from itertools import product
import pandas as pd
import pickle
import warnings

try:
    from Algoport import cdf_2
    cdf_2_imported = True
except:
    cdf_2_imported = False
    warnings.warn('Could not import the C version of integrand for stable CDF. '
                  'This will result in slowed down performance for Omega Ratio. '
                  'Currently the C version is only prebuilt for Python 3.8 environment.')

class MarkovChainProcess:
    def __init__(self):
        """
        Implementation of the Markov Chain Process model of the returns.
        For efficiency reasons, the estimation is performed for a set of assets simultaneously, thus the single
        object estimates N univariate processes at once.
        """
        # Number of states in the process. Integer for univariate case, tuple for bivariate
        self.n_states = None
        # Dimentionality of the process. 1 for univariate, 2 for bivariate
        self.dim = None
        # Unconditional probabilities for the states
        self.unconditional_returns_prob = None
        # Multiplier for future wealth computation
        self.u = None
        # Transition matrix of the process
        self.transition_matrix = None
        # Approximated return values of the states
        self.states = None
        # Last state in the data used for estimation
        self.last_state = None
        # Initial data passed to self.fit. A 2d array of NxM shape, where N is the number of assets and M is the number
        # of observations.
        self.data = None
        # Portfolio weights passed in the fit call
        self.weights = None
        # Initial wealth value used for evaluation of the future wealth of the Markov process
        self.W0 = 1
        # Whether the process begins with unconditional probabilities or conditional ones.
        # Purely for passing into the internal fit_bi
        self.unconditional = True

    def adjust_W0(self, current_weights, new_weights, transaction_cost):
        """
        Adjusts the initial wealth based on the proportion of portfolio that is to be redistributed.
        :param current_weights:
        :param new_weights:
        :param transaction_cost:
        :return:
        """
        sales = new_weights - current_weights
        fraction_sold = (1 - current_weights.sum()) - sales[sales<0].sum()
        self.W0 = 1 - fraction_sold * transaction_cost

    def fit(self, arr, N, weights=None, current_weights=None, transaction_cost=0, unconditional_start=True):
        """
        Fit the Markov Chain Process either for all the assets in arr or for a portfolio constructed by weights
        :param arr: np.array of shape (N, M), where N is the number of assets and M is the number of observations
        :param N: int, number of states of the process.
        :param weights: 1d np.array of length N. Portfolio weights of the assets.
        :param current_weights: 1d np.array of length N. Previous portfolio weights for the same assets.
        :param transaction_cost: float, % of the wealth that is being lost on sale of the assets.
        :return: None, fills in the attributes initialized in self.__init__
        """
        if current_weights is not None:
            self.adjust_W0(current_weights=current_weights, new_weights=weights, transaction_cost=transaction_cost)
        if isinstance(arr, pd.DataFrame):
            arr = np.array(arr)
        if isinstance(arr, pd.Series):
            arr = np.array(arr)
        # Write the N states
        self.n_states = N
        # The initial dimentionality is always 1, bivariate processes are only used recursively
        self.dim = 1
        # Write the weights
        self.weights = weights
        # Write the data
        self.data = arr
        # If weights are provided - construct the portfolio series.
        if weights is not None:
            arr = weights @ arr
        if len(arr.shape) == 1:
            arr = arr.reshape((1, len(arr)))
        # Find the boundaries of observed process
        z_min = arr.min(axis=1)
        z_max = arr.max(axis=1)
        # Compute the multiplier
        self.u = (z_max / z_min) ** (1 / N)
        # Estimate intervals a for the process
        a = [((z_min / z_max) ** (i / N)) * z_max for i in range(N + 1)]
        a = np.apply_along_axis(lambda x: np.flip(np.sort(x)), 0, a)
        # Estimate the states as a geometric averages of interval boundaries
        z_i = np.array([np.sqrt(a[i] * a[i - 1]) for i in range(1, N + 1)])
        self.states = z_i.T
        # Count the number of times process(es) were in each state
        counts = np.zeros((arr.shape[0], arr.shape[1], self.n_states))
        for i in range(1, len(a)):
            if i == 1:
                counts[:, :, i - 1] = np.apply_along_axis(lambda x: x >= a[i, :], 0, arr)
            elif i == len(a) - 1:
                counts[:, :, i - 1] = np.apply_along_axis(lambda x: x <= a[i - 1, :], 0, arr)
            else:
                counts[:, :, i - 1] = np.apply_along_axis(lambda x: (x < a[i - 1, :]) * (x >= a[i, :]), 0, arr)
        # Construct the flow of the states, i.e. ordered array of tuples (previous state, state)
        transitions_flow = np.array(
            [sliding_window_view(np.where(counts[i] == 1)[1], 2) for i in range(counts.shape[0])])
        # Get the last state for each asset
        self.last_state = transitions_flow[:, -1, -1]
        # Count the transitions between states
        prep = [np.unique(transitions_flow[i], return_counts=True, axis=0) for i in range(transitions_flow.shape[0])]
        # Estimate the transition matrix(es)
        TM = np.zeros((arr.shape[0], self.n_states, self.n_states))
        for i in range(len(prep)):
            TM[i, prep[i][0][:, 0], prep[i][0][:, 1]] = prep[i][1]
        self.transition_matrix = np.array(
            [np.apply_along_axis(lambda arr: np.divide(arr, arr.sum(), where=arr.sum() != 0), 1, TM[i]) for i in
             range(TM.shape[0])])
        # Estimate the unconditional probabilities as number of times a given state is observed over the total
        # number of observations
        if unconditional_start:
            self.unconditional = True
            self.unconditional_returns_prob = np.apply_along_axis(lambda x: x / counts.sum(axis=(1, 2)), 0,
                                                                  counts.sum(axis=1))
        else:
            self.unconditional = False
            self.unconditional_returns_prob = self.transition_matrix[:, self.last_state, :]
            self.unconditional_returns_prob = self.unconditional_returns_prob.reshape((self.unconditional_returns_prob.shape[0], self.unconditional_returns_prob.shape[2]))

    def fit_bi(self, arr1, arr2, N, M, weights=None, unconditional_start=True):
        """
        Fit bivariate Markov Chain Process.
        :param arr1: 1d np.array of observations for the first asset or 2d array of all assets (then, weights shall be provided)
        :param arr2: 1d np.array of observations for the second asset
        :param N: number of states for the first asset
        :param M: number of states for the second asset
        :param weights:
        :return:
        """
        self.dim = 2
        self.n_states = (N, M)

        if weights is not None:
            arr1 = weights @ arr1

        # todo change initial calculation to 2d form.
        z_min1 = arr1.min()
        z_max1 = arr1.max()
        z_min2 = arr2.min()
        z_max2 = arr2.max()

        self.u = ((z_max1 / z_min1) ** (1 / N), (z_max2 / z_min2) ** (1 / M))

        a = [((z_min1 / z_max1) ** (i / N)) * z_max1 for i in range(N + 1)]
        a = np.flip(np.sort(a))
        b = [((z_min2 / z_max2) ** (i / M)) * z_max2 for i in range(M + 1)]
        b = np.flip(np.sort(b))

        intervals_a = sliding_window_view(a, 2)
        intervals_b = sliding_window_view(b, 2)

        intervals = np.zeros((N, M, 4))
        for i, i_val in enumerate(intervals_a):
            for j, j_val in enumerate(intervals_b):
                intervals[i, j, :] = [i_val[0], i_val[1], j_val[0], j_val[1]]

        z_i_x = [np.sqrt(a[i] * a[i - 1]) for i in range(1, N + 1)]
        z_i_y = [np.sqrt(b[i] * b[i - 1]) for i in range(1, M + 1)]

        self.states = np.zeros((N, M, 2))
        for x, val_x in enumerate(z_i_x):
            for y, val_y in enumerate(z_i_y):
                self.states[x, y, :] = [val_x, val_y]

        arr = np.vstack((arr1, arr2))

        counts = np.zeros((list(intervals.shape)[:-1] + [arr.shape[1]]))

        for i_x in range(N):
            for i_y in range(M):
                counts[i_x, i_y, :] = np.all(
                    (arr.T >= intervals[i_x, i_y, [1, 3]]) * (arr.T <= intervals[i_x, i_y, [0, 2]]), axis=1)

        i, j, t = np.where(counts == 1)
        order = [np.where(t == o)[0][0] for o in np.sort(t)]

        transitions_flow = sliding_window_view(np.vstack((i[order], j[order])).T, (2, 2)).squeeze()
        self.last_state = transitions_flow[-1][-1, :]
        trans_indices, counts_ = np.unique(transitions_flow, return_counts=True, axis=0)
        TM = np.zeros((N, M, N, M))

        TM[trans_indices[:, 0, 0], trans_indices[:, 0, 1], trans_indices[:, 1, 0], trans_indices[:, 1, 1]] = counts_
        TM_sum = TM.sum(axis=(2, 3))
        TM_sum[TM_sum == 0] = 1
        trans = np.zeros(TM.shape)
        for i in range(N):
            for j in range(M):
                trans[i, j, :, :] = TM[i, j, :, :] / TM_sum[i, j]
        self.transition_matrix = trans
        if unconditional_start:
            self.unconditional_returns_prob = counts.sum(axis=2) / counts.sum()
        else:
            self.unconditional_returns_prob = self.transition_matrix[self.last_state[0], self.last_state[1], :, :]

    def expected_wealth(self, T):
        """
        Compute expected wealth at time T.
        :param T: int, number of steps in the future for which expected wealth shall be computed.
        :return: float, expected wealth
        """
        if self.dim == 1:
            probabilities = self.compute_wealth_distribution(steps=T).squeeze()
            wealths = self.compute_wealth(steps=T).squeeze()
            return probabilities @ wealths
        elif self.dim == 2:
            probabilities = self.compute_wealth_distribution(steps=T)
            probabilities = np.stack((probabilities, probabilities), axis=2)
            W = self.compute_wealth(steps=T)
            EW = (probabilities * W).sum(axis=(0,1))
            return EW

    def simulate(self, n_steps):
        """
        Simulate realization Markov Chain Process for n_steps into the future.
        :param n_steps: int,
        :return: None, plots the simulation
        """
        #todo review
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
        """
        Operation shifting the diagonal of the array
        :param arr:
        :return:
        """
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

    def compute_Q(self, steps, TM, p0):
        """
        Iteratively compute the future Q 2d matrix of NxM, where N is number of possible wealth values,
        M is the number of states, Q(n,m) is the probability to be in m'th state while having obtained n'th wealth
        by this point
        :param steps: int, steps in the future
        :param TM: 2d np.array of shape (MxM), transition matrix
        :param p0: 1d np.array, initial
        :return:
        """
        Q_prev = p0
        for s in range(1, steps + 1):
            Q_k = self.diagM(Q_prev @ TM)
            Q_prev = Q_k
        return Q_prev

    def compute_wealth_distribution(self, steps):
        """
        Estimate multinomial probability distribution of future wealth
        :param steps: int, T steps in the future
        :return: np.array, 1d for univariate process, 2d for bivariate
        """
        if self.dim == 1:
            Qs = []
            for p0, TM in zip(self.unconditional_returns_prob, self.transition_matrix):
                Qs.append(self.compute_Q(steps=steps, p0=p0, TM=TM))
            identity = np.ones(self.n_states)
            return np.array([Q @ identity for Q in Qs])
        elif self.dim == 2:
            p_prev = self.unconditional_returns_prob
            p_prev = p_prev.reshape(1, 1, p_prev.shape[0], p_prev.shape[1])
            N = self.n_states[0]
            M = self.n_states[1]
            i_x = np.arange(N)
            i_y = np.arange(M)
            i = np.array(list(product(i_x, i_y)))
            for t in range(1, steps+1):
                upper_boundary = np.array(p_prev.shape[0:2])
                d_N = 1 + t * (N - 1)
                d_M = 1 + t * (M - 1)
                p = np.zeros((d_N, d_M, N, M))

                l_x = np.arange(d_N)
                l_y = np.arange(d_M)

                l = np.array(list(product(l_x, l_y)))
                l_full = np.repeat(l[:, np.newaxis, :], len(i), axis=1)
                i_full = np.resize(i, (l.shape[0], i.shape[0], 2))
                dif = l_full - i_full
                mask = np.all((dif >= 0) & (dif < upper_boundary), axis=2)
                l_h = dif[mask]
                h = i_full[mask]
                init = p_prev[l_h[:, 0], l_h[:, 1], h[:, 0], h[:, 1]]

                rez = np.repeat(init[np.newaxis, :], N * M, axis=1).reshape((len(init), N, M)) * self.transition_matrix[
                                                                                                 h[:, 0], h[:, 1], :, :]
                order = mask.sum(axis=1)
                cord = np.cumsum(order)

                rez_new = np.array([np.sum(x, axis=0) for x in np.split(rez, cord)[:-1]])
                p[l[:, 0], l[:, 1], :, :] = rez_new
                p_prev = p

            return p.sum(axis=(2,3))

    def compute_wealth(self, steps, asset=None):
        if self.dim == 1:
            if asset is None:
                wealth = np.array(
                    [[self.W0 * (state[0] ** steps) * (u ** (1 - i)) for i in range(1, (self.n_states - 1) * steps + 2)] for
                     state, u in zip(self.states, self.u)])
                return wealth
            else:
                wealth = np.array([self.W0 * (self.states[asset, 0] ** steps) * (self.u[asset] ** (1 - i)) for i in
                                   range(1, (self.n_states - 1) * steps + 2)])
                return wealth
        elif self.dim == 2:
            N = self.n_states[0]
            M = self.n_states[1]
            W_x = 1 + (N - 1) * steps
            W_y = 1 + (M - 1) * steps
            W = np.zeros((W_x, W_y, 2))
            z = self.states[0, 0]
            for l_x in range(W_x):
                for l_y in range(W_y):
                    W[l_x, l_y] = self.W0 * (z ** steps) * np.array([self.u[0] ** (0 - l_x), self.u[1] ** (0 - l_y)])
            return W

    def time_distribution(self, k, gain=True, thresh=1.05):
        distributions = []
        for i, (p0, TM) in enumerate(zip(self.unconditional_returns_prob, self.transition_matrix)):
            probabilities = []
            Q_prev = p0
            for step in range(1, k + 1):
                Q_k = self.diagM(Q_prev @ TM)
                wealth_k = self.compute_wealth(step, asset=i)
                if gain:
                    y = wealth_k >= thresh
                else:
                    y = wealth_k <= thresh
                if step == k:
                    probabilities.append(Q_k.sum())
                else:
                    probabilities.append(Q_k[y].sum())
                Q_k[y] = 0
                Q_prev = Q_k
            distributions.append(probabilities)
        return np.array(distributions).squeeze()

    def quant(self, wealth, cdf, q):
        mask = cdf >= q
        return wealth[mask].min()

    def stable_params_est(self, wealth, cdf):
        def interp2d(x, y, table, val_x, val_y):
            x1 = x[val_x >= x].argmax()
            x2 = x[val_x <= x].argmin()
            y1 = y[val_y >= y].argmax()
            y2 = y[val_y <= y].argmin()

            val = np.array([val_x, val_y])
            w11 = np.sqrt(((np.array([x[x1], y[y1]]) - val) ** 2).sum())
            if w11 == 0:
                return table[x1, y1]
            w12 = np.sqrt(((np.array([x[x1], y[y2]]) - val) ** 2).sum())
            if w12 == 0:
                return table[x1, y2]
            w21 = np.sqrt(((np.array([x[x2], y[y1]]) - val) ** 2).sum())
            if w21 == 0:
                return table[x2, y1]
            w22 = np.sqrt(((np.array([x[x2], y[y2]]) - val) ** 2).sum())
            if w22 == 0:
                return table[x2, y2]
            weights = np.array([[w11, w12],
                                [w21, w22]])
            weights = 1 / weights
            weights = weights / weights.sum()
            values = np.array([[table[x1, y1], table[x1, y2]],
                               [table[x2, y1], table[x2, y2]]])
            return np.sum(weights * values)

        def f_1(v_alpha, v_beta):
            if v_alpha < 2.439 or np.isnan(v_alpha) or np.isinf(v_alpha):
                return 2

            v_a_r = np.array([2.439, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.5, 4, 5, 6, 8, 10, 15, 25])
            v_b_r = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1])
            table = np.array([[2, 2, 2, 2, 2, 2, 2],
                              [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
                              [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
                              [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
                              [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
                              [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
                              [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
                              [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
                              [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
                              [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
                              [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
                              [0.896, 0.892, 0.887, 0.883, 0.855, 0.823, 0.769],
                              [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
                              [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.595],
                              [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513]])
            if v_beta < 0:
                v_beta = -v_beta
            alpha = interp2d(x=v_a_r, y=v_b_r, table=table, val_x=v_alpha, val_y=v_beta)

            if alpha <= 1:
                alpha = 1.01
            elif alpha > 2:
                alpha = 2
            return alpha

        def f_2(v_alpha, v_beta):
            if v_alpha < 2.439 or np.isnan(v_alpha) or np.isinf(v_alpha):
                return np.sign(v_beta)
            v_a_r = np.array([2.439, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.5, 4, 5, 6, 8, 10, 15, 25])
            v_b_r = np.array([0, 1, 0.2, 0.3, 0.5, 0.7, 1])
            table = np.array([[0, 2.160, 1, 1, 1, 1, 1],
                              [0, 1.592, 3.390, 1, 1, 1, 1],
                              [0, 0.759, 1.800, 1, 1, 1, 1],
                              [0, 0.482, 1.048, 1.694, 1, 1, 1],
                              [0, 0.360, 0.760, 1.232, 2.229, 1, 1],
                              [0, 0.253, 0.518, 0.823, 1.575, 1, 1],
                              [0, 0.203, 0.410, 0.632, 1.244, 1.906, 1],
                              [0, 0.165, 0.332, 0.499, 0.943, 1.560, 1],
                              [0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
                              [0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
                              [0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
                              [0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
                              [0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
                              [0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
                              [0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274]])

            if v_beta < 0:
                beta = -interp2d(x=v_a_r, y=v_b_r, table=table, val_x=v_alpha, val_y=-v_beta)
            else:
                beta = interp2d(x=v_a_r, y=v_b_r, table=table, val_x=v_alpha, val_y=v_beta)

            if beta < -1:
                beta = -1
            elif beta > 1:
                beta = 1

            return beta

        def f_3(alpha, beta):
            a_r = np.arange(2, 0.49, -0.1)
            b_r = np.arange(0, 1.1, 0.25)
            table = np.array([[1.908, 1.908, 1.908, 1.908, 1.908],
                              [1.914, 1.915, 1.916, 1.918, 1.921],
                              [1.921, 1.922, 1.927, 1.936, 1.947],
                              [1.927, 1.930, 1.943, 1.961, 1.987],
                              [1.933, 1.940, 1.962, 1.997, 2.043],
                              [1.939, 1.952, 1.988, 2.045, 2.116],
                              [1.946, 1.967, 2.022, 2.106, 2.211],
                              [1.955, 1.984, 2.067, 2.188, 2.333],
                              [1.965, 2.007, 2.125, 2.294, 2.491],
                              [1.980, 2.040, 2.205, 2.435, 2.696],
                              [2.000, 2.085, 2.311, 2.624, 2.973],
                              [2.040, 2.149, 2.461, 2.886, 3.356],
                              [2.098, 2.244, 2.676, 3.265, 3.912],
                              [2.189, 2.392, 3.004, 3.844, 4.775],
                              [2.337, 2.635, 3.542, 4.808, 6.247],
                              [2.588, 3.073, 4.534, 6.636, 9.144]])

            if beta < 0:
                beta = -beta

            #todo it shouldn't be possible!
            if alpha < 0.5:
                alpha = 0.5

            v_scale = interp2d(x=a_r, y=b_r, table=table, val_x=alpha, val_y=beta)

            return v_scale

        def f_4(alpha, beta):
            a_r = np.arange(2, 0.49, -0.1)
            b_r = np.arange(0, 1.1, 0.25)

            table = np.array([[0, 0, 0, 0, 0],
                              [0, -0.017, -0.032, -0.049, -0.064],
                              [0, -0.030, -0.061, -0.092, -0.123],
                              [0, -0.043, -0.088, -0.132, -0.179],
                              [0, -0.056, -0.111, -0.170, -0.232],
                              [0, -0.066, -0.134, -0.206, -0.283],
                              [0, -0.075, -0.154, -0.241, -0.335],
                              [0, -0.084, -0.173, -0.276, -0.390],
                              [0, -0.090, -0.192, -0.310, -0.447],
                              [0, -0.095, -0.208, -0.346, -0.508],
                              [0, -0.098, -0.223, -0.383, -0.576],
                              [0, -0.099, -0.237, -0.424, -0.652],
                              [0, -0.096, -0.250, -0.469, -0.742],
                              [0, -0.089, -0.262, -0.520, -0.853],
                              [0, -0.078, -0.272, -0.581, -0.997],
                              [0, -0.061, -0.279, -0.659, -1.198]])

            if beta < 0:
                v_loc = -interp2d(x=a_r, y=b_r, table=table, val_x=alpha, val_y=-beta)
            else:
                v_loc = interp2d(x=a_r, y=b_r, table=table, val_x=alpha, val_y=beta)
            return v_loc

        q95 = self.quant(wealth, cdf, 0.95)
        q75 = self.quant(wealth, cdf, 0.75)
        q50 = self.quant(wealth, cdf, 0.5)
        q25 = self.quant(wealth, cdf, 0.25)
        q05 = self.quant(wealth, cdf, 0.05)
        v_alpha = (q95 - q05) / (q75 - q25)
        v_beta = (q95 + q05 - 2 * q50) / (q95 - q05)

        try:
            alpha = f_1(v_alpha, v_beta)
        except:
            log = {'q95': q95,
                   'q75': q75,
                   'q50': q50,
                   'q25': q25,
                   'q05': q05,
                   'v_alpha': v_alpha,
                   'v_beta': v_beta,
                   'wealth': wealth,
                   'cdf': cdf}
            with open('log_params_est.pickle', 'wb') as f:
                pickle.dump(log, f)
        beta = f_2(v_alpha, v_beta)
        scale = (q75 - q25) / f_3(alpha, beta)
        loc = q50 + scale * f_4(alpha, beta) - beta * scale * np.tan(np.pi * alpha / 2)

        return {'alpha': alpha, 'beta': beta, 'loc': loc, 'scale': scale}

    def CVaR(self, alpha, beta, loc, scale, q=0.05):
        VaR = -levy_stable.ppf(q, alpha, beta, 0, 1)
        beta_ = -np.sign(VaR) * beta
        x0 = (1 / alpha) * np.arctan(beta_ * np.tan(np.pi * alpha / 2))
        g = lambda x: np.sin(alpha * (x0 + x) - 2 * x) / (np.sin(alpha * (x0 + x))) - (
                (alpha * (np.cos(x) ** 2)) / (np.sin(alpha * (x0 + x)) ** 2))
        v = lambda x: (np.cos(alpha * x0) ** (1 / (alpha - 1))) * (
                    (np.cos(x) / np.sin(alpha * (x0 + x))) ** (alpha / (alpha - 1))) \
                      * (np.cos(alpha * x0 + (alpha - 1) * x) / np.cos(x))
        CVaR_st = (alpha / (1 - alpha)) * (np.abs(VaR) / (np.pi * q)) * \
                  integrate.quad(lambda x: g(x) * np.exp(-(np.abs(VaR) ** (alpha / (alpha - 1))) * v(x)), -x0,
                                 np.pi / 2)[0]
        return CVaR_st * scale - loc

    def covariance(self, T):
        wealth = self.compute_wealth(steps=T)
        distribution = self.compute_wealth_distribution(steps=T)
        mult = np.stack((distribution, distribution), axis=2)
        mean = np.sum(wealth * mult, axis=(0, 1))
        return (np.prod((wealth - mean), axis=2) * distribution).sum()

    def stable_cdf(self, X, alpha, beta, loc, scale):
        """
        Evaluate cumulative density function of alpha-stable distribution at X.
        The formulation is given for the standardized case (loc = 0, scale = 1), so parameters loc and scale provided
        are used to standardize X.
        :param X: float, value to evaluate cdf at
        :param alpha: float, > 1, alpha parameter of the distribution
        :param beta: float, from -1 to 1, beta parameter of the stable distribution
        :param loc: float, location parameter of the stable distribution
        :param scale: float, scale parameter of the stable distribution
        :return: float, cdf value
        """
        # Standardize X
        X = (X - loc) / scale
        x0 = -beta * np.tan(np.pi * alpha / 2)
        x1 = (1 / alpha) * np.arctan(-x0)
        if alpha < 1:
            c1 = (1 / np.pi) * (np.pi / 2 - x1)
        elif alpha > 1:
            c1 = 1
        v = lambda x: (np.cos(alpha * x1) ** (1 / (alpha - 1))) * (
                    (np.cos(x) / np.sin(alpha * (x1 + x))) ** (alpha / (alpha - 1))) \
                      * (np.cos(alpha * x1 + (alpha - 1) * x) / np.cos(x))
        func = lambda x: c1 + np.sign(1 - alpha) / np.pi * \
                         integrate.quad(lambda y: np.exp(-((x - x0) ** (alpha / (alpha - 1))) * v(y)), -x1, np.pi / 2)[
                             0]
        if X > x0:
            cdf_val = func(X)
        elif np.round(X - x0, 5) == 0:
            cdf_val = (1 / np.pi) * (np.pi / 2 - x1)
        else:
            cdf_val = 1 - self.stable_cdf(-X, alpha, -beta, loc, scale)
        return cdf_val

    def stable_cdf_c(self, X, alpha, beta, loc, scale):
        """
        Evaluate cumulative density function of alpha-stable distribution at X.
        The formulation is given for the standardized case (loc = 0, scale = 1), so parameters loc and scale provided
        are used to standardize X.
        :param X: float, value to evaluate cdf at
        :param alpha: float, > 1, alpha parameter of the distribution
        :param beta: float, from -1 to 1, beta parameter of the stable distribution
        :param loc: float, location parameter of the stable distribution
        :param scale: float, scale parameter of the stable distribution
        :return: float, cdf value
        """
        # Standardize X
        X = (X - loc) / scale
        x0 = -beta * np.tan(np.pi * alpha / 2)
        x1 = (1 / alpha) * np.arctan(-x0)
        if alpha < 1:
            c1 = (1 / np.pi) * (np.pi / 2 - x1)
        elif alpha > 1:
            c1 = 1

        inter = LowLevelCallable.from_cython(cdf_2, 'integrand1')

        func = lambda x: c1 + np.sign(1 - alpha) / np.pi * \
                         integrate.quad(inter, -x1, np.pi / 2, args=(x, x1, x0, alpha))[0]

        if X > x0:
            cdf_val = func(X)
        elif X  == x0:
            cdf_val = (1 / np.pi) * (np.pi / 2 - x1)
        else:
            cdf_val = 1 - self.stable_cdf_c(-X, alpha, -beta, -loc, scale)
        return cdf_val

    def omega_ratio(self, thresh, alpha, beta, loc, scale):
        """

        :param thresh:
        :param alpha:
        :param beta:
        :param loc:
        :param scale:
        :return:
        """
        num = integrate.quad(lambda x: 1 - self.stable_cdf(x, alpha, beta, loc, scale), thresh, np.inf)[0]
        den = integrate.quad(lambda x: self.stable_cdf(x, alpha, beta, loc, scale), -np.inf, thresh)[0]
        if den == 0:
            return 1e20
        return num / den

    def omega_ratio_c(self, thresh, alpha, beta, loc, scale):
        """

        :param thresh:
        :param alpha:
        :param beta:
        :param loc:
        :param scale:
        :return:
        """
        num = integrate.quad(lambda x: 1 - self.stable_cdf_c(x, alpha, beta, loc, scale), thresh, np.inf)[0]
        den = integrate.quad(lambda x: self.stable_cdf_c(x, alpha, beta, loc, scale), -np.inf, thresh)[0]
        if den == 0:
            return 1e20
        return num / den

    def MSG_omega_ratio(self, T, thresh=1.1):
        """

        :param T:
        :return:
        """
        wealth = np.flip(self.compute_wealth(steps=T)).squeeze()
        emp_pdf = np.flip(self.compute_wealth_distribution(steps=T).squeeze())
        emp_cdf = emp_pdf.cumsum()

        try:
            alpha, beta, loc, scale = self.stable_params_est(wealth, emp_cdf).values()
        except:
            return 0

        if loc > thresh:
            thresh = (loc + thresh) / 2

        # The check is done on import. Use C version if managed to import the dll or Python if not.
        if cdf_2_imported:
            OR = self.omega_ratio_c(thresh, alpha, beta, loc, scale)
        else:
            OR = self.omega_ratio(thresh, alpha, beta, loc, scale)
        return OR

    def MSG_stable_ratio(self, T):
        """

        :param T:
        :return:
        """
        wealth = np.flip(self.compute_wealth(steps=T)).squeeze()
        emp_pdf = np.flip(self.compute_wealth_distribution(steps=T).squeeze())
        emp_cdf = emp_pdf.cumsum()
        try:
            alpha, beta, loc, scale = self.stable_params_est(wealth, emp_cdf).values()
        except:
            return 0
        CVaR = self.CVaR(alpha, beta, loc, scale)
        ratio = loc / (1 + CVaR)
        return ratio

    def MSG_time_to_gain(self, T, thresh=1.05):
        """

        :param T:
        :param thresh:
        :return:
        """
        steps = np.arange(1, T+1)
        distribution = self.time_distribution(k=T, thresh=thresh)
        return np.sum(np.apply_along_axis(lambda x: x * steps, 1, distribution), axis=1)

    def MSG_time_to_lose(self, T, thresh=0.95):
        """

        :param T:
        :param thresh:
        :return:
        """
        steps = np.arange(1, T+1)
        distribution = self.time_distribution(k=T, gain=False, thresh=thresh)
        return np.sum(np.apply_along_axis(lambda x: x * steps, 1, distribution), axis=1)

    def MSG_wealth_sd(self, T):
        """

        :param T:
        :return:
        """
        # todo figure out how if this implementation is correct
        wealth = self.compute_wealth(steps=T)
        distribution = self.compute_wealth_distribution(steps=T)
        if self.dim == 1:
            mean = np.sum(wealth*distribution, axis=1)
            return np.sqrt(np.sum((np.apply_along_axis(lambda x: x - mean, 0, wealth) ** 2) * distribution, axis=1))
        elif self.dim == 2:
            distribution = np.stack((distribution, distribution), axis=2)
            mean = np.sum(wealth * distribution, axis=(0,1))
            return np.sqrt(np.sum((wealth - mean) ** 2) * distribution, axis=1)

    def MSG_wealth_mean(self, T):
        """

        :param T:
        :return:
        """
        wealth = self.compute_wealth(steps=T)
        distribution = self.compute_wealth_distribution(steps=T)
        return np.sum(wealth*distribution, axis=1)

    def MSG_square_root_utility(self, T):
        """

        :param T:
        :return:
        """
        wealth = self.compute_wealth(steps=T)
        wealth = np.sqrt(wealth) * 2
        distribution = self.compute_wealth_distribution(steps=T)
        return np.sum(wealth * distribution, axis=1)

    def MSG_CVaR_log_wealth(self, T):
        """

        :param T:
        :return:
        """
        wealths = np.flip(self.compute_wealth(steps=T))
        emp_pdf = np.flip(self.compute_wealth_distribution(steps=T))
        emp_cdfs = emp_pdf.cumsum(axis=1)
        CVaRs = []
        i = 0
        for wealth, cdf in zip(wealths, emp_cdfs):
            alpha, beta, loc, scale = self.stable_params_est(np.log(wealth), cdf).values()
            CVaR = self.CVaR(alpha, beta, loc, scale)
            CVaRs.append(CVaR)
            i += 1
        return np.array(CVaRs)

    def MSG_stable_location(self, T):
        """

        :param T:
        :return:
        """
        wealths = np.flip(self.compute_wealth(steps=T))
        emp_pdf = np.flip(self.compute_wealth_distribution(steps=T))
        emp_cdfs = emp_pdf.cumsum(axis=1)
        locs = []
        for wealth, cdf in zip(wealths, emp_cdfs):
            loc = self.stable_params_est(wealth, cdf)['loc']
            locs.append(loc)
        return np.array(locs)

    def MSG_sharpe_ratio(self, T):
        mean = self.MSG_wealth_mean(T=T)
        sd = self.MSG_wealth_sd(T=T)
        return (mean - 1) / sd

    def MSG_corr_usb(self, T):
        usb = np.max(self.data, axis=0)
        chain_usb = MarkovChainProcess()
        chain_usb.fit(usb, N = self.n_states)
        sdu = chain_usb.MSG_wealth_sd(T=T)[0]
        sd = self.MSG_wealth_sd(T=T)
        corrs = []

        if self.weights is None:
            for i in range(self.data.shape[0]):
                ass = self.data[i]
                chain_bi = MarkovChainProcess()
                chain_bi.fit_bi(ass, usb, N=self.n_states, M=self.n_states, unconditional_start=self.unconditional)
                cov = chain_bi.covariance(T=T)
                corr = cov / (sdu * sd[i])
                corrs.append(corr)
        else:
            ass = self.weights @ self.data
            chain_bi = MarkovChainProcess()
            chain_bi.fit_bi(ass, usb, N=self.n_states, M=self.n_states, unconditional_start=self.unconditional)
            cov = chain_bi.covariance(T=T)
            corr = cov / (sdu * sd[0])
            corrs.append(corr)

        return np.array(corrs)

    def MSG_corr_lsb(self, T):
        lsb = np.min(self.data, axis=0)
        chain_lsb = MarkovChainProcess()
        chain_lsb.fit(lsb, N = self.n_states)
        sdl = chain_lsb.MSG_wealth_sd(T=T)[0]
        sd = self.MSG_wealth_sd(T=T)
        corrs = []

        if self.weights is None:
            for i in range(self.data.shape[0]):
                ass = self.data[i]
                chain_bi = MarkovChainProcess()
                chain_bi.fit_bi(ass, lsb, N=self.n_states, M=self.n_states, unconditional_start=self.unconditional)
                cov = chain_bi.covariance(T=T)
                corr = cov / (sdl * sd[i])
                corrs.append(corr)
        else:
            ass = self.weights @ self.data
            chain_bi = MarkovChainProcess()
            chain_bi.fit_bi(ass, lsb, N=self.n_states, M=self.n_states, unconditional_start=self.unconditional)
            cov = chain_bi.covariance(T=T)
            corr = cov / (sdl * sd[0])
            corrs.append(corr)

        return np.array(corrs)

    def MSG_corr_ratio(self, T):
        corr_usb = self.MSG_corr_usb(T=T)
        corr_lsb = self.MSG_corr_lsb(T=T)
        ratio = (1 - corr_usb) / (1 - corr_lsb)
        return ratio
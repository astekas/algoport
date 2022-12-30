### A module containing the models for Portfolio Optimization ###
from qpsolvers import solve_qp
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
import inspect
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import minimize as sc_minimize
from scipy.optimize import LinearConstraint as LC
from Markov import MarkovChainProcess
import Metrics
from time import perf_counter
from mealpy.bio_based import SMA

# Meta-heuristics based optimization
class PortfolioProblem(ElementwiseProblem):
    def __init__(self, n_var, function, data, function_kwargs, **kwargs):
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=1.0, **kwargs)
        self.function = function
        self.data = data
        self.function_kwargs = function_kwargs

    def _evaluate(self, x, out, *args, **kwargs):
        args = [self.data, self.function_kwargs]
        out["F"] = -self.function(x, args)

class PortfolioRepair(Repair):

    def _do(self, problem, X, **kwargs):
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)

class PortfolioOptimizer:
    def __init__(self, model=None, metric=None, model_metric=None, model_kwargs=None, optimizer_kwargs=None, config=None):
        self.model = model
        self.metric = metric
        self.model_metric = model_metric
        if model_kwargs is None:
            self.model_kwargs = {'init': {},
                                 'fit': {}}
        else:
            self.model_kwargs = model_kwargs
        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = optimizer_kwargs

        self.config = config

        self.current_weights = None

        self.values = []
        self.optimization_times = []

        if hasattr(self.model, 'fit'):
            expected_args = inspect.getfullargspec(self.model.fit).args
            if 'weights' in expected_args:
                self.model_with_weights = True
            else:
                self.model_with_weights = False
            if 'current_weights' in expected_args:
                self.model_with_current_weights = True
            else:
                self.model_with_current_weights = False
        else:
            raise ValueError('OPTIMIZATION ERROR. Model should have the "fit" method.')

    def prepare_model(self, returns):
        if self.model is not None:
            if inspect.isclass(self.model):
                self.model = self.model(**self.model_kwargs['init'])
            if not self.model_with_weights and not self.model_with_current_weights:
                self.model.fit(returns, **self.model_kwargs['fit'])

    def func(self, weights, returns):
        if self.model is not None and self.model_with_weights and not self.model_with_current_weights:
            self.model.fit(returns, weights=weights, **self.model_kwargs['fit'])
        elif self.model is not None and self.model_with_weights and self.model_with_current_weights:
            self.model.fit(returns, weights=weights, current_weights=self.current_weights, **self.model_kwargs['fit'])
        elif self.model is not None:
            self.model.fit(returns, **self.model_kwargs['fit'])
        if self.model_metric is not None:
            metric_name, positive, kwargs = self.model_metric
            metric = getattr(self.model, metric_name)
            expected_args = inspect.getfullargspec(metric).args
            if 'weights' in expected_args:
                kwargs['weights'] = weights
            if 'returns' in expected_args:
                kwargs['returns'] = returns
            res = metric(**kwargs)
            if positive:
                res = - res
            if isinstance(res, np.ndarray):
                res = res.squeeze()
            return res
        if self.metric is not None:
            metric, positive, kwargs = self.metric
            if isinstance(metric, str):
                try:
                    metric = getattr(Metrics, metric)
                except:
                    raise ValueError(f'OPTIMIZER ERROR. Metric {metric} not found in Metrics.')
            expected_args = inspect.getfullargspec(metric).args
            if 'weights' in expected_args:
                kwargs['weights'] = weights
            if 'returns' in expected_args:
                kwargs['returns'] = returns
            res = metric(**kwargs)
            if positive:
                res = - res
            if isinstance(res, np.ndarray):
                res = res.squeeze()
            return res

    def optimization(self, returns, **kwargs):
        weights = np.ones(len(returns))
        weights = weights / np.sum(weights)
        return weights

    def optimize(self, returns, current_weights=None):
        if current_weights is not None:
           self.current_weights = current_weights
        else:
            self.current_weights = None
        self.prepare_model(returns=returns)
        start = perf_counter()
        out = self.optimization(returns=returns, **self.optimizer_kwargs)
        if len(out) == 2:
            res, val = out
        else:
            res = out
            val = None
        self.values.append(val)
        self.optimization_times.append(perf_counter() - start)
        return res

class PyMOO(PortfolioOptimizer):

    def optimization(self, returns, algorithm='SMSEMOA', n_gen_termination=10):
        n_var = len(returns)
        problem = PortfolioProblem(n_var=n_var, function=self.func, data=returns, function_kwargs=None)
        algorithm = SMSEMOA(repair=PortfolioRepair(),
                            termination=('n_gen', n_gen_termination))
        res = minimize(problem,
                       algorithm,
                       verbose=False)
        X = res.opt.get("X")[0]
        F = res.opt.get("F")[0]
        print(X)
        return X, F

class MealPy(PortfolioOptimizer):

    def optimization(self, returns, **kwargs):
        def f(solution):
            res = self.func(solution, returns)
            penalty = np.abs(np.sum(solution) - 1)


            return res + np.abs(res * penalty)

        problem = {
            "fit_func": f,
            "lb": [0, ] * len(returns),
            "ub": [1, ] * len(returns),
            "minmax": "min",
        }
        sma_model = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
        best_position, best_fitness_value = sma_model.solve(problem, mode="process", n_workers=7)
        return best_position, best_fitness_value

class SciPy(PortfolioOptimizer):
    def optimization(self, returns, algorithm='trust-constr'):
        if algorithm == 'trust-constr':
            lb = np.zeros(returns.shape[0])
            ub = np.ones(returns.shape[0])
            bounds = list(zip(lb,ub))
            A = np.ones((2, returns.shape[0]))
            delta = 1e-05
            lb_const = np.array([1-delta, 1])
            ub_const = np.array([1+delta, 1])
            feas = np.array([True, True])
            constraints = LC(A=A, lb=lb_const, ub=ub_const, keep_feasible=feas)
        elif algorithm == 'SLSQP':
            lb = np.zeros(returns.shape[0])
            ub = np.ones(returns.shape[0])
            bounds = list(zip(lb, ub))
            def constraint(x):
                return np.sum(x) - 1
            constraints = [{'type': 'eq', 'fun': constraint}]
        elif algorithm == 'COBYLA':
            bounds = None
            # Since COBYLA only takes inequality constraints - define the sum(weights) = 1 as 2 constraints
            def constraint_lower(x):
                return np.sum(x) - 1
            def constraint_upper(x):
                return -(np.sum(x) - 1)
            def constraint_bounds(x):
                return np.sum(x[x<0])
            constraints = [{'type': 'ineq', 'fun': constraint_lower},
                           {'type': 'ineq', 'fun': constraint_upper},
                           {'type': 'ineq', 'fun': constraint_bounds}]
        else:
            raise ValueError('Unknown algorithm for optimizer scipy. Available include COBYLA, trust-constr and SLSQP')

        init = np.ones(returns.shape[0]) / returns.shape[0]

        res = sc_minimize(self.func, x0=init, args=[returns], bounds=bounds, method=algorithm, constraints=constraints)
        weights = res.x
        weights[weights<1e-3] = 0
        print('Final weights - ', res.x)
        #todo change
        return res.x, res.fun

# Classical quadratic optimization.
class MVOptimization(PortfolioOptimizer):
    def optimization(self, returns, alpha=0.05):
        if self.model is not None and hasattr(self.model, 'returns'):
            returns = self.model.returns
            print(returns.shape)
        mean = np.mean(returns, axis=1)
        covariance = np.cov(returns)

        P = csc_matrix(covariance)
        q = alpha * mean
        A = csc_matrix(np.ones((1, len(q))))
        lb = np.zeros(len(q))
        ub = np.ones(len(q))
        weights = solve_qp(P, q, A=A, b=1, lb=lb, ub=ub, solver="osqp")
        return weights

class SimplexOptimization(PortfolioOptimizer):
    def improve_by_increasing(self, x, value, func, n_assets, tol, m, p):
        improved = False
        global_best = x
        global_best_val = value

        for a in range(n_assets):
            temp_best = global_best
            temp_best_val = global_best_val

            e = np.zeros(n_assets)
            e[a] = 1
            beta = (np.arange(1, m) / m) ** p

            x = temp_best
            for b in beta:
                x = (1 - b) * x + b * e
                new_val = func(x)
                if temp_best_val - new_val > tol:
                    temp_best = x
                    temp_best_val = new_val
                    improved = True
            if global_best_val - temp_best_val > tol:
                global_best_val = temp_best_val
                global_best = temp_best

        return global_best, global_best_val, improved

    def improve_by_decreasing(self, x, value, func, n_assets, tol, m, p):
        improved = False
        global_best = x
        global_best_val = value
        for a in range(n_assets):
            temp_best = global_best
            temp_best_val = global_best_val
            x = temp_best

            e = np.zeros(n_assets)
            e[a] = 1
            p_x = (x - (e * x[a])) / (1 - x[a])
            beta = (np.arange(1, m) / m) ** p

            for b in beta:
                x = (1 - b) * x + b * p_x
                new_val = func(x)
                if temp_best_val - new_val > tol:
                    temp_best = x
                    temp_best_val = new_val
                    improved = True
            if global_best_val - temp_best_val > tol:
                global_best_val = temp_best_val
                global_best = temp_best

        return global_best, global_best_val, improved

    def repair(self, x, precision):
        x[x<precision] = 0
        return x / x.sum()

    def optimization(self, returns, func=None, tol=0.001, prec=1e-3, m=10, p=1):
        if func == None:
            def func(weights):
                res = self.func(weights, returns)
                return res
        n_assets = len(returns)
        x_i = np.ones(n_assets) / n_assets

        val_i = func(x_i)
        x, value, improved = self.improve_by_increasing(x=x_i, value=val_i, func=func, n_assets=n_assets, tol=tol, m=m, p=p)
        while improved:
            x, value, improved = self.improve_by_decreasing(x=x, value=value, func=func, n_assets=n_assets, tol=tol, m=m, p=p)
            if improved:
                x, value, improved = self.improve_by_increasing(x=x, value=value, func=func, n_assets=n_assets, tol=tol, m=m,
                                                                p=p)
        print(x, value)
        return self.repair(x, precision=prec), value

import functools
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

np.random.seed(100)

# Constants
# n_stocks = 500
# n_sectors = 5
# abs_starting_weights = 0.02
# max_weight = 0.02
# total_turnover = 3
# individual_turnover = 0.025
# target_std = 0.006
# sector_abs_weight = 1e-3
# eps = 1e-8

n_stocks = 2500
n_sectors = 15
abs_starting_weights = 0.0006
max_weight = 0.002
total_turnover = 0.5
individual_turnover = 0.003
target_std = 0.0005
sector_abs_weight = 1e-3
eps = 1e-8

# X matrices
zeros = np.zeros(n_stocks)
ones = np.ones(n_stocks)
diagonal_a = np.eye(n_stocks)
double_diagonal = np.hstack((diagonal_a, diagonal_a))
diagonal_zeros = np.hstack((diagonal_a, np.zeros((n_stocks, n_stocks))))
zeros_diagonal = np.hstack((np.zeros((n_stocks, n_stocks)), diagonal_a))
ones_zeros = np.hstack((ones, zeros))
zeros_ones = np.hstack((zeros, ones))
diagonal_negdiagonal = np.hstack((diagonal_a, -diagonal_a))

# Positive/negative add to 1 and -1 respectively
current_weights = np.random.uniform(
    -abs_starting_weights, abs_starting_weights, n_stocks
)
current_weights -= current_weights.mean()
current_weights /= abs(current_weights).sum()
current_weights *= 2
current_weights = np.hstack((current_weights, zeros))


@functools.cache
def make_weight(n_stock, top_weight):
    hello = [0, 1, 2, 3]
    return hello


# current_weights = np.random.uniform(-abs_starting_weights, abs_starting_weights, n_stocks)
# current_weights[0] = -abs_starting_weights
# current_weights[1] =

# Positive/negative add to 1 and -1 respectively
targ_w = current_weights[:n_stocks] + np.random.normal(0, target_std, n_stocks)
targ_w -= targ_w.mean()
targ_w /= np.abs(targ_w).sum()
targ_w *= 2
target_weights = np.hstack((targ_w, zeros))
# target_weights = np.random.uniform(-abs_starting_weights, abs_starting_weights, n_stocks)
# target_weights -= target_weights.mean()
# target_weights /= abs(target_weights).sum()
# target_weights *= 2
# target_weights = np.hstack((target_weights, zeros))
# target_w = target_weights[:n_stocks]

# Strict limits from individual_turnover and abs_weight
lb_stocks = np.maximum(-max_weight, current_weights[:n_stocks] - individual_turnover)
ub_stocks = np.minimum(max_weight, current_weights[:n_stocks] + individual_turnover)

# Check if allows positive
pos_mask = targ_w > 0
adjust_pos = np.logical_and(ub_stocks > 0, lb_stocks <= 0)
force_pos = np.logical_and(pos_mask, adjust_pos)

# Truncate lower bound for positive
lb_stocks[force_pos] = 0.0

# Positive optimized variables
already_pos = np.logical_and(lb_stocks >= 0, ub_stocks > 0)
all_pos = np.logical_or(force_pos, already_pos)

# Check if allows negative
neg_mask = targ_w < 0
adjust_neg = np.logical_and(ub_stocks > 0, lb_stocks <= 0)
force_neg = np.logical_and(neg_mask, adjust_neg)

# Truncate upper bound for negative
ub_stocks[force_neg] = 0.0

# Negative optimized variables
already_neg = np.logical_and(ub_stocks <= 0, lb_stocks < 0)
all_neg = np.logical_or(force_neg, already_neg)

# Gross exposure x matrices
all_pos_padded = np.hstack((all_pos, np.zeros(n_stocks, dtype=bool))).astype(float)
all_neg_padded = np.hstack((all_neg, np.zeros(n_stocks, dtype=bool))).astype(float)
force_zero = target_weights[:n_stocks] == 0
force_zero_padded = np.hstack((force_zero, np.zeros(n_stocks, dtype=bool))).astype(
    float
)

# Form sector weights
sector_weights = np.random.rand(n_sectors, n_stocks)
column_sum = sector_weights.sum(axis=0, keepdims=True)
sector_weights = sector_weights / column_sum
sector_weights = np.hstack((sector_weights, np.zeros((n_sectors, n_stocks))))


def objective(w, w0):
    diff = w[:n_stocks] - w0[:n_stocks]
    return np.inner(diff, diff)


def objective_gradient(w, w0):
    grad = 2.0 * (w[:n_stocks] - w0[:n_stocks])
    grad = np.hstack((grad, np.zeros(n_stocks)))
    return grad


# def ReLU_a(n):
#     left_side = np.eye(n)
#     right_side = -np.eye(n)
#     return np.concatenate((left_side, right_side, np.zeros((n_stocks,n_stocks))), axis=1)

# Constraints

# Zero edge case
zeros_case = LinearConstraint(force_zero_padded, lb=eps, ub=eps)

# Gross exposure
long_sum = LinearConstraint(all_pos_padded, lb=1.0, ub=1.0)

# Force positives
positive_force = LinearConstraint(np.diag(all_pos_padded), lb=0.0, ub=math.inf)

# Force negatives
negative_force = LinearConstraint(np.diag(all_neg_padded), lb=-math.inf, ub=0.0)

# Dollar neutral
dollar_neutral = LinearConstraint(ones_zeros, lb=eps, ub=eps)

Individual weights
individual_weight = LinearConstraint(diagonal_zeros, lb=lb_stocks, ub=ub_stocks)

# Turnover
total_turnover_geq = LinearConstraint(
    double_diagonal, lb=current_weights[:n_stocks], ub=np.inf
)

total_turnover_leq = LinearConstraint(
    diagonal_negdiagonal, lb=-np.inf, ub=current_weights[:n_stocks]
)

sum_v = LinearConstraint(zeros_ones, lb=0.0, ub=total_turnover)

positive_v = LinearConstraint(zeros_diagonal, lb=0.0, ub=math.inf)

# Sector neutral
sector_neutrality = LinearConstraint(
    sector_weights, lb=-sector_abs_weight, ub=sector_abs_weight
)

# Form constraint list
constraints = [
    dollar_neutral,
    long_sum,
    positive_force,
    negative_force,
    sector_neutrality,
    individual_weight,
    positive_v,
    sum_v,
    total_turnover_geq,
    total_turnover_leq,
]

if force_zero.any():
    constraints.append(zeros_case)

# Total Stock Optimization
breakpoint()

# start timer
start_time = time.perf_counter()

result = minimize(
    fun=objective,
    x0=current_weights,
    args=(target_weights,),
    # jac = objective_gradient,
    method="SLSQP",
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-6, "disp": True},
)

optimize_weights = result.x[:n_stocks]
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Seconds: {elapsed_time}")
print(f"n_stocks: {n_stocks}")
print(f"abs_starting_weights:{abs_starting_weights}")
print(f"n_sectors: {n_sectors}")
print(f"max_weight:{max_weight}")
print(f"total_turnover: {total_turnover}")
print(f"target_std: {target_std}")
print(f"sector_abs_weight: {sector_abs_weight}")
print(f"eps: {eps}")
breakpoint()

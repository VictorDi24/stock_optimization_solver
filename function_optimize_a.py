import functools
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


# defines constants
def get_config():
    return {
        "n_stocks": 2500,
        "n_sectors": 15,
        "abs_starting_weights": 0.0006,
        "max_weight": 0.002,
        "total_turnover": 0.5,
        "individual_turnover": 0.003,
        "target_std": 0.0005,
        "sector_abs_weight": 1e-3,
        "eps": 1e-8,
    }


def generate_weights():
    config = get_config()

    current_weights = np.random.uniform(
        -config["abs_starting_weights"],
        config["abs_starting_weights"],
        config["n_stocks"],
    )
    current_weights -= current_weights.mean()
    current_weights /= abs(current_weights).sum()
    current_weights *= 2
    current_weights = np.hstack((current_weights, np.zeros(config["n_stocks"])))

    targ_w = current_weights[: config["n_stocks"]] + np.random.normal(
        0, config["target_std"], config["n_stocks"]
    )
    targ_w -= targ_w.mean()
    targ_w /= np.abs(targ_w).sum()
    targ_w *= 2
    target_weights = np.hstack((targ_w, np.zeros(config["n_stocks"])))

    return current_weights, target_weights


def generate_bounds_masks(current_weights, target_weights, config):
    # Strict limits from individual_turnover and abs_weight
    lb_stocks = np.maximum(
        -config["max_weight"],
        current_weights[: config["n_stocks"]] - config["individual_turnover"],
    )
    ub_stocks = np.minimum(
        config["max_weight"],
        current_weights[: config["n_stocks"]] + config["individual_turnover"],
    )

    # Check if allows positive
    pos_mask = target_weights[: config["n_stocks"]] > 0
    adjust_pos = np.logical_and(ub_stocks > 0, lb_stocks <= 0)
    force_pos = np.logical_and(pos_mask, adjust_pos)

    # Truncate lower bound for positive
    lb_stocks[force_pos] = 0.0

    # Positive optimized variables
    already_pos = np.logical_and(lb_stocks >= 0, ub_stocks > 0)
    all_pos = np.logical_or(force_pos, already_pos)

    # Check if allows negative
    neg_mask = target_weights[: config["n_stocks"]] < 0
    adjust_neg = np.logical_and(ub_stocks > 0, lb_stocks <= 0)
    force_neg = np.logical_and(neg_mask, adjust_neg)

    # Truncate upper bound for negative
    ub_stocks[force_neg] = 0.0

    # Negative optimized variables
    already_neg = np.logical_and(ub_stocks <= 0, lb_stocks < 0)
    all_neg = np.logical_or(force_neg, already_neg)

    # Gross exposure x matrices
    all_pos_padded = np.hstack(
        (all_pos, np.zeros(config["n_stocks"], dtype=bool))
    ).astype(float)
    all_neg_padded = np.hstack(
        (all_neg, np.zeros(config["n_stocks"], dtype=bool))
    ).astype(float)
    force_zero = target_weights[: config["n_stocks"]] == 0
    force_zero_padded = np.hstack(
        (force_zero, np.zeros(config["n_stocks"], dtype=bool))
    ).astype(float)

    return lb_stocks, ub_stocks, all_pos_padded, all_neg_padded, force_zero_padded


def create_sectors(config):
    sector_weights = np.random.rand(config["n_sectors"], config["n_stocks"])
    column_sum = sector_weights.sum(axis=0, keepdims=True)
    sector_weights = sector_weights / column_sum
    sector_weights = np.hstack(
        (sector_weights, np.zeros((config["n_sectors"], config["n_stocks"])))
    )
    return sector_weights


def build_constraints(
    current_weights,
    lb_stocks,
    ub_stocks,
    all_pos_padded,
    all_neg_padded,
    force_zero_padded,
    sector_weights,
    config,
):
    zeros = np.zeros(config["n_stocks"])
    ones = np.ones(config["n_stocks"])
    diagonal_a = np.eye(config["n_stocks"])
    double_diagonal = np.hstack((diagonal_a, diagonal_a))
    diagonal_zeros = np.hstack(
        (diagonal_a, np.zeros((config["n_stocks"], config["n_stocks"])))
    )
    zeros_diagonal = np.hstack(
        (np.zeros((config["n_stocks"], config["n_stocks"])), diagonal_a)
    )
    ones_zeros = np.hstack((ones, zeros))
    zeros_ones = np.hstack((zeros, ones))
    diagonal_negdiagonal = np.hstack((diagonal_a, -diagonal_a))

    zeros_case = LinearConstraint(force_zero_padded, lb=config["eps"], ub=config["eps"])
    # Gross exposure
    long_sum = LinearConstraint(all_pos_padded, lb=1.0, ub=1.0)

    # Force positives
    positive_force = LinearConstraint(np.diag(all_pos_padded), lb=0.0, ub=math.inf)

    # Force negatives
    negative_force = LinearConstraint(np.diag(all_neg_padded), lb=-math.inf, ub=0.0)

    # Dollar neutral
    dollar_neutral = LinearConstraint(ones_zeros, lb=config["eps"], ub=config["eps"])

    # Individual weights
    individual_weight = LinearConstraint(diagonal_zeros, lb=lb_stocks, ub=ub_stocks)

    # Turnover
    total_turnover_geq = LinearConstraint(
        double_diagonal, lb=current_weights[: config["n_stocks"]], ub=math.inf
    )

    total_turnover_leq = LinearConstraint(
        diagonal_negdiagonal, lb=-math.inf, ub=current_weights[: -config["n_stocks"]]
    )

    sum_v = LinearConstraint(zeros_ones, lb=0.0, ub=config["total_turnover"])

    positive_v = LinearConstraint(zeros_diagonal, lb=0.0, ub=math.inf)

    # Sector neutral
    sector_neutrality = LinearConstraint(
        sector_weights, lb=-config["sector_abs_weight"], ub=config["sector_abs_weight"]
    )

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

    if force_zero_padded[: config["n_stocks"]].any():
        constraints.append(zeros_case)

    return constraints


def run_optimize(current_weights, target_weights, constraints, config):
    def objective(w, w0):
        diff = w[: config["n_stocks"]] - w0[: config["n_stocks"]]
        return np.inner(diff, diff)

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

    optimize_weights = result.x[: config["n_stocks"]]
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Seconds: {elapsed_time}")
    print(f"n_stocks: {config['n_stocks']}")
    print(f"abs_starting_weights: {config['abs_starting_weights']}")
    print(f"n_sectors: {config['n_sectors']}")
    print(f"max_weight: {config['max_weight']}")
    print(f"total_turnover: {config['total_turnover']}")
    print(f"target_std: {config['target_std']}")
    print(f"sector_abs_weight: {config['sector_abs_weight']}")
    print(f"eps: {config['eps']}")
    return optimize_weights, result.x


def run_optimization_pipeline():
    config = get_config()

    current_w, target_w = generate_weights()
    sector_w = create_sectors(config)  # Assume this exists

    lb, ub, all_pos, all_neg, force_zero = generate_bounds_masks(
        current_w, target_w, config
    )
    constraints = build_constraints(
        current_w, lb, ub, all_pos, all_neg, force_zero, sector_w, config
    )
    breakpoint()
    optimize_weights, result_x = run_optimize(current_w, target_w, constraints, config)
    breakpoint()
    return optimize_weights, result_x


optimize_weights, result_x = run_optimization_pipeline()

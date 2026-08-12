import functools
import math
import operator
import time

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy import stats


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


# defines constants
@functools.lru_cache(maxsize=1)
def get_config():
    return {
        # "n_stocks": weights_one.shape[0],
        # "n_stocks": 2500,
        "number_buffer": 2,
        "n_sectors": 15,
        "abs_starting_weights": 0.012,
        "max_weight": 0.008,
        "total_turnover": 0.8,
        "individual_turnover": 0.005,
        "target_std": 0.0005,
        "sector_abs_weight": 1e-3,
        "eps": 1e-8,
        "speed": 0,
        "day": 1,
        "alpha": 1,
        "beta": 0.00,
        "number_gates": 2,
    }


def generate_weights(config):
    start = time.perf_counter()
    temp = np.load("/home/cding/IdeaProjects/cvxpy_optimize/short_arr_a.npz")

    # weights = temp["x"]
    on_off = temp["y"]
    delta = temp["z"]
    weights = delta[
        config["day"] : config["day"] + config["number_gates"], :, config["speed"]
    ]
    day_on_off = on_off[config["day"] + 1]
    day_weights = weights[:, day_on_off]
    day_weights[np.isnan(day_weights)] = 0

    day_weights /= abs(day_weights).sum(axis=1, keepdims=True)
    day_weights -= day_weights.mean(axis=1, keepdims=True)
    day_weights *= 2

    weights_one = day_weights[0, :]
    weights_two = day_weights[1, :]
    end = time.perf_counter()
    print(f"load time: {end - start:.4f} seconds")

    # x = cp.Variable(2 * weights_one.shape[0] + 1, name="x")
    x = cp.Variable(
        config["number_buffer"] * weights_one.shape[0] + config["number_gates"],
        name="x",
    )

    weights_a = np.hstack((weights_one, np.zeros(weights_one.shape[0])))
    weights_b = np.hstack((weights_two, np.zeros(weights_two.shape[0])))

    return weights_a, weights_b, x


def generate_bounds_masks(weights_one, weights_two, config):
    # Strict limits from individual_turnover and abs_weight
    lb_stocks = np.maximum(
        -config["max_weight"],
        weights_one[: weights_one.shape[0] // config["number_buffer"]]
        - config["individual_turnover"],
    )
    ub_stocks = np.minimum(
        config["max_weight"],
        weights_one[: weights_one.shape[0] // config["number_buffer"]]
        + config["individual_turnover"],
    )

    # Check if allows positive
    pos_mask = weights_two[: weights_two.shape[0] // config["number_buffer"]] > 0
    adjust_pos = np.logical_and(ub_stocks >= 0, lb_stocks <= 0)
    force_pos = np.logical_and(pos_mask, adjust_pos)

    # Truncate lower bound for positive
    lb_stocks[force_pos] = 0.0

    # Positive optimized variables
    already_pos = np.logical_and(lb_stocks >= 0, ub_stocks >= 0)
    all_pos = np.logical_or(force_pos, already_pos)

    # Check if allows negative
    neg_mask = weights_two[: weights_two.shape[0] // config["number_buffer"]] < 0
    adjust_neg = np.logical_and(ub_stocks >= 0, lb_stocks <= 0)
    force_neg = np.logical_and(neg_mask, adjust_neg)

    # Truncate upper bound for negative
    ub_stocks[force_neg] = 0.0

    # Negative optimized variables
    already_neg = np.logical_and(ub_stocks <= 0, lb_stocks <= 0)
    all_neg = np.logical_or(force_neg, already_neg)

    # Gross exposure x matrices
    all_pos_padded = np.hstack(
        (all_pos, np.zeros(all_pos.shape[0], dtype=bool))
    ).astype(float)
    all_neg_padded = np.hstack(
        (all_neg, np.zeros(all_neg.shape[0], dtype=bool))
    ).astype(float)
    force_zero = weights_two[: weights_two.shape[0] // config["number_buffer"]] == 0
    force_zero_padded = np.hstack(
        (force_zero, np.zeros(force_zero.shape[0], dtype=bool))
    ).astype(float)

    return lb_stocks, ub_stocks, all_pos_padded, all_neg_padded, force_zero_padded


def create_sectors(config, weights_two):
    sector_weights = np.random.rand(
        config["n_sectors"], int(weights_two.shape[0] / config["number_buffer"])
    )
    column_sum = sector_weights.sum(axis=0, keepdims=True)
    sector_weights = sector_weights / column_sum
    sector_weights = np.hstack(
        (
            sector_weights,
            np.zeros(
                (config["n_sectors"], weights_two.shape[0] // config["number_buffer"])
            ),
        )
    )
    return sector_weights


def sector_constraint(sector_weights, config, x):
    sector_neutrality = [
        sector_weights @ x[: int(sector_weights.shape[1])]
        >= -config["sector_abs_weight"],
        sector_weights @ x[: int(sector_weights.shape[1])]
        <= config["sector_abs_weight"],
    ]
    return sector_neutrality


def net_exposure_constraint(all_pos_padded, config, x):
    long_sum = [
        all_pos_padded @ x[: int(all_pos_padded.shape[0])] <= 1.0 + config["eps"],
        all_pos_padded @ x[: int(all_pos_padded.shape[0])] >= 1.0 - config["eps"],
    ]
    return long_sum


def force_sign_constraint(all_pos_padded, all_neg_padded, config, x):
    n_stocks = all_pos_padded.shape[0]
    pos_idx = all_pos_padded.astype(bool)
    positive_force = [x[:n_stocks][pos_idx] >= -config["eps"]]
    neg_idx = all_neg_padded.astype(bool)
    negative_force = [x[:n_stocks][neg_idx] <= config["eps"]]

    return positive_force + negative_force


def force_zero_constraint(force_zero_padded, config, x):

    zero_idx = force_zero_padded.astype(bool)
    zeros_case = [
        x[: force_zero_padded.shape[0]][zero_idx] <= config["eps"],
        x[: force_zero_padded.shape[0]][zero_idx] >= -config["eps"],
    ]
    return zeros_case


def dollar_neutral_constraint(ones_zeros, config, x):
    dollar_neutrality = [
        ones_zeros @ x[: ones_zeros.shape[1]] <= config["eps"],
        ones_zeros @ x[: ones_zeros.shape[1]] >= -config["eps"],
    ]
    return dollar_neutrality


def individual_constraint(diagonal_zeros, config, lb_stocks, ub_stocks, x):
    individual_weight = [
        diagonal_zeros @ x[: diagonal_zeros.shape[1]] >= lb_stocks,
        diagonal_zeros @ x[: diagonal_zeros.shape[1]] <= ub_stocks,
    ]
    return individual_weight


def total_turnover_component_constraint(
    double_diagonal, diagonal_negdiagonal, config, weights_one, x
):
    total_turnover_geq = [
        double_diagonal @ x[: double_diagonal.shape[1]]
        >= weights_one[: double_diagonal.shape[0]],
        double_diagonal @ x[: double_diagonal.shape[1]] <= math.inf,
    ]

    total_turnover_leq = [
        diagonal_negdiagonal @ x[: double_diagonal.shape[1]] >= -math.inf,
        diagonal_negdiagonal @ x[: double_diagonal.shape[1]]
        <= weights_one[: double_diagonal.shape[0]],
    ]
    return total_turnover_geq + total_turnover_leq


def v_constraint(zeros_ones, zeros_diagonal, config, x):
    sum_v = [
        zeros_ones @ x[: zeros_ones.shape[1]] >= 0.0,
        zeros_ones @ x[: zeros_ones.shape[1]]
        <= config["total_turnover"] + x[config["alpha_index"]],
    ]

    positive_v = [
        zeros_diagonal @ x[: zeros_diagonal.shape[1]] >= 0.0,
        zeros_diagonal @ x[: zeros_diagonal.shape[1]] <= math.inf,
    ]
    return sum_v, positive_v


def alpha_constraint(config, x):
    alpha_gate = [x[config["alpha_index"]] >= 0]

    return alpha_gate


def objective_a(x, weights_two, config):
    diff = (
        x[: int(weights_two.shape[0] / config["number_buffer"])]
        - weights_two[: int(weights_two.shape[0] / config["number_buffer"])]
    )
    return cp.sum_squares(diff)


def objective_b(x, weights_two, config):
    diff = (
        x[: int(weights_two.shape[0] / config["number_buffer"])]
        - weights_two[: int(weights_two.shape[0] / config["number_buffer"])]
    )
    return cp.sum_squares(diff) + config["alpha"] * (x[-config["alpha_index"]]) ** 2


def objective_c(x, weights_two, config):
    diff = (
        x[: int(weights_two.shape[0] / config["number_buffer"])]
        - weights_two[: int(weights_two.shape[0] / config["number_buffer"])]
    )
    return (
        cp.sum_squares(diff)
        + config["alpha"] * (x[config["alpha_index"]]) ** 2
        + config["beta"] * (x[config["beta_index"]]) ** 2
    )


def print_details(
    optimize_weights, weights_two, all_pos, all_neg, result_x, sector_w, config
):
    n_stocks = optimize_weights.shape[0]
    print("---------------- VERIFICATION: --------------")
    if optimize_weights is not None and result_x is not None:
        print(f"net exposure: {optimize_weights.sum()}")
        print(f"short sum: {np.minimum(optimize_weights, 0).sum()}")
        print(f"long sum: {np.maximum(optimize_weights, 0).sum()}")

        pos1 = (optimize_weights[:n_stocks] >= 0).sum()
        print(f"force positive: {pos1 == all_pos[:n_stocks].sum()}")
        neg1 = (optimize_weights[:n_stocks] <= 0).sum()
        print(f"force negative: {neg1 == all_neg[:n_stocks].sum()}")
        print(
            f"sector weights: {sector_w @ result_x[: config['number_buffer'] * n_stocks] <= config['sector_abs_weight']}"
        )
        print(
            f"positive v: {(result_x[n_stocks : -config['number_gates']] >= 0).sum() == n_stocks}"
        )
        print(
            f"over total turnover cap?: {result_x[n_stocks : -config['number_gates']].sum() > config['total_turnover'] + config['eps']}"
        )
        # print(
        #     f"turnover: {result_x[n_stocks : -config['number_gates']].sum() <= config['total_turnover'] + config['eps']}"
        # )
    else:
        print("optimization failed")

    print("----------------- OPTIMIZATION DETAILS: --------------")
    print(f"n_stocks: {n_stocks}")
    # print(f"abs_starting_weights: {config['abs_starting_weights']}")
    print(f"n_sectors: {config['n_sectors']}")
    print(f"max_weight: {config['max_weight']}")
    print(f"total_turnover: {config['total_turnover']}")
    print(f"target_std: {config['target_std']}")
    print(f"sector_abs_weight: {config['sector_abs_weight']}")
    print(f"eps: {config['eps']}")
    print(f"alpha: {config['alpha']}")
    print(f"beta: {config['beta']}")
    print("----------------- POST RUN STATISTICS: --------------")
    print(
        f"no buffer objective: {((optimize_weights - weights_two[:n_stocks]) ** 2).sum()}"
    )
    correlation = stats.pearsonr(optimize_weights, weights_two[:n_stocks])
    print(f"correlation: {correlation.statistic}")


def run_optimize():
    config = get_config()
    weights_one, weights_two, x = generate_weights(config)
    config["alpha_index"] = weights_one.shape[0]
    config["beta_index"] = weights_one.shape[0] + 1

    n_stocks = int(weights_one.shape[0] / config["number_buffer"])
    zeros = np.zeros(n_stocks)
    ones = np.ones(n_stocks)
    diagonal_a = sp.eye(n_stocks, format="csr")
    double_diagonal = sp.hstack((diagonal_a, diagonal_a), format="csr")
    diagonal_zeros = sp.hstack(
        (diagonal_a, np.zeros((n_stocks, n_stocks))), format="csr"
    )
    zeros_diagonal = sp.hstack(
        (np.zeros((n_stocks, n_stocks)), diagonal_a), format="csr"
    )
    ones_zeros = sp.hstack((sp.coo_matrix(ones), sp.coo_matrix(zeros)), format="csr")
    zeros_ones = sp.hstack((sp.coo_matrix(zeros), sp.coo_matrix(ones)), format="csr")
    diagonal_negdiagonal = sp.hstack((diagonal_a, -diagonal_a), format="csr")
    negdiagonal_diagonal = sp.hstack((-diagonal_a, diagonal_a), format="csr")

    lb_stocks, ub_stocks, all_pos_padded, all_neg_padded, force_zero_padded = (
        generate_bounds_masks(
            weights_one,
            weights_two,
            config,
        )
    )

    # Constraints
    sector_weights = create_sectors(config, weights_two)
    sector_cons = sector_constraint(sector_weights, config, x)
    long_sum = net_exposure_constraint(all_pos_padded, config, x)
    force_sign = force_sign_constraint(all_pos_padded, all_neg_padded, config, x)
    dollar_neutral = dollar_neutral_constraint(ones_zeros, config, x)
    individual_weight = individual_constraint(
        diagonal_zeros, config, lb_stocks, ub_stocks, x
    )
    total_turnover = total_turnover_component_constraint(
        double_diagonal, diagonal_negdiagonal, config, weights_one, x
    )
    sum_v, positive_v = v_constraint(zeros_ones, zeros_diagonal, config, x)
    alpha_cons = alpha_constraint(config, x)
    force_zero = force_zero_constraint(force_zero_padded, config, x)

    if force_zero_padded[:n_stocks].any():
        constraints = [
            sector_cons,
            long_sum,
            force_sign,
            dollar_neutral,
            individual_weight,
            total_turnover,
            sum_v,
            positive_v,
            alpha_cons,
            force_zero,
        ]
        constraints = functools.reduce(operator.add, constraints)
    else:
        constraints = [
            sector_cons,
            long_sum,
            force_sign,
            dollar_neutral,
            individual_weight,
            total_turnover,
            sum_v,
            positive_v,
            alpha_cons,
        ]
        constraints = functools.reduce(operator.add, constraints)

    TOL = 1e-10
    MAX_ITER = 100000

    # Clarabel Configuration
    clarabel_opts = {
        "solver": cp.CLARABEL,
        "tol_feas": TOL,  # Primal & Dual feasibility threshold
        "tol_gap_abs": TOL,  # Absolute duality gap
        "tol_gap_rel": TOL,  # Relative duality gap
        "max_iter": MAX_ITER,
        "verbose": False,  # Disable stdout logging to ensure raw speed evaluation
    }

    # PIQP Configuration
    piqp_opts = {
        "solver": cp.PIQP,
        "eps_abs": TOL,  # Absolute tolerance (Primal, Dual, and Gap)
        "eps_rel": 0.0,  # Set relative to 0.0 or 1e-9 to enforce absolute alignment
        "max_iter": MAX_ITER,
        "verbose": False,
    }

    problem = cp.Problem(
        cp.Minimize(objective_b(x, weights_two, config)),
        constraints=constraints,
    )

    start_time = time.perf_counter()
    problem.solve(**clarabel_opts)
    # problem.solve(**piqp_opts)
    end_time = time.perf_counter()
    print(f"Solver time: {end_time - start_time:.4f} seconds")
    if problem.status == "optimal":
        optimize_weights = x.value[:n_stocks]
    else:
        optimize_weights = None

    print("Status:", problem.status)
    print("Optimal objective value:", problem.value)
    print_details(
        optimize_weights,
        weights_two,
        all_pos_padded,
        all_neg_padded,
        x.value,
        sector_weights,
        config,
    )

import numpy as np
import cvxpy as cp

def lp_problem():
    """Pure LP: only zero/nonneg cones, no SOC or PSD involved."""
    x = cp.Variable(3)
    constraints = [x >= 0, cp.sum(x) == 1, x[0] + 2 * x[1] <= 1.5]
    prob = cp.Problem(cp.Minimize(x[0] - 2 * x[1] + x[2]), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)
    return prob, x


def qp_problem():
    """QP: exercises Clarabel's native quadratic-objective support
    (no epigraph reformulation needed, unlike ECOS/SOCP-only solvers)."""
    np.random.seed(0)
    n = 10
    A = np.random.randn(n, n)
    P = A.T @ A + 1e-3 * np.eye(n)  # PSD
    q = np.random.randn(n)

    x = cp.Variable(n)
    constraints = [x >= -1, x <= 1, cp.sum(x) == 0]
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)
    return prob, x


def socp_problem():
    """SOCP: an explicit second-order cone constraint."""
    x = cp.Variable(3)
    constraints = [
        cp.SOC(x[2], x[0:2]),  # ||x[0:2]||_2 <= x[2]
        x[0] + x[1] == 1,
    ]
    prob = cp.Problem(cp.Minimize(x[2]), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)
    return prob, x


def norm1_problem():
    """L1-norm constraint: DCP compiles this to cone form automatically,
    no manual auxiliary/epigraph variables required."""
    np.random.seed(1)
    n = 8
    x0 = np.random.randn(n) * 0.1
    x = cp.Variable(n)
    constraints = [cp.norm1(x - x0) <= 0.5, cp.sum(x) == 0]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)
    return prob, x


def infeasible_problem():
    """Deliberately infeasible, to see the status string cvxpy/Clarabel
    return instead of an opaque exit code."""
    x = cp.Variable(2)
    constraints = [x >= 1, x <= 0]
    prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)
    return prob, x


def solver_options_demo():
    """Same QP solved twice: default max_iter vs. an artificially low one,
    to see how solver_stats and status change."""
    prob1, _ = qp_problem()
    print(
        "\ndefault settings -> status:", prob1.status,
        "| iters:", prob1.solver_stats.num_iters,
        "| solve_time:", prob1.solver_stats.solve_time,
    )

    x2 = cp.Variable(10)
    np.random.seed(0)
    A = np.random.randn(10, 10)
    P = A.T @ A + 1e-3 * np.eye(10)
    q = np.random.randn(10)
    constraints = [x2 >= -1, x2 <= 1, cp.sum(x2) == 0]
    prob2 = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x2, P) + q @ x2), constraints)
    prob2.solve(solver=cp.CLARABEL, max_iter=5, verbose=True)
    print("max_iter=5 -> status:", prob2.status)
    return prob1, prob2


def warm_start_demo():
    """Solve a parametrized least-squares problem twice; second solve
    is warm-started to show the effect on solve_time."""
    m, n = 200, 100
    np.random.seed(2)
    A = np.random.randn(m, n)
    b = cp.Parameter(m)

    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

    b.value = np.random.randn(m)
    prob.solve(solver=cp.CLARABEL)
    t_cold = prob.solver_stats.solve_time

    b.value = np.random.randn(m)
    prob.solve(solver=cp.CLARABEL, warm_start=True)
    t_warm = prob.solver_stats.solve_time

    print(f"cold solve: {t_cold:.4f}s | warm-started solve: {t_warm:.4f}s")
    return prob, x


if __name__ == "__main__":
    check_environment()

    print("\n--- LP ---")
    lp_prob, lp_x = lp_problem()

    print("\n--- QP ---")
    qp_prob, qp_x = qp_problem()

    print("\n--- SOCP ---")
    socp_prob, socp_x = socp_problem()

    print("\n--- L1 norm (auto conic reformulation) ---")
    norm1_prob, norm1_x = norm1_problem()

    print("\n--- Infeasible ---")
    infeas_prob, infeas_x = infeasible_problem()

    print("\n--- Solver options ---")
    opt_prob1, opt_prob2 = solver_options_demo()

    print("\n--- Warm start ---")
    ws_prob, ws_x = warm_start_demo()

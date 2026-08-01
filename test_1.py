import cvxpy as cp
import numpy as np

# Create variables
x = cp.Variable(2, name="x")

# Formulate objective and constraints
objective = cp.Minimize(0.5 * cp.sum_squares(x) + np.array([1.0, 2.0]) @ x)
constraints = [
    x >= 0,
    x[0] + x[1] == 1
]

# Form problem
problem = cp.Problem(objective, constraints)

# Solve using Clarabel
problem.solve(solver=cp.CLARABEL, verbose=True, max_iter=100)

# Access results
print("Status:", problem.status)
print("Optimal value:", problem.value)
print("Optimal x:", x.value)

from scipy.optimize import linprog
import warnings

# Coefficients of the objective function
c = [2, -3, 0, -5]

# Coefficients of the inequality constraints
A = [[-1, 1, -1, -1], [2, 4, 0, 0], [0, 0, 1, 1]]

# Right-hand side of the inequality constraints
b = [8, 10, 3]

# Bounds for the variables
x0_bounds = (0, None)
x1_bounds = (0, None)
x2_bounds = (0, None)
x3_bounds = (0, None)

# Solve the problem
warnings.filterwarnings("ignore", category=DeprecationWarning)
# method simplex will be removed in 1.11
res = linprog(
    c,
    A_ub=A,
    b_ub=b,
    bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds],
    method="simplex",
)

print("Optimal value:", res.fun)
print("x:", res.x)

# Right-hand side of the inequality constraints
b = [3, 0, 6]

# Solve the problem
res = linprog(
    c,
    A_ub=A,
    b_ub=b,
    bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds],
    method="simplex",
)

print("Optimal value:", res.fun)
print("x:", res.x)

# Right-hand side of the inequality constraints
b = [0, 3, 9]

# Solve the problem
res = linprog(
    c,
    A_ub=A,
    b_ub=b,
    bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds],
    method="simplex",
)

print("Optimal value:", res.fun)
print("x:", res.x)

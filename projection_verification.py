
"""
Econometrics I - Problem Set 1, Question 5
Numerical verification of projection matrices
and the Frisch-Waugh-Lovell theorem.

This supplements, but does not replace,
the algebraic proof in the LaTeX submission.
"""

import numpy as np


# 1. Reproducible example data
rng = np.random.default_rng(42)
n = 100

Z = rng.normal(size=(n, 1))
W = rng.normal(size=(n, 1))
X = rng.normal(size=(n, 1))
epsilon = rng.normal(size=(n, 1))

Y = 2.0 * X + 1.5 * W - 0.5 * Z + epsilon


# 2. Define the residual-maker matrix
def residual_maker(A):
    """
    M_A = I - A(A'A)^(-1)A'
    """
    identity = np.eye(A.shape[0])
    projection = A @ np.linalg.solve(A.T @ A, A.T)
    return identity - projection


# 3. Construct the matrices
M_Z = residual_maker(Z)

WZ = np.column_stack((W, Z))
M_WZ = residual_maker(WZ)

M_ZW = M_Z @ W
M_MZW = residual_maker(M_ZW)


# 4. Verify the matrix identity
lhs = M_WZ
rhs = M_MZW @ M_Z

max_error = np.max(np.abs(lhs - rhs))
frobenius_error = np.linalg.norm(lhs - rhs, ord="fro")

print("QUESTION 5: PROJECTION IDENTITY")
print("--------------------------------------")
print(f"Maximum absolute error: {max_error:.12e}")
print(f"Frobenius error:        {frobenius_error:.12e}")

assert np.allclose(lhs, rhs, atol=1e-10, rtol=0)

print("Projection identity verified.")


# 5. Three-step regression procedure

# Step 1: Residualise Y, X and W on Z
R_Y = M_Z @ Y
R_X = M_Z @ X
R_W = M_Z @ W

# Step 2: Residualise R_Y and R_X on R_W
M_RW = residual_maker(R_W)

R_Y_tilde = M_RW @ R_Y
R_X_tilde = M_RW @ R_X

# Step 3: Regress residualised Y on residualised X
beta_three_step = np.linalg.lstsq(
    R_X_tilde, R_Y_tilde, rcond=None
)[0][0, 0]


# 6. Full OLS regression
design = np.column_stack((X, W, Z))

beta_full = np.linalg.lstsq(
    design, Y, rcond=None
)[0][0, 0]


# 7. Compare coefficient estimates
coefficient_error = abs(beta_three_step - beta_full)

print()
print("QUESTION 5: FWL VERIFICATION")
print("--------------------------------------")
print(f"Full OLS beta_1:      {beta_full:.10f}")
print(f"Three-step beta_1:    {beta_three_step:.10f}")
print(f"Absolute difference:  {coefficient_error:.12e}")

assert np.isclose(
    beta_full, beta_three_step,
    atol=1e-10, rtol=0
)

print("FWL equivalence verified.")


"""
Econometrics I - Problem Set 1, Question 2(d)
Optimal population CART of depth one.

Model:
y = max(x, 0) + x * epsilon

x and epsilon are independent standard normals.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar


def cart_predictions(t):
    """Optimal left/right predictions at threshold t."""
    if t < 0:
        a = 0.0
        b = norm.pdf(0) / norm.sf(t)
    else:
        a = (norm.pdf(0) - norm.pdf(t)) / norm.cdf(t)
        b = norm.pdf(t) / norm.sf(t)
    return a, b


def cart_mse(t):
    """Population expected squared prediction error."""
    a, b = cart_predictions(t)
    return (
        1.5
        - a**2 * norm.cdf(t)
        - b**2 * norm.sf(t)
    )


# Negative thresholds have their best MSE at t=0.
# Optimise over a sufficiently broad nonnegative region.
result = minimize_scalar(
    cart_mse,
    bounds=(0, 5),
    method="bounded",
    options={"xatol": 1e-12},
)

if not result.success:
    raise RuntimeError("Optimisation failed.")

threshold = result.x
a, b = cart_predictions(threshold)
mse_cart = cart_mse(threshold)

# Check the boundary.
if cart_mse(0) < mse_cart:
    threshold = 0.0
    a, b = cart_predictions(threshold)
    mse_cart = cart_mse(threshold)

mse_cef = 1.0
mse_blp = 1.25 - 1 / (2 * np.pi)

print("QUESTION 2(d)")
print("-----------------------------")
print(f"Optimal threshold: {threshold:.8f}")
print(f"Left prediction:   {a:.8f}")
print(f"Right prediction:  {b:.8f}")
print(f"CART MSE:          {mse_cart:.8f}")
print(f"CEF MSE:           {mse_cef:.8f}")
print(f"BLP MSE:           {mse_blp:.8f}")
print(f"FOC check:         {(a+b)/2:.8f}")

# Plot the three predictors.
x = np.linspace(-3, 3, 1001)
cef = np.maximum(x, 0)
blp = 1 / np.sqrt(2 * np.pi) + 0.5 * x

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x, cef, label="CEF", linewidth=2)
ax.plot(x, blp, label="BLP", linewidth=2)

# Plot CART in two pieces to show the discontinuity.
ax.hlines(a, -3, threshold, label="CART", linewidth=2)
ax.hlines(b, threshold, 3, linewidth=2)
ax.axvline(threshold, linestyle="--", alpha=0.5)

ax.set_xlabel("x")
ax.set_ylabel("Predicted y")
ax.set_title("Problem 2(d): CEF, BLP and optimal CART")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()

fig.savefig("cart_comparison.png", dpi=300)
plt.show()

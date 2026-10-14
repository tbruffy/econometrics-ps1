# Econometrics I — Problem Set 1

Supplementary Python calculations for the
Econometrics I Problem Set 1.

The analytical solutions are presented separately
in the LaTeX submission.

## Contents

### Question 2(d): Optimal CART

File: `cart_optimization.py`

Numerically minimises the population mean squared
error of a depth-one CART predictor.

The model is:

    y = max(x, 0) + x * epsilon

where x and epsilon are independent N(0,1).

The script:

- Derives optimal leaf predictions for a given split.
- Numerically optimises the CART threshold.
- Calculates CEF, BLP and CART mean squared errors.
- Produces `cart_comparison.png`.

### Question 5: Projection Matrices

File: `projection_verification.py`

Numerically checks the matrix identity:

    M_(W,Z) = M_(M_Z W) M_Z

The script also confirms that the three-step
residualisation procedure produces the same OLS
coefficient as the full multiple regression.

A fixed random seed ensures reproducibility.

Numerical verification supplements, rather than
replaces, the analytical proof.

## How to reproduce

Install Python 3 and the dependencies:

    python -m pip install -r requirements.txt

Run Question 2(d):

    python cart_optimization.py

Run Question 5:

    python projection_verification.py

The Question 2(d) script saves the figure
`cart_comparison.png` in the working directory.

## Dependencies

- NumPy
- SciPy
- Matplotlib

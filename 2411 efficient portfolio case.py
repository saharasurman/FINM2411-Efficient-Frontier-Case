import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


mu_E = 0.095  # Expected return of equities
sigma_E = 0.15  # Standard deviation of equities
mu_D = 0.055  # Expected return of bonds
sigma_D = 0.065  # Standard deviation of bonds
cor_ED = -0.7  # Correlation between equities and bonds
b = 0.003  # Risk aversion coefficient


def calculate_utility(E_Wt, Var_Wt, b, W_0):
    """
    Calculating the utility when weight is t
    """
    
    E_U_Wt = W_0 * (E_Wt - b * ( Var_Wt + E_Wt**2 ))

    return E_U_Wt


def calc_expected_return(w_E, w_D, mu_E, mu_D): 
    """
    Calculating the expected return of the portfolio. 
    """

    E_rp = (w_E * mu_E) + (w_D * mu_D)
    return E_rp


def calc_variance(w_E, w_D, sigma_E, sigma_D, cor_ED): 
    """
    Calculating the variance of the portfolio.
    """

    Var_rp = (w_E**2 * sigma_E**2) + (w_D**2 * sigma_D**2) + (2 * w_E * w_D * cor_ED * sigma_E * sigma_D)
    return Var_rp

def maximise_utility(w_D): 

    w_E = 1 - w_D
    w_0 = w_E + w_D

    E_rp = calc_expected_return(w_E, w_D, mu_E, mu_D)

    Var_rp = calc_variance(w_E, w_D, sigma_E, sigma_D, cor_ED)

    E_U_wt = calculate_utility(E_rp, Var_rp, b, w_0)

    return E_U_wt  # We want to maximize utility, so we return the negative value




# Define the objective function for minimization (negative utility)
def objective(w_D):
    return -maximise_utility(w_D)

# Perform the optimization within bounds [0, 1]
result = minimize_scalar(objective, bounds=(0, 1), method='bounded')

# Extract the optimal weight in bonds
optimal_w_D = result.x
max_utility = -result.fun  # Negate to get the actual maximum utility

# Output the results
print(f"Optimal weight in bonds (w_D): {optimal_w_D:.4f}")
print(f"Maximum expected utility: {max_utility:.4f}")
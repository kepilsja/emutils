import numpy as np

def n_ge(x):
    return (15.9871824405994 + 1/(x**2) + 2.2955370277312/(x**2 - 0.213744089567103**2))**0.5

def n_si(x):
    return (11.67316 + 1/(x**2) + 0.004482633/(x**2 - 1.108205**2))**0.5

def n_air(x):
    return 1.00027278 * np.ones_like(x)
import numpy as np

def gaussian(x, A, x0, sigma):
    """Gaussian function: A * exp(-(x - x0)^2 / (2 * sigma^2))"""
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))
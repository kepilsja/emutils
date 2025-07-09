import numpy as np

def n_ge(x):
    '''
    Formula from [Burnett et al. 2016](https://refractiveindex.info/?shelf=main&book=Ge&page=Burnett)
    applicable for 0.4 $\mu$m < x < 11.0 $\mu$m.
    '''
    applicable_range = (0.4, 11.0)
    if x < applicable_range[0] or x > applicable_range[1]:
        raise ValueError(f'{x = } is outside the range {applicable_range}')
    return (1+0.4886331/(1-1.393959/x**2)+14.5142535/(1-0.1626427/x**2)+0.0091224/(1-752.190/x**2))**.5
    # old formula, unknown source
    # return (15.9871824405994 + 1/(x**2) + 2.2955370277312/(x**2 - 0.213744089567103**2))**0.5

def n_si(x):
    '''
    Formula from [Chandler-Horowitz and Amirtharaj 2005](https://refractiveindex.info/?shelf=main&book=Si&page=Chandler-Horowitz)
    applicable for 2.5 $\mu$m < x <22.2 $\mu$m.
    '''
    applicable_range = (2.5, 22.2)
    if x < applicable_range[0] or x > applicable_range[1]:
        raise ValueError(f'{x = } is outside the range {applicable_range}')
    return (11.67316 + 1/(x**2) + 0.004482633/(x**2 - 1.108205**2))**0.5

def n_air(x):
    return 1.00027278 * np.ones_like(x)
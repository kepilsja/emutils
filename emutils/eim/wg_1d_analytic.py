import numpy as np
from scipy.optimize import fsolve

def solve_1D_analytic(lbd0, t, n1, n2, n3, mode='both'):
    """
    Computes the effective indices for TE and TM modes in a 3-layer waveguide.
    
    Parameters:
    lambda_ : float
        Wavelength of the incident light.
    t : float
        Thickness of the core layer.
    n1, n2, n3 : float
        Refractive indices of the cladding, core, and substrate, respectively.
    mode : str, optional
        Specifies which mode to calculate ('TE', 'TM', or 'both'). Default is 'both'.

    Returns:
    nTE : list
        Effective indices for TE modes.
    TEparam : list
        Parameters for TE modes.
    nTM : list
        Effective indices for TM modes.
    TMparam : list
        Parameters for TM modes.
    """
    k0 = 2 * np.pi / lbd0
    b0 = np.linspace(max(n1, n3) * k0, n2 * k0, 1000)[:-1]  # k0*n3 < b < k0*n2
    
    nTE, TEparam, nTM, TMparam = [], [], [], []
    
    if mode in ['both', 'TE']:
        te0 = TE_eq(b0, k0, n1, n2, n3, t)[0]
        intervals = (te0 >= 0).astype(int) - (te0 < 0).astype(int)
        izeros = np.where(np.diff(intervals) < 0)[0]
        X0 = np.array([b0[izeros], b0[izeros + 1]]).T
        
        for x in X0:
            root = fsolve(lambda x: TE_eq(x, k0, n1, n2, n3, t)[0], x)[0] / k0
            nTE.append(root)
            TEparam.append(TE_eq(root * k0, k0, n1, n2, n3, t)[1:])
        
        nTE = nTE[::-1]
        TEparam = TEparam[::-1]
    
    if mode in ['both', 'TM']:
        tm0 = TM_eq(b0, k0, n1, n2, n3, t)[0]
        intervals = (tm0 >= 0).astype(int) - (tm0 < 0).astype(int)
        izeros = np.where(np.diff(intervals) < 0)[0]
        X0 = np.array([b0[izeros], b0[izeros + 1]]).T
        
        for x in X0:
            root = fsolve(lambda x: TM_eq(x, k0, n1, n2, n3, t)[0], x)[0] / k0
            nTM.append(root)
            TMparam.append(TM_eq(root * k0, k0, n1, n2, n3, t)[1:])
        
        nTM = nTM[::-1] if nTM else []
        TMparam = TMparam[::-1] if nTM else []
    
    return (nTE, TEparam) if mode=='TE' else (nTM, TMparam) if mode =='TM' else (nTE, TEparam, nTM, TMparam)

def TE_eq(b0, k0, n1, n2, n3, t):
    """
    Computes the TE mode equation.
    """
    h0 = np.sqrt((n2 * k0) ** 2 - b0 ** 2)
    q0 = np.sqrt(b0 ** 2 - (n1 * k0) ** 2)
    p0 = np.sqrt(b0 ** 2 - (n3 * k0) ** 2)
    te0 = np.tan(h0 * t) - (p0 + q0) / h0 / (1 - (p0 * q0) / h0 ** 2)
    return te0, h0, q0, p0

def TM_eq(b0, k0, n1, n2, n3, t):
    """
    Computes the TM mode equation.
    """
    h0 = np.sqrt((n2 * k0) ** 2 - b0 ** 2)
    q0 = np.sqrt(b0 ** 2 - (n1 * k0) ** 2)
    p0 = np.sqrt(b0 ** 2 - (n3 * k0) ** 2)
    pbar0 = (n2 / n3) ** 2 * p0
    qbar0 = (n2 / n1) ** 2 * q0
    tm0 = np.tan(h0 * t) - h0 * (pbar0 + qbar0) / (h0 ** 2 - pbar0 * qbar0)
    return tm0, h0, q0, p0

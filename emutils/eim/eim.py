from typing import Tuple
import numpy as np

from ..materials import *

um = 1e-6
nm = 1e-9

def eim_rib(w: float, core_h: float, clad_h:float , lbd:float,
            n: Tuple[float, float, float], mode='TM') -> float:
    '''
    Calculate the effective index of a rib waveguide using effective index
    method approximation.

    Rib waveguide cross section:
               cladding
              __________
    _________|          |_________
                 core
    ______________________________

               substrate

        II   |     I    |    II      <--- vertical sections

    ------------------------------------------------
    
    Parameters:
    w : float
        Width of the I section in microns.
    core_h : float
        Height of the I section core layer in microns.
    clad_h : float            
        Height of the II section core layer in microns.
    lbd : float
        Wavelength of the incident light in microns.
    n : tuple[int, int, int]
        Refractive indices of the top cladding, core, and substrate, respectively.
    mode : str, optional
        Specifies which mode to calculate ('TE', 'TM'). Default is 'TM'.

    Returns:
    n_eff : float
        Effective index of the rib waveguide.
    '''
    assert mode in ['TM', 'TE'], "Invalid mode. Choose 'TM' or 'TE'."
    assert w > 0, "Width must be positive."
    assert core_h > 0, "Core height must be positive."
    assert clad_h > 0, "Cladding height must be positive."
    assert lbd > 0, "Wavelength must be positive."
    assert all([ni>0 for ni in n]), "Refractive indices must be positive."

    from .wg_1d_analytic import solve_1D_analytic

    n1, n2, n3 = n
    w *= um
    core_h *= um
    clad_h *= um
    lbd *= um
    
    core_n_eff = solve_1D_analytic(lbd, core_h, n1, n2, n3, mode)[0]
    clad_n_eff = solve_1D_analytic(lbd, clad_h, n1, n2, n3, mode)[0]
    
    if not core_n_eff:
        raise ValueError("No solution in section I found for the given parameters.")
    if not clad_n_eff:
        raise ValueError("No solution in section II found for the given parameters.")

    n_eff = solve_1D_analytic(lbd, w, clad_n_eff[0], core_n_eff[0], clad_n_eff[0],
                              mode='TM' if mode=='TE' else 'TM')[0]

    if not n_eff:
        raise ValueError("No solution in the rib waveguide found for the given parameters.")

    return n_eff[0]

    
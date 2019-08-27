"""tests for toolbox.py module."""
import numpy as np
from toolbox import *


def lj(r):
    """Return lennard jones force and potential."""
    F = 4 * epsilon / (r-R) * (12 * (sigma / (r-R))**12 - 6 * (sigma / (r-R))**6)
    V = 4 * epsilon * ((sigma / (r-R))**12 - (sigma / (r-R))**6)
    return(V, F)


def test_force_to_potential():
    """Uses a lj potential to test force_to_potential()."""
    epsilon = 2.0
    sigma = 1.0
    R = 25
    r_values = np.linspace(0.8+R, 2.7*sigma+R, 50)
    V_true, F_true = lj(r_values)

    V = force_to_potential(r_values, F_true)


def test_potential_to_force():
    """Uses a lj potential to test force_to_potential()."""
    epsilon = 2.0
    sigma = 1.0
    R = 25
    r_values = np.linspace(0.8+R, 2.7*sigma+R, 50)
    V_true, F_true = lj(r_values)

    F = potential_to_force(r_values, V_true)

def test_vector_autocorr():
    """ """
    unit_vecs = np.array([[0,0,1], [0,0,1]])
    print(unit_vecs.shape)
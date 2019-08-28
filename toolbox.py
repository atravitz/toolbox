"""Contains functions that are commonly used in polymer-colloid studies."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import integrate

def lj(r, epsilon, sigma, R):
    """Return lennard jones force and potential."""
    F = 4 * epsilon / (r-R) * (12 * (sigma / (r-R))**12 - 6 * (sigma / (r-R))**6)
    V = 4 * epsilon * ((sigma / (r-R))**12 - (sigma / (r-R))**6)
    return(V, F)

def force_to_potential(dist, force):
    """
    Return potential (V) for tabulated force using trapezoidal integration.

    Assumes the potential goes to 0 at last data point.
    Keyword arguments:
    dist -- 1D numpy array containing interparticle distances
    force -- 1D numpy array containing the corresponding force values
    """


    V_trap = -integrate.cumtrapz(y=force, x=dist)
    V_trap -= V_trap[-1]
    return V_trap


def potential_to_force(dist, potential):
    """
    Return force for a given tabulated potential using numpy gradient method.

    Assumes a uniform spacing between dist values.
    Keyword arguments:
    dist -- 1D numpy array containing interparticle distances
    force -- 1D numpy array containing the corresponding potential values
    """
    interval = dist[1]-dist[0]
    F = -np.gradient(potential, interval)
    return F


def statepoint_df(project):
    """Return a pandas dataframe for all statepoints in a project."""
    df_index = pd.DataFrame(project.index())
    df_index = df_index.set_index(['_id'])
    statepoints = {doc['_id']: doc['statepoint'] for doc in project.index()}
    df = pd.DataFrame(statepoints).T.join(df_index)
    return df


def periodicBC(vec, box):
    """ periodic BC from E. Teich's PHYS 514 hw 3 (courtesy of E. Gull) """
    newvec=np.copy(vec)
    if (newvec[0] > box[0]/2.): newvec[0] -= box[0]
    if (newvec[0] < -box[0]/2.): newvec[0] += box[0]
    if (newvec[1] > box[1]/2.): newvec[1] -= box[1]
    if (newvec[1] < -box[1]/2.): newvec[1] += box[1]
    if (newvec[2] > box[2]/2.): newvec[2] -= box[2]
    if (newvec[2] < -box[2]/2.): newvec[2] += box[2]
    return newvec


def vector_autocorr(vec):
    """perform autocorrelation on a vector,
       returns a list containing correlation values
       """
    mean = np.mean(vec, axis=0)
    var = np.var(vec, axis=0)
    n = len(vec)
    R = np.zeros(n)
    for k in np.arange(0,n):
        sum = 0
        for t in np.arange(0, n-k):
            sum += np.dot(vec[t],vec[t+k])
        R[k]=sum/((n-k))
    return R

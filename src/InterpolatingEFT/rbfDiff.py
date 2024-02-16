"""
Plots the rbf-interpolated histograms into "out"
"""

import os
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, List, Dict
from scipy.optimize import minimize
from dataSplitter import dataToDict
from rbfSpline import rbfSpline
from interpPlotting import eps
from utils import Data
from dataLoader import NLLDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
hep.style.use("CMS")

def minimizeWrapper(coeffs: npt.NDArray[np.float32], 
                    pois: Optional[List[str]],  
                    spline: Optional[rbfSpline]=None) -> float:
    """
    Wrapper to pass the spline function to scipy.optimize.minimize

    Args:
        coeffs (npt.NDArray[np.float32]): The values of the WCs
        pois (Optional[List[str]]): The poi list
        spline (Optional[rbfSpline], optional): The RBF spline.\
            Defaults to None.

    Returns:
        float: The value of the interpolated function
    """
    if spline and pois:
        pois_map = {poi: coeffs[i] for i, poi in enumerate(pois)}
        return spline.evaluate(pois_map)
    else:
        return np.nan

def interpolate(data: List[Dict[str, float]], 
                eps: float) -> rbfSpline:
    """
    Interpolates the data using RBFs
    """

    # Initialise RBF
    spline = rbfSpline(len(list(data[0].keys()))-1, use_scipy_interp=False)
    spline.initialise(data, "deltaNLL", eps=eps, rescaleAxis=True)

    return spline

# if __name__ == "__main__":
#     interpolate(0.1)
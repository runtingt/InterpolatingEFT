"""
Class definitions for interpolators
"""

import os
import pandas as pd
import numpy as np
import uproot
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from InterpolatingEFT.utils import Data
from InterpolatingEFT.dataSplitter import loadAndSplit
from InterpolatingEFT.dataLoader import NLLDataset, loadData
from InterpolatingEFT.rbfSplineFast import rbfSplineFast
from scipy.optimize import OptimizeResult, minimize

class Combine1D:
    """
    Holds the data for the 1D scans from Combine
    """
    def __init__(self, data_config: Data, poi: str) -> None:
        """
        Grabs the data from the Combine output
        """
        self.data_dir = 'data'
        self.stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
                     f"{data_config['type']}.{data_config['attribute']}.")
        self.data = loadData(
            os.path.join(os.path.abspath(self.data_dir), 
                         f"{self.stem}{poi}.root"),
            list(data_config["POIs"].keys()), include_best=True)
        
class Combine2D:
    """
    Holds the data for the 1D scans from Combine
    """
    def __init__(self, data_config: Data, pair_name: str) -> None:
        """
        Grabs the data from the Combine output
        """
        # Extract the contour from the ROOT (NLL) plot
        file = uproot.open(f"contours/{pair_name}.root")
        assert isinstance(file, uproot.ReadOnlyDirectory)
        self.contours = {}
        for ci in [68, 95]: 
            self.contours[ci] = file[f'graph{ci}_default_0;1'].values()
        
        # Get data
        self.data_dir = 'data'
        self.stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
                     f"{data_config['type']}.{data_config['attribute']}.")
        self.data = loadData(
            os.path.join(os.path.abspath(self.data_dir), 
                         f"{self.stem}{pair_name}.root"),
            list(data_config["POIs"].keys()), include_best=True)
        
        file.close()

class Interpolator(ABC):
    """
    Abstract base class for a generic interpolator

    Args:
        ABC (ABC): Helper class for abstract base classes
    """
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialises the Interpolator
        """
        super().__init__()

    @abstractmethod
    def initialise(self, data_config: Data) -> None:
        """
        Loads data and computes weights for the interpolator

        Args:
            data_config (Data): Options for the data
        """
        pass
        
    @abstractmethod
    def evaluate(self, point: Any) -> float:
        """
        Evaluates the interpolator at a specified point

        Args:
            point (Any): The point to evaluate at

        Returns:
            float: The output of the interpolator at that point
        """
        return 0
        
    @abstractmethod
    def minimize(self, free_keys: List[str], 
                 fixed_vals: Dict[str, List[float]]) -> OptimizeResult:
        """
        Minimize the interpolator using SciPy

        Args:
            free_keys (List[str]): The keys to minimise over
            fixed_vals (Dict[str, List[float]]): The fixed values in the fit

        Returns:
            OptimizeResult: The result of optimising
        """
        pass

class rbfInterpolator(Interpolator):
    """
    Interpolates a surface using radial basis functions

    Args:
        Interpolator (Interpolator): The abstract base class
    """
    def __init__(self) -> None:
        """
        Initialises the Interpolator
        """
        super().__init__()
        self.data_dir = 'data'

    def initialise(self, data_config: Data):
        """
        Loads data and computes weights for the interpolator

        Args:
            data_config (Data): Options for the data
        """
        super().initialise(data_config)
        
        self.pois = list(data_config["POIs"].keys())
        self.bounds = [tuple(*val.values()) 
                       for val in data_config["POIs"].values()]
        self.stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
                     f"{data_config['type']}.{data_config['attribute']}.")
            
        # Get a splits as dataframes
        datasets = [] 
        data_train, data_test  = loadAndSplit(
            os.path.join(os.path.abspath(self.data_dir), 
                         f"{self.stem}{'_'.join(self.pois)}.root"), 
            data_config, include_best=True,
            split=data_config["fraction"])
        for data in [data_train, data_test]:
            assert isinstance(data.dataset, NLLDataset)
            data_tup = data.dataset[list(data.indices)]
            datasets.append(NLLDataset(*data_tup, data.dataset.POIs))
        self.best = datasets[0].X[np.argmin(datasets[0].Y)].detach().numpy()
        self.data_train = datasets[0].toFrame()
        self.data_test = datasets[1].toFrame()
        
        # Build interpolator
        self.spline = rbfSplineFast(len(self.data_train.columns)-1)
        self.spline.initialise(self.data_train, "deltaNLL", radial_func="cubic",
                               eps=data_config['interpolator']['eps'],
                               rescaleAxis=True)
        
    def evaluate(self, point: pd.DataFrame) -> float:
        """
        Evaluates the interpolator at a specified point

        Args:
            point (Any): The point to evaluate at

        Returns:
            float: The output of the interpolator at that point
        """
        super().evaluate(point)
        return self.spline.evaluate(point)
    
    def _minimizeWrapper(self, coeffs: List[float], free_keys: List[str],
                         fixed_vals: Dict[str, List[float]]) -> float:
        """
        Handles the passing of parameters from SciPy to the interpolator

        Args:
            coeffs (List[float]): The values of the WCs from SciPy
            free_keys (List[str]): The WCs that are floating in the fit
            fixed_vals (Dict[str, List[float]]): The fixed WC/value pairs

        Returns:
            float: The value of the function at the specified point
        """
        super().minimize(free_keys, fixed_vals)
            
        # Create a blank df
        d = {poi: np.nan for poi in self.pois}
        
        # Fill in the free values
        for key, coeff in zip(free_keys, coeffs):
            d[key] = coeff
        
        # Fill in the fixed values
        for key, val in fixed_vals.items():
            d[key] = val[0]

        return self.evaluate(pd.DataFrame([d]))
    
    def minimize(self, free_keys: List[str], 
                 fixed_vals: Dict[str, List[float]]) -> OptimizeResult:
        """
        Minimize the interpolator using SciPy

        Args:
            free_keys (List[str]): The keys to minimise over
            fixed_vals (Dict[str, List[float]]): The fixed values in the fit

        Returns:
            OptimizeResult: The result of optimising
        """
        super().minimize(free_keys, fixed_vals)
        assert (len(free_keys) + len(fixed_vals)) == len(self.pois)
        
        # Get start point and bounds for free POIs
        start = []
        bounds = []
        for key in free_keys:
            idx = self.pois.index(key)
            start.append(self.best[idx])
            bounds.append(self.bounds[idx])
    
        res = minimize(self._minimizeWrapper, x0=start, bounds=bounds,
                       args=(free_keys, fixed_vals))
        return res

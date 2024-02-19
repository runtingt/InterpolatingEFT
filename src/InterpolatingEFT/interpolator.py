import os
import pandas as pd
import numpy as np
import uproot
from copy import deepcopy
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from utils import Data
from dataSplitter import loadAndSplit
from dataLoader import NLLDataset, loadData
from rbfSplineFast import rbfSplineFast
from scipy.optimize import OptimizeResult, minimize

class Combine1D:
    def __init__(self, data_config: Data, poi: str) -> None:
        self.data_dir = 'data'
        self.stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
                     f"{data_config['type']}.{data_config['attribute']}.")
        self.data_1d = loadData(
            os.path.join(os.path.abspath(self.data_dir), 
                         f"{self.stem}{poi}.root"),
            [poi], include_best=True)
        
class Combine2D:
    def __init__(self, pair_name: str) -> None:
        # Extract the contour from the ROOT (NLL) plot
        file = uproot.open(f"contours/{pair_name}.root")
        assert isinstance(file, uproot.ReadOnlyDirectory)
        self.data_2d = {}
        for ci in [68, 95]: 
            self.data_2d[ci] = file[f'graph{ci}_default_0;1'].values()
        file.close()

class Interpolator(ABC):
    """
    Abstract base class for a generic interpolator

    Args:
        ABC (ABC): Helper class for abstract base classes
    """
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialise(self, data_config: Data):
        pass
        
    @abstractmethod
    def evaluate(self, point: Any) -> float:
        return 0
        
    @abstractmethod
    def minimize(self, free_keys: List[str], 
                 fixed_vals: Dict[str, List[float]]) -> OptimizeResult:
        pass

class rbfInterpolator(Interpolator):
    def __init__(self) -> None:
        super().__init__()
        self.data_dir = 'data'

    def initialise(self, data_config: Data):
        super().initialise(data_config)
        
        self.pois = list(data_config["POIs"].keys())
        self.bounds = [tuple(*val.values()) 
                       for val in data_config["POIs"].values()]
        self.stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
                     f"{data_config['type']}.{data_config['attribute']}.")
            
        # Get a subset as a dataframe
        self.data, _ = loadAndSplit(
            os.path.join(os.path.abspath(self.data_dir), 
                         f"{self.stem}{'_'.join(self.pois)}.root"), 
            data_config, include_best=True,
            split=data_config["fraction"])
        assert isinstance(self.data.dataset, NLLDataset)
        data_tup = self.data.dataset[list(self.data.indices)]
        data_rbf = NLLDataset(*data_tup, self.data.dataset.POIs)
        self.best = data_rbf.X[np.argmin(data_rbf.Y)].detach().numpy()
        self.data = data_rbf.toFrame()
        
        # Build interpolator
        self.spline = rbfSplineFast(len(self.data.columns)-1)
        self.spline.initialise(self.data, "deltaNLL", radial_func="cubic",
                               eps=data_config['interpolator']['eps'],
                               rescaleAxis=True)
        
    def evaluate(self, point: pd.DataFrame) -> float:
        super().evaluate(point)
        return self.spline.evaluate(point)
    
    def _minimizeWrapper(self, coeffs: List[float], free_keys: List[str],
                         fixed_vals: Dict[str, List[float]]) -> float:
        super().minimize(free_keys, fixed_vals)
            
        # Create a blank df
        d = {poi: [np.nan] for poi in self.pois}
        
        # Fill in the free values
        for key, coeff in zip(free_keys, coeffs):
            d[key] = [coeff]
        
        # Fill in the fixed values
        for key, val in fixed_vals.items():
            d[key] = val
        
        return self.evaluate(pd.DataFrame(d))
    
    def minimize(self, free_keys: List[str], 
                 fixed_vals: Dict[str, List[float]]) -> OptimizeResult:
        super().minimize(free_keys, fixed_vals)
        assert (len(free_keys) + len(fixed_vals)) == len(self.pois)
        
        #  Get start point and bounds for free POIs
        start = []
        bounds = []
        for key in free_keys:
            idx = self.pois.index(key)
            start.append(self.best[idx])
            bounds.append(self.bounds[idx])
    
        res = minimize(self._minimizeWrapper, x0=start, bounds=bounds,
                       args=(free_keys, fixed_vals))
        return res

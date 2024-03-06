"""
Class definitions for interpolators
"""

import os
import pandas as pd
import numpy as np
import uproot
<<<<<<< HEAD
import numpy.typing as npt
from typing import Any, Dict, List, Tuple
=======
from numpy import typing as npt
from typing import Any, Dict, List
>>>>>>> f1a859125d5ad7a565ea2b2ad664140f2a2c5806
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
        channel = data_config['channel']
        model = data_config['model']
        filename = (f"contours/plot.{channel}.{model}.observed." +
                    f"nominal.{pair_name}.root")
        file = uproot.open(filename)
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
    
class combineInterpolator(Interpolator):
    """
    Interpolates a surface using a grid of points

    :param Interpolator: The abstract base class
    :type Interpolator: Interpolator
    """
    def __init__(self) -> None:
        """
        Initialises the interpolator
        """
        self.data_dir = 'data'
        super().__init__()
    def initialise(self, data_config: Data) -> None:
        """
        Loads data for the interpolator

        :param data_config: Options for the data
        :type data_config: Data
        """
        super().initialise(data_config)
        
        # Setup
        self.pois = list(data_config["POIs"].keys())
        self.bounds = [tuple(*val.values()) 
                       for val in data_config["POIs"].values()]
        self.stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
                     f"{data_config['type']}.{data_config['attribute']}.")

        # Grab data
        data_train, data_test, best  = loadAndSplit(
            os.path.join(os.path.abspath(self.data_dir), 
                        f"{self.stem}{'_'.join(self.pois)}.root"), 
            data_config, include_best=True, split=1)
        
        # Get splits as dataframes
        datasets = []
        for data in [data_train, data_test]:
            assert isinstance(data.dataset, NLLDataset)
            data_tup = data.dataset[list(data.indices)]
            datasets.append(NLLDataset(*data_tup, data.dataset.POIs))
        datasets[0].append(best)
        self.best = best.X.detach().numpy()[0]
        self.data_train = datasets[0].toFrame()
        self.data_test = datasets[1].toFrame()
        
    def evaluate(self, point: Any) -> float:
        """
        Evalutes the interpolator at a given point
        NOTE: This will return np.inf for any point not in the dataset

        :param point: The point to evaluate the dataset at
        :type point: Any
        :return: The interpolator evaluated at the point
        :rtype: float
        """
        assert isinstance(point, pd.DataFrame)
        # Convert to numpy arrays
        point_arr = point.to_numpy()
        data_arr = self.data_train[self.pois].to_numpy()
        
        # Match the entire row
        match = np.argwhere(np.sum(data_arr == point_arr, 
                                   axis=1) == 4).flatten()
        
        # Return matched value
        if match.size:
            return self.data_train.iloc[match]['deltaNLL'].to_numpy()
        else:
            return np.nan
    
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
        if free_keys == self.pois:
            return OptimizeResult({'x': [999]*len(self.pois), 'fun': 0.0})
        
        # Get the rows which match the fixed values
        data = self.data_train
        for key, value in fixed_vals.items():
            data = data[data[key] == value[0]]

        # Minimise deltaNLL       
        best = data.iloc[data['deltaNLL'].argmin()]
        
        return OptimizeResult({'x': best[self.pois].to_numpy(),
                               'fun': best['deltaNLL']})

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
        data_train, data_test, best = loadAndSplit(
            os.path.join(os.path.abspath(self.data_dir), 
                         f"{self.stem}{'_'.join(self.pois)}.root"), 
            data_config, include_best=True,
            split=data_config["fraction"])
        for data in [data_train, data_test]:
            assert isinstance(data.dataset, NLLDataset)
            data_tup = data.dataset[list(data.indices)]
            datasets.append(NLLDataset(*data_tup, data.dataset.POIs))
        datasets[0].append(best)
        self.best = best.X.detach().numpy()[0]
        self.data_train = datasets[0].toFrame()
        self.data_test = datasets[1].toFrame()
        
        # Build interpolator
        self.spline = rbfSplineFast(len(self.data_train.columns)-1)
        self.spline.initialise(self.data_train, "deltaNLL", radial_func="cubic",
                               eps=data_config['interpolator']['eps'],
                               rescaleAxis=True)
        
    def evaluate(self, point: pd.DataFrame) -> Tuple[float, 
                                                     npt.NDArray[np.float32]]:
        """
        Evaluates the interpolator at a specified point

        Args:
            point (Any): The point to evaluate at

        Returns:
            float: The output of the interpolator at that point
        """
        super().evaluate(point)
        return self.spline.evaluate(point), self.spline.evaluate_grad(point)[0]
    
    def evaluate_no_if(self, point: pd.DataFrame) -> float:
        """
        Evaluates the interpolator at a specified point

        Args:
            point (Any): The point to evaluate at

        Returns:
            float: The output of the interpolator at that point
        """
        super().evaluate(point)
        return self.spline.evaluate_no_if(point)
    
    def evaluate_no_pandas(self, point: npt.NDArray[np.float32]) -> float:
        """
        Evaluates the interpolator at a specified point

        Args:
            point npt.NDArray[np.float32]: The point to evaluate at

        Returns:
            float: The output of the interpolator at that point
        """
        super().evaluate(point)
        return self.spline.evaluate_no_pandas(point)
    
    def evaluate_no_if_no_pandas(self, point: npt.NDArray[np.float32]) -> float:
        """
        Evaluates the interpolator at a specified point

        Args:
            point npt.NDArray[np.float32]: The point to evaluate at

        Returns:
            float: The output of the interpolator at that point
        """
        super().evaluate(point)
        return self.spline.evaluate_no_if_no_pandas(point)
    
    def _minimizeWrapper(self, coeffs: List[float], free_keys: List[str],
                         fixed_vals: Dict[str, List[float]]
                         ) -> Tuple[float, npt.NDArray[np.float32]]:
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
        mask = {poi: False for poi in self.pois}
        
        # Fill in the free values
        for key, coeff in zip(free_keys, coeffs):
            d[key] = coeff
            mask[key] = True
        
        # Fill in the fixed values
        for key, val in fixed_vals.items():
            d[key] = val[0]

        val, grad = self.evaluate(pd.DataFrame([d]))
        return val, grad[list(mask.values())]
    
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
                       jac=True, args=(free_keys, fixed_vals), options={'maxls':50})
        return res

import os
import numpy as np
import timeit
from utils import loadConfig
from rbfSpline import rbfSpline
from dataLoader import NLLDataset
from typing import List, Optional
from dataSplitter import loadAndSplit, dataToDict, dataToFrame

import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")


# ----------------------
import numpy as np
import sys

class rbfSplineFast:
    def __init__(self,ndim=1):
        self._ndim = ndim
        self._initialised = False
    
    def log(self, func, val):
        np.set_printoptions(precision=3, linewidth=200)
        print(f"{type(self).__name__}.{func.__name__} {val}")
    
    def _initialise(self,input_data,target_col,eps,rescaleAxis):

        self._input_data = input_data
        self._target_col = target_col
        self._input_points = input_data.drop(target_col, 
                                             axis="columns").to_numpy()
        
        self._eps = eps  
        self._rescaleAxis = rescaleAxis

        self._M = len(input_data) # Num points
        if self._M < 1 : sys.exit("Error - rbf_spline must be initialised with at least one basis point")
        
        self._parameter_keys = list(input_data.columns)
        self._parameter_keys.remove(target_col)

        if self._ndim!=len(self._parameter_keys): 
            sys.exit("Error - initialise given points with more dimensions (%g) than ndim (%g)"%(len(self._parameter_keys),self._ndim))

        self._axis_pts = self._M**(1./self._ndim)
       
        self.calculateWeights()

    def initialise(self,input_points,target_col,eps=10.,rescaleAxis=True):
        self._initialise(input_points,target_col,eps,rescaleAxis)
    
    def diff2(self, points1, points2):
        # The interpolator must have been initialised on points2
        v = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        if self._rescaleAxis: v=self._axis_pts*v/(np.max(points2, axis=0) - np.min(points2, axis=0))
        return np.power(v, 2)

    def getDistSquare(self, col):
        return self.diff2(col, col)
        
    def getDistFromSquare(self, point, inp):
        dk2 = np.sum(self.diff2(point, inp), axis=-1).flatten()
        return dk2

    def radialFunc(self,d2):
        # expo = (d2/(self._eps*self._eps))
        # return np.exp(-1*expo)
        # return np.sqrt(1+d2/(self._eps*self._eps))
        return np.power(d2/(self._eps*self._eps), 3/2)

    def evaluate(self,point):
        if not self._initialised:
            print("Error - must first initialise spline with set of points before calling evaluate()") 
            return np.nan
        if not set(point.keys())==set(self._parameter_keys): 
            print ("Error - must have same variable labels, you provided - ",point.keys(),", I only know about - ",self._parameter_keys)
            return np.nan
        vals = self.radialFunc(self.getDistFromSquare(point.to_numpy(), self._input_points))
        # self.log(self.evaluate, self._weights)
        # self.log(self.evaluate, vals)
        weighted_vals = self._weights * vals
        return sum(weighted_vals)

    def calculateWeights(self) : 
        A = np.zeros((self._M, self._M))
        inp = self._input_points
        B = self._input_data[self._target_col].to_numpy()
        
        d2 = np.sum(self.diff2(inp, inp), axis=2)
        A = self.radialFunc(d2) + np.eye(self._M)
    
        self._weights = np.linalg.solve(A,B)
        self._initialised=True

# ----------------------
def RBFWrapper(coeffs: List[float], spline: Optional[rbfSpline | rbfSplineFast],
               pois: Optional[List[str]], fixed_idx: Optional[int],
               fixed_val: Optional[int]):
    if (spline is None) or (pois is None):
        return np.nan
    else:
        coeffs = list(coeffs)
        if (fixed_idx is not None) and (fixed_val is not None):
            coeffs.insert(fixed_idx, fixed_val)
        pois_map = {poi: coeffs[i] for i, poi in enumerate(pois)}
        return spline.evaluate(pois_map)

if __name__ == "__main__":
    # Setup
    os.chdir("../..") # Testing only
    data_dir = "data"
    data_config = loadConfig("configs/default.yaml")["data"]
    pois = list(data_config["POIs"].keys())
    bounds = [tuple(*val.values()) for val in data_config["POIs"].values()]
    stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
            f"{data_config['type']}.{data_config['attribute']}.")
    
    # Get data
    SPLIT = 0.002
    data_rbf, _ = loadAndSplit(os.path.join(os.path.abspath(data_dir), 
                                              f"{stem}{'_'.join(pois)}.root"),
                                 data_config, split=SPLIT, include_best=False)
    assert isinstance(data_rbf.dataset, NLLDataset)
    data_rbf_tup = data_rbf.dataset[list(data_rbf.indices)]
    data_rbf = NLLDataset(*data_rbf_tup, data_rbf.dataset.POIs)
    data_dict = dataToDict(data_rbf)
    data_df = dataToFrame(data_rbf)
    
    # Setup interpolators
    spline = rbfSpline(len(list(data_dict[0].keys()))-1, use_scipy_interp=False)
    spline.initialise(data_dict, "deltaNLL", eps=10, rescaleAxis=True)
    spline_f = rbfSplineFast(len(list(data_dict[0].keys()))-1)
    spline_f.initialise(data_df, "deltaNLL", eps=10, rescaleAxis=True)
    
    # Evaluate
    import datetime
    pt = {"lchgXE3": 0, "lchwXE2": 0, "lctgreXE1": 0}
    s = datetime.datetime.now()
    print(spline.evaluate(pt))
    print(datetime.datetime.now() - s)
    print()
    pt = pd.DataFrame({'lchgXE3': np.array([0]), 
                       'lchwXE2': np.array([0]), 
                       'lctgreXE1': np.array([0])})
    s = datetime.datetime.now()
    print(spline_f.evaluate(pt))
    print(datetime.datetime.now() - s)
    
    """
    num = 10
    spline = rbfSpline(len(list(data_dict[0].keys()))-1, use_scipy_interp=False)
    print('Timing old')
    print(timeit.timeit('spline.initialise(data_dict, "deltaNLL", eps=10, rescaleAxis=False)', globals=globals(), number=num)/num)
    spline_f = rbfSplineFast(len(list(data_dict[0].keys()))-1)
    print('Timing new')
    print(timeit.timeit('spline_f.initialise(data_df, "deltaNLL", eps=10, rescaleAxis=False)', globals=globals(), number=num)/num)
    """

import os
import timeit
import numpy as np
import pandas as pd
from utils import loadConfig
from rbfDiff import interpolate
from rbfSpline import rbfSpline
from rbfSplineVec import rbfSplineVec
from rbfSplineFast import rbfSplineFast
from dataLoader import NLLDataset
from typing import List, Optional
from scipy.interpolate import RBFInterpolator
from dataSplitter import loadAndSplit, dataToDict, dataToFrame

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

def RBFWrapper(coeffs: List[float], spline: Optional[rbfSpline | rbfSplineVec],
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
    
def RBFFastWrapper(coeffs: List[float], spline: Optional[rbfSplineFast],
               pois: Optional[List[str]], fixed_idx: Optional[int],
               fixed_val: Optional[int]):
    if (spline is None) or (pois is None):
        return np.nan
    else:
        coeffs = list(coeffs)
        if (fixed_idx is not None) and (fixed_val is not None):
            coeffs.insert(fixed_idx, fixed_val)
        pois_map = pd.DataFrame({poi: np.array([coeffs[i]]) for i, poi in enumerate(pois)})
        return spline.evaluate(pois_map)
    
def scipyWrapper(coeffs: List[float], interp: Optional[RBFInterpolator],
                 fixed_idx: Optional[int], fixed_val: Optional[int]):
    if (interp is None) or (pois is None):
        return np.nan
    else:
        coeffs = list(coeffs)
        if (fixed_idx is not None) and (fixed_val is not None):
            coeffs.insert(fixed_idx, fixed_val)
        coeffs_vec = np.expand_dims(coeffs, 0)
        return interp(coeffs_vec)

def timerbf():
    # Get a subset as a dict
    data_rbf, _ = loadAndSplit(os.path.join(os.path.abspath(data_dir), 
                                            f"{stem}{'_'.join(pois)}.root"),
                               data_config, split=SPLIT, include_best=False)
    assert isinstance(data_rbf.dataset, NLLDataset)
    data_rbf_tup = data_rbf.dataset[list(data_rbf.indices)]
    data_rbf = NLLDataset(*data_rbf_tup, data_rbf.dataset.POIs)
    data_train_dict = dataToDict(data_rbf)
    spline = interpolate(data_train_dict, eps=10)
    
    for x in xs:
        r = RBFWrapper([-0.23, -2.97], spline, pois, 0, x)
def timerbfVec():
    # Get a subset as a dict
    data_rbf_v, _ = loadAndSplit(os.path.join(os.path.abspath(data_dir), 
                                              f"{stem}{'_'.join(pois)}.root"),
                                 data_config, split=SPLIT, include_best=False)
    assert isinstance(data_rbf_v.dataset, NLLDataset)
    data_rbf_v_tup = data_rbf_v.dataset[list(data_rbf_v.indices)]
    data_rbf_v = NLLDataset(*data_rbf_v_tup, data_rbf_v.dataset.POIs)
    data_train_v_dict = dataToDict(data_rbf_v)
    
    spline_v = rbfSplineVec(len(list(data_train_v_dict[0].keys()))-1, use_scipy_interp=False)
    spline_v.initialise(data_train_v_dict, "deltaNLL", eps=10, rescaleAxis=True)
    
    for x in xs:
        v = RBFWrapper([-0.23, -2.97], spline_v, pois, 0, x)
def timerbffast():
    # Get a subset as a df
    data_rbf_f, _ = loadAndSplit(os.path.join(os.path.abspath(data_dir), 
                                              f"{stem}{'_'.join(pois)}.root"),
                                 data_config, split=SPLIT, include_best=False)
    assert isinstance(data_rbf_f.dataset, NLLDataset)
    data_rbf_f_tup = data_rbf_f.dataset[list(data_rbf_f.indices)]
    data_rbf_f = NLLDataset(*data_rbf_f_tup, data_rbf_f.dataset.POIs)
    data_train_f_frame = dataToFrame(data_rbf_f)
    
    spline_f = rbfSplineFast(len(data_train_f_frame.columns)-1)
    spline_f.initialise(data_train_f_frame, "deltaNLL", eps=10, rescaleAxis=True)
    
    for x in xs:
        f = RBFFastWrapper([-0.23, -2.97], spline_f, pois, 0, x)
def timescipy():
    # Get subset as NLL Dataset
    data_scipy, _ = loadAndSplit(os.path.join(os.path.abspath(data_dir), 
                                            f"{stem}{'_'.join(pois)}.root"),
                                data_config, split=SPLIT, include_best=False)
    assert isinstance(data_scipy.dataset, NLLDataset)
    data_scipy_tup = data_scipy.dataset[list(data_scipy.indices)]
    data_scipy = NLLDataset(*data_scipy_tup, data_scipy.dataset.POIs)
    interp = RBFInterpolator(data_scipy.X, data_scipy.Y, 
                            kernel='cubic', epsilon=5)
    
    for x in xs:
        s = scipyWrapper([-0.23, -2.97], interp, 0, x)

if __name__ == "__main__":
    # Setup
    os.chdir("../..") # Testing only
    data_dir = "data"
    data_config = loadConfig("configs/default.yaml")["data"]
    pois = list(data_config["POIs"].keys())
    bounds = [tuple(*val.values()) for val in data_config["POIs"].values()]
    stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
            f"{data_config['type']}.{data_config['attribute']}.")
    
    # Scan through split fraction
    times_rbf = []
    times_rbfvec = []
    times_rbffast = []
    times_scipy = []
    splits_run = []
    SPLITS = np.arange(0.1, 1.1, 0.1)
    for SPLIT in SPLITS:
        print(SPLIT)
        splits_run.append(SPLIT)
        
        # Rather than minimize, just evaluate at every point in the dataset
        num = 10
        samples = 100
        xs = np.linspace(-35, 30, samples)

        print('Timing rbf')
        times_rbf.append(timeit.timeit('timerbf()', globals=globals(), number=num)/num)
        # print('Timing rbfVec')
        # times_rbfvec.append(timeit.timeit('timerbfVec()', globals=globals(), number=num)/num)
        # print('Timing rbffast')
        # times_rbffast.append(timeit.timeit('timerbffast()', globals=globals(), number=num)/num)
        print('Timing scipy')
        times_scipy.append(timeit.timeit('timescipy()', globals=globals(), number=num)/num)
    
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(splits_run, np.array(times_rbf), label='rbfSpline')
        # ax.plot(splits_run, np.array(times_rbfvec)/np.array(times_scipy), label='rbfSpline (vectorised)')
        # ax.plot(splits_run, np.array(times_rbffast)/np.array(times_scipy), label='rbfSpline (no loops)')
        ax.plot(splits_run, np.array(times_scipy), label='SciPy')
        ax.set_xlabel('Dataset fraction')
        ax.set_ylabel(f'Average runtime [s]')
        ax.legend(loc='upper left')
        plt.savefig(f"out/debug/timings_slow.png", facecolor='white', 
                    bbox_inches='tight', dpi=125)

    # print(times_rbf, times_rbfvec, times_rbffast,times_scipy, sep='\n')
"""
Script for debugging various models, plotting bugs etc
"""
import os
import torch
import numpy as np
import pandas as pd
import mplhep as hep
from copy import deepcopy
from utils import loadConfig
from dataLoader import loadData, NLLDataset
from dataSplitter import loadAndSplit, dataToDict, dataToFrame
from rbfDiff import interpolate
from rbfSpline import rbfSpline
from rbfSplineFast import rbfSplineFast
from matplotlib import pyplot as plt
from typing import Optional, List
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
hep.style.use('CMS')

def add_to_single(ax, data: pd.DataFrame, label: str, poi: str):
    # Setup
    xs = data[poi]
    ys = 2*data['deltaNLL']
    
    # Mask
    mask = np.where(ys <= 10)[0]
    xs = xs[mask]
    ys = ys[mask]

    # Fit
    xs_fine = np.linspace(np.min(xs), np.max(xs), 100)
    fit = np.poly1d(np.polyfit(xs, ys, 8))
    
    # Plot
    ax.plot(xs_fine, fit(xs_fine), lw=2)
    ax.scatter(xs, ys, s=50, label=f'{label}')
    ax.set_ylim(-0.5, 17)
    
    # Annotate
    ax.set_xlabel(poi)
    ax.set_ylabel(r"$-2\,\Delta\,\ln{\mathcal{L}}$")
    ax.legend(loc='upper right')
    
    return ax
    
def minimizeScipy(coeffs: List[float], interp: Optional[RBFInterpolator],
                    fixed_idx: Optional[int], fixed_val: Optional[int]):
    if (interp is None) or (pois is None):
        return np.nan
    else:
        coeffs = list(coeffs)
        if (fixed_idx is not None) and (fixed_val is not None):
            coeffs.insert(fixed_idx, fixed_val)
        coeffs_vec = np.expand_dims(coeffs, 0)
        return interp(coeffs_vec)
    
def minimiseRBF(coeffs: List[float], spline: Optional[rbfSpline],
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

if __name__ == "__main__":
    # Setup
    os.chdir("../..") # Testing only
    data_dir = "data"
    data_config = loadConfig("configs/default.yaml")["data"]
    pois = list(data_config["POIs"].keys())
    bounds = [tuple(*val.values()) for val in data_config["POIs"].values()]
    stem = (f"scan.{data_config['channel']}.{data_config['model']}." +
            f"{data_config['type']}.{data_config['attribute']}.")
    
    # Load data
    data_all = loadData(os.path.join(os.path.abspath(data_dir), 
                                     f"{stem}{'_'.join(pois)}.root"), 
                        pois, include_best=True)
    best = data_all[data_all['deltaNLL'] == 0]
    best_point = best[pois].to_numpy().flatten()
    print(data_all.describe())
    print(best_point)
    
    # Get subset
    data_scipy, _ = loadAndSplit(os.path.join(os.path.abspath(data_dir), 
                                              f"{stem}{'_'.join(pois)}.root"),
                                 data_config, split=1, include_best=False)
    assert isinstance(data_scipy.dataset, NLLDataset)
    
    # Get the underlying dataset
    data_scipy_tup = data_scipy.dataset[list(data_scipy.indices)]
    data_scipy = NLLDataset(*data_scipy_tup, data_scipy.dataset.POIs)
    data_scipy.append(NLLDataset(torch.tensor(best_point).unsqueeze(0),
                                 torch.tensor(best['deltaNLL']).unsqueeze(0),
                                 data_scipy.POIs))
    
    # Get a possibly different subset as a dict
    data_rbf, _ = loadAndSplit(os.path.join(os.path.abspath(data_dir), 
                                        f"{stem}{'_'.join(pois)}.root"),
                                 data_config, split=.25, include_best=False)
    assert isinstance(data_rbf.dataset, NLLDataset)
    data_rbf_tup = data_rbf.dataset[list(data_rbf.indices)]
    data_rbf = NLLDataset(*data_rbf_tup, data_rbf.dataset.POIs)
    data_rbf.append(NLLDataset(torch.tensor(best_point).unsqueeze(0),
                           torch.tensor(best['deltaNLL']).unsqueeze(0),
                           data_rbf.POIs))
    data_train_dict = dataToDict(data_rbf)
    data_train_frame = dataToFrame(data_rbf)
    
    # Build interpolators
    interp = RBFInterpolator(data_scipy.X, data_scipy.Y, 
                             kernel='cubic', epsilon=5)
    best_scipy = minimize(minimizeScipy, x0=best_point, 
                          args=(interp, None, None), bounds=bounds)

    spline = interpolate(data_train_dict, eps=0.2)
    best_rbf = minimize(minimiseRBF, x0=best_point, 
                        args=(spline, pois, None, None), bounds=bounds)
    
    spline_f = rbfSplineFast(len(data_train_frame.columns)-1)
    spline_f.initialise(data_train_frame, "deltaNLL", eps=0.2, rescaleAxis=True)
    best_rbf_f = minimize(RBFFastWrapper, x0=best_point, 
                          args=(spline_f, pois, None, None), bounds=bounds)
    print(best_rbf_f, best_rbf, best_scipy)
    print(spline_f.evaluate(pd.DataFrame({'lchgXE3': [-9.221493],
                                          'lchwXE2': [-0.2371213],
                                          'lctgreXE1': [-2.9788358]})))
    
    # ------------
    # Plot singles
    # ------------
    for i, poi in enumerate(pois[:]):        
        # Plot the scanned values
        print('Combine')
        data_1d = loadData(os.path.join(os.path.abspath(data_dir),
                                        f"{stem}{poi}.root"), 
                           [poi], include_best=True)
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        add_to_single(ax, data_1d, "Combine scan", poi)
        
        # # Plot the values profiled from the 3D grid
        # profiled = data_all.groupby(poi).min()
        # d = pd.DataFrame({poi: np.array(profiled.index), 
        #                   'deltaNLL': np.array(profiled['deltaNLL'])})
        # add_to_single(ax, d, f"Combine profiled ({len(data_all)})", poi)
        
        # # ------------------
        # # Profile with scipy
        # # ------------------
        # print('SciPy')
        # xs = np.linspace(bounds[i][0], bounds[i][1], 250)
        # ys = []
        # start = list(deepcopy(best_point))
        # start.pop(i)
        # bounds_for_free = deepcopy(bounds)
        # bounds_for_free.pop(i)
        # bounds_for_free = tuple(bounds_for_free)
        # for x in xs:
        #     res = minimize(minimizeScipy, x0=start,
        #                    args=(interp, i, x), bounds=bounds_for_free)
        #     ys.append(2*(res['fun']-best_scipy['fun']))
        # ys = np.array(ys)
        # mask = np.where(ys <= 10)[0]
        # xs = xs[mask]
        # ys = ys[mask]
        # ax.plot(xs, ys, lw=2, label=f"SciPy RBF ({len(data_scipy.X)})")
        
        # # Mark the 'training' data
        # xs = data_scipy.X[:, i].unique().detach().numpy()
        # ys = []
        # for x in xs:
        #     res = minimize(minimizeScipy, x0=start,
        #                    args=(interp, i, x), bounds=bounds_for_free)
        #     ys.append(2*(res['fun']-best_scipy['fun']))
        # ys = np.array(ys)
        # mask = np.where(ys <= 10)[0]
        # xs = xs[mask]
        # ys = ys[mask]
        # ax.scatter(xs, ys, s=50)
        # ax.legend(loc='upper right')
        
        # # -----------------
        # # Profile with RBF
        # # -----------------
        # print('rbf')
        # xs = np.linspace(bounds[i][0], bounds[i][1], 200)
        # ys = []
        # start = list(deepcopy(best_point))
        # start.pop(i)
        # bounds_for_free = deepcopy(bounds)
        # bounds_for_free.pop(i)
        # bounds_for_free = tuple(bounds_for_free)
        # for x in xs:
        #     print(x)
        #     res = minimize(minimiseRBF, x0=start,
        #                    args=(spline, pois, i, x), bounds=bounds_for_free)
        #     ys.append(2*(res['fun']-best_rbf['fun']))
        # ys = np.array(ys)
        # mask = np.where(ys <= 10)[0]
        # xs = xs[mask]
        # ys = ys[mask]
        # rbf_xs = xs
        # rbf_ys = ys
        # ax.plot(xs, ys, lw=2, label=f"rbfSpline ({len(data_train_dict)})")
        
        # # Mark the 'training' data
        # xs = data_rbf.X[:, i].unique().detach().numpy()
        # ys = []
        # for x in xs:
        #     print(x)
        #     res = minimize(minimiseRBF, x0=start,
        #                    args=(spline, pois, i, x), bounds=bounds_for_free)
        #     ys.append(2*(res['fun']-best_rbf_f['fun']))
    
        # ys = np.array(ys)
        # mask = np.where(ys <= 10)[0]
        # xs = xs[mask]
        # ys = ys[mask]
        # ax.scatter(xs, ys, s=50)
        # ax.legend(loc='upper right')
                
        # -----------------
        # Profile with RBF (fast)
        # -----------------
        print('rbffast')
        xs = np.linspace(bounds[i][0], bounds[i][1], 200)
        ys = []
        start = list(deepcopy(best_point))
        start.pop(i)
        bounds_for_free = deepcopy(bounds)
        bounds_for_free.pop(i)
        bounds_for_free = tuple(bounds_for_free)
        for x in xs:
            res = minimize(RBFFastWrapper, x0=start,
                           args=(spline_f, pois, i, x), bounds=bounds_for_free)
            print(x, res['x'], 2*(res['fun']-best_rbf['fun']))
            ys.append(2*(res['fun']-best_rbf['fun']))
        ys = np.array(ys)
        mask = np.where(ys <= 10)[0]
        xs = xs[mask]
        ys = ys[mask]
        rbf_f_ys = ys
        ax.plot(xs, ys, lw=2, label=f"rbfSpline no loops ({len(data_train_dict)})")
        
        # Mark the 'training' data
        xs = data_rbf.X[:, i].unique().detach().numpy()
        ys = []
        for x in xs:
            print(x)
            res = minimize(RBFFastWrapper, x0=start,
                           args=(spline_f, pois, i, x), bounds=bounds_for_free)
            ys.append(2*(res['fun']-best_rbf_f['fun']))
    
        ys = np.array(ys)
        mask = np.where(ys <= 10)[0]
        xs = xs[mask]
        ys = ys[mask]
        ax.scatter(xs, ys, s=50)
        ax.legend(loc='upper right')
        
        plt.savefig(f"out/debug/{poi}_fast.png", facecolor='white', 
                    bbox_inches='tight', dpi=125)
        
        # # Residuals plot
        # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        # ax.scatter(rbf_xs, rbf_ys-rbf_f_ys)
        # ax.set_ylim((-0.002, 0.002))
        # ax.set_xlabel(poi)
        # ax.set_ylabel(r'$\mathrm{rbfSpline} - \mathrm{rbfSpline(no loops)}$')
        # plt.savefig(f"out/debug/{poi}_diff.png", facecolor='white', 
        #             bbox_inches='tight', dpi=125)
        
        

"""
For splits 1.0, 0.25

Scipy init time:  0:00:00.192090
rbfSpline init time:  0:00:01.550445
Scipy interp time:  0:00:00.889456
rbfSpline interp time:  0:00:27.073739
"""

"""
Script for debugging various models, plotting bugs etc
"""
import os
import tqdm
import torch
import uproot
import numpy as np
import pandas as pd
import mplhep as hep
from copy import deepcopy
from utils import loadConfig
from dataLoader import loadData, NLLDataset
from dataSplitter import loadAndSplit, dataToFrame
from rbfSplineFast import rbfSplineFast
from matplotlib import pyplot as plt
from matplotlib import colors
from typing import Optional, List
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from itertools import combinations
hep.style.use('CMS')

def add_to_pair(ax, xs, ys, ls, label):
    ax.plot(xs, ys, ls=ls, label=label, c='k')
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

# NOTE will need re-writing to handle multiple free POIs - rewrite for arb. dim
def RBFFastWrapper2D(coeff: List[float], spline: Optional[rbfSplineFast],
                     pois: Optional[List[str]], free_idx: Optional[int],
                     fixed_vals: Optional[List[float]]):
    if (spline is None) or (pois is None):
        return np.nan
    else:
        if (free_idx is not None) and (fixed_vals is not None):
            # print(fixed_vals)
            coeffs = deepcopy(fixed_vals)
            coeffs.insert(free_idx, coeff[0])
            # print({poi: np.array([coeffs[i]]) for i, poi in enumerate(pois)})
            pois_map = pd.DataFrame({poi: np.array([coeffs[i]]) for i, poi in enumerate(pois)})
        return spline.evaluate(pois_map)

if __name__ == "__main__":
    # Setup
    os.chdir("../..") # Testing only
    data_dir = "data"
    data_config = loadConfig("configs/default.yaml")["data"]
    pois = list(data_config["POIs"].keys())
    poi_pairs = list(combinations(pois, 2))
    poi_pair_idxs = list(combinations(range(len(pois)), 2))
    
    free_idxs = []
    for pair_idxs in poi_pair_idxs:
        for idx in range(len(pois)):
            if idx not in pair_idxs:
                free_idxs.append(idx)
    
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
    
    # Get a possibly different subset as a dataframe
    data_rbf, _ = loadAndSplit(os.path.join(os.path.abspath(data_dir), 
                                        f"{stem}{'_'.join(pois)}.root"),
                                 data_config, split=.25, include_best=False)
    assert isinstance(data_rbf.dataset, NLLDataset)
    data_rbf_tup = data_rbf.dataset[list(data_rbf.indices)]
    data_rbf = NLLDataset(*data_rbf_tup, data_rbf.dataset.POIs)
    data_rbf.append(NLLDataset(torch.tensor(best_point).unsqueeze(0),
                           torch.tensor(best['deltaNLL']).unsqueeze(0),
                           data_rbf.POIs))
    data_train_frame = dataToFrame(data_rbf)
    
    # Build interpolator
    spline = rbfSplineFast(len(data_train_frame.columns)-1)
    spline.initialise(data_train_frame, "deltaNLL", eps=0.2, rescaleAxis=True)
    
    # Get min
    best_rbf = minimize(RBFFastWrapper, x0=best_point, 
                        args=(spline, pois, None, None), bounds=bounds)
    print(best_rbf)
    
    # ------------
    # Plot singles
    # ------------
    for i, (free_idx, fixed_idxs, pair) in enumerate(zip(free_idxs[:], 
                                                         poi_pair_idxs[:],
                                                         poi_pairs[:])):
        pair = list(pair)
        pair_name = '_'.join(pair)
        
        # Plot the scanned values
        print('Combine ', pair)
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        # Extract the contour from the ROOT (NLL) plot
        file = uproot.open(f"contours/{pair_name}.root")
        assert isinstance(file, uproot.ReadOnlyDirectory)
        for ci, ls in zip([68, 95], ['-', '--']):
            xs, ys = file[f'graph{ci}_default_0;1'].values()
            ax = add_to_pair(ax, xs, ys, ls=ls, label=f'Combine {ci}% CL')
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
        file.close()
    
        # -----------------
        # Profile with RBF
        # -----------------
        print('rbf', pair)
        # Setup
        start = best_point[idx]
        bounds_for_free = bounds[idx]
        bounds_for_free = tuple(bounds_for_free)
        
        # Build grid
        x = np.linspace(bounds[fixed_idxs[0]][0], bounds[fixed_idxs[0]][1], 100)
        y = np.linspace(bounds[fixed_idxs[1]][0], bounds[fixed_idxs[1]][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.full(X.shape, np.nan).flatten()

        # print(bounds_for_free)
        for i, (x, y) in tqdm.tqdm(enumerate(zip(X.flatten(), Y.flatten())),
                                   total=len(Z)):
            # print(x, y)
            res = minimize(RBFFastWrapper2D, x0=start,
                           args=(spline, pois, free_idx, [x, y]))
            # val = spline.evaluate(pd.DataFrame({'lchgXE3': [x],
            #                               'lchwXE2': [0],
            #                               'lctgreXE1': [y]}))
            Z[i] = 2*(res['fun'] - best_rbf['fun'])
            # Z[i] = 2*(val- best_rbf['fun'])
        
        # Reshape and plot
        Z = Z.reshape(X.shape)
        shifts = [2.30, 5.99]
        labels = ["68%", "95%"]
        styles = ['-', '--']
        for shift, label, style in zip(shifts, labels, styles):
            ax.contour(X, Y, Z, levels=[shift], 
                       zorder=0, colors='#5790fc', linestyles=style)
            ax.plot(1, label=f'RBF {label} CL ({len(data_rbf)})', color='#5790fc', linestyle=style)
        ax.legend(loc='upper right')
            
        plt.savefig(f"out/debug/{pair_name}.png", facecolor='white', 
                    bbox_inches='tight', dpi=125)
        

"""
For splits 1.0, 0.25

Scipy init time:  0:00:00.192090
rbfSpline init time:  0:00:01.550445
Scipy interp time:  0:00:00.889456
rbfSpline interp time:  0:00:27.073739
"""

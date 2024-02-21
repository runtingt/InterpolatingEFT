"""
Plots all profiled fits
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from InterpolatingEFT.utils import Data
from InterpolatingEFT.interpolator import rbfInterpolator, Combine1D
from itertools import combinations
from typing import List, Tuple
hep.style.use('CMS')

def plotScan1D(poi: str, label: str, 
               out: str ='out/default') -> None:
    """
    Plot the 1D scan for the specified POI

    Args:
        poi (str): The poi to plot
        label (str): The legend entry
        out (str, optional): The output dir. Defaults to 'out/default'.
    """
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))    
    for interp in ['combine', 'rbf']:
        # Load and unpack
        with open(os.path.join(out, f"{poi}_{interp}.pkl"), "rb") as f:
            d = pickle.load(f)
            xs = np.array(d['x'])
            ys = np.array(d['y'])
            best = np.array(d['best'])
        
        # Plot
        ys = 2*(ys-best)
        mask = np.where(ys <= 10)[0]
        xs = xs[mask]
        ys = ys[mask]
        if interp != 'combine':
            ax.plot(xs, ys, lw=2, label=label)
        else:
            xs_fine = np.linspace(np.min(xs), np.max(xs), 100)
            fit = np.poly1d(np.polyfit(xs, ys, 8))
            ax.plot(xs_fine, fit(xs_fine), lw=2)
            ax.scatter(xs, ys, s=50, label="Combine")
    
    # Annotate
    ax.set_xlabel(poi)
    ax.set_ylabel(r"$-2\,\Delta\,\ln{\mathcal{L}}$")
    ax.set_ylim(-0.5, 12)
    ax.legend(loc='upper right')
    
    # Save
    plt.savefig(os.path.join(out, f"{poi}.png"), facecolor='white',
                bbox_inches='tight', dpi=125)
    
def plotAllScan1D(pois: List[str], labels: List[str],
                  out: str ='out/default') -> None:
    """
    Plots the 1D scan for the specified POIs

    Args:
        poi (str): The pois to plot
        label (str): The legend entries
        out (str, optional): The output dir. Defaults to 'out/default'.
    """
    for poi, label in zip(pois, labels):
        plotScan1D(poi, label, out=out)
        
def plotScan2D(pair: Tuple[str, str], label: str, 
               out: str ='out/default') -> None:
    """
    Plots the 2D scan for the specified POI pair

    Args:
        pair (str): The pair to plot
        label (str): The legend entry
        out (str, optional): The output dir. Defaults to 'out/default'.
    """
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))    
    for interp in ['combine', 'rbf']:
        # Load and unpack
        with open(os.path.join(out, f"{pair[0]}_{pair[1]}_{interp}.pkl"),
                  "rb") as f:
            d = dict(pickle.load(f))
            
        # Plot contours
        styles = ['-', '--']
        if interp != 'combine':
            X = np.array(d['x'])
            Y = np.array(d['y'])
            Z = np.array(d['z'])
            best = np.array(d['best'])
        
            # Plot
            Z = 2*(Z-best)
            shifts = [2.30, 5.99]
            cls = ["68%", "95%"]
            for shift, cl, style in zip(shifts, cls, styles):
                ax.contour(X, Y, Z, levels=[shift], 
                        zorder=0, colors='#5790fc', linestyles=style)
                ax.plot(1, label=f"{label} {cl} CL", color='#5790fc', 
                        linestyle=style)
        else:
            styles = ['-', '--']
            for ci, ls in zip(d.keys(), styles):
                ax.plot(d[ci][0], d[ci][1], ls=ls, 
                        label=f"Combine {ci}% CL", c='k')
    
    # Annotate
    ax.set_xlabel(pair[0])
    ax.set_ylabel(pair[1])
    ax.legend(loc='upper right')
    
    # Save
    plt.savefig(os.path.join(out, f"{pair[0]}_{pair[1]}.png"), 
                facecolor='white', bbox_inches='tight', dpi=125)
    
def plotAllScan2D(pois: List[str], labels: List[str], 
              out: str ='out/default') -> None:
    """
    Plots the 2D scan for the specified POI pairs

    Args:
        pois (str): The pois to plot, pairwise
        label (str): The legend entries
        out (str, optional): The output dir. Defaults to 'out/default'.
    """
    poi_pairs = list(combinations(pois, 2))
    for pair, label in zip(poi_pairs, labels):
        plotScan2D(pair, label, out=out)
        
def plotDiff1D(poi: str, interp: rbfInterpolator, data_config: Data,
               out: str='out/default') -> None:
    """    
    Plots the difference between the interpolated and \
        true value from the 1D scan for a given poi

    :param poi: The parameter to plot for
    :type poi: str
    :param interp: The interpolator to use
    :type interp: rbfInterpolator
    :param data_config: Options for the data
    :type data_config: Data
    :param out: The output directory, defaults to 'out/default'
    :type out: str, optional
    """    
    # Rather than to the grid (because it isn't very dense), we should just
    # do it to the combine scan! This means we don't need the train/test datasets,
    # just the interpolator and the data from Combine for the 1D poi. We have both of
    # these already! We need Combine1D 
    
    # Initialise Combine1D interpolator
    data_test = Combine1D(data_config, poi).data_1d
    xs = data_test[poi].to_numpy()
    ys = data_test['deltaNLL'].to_numpy()
    mask = np.where(2*ys <= 10)[0]
    
    # TODO vectorise
    # Compute differences to true value
    diffs = np.full(ys.shape, np.nan)
    scan_points = data_test[interp.pois]
    for i in range(len(scan_points)):
        diffs[i] = ys[i] - interp.evaluate(scan_points.iloc[i:i+1])
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.scatter(xs[mask], diffs[mask], label='test')
    ax.set_xlabel(poi)
    ax.set_ylabel(r"True - predicted")
    plt.savefig(os.path.join(out, f"{poi}_diff.png"), facecolor='white',
                bbox_inches='tight', dpi=125)

def plotAllDiff1D(interp: rbfInterpolator, data_config: Data,
                  out: str='out/default') -> None:
    """
    Plots all the 1D difference plots

    :param interp: The interpolator to use
    :type interp: rbfInterpolator
    :param data_config: Options for the data
    :type data_config: Data
    :param out: The output directory, defaults to 'out/default'
    :type out: str, optional
    """
    for poi in interp.pois:
        plotDiff1D(poi, interp, data_config, out=out)

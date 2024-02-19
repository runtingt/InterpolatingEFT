"""
Plots all profiled fits
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from itertools import combinations
from typing import List, Tuple
hep.style.use('CMS')

def plot1D(poi: str, label: str, out: str ='out/default') -> None:
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
    
def plotAll1D(pois: List[str], labels: List[str], out: str ='out/default') -> None:
    """
    Plots the 1D scan for the specified POIs

    Args:
        poi (str): The pois to plot
        label (str): The legend entries
        out (str, optional): The output dir. Defaults to 'out/default'.
    """
    for poi, label in zip(pois, labels):
        plot1D(poi, label, out=out)
        
def plot2D(pair: Tuple[str, str], label: str, out: str ='out/default') -> None:
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
    
def plotAll2D(pois: List[str], labels: List[str], out: str ='out/default') -> None:
    """
    Plots the 2D scan for the specified POI pairs

    Args:
        pois (str): The pois to plot, pairwise
        label (str): The legend entries
        out (str, optional): The output dir. Defaults to 'out/default'.
    """
    poi_pairs = list(combinations(pois, 2))
    for pair, label in zip(poi_pairs, labels):
        plot2D(pair, label, out=out)

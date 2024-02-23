"""
Plots all profiled fits
"""

import os
import pickle
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from InterpolatingEFT.utils import Data
from InterpolatingEFT.interpolator import rbfInterpolator, Combine1D, Combine2D
from itertools import combinations
from typing import List, Tuple
hep.style.use('CMS')
colors = ["#5790fc", "#f89c20"]

def plotScan1D(poi: str, label: str, 
               out: str ='out/default',
               ax: plt.Axes=None) -> None:
    """
    Plots the 1D scan for the specified POI

    :param poi: The poi to plot
    :type poi: str
    :param label: The legend entry
    :type label: str
    :param ax: Axes to plot on, defaults to None
    :type: plt.Axes, optional
    :param out: The output dir, defaults to 'out/default'
    :type out: str, optional
    """
    # Setup
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        save = True
    else:
        save = False
    
    # Plot
    for i, interp in enumerate(['combine', 'rbf']):
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
            ax.plot(xs, ys, lw=10, label=label)
        else:
            xs_fine = np.linspace(np.min(xs), np.max(xs), 100)
            fit = np.poly1d(np.polyfit(xs, ys, 8))
            ax.plot(xs_fine, fit(xs_fine), lw=10)
            ax.scatter(xs, ys, s=250, label="Combine")
        ax.set_ylim(-0.5, 12)
    
        if not save:
            # Add limits to plot
            shifts = [1.00, 3.84]
            styles = ['--', '-.']
            x = xs_fine if i == 0 else xs
            y = fit(xs_fine) if i == 0 else ys
            for shift, style in zip(shifts, styles):
                idxs = np.argwhere(np.diff(np.sign(y-shift))).flatten()
                for idx in idxs:
                    ax.plot([x[idx], x[idx]], [-0.5, y[idx]], 
                            lw=10, ls=style, c=colors[i])
                ax.plot([np.min(x)*2, np.max(x)*2], [shift, shift], 
                        lw=5, ls='-', c='k', alpha=0.1, zorder=-5)
           
    # Annotate
    if save:
        ax.set_xlabel(poi)
        ax.set_ylabel(r"$-2\,\Delta\,\ln{\mathcal{L}}$")
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
               ax: plt.Axes=None,
               out: str ='out/default') -> None:
    """
    Plots the 2D scan for the specified POI pair

    Args:
        pair (str): The pair to plot
        label (str): The legend entry
        ax (plt.Axes, optional): Axes to plot on
        out (str, optional): The output dir. Defaults to 'out/default'.
    """
    # Setup
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        save = True
    else:
        save = False
     
    # Plot
    for interp in ['combine', 'rbf']:
        # Load and unpack
        with open(os.path.join(out, f"{pair[0]}_{pair[1]}_{interp}.pkl"),
                  "rb") as f:
            d = dict(pickle.load(f))
            
        # Plot contours
        if save:
            styles = ['-', '--']
        else:
            styles = ['dashed', 'dashdot']
        if interp == 'combine':
            for ci, ls in zip(d.keys(), styles):
                ax.plot(d[ci][0], d[ci][1], ls=ls, lw=10,
                        label=f"Combine {ci}% CL", color='#5790fc')
        else:
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
                           zorder=5, colors='#f89c20', linestyles=style, linewidths=10)
                ax.plot(1, label=f"{label} {cl} CL", color='#f89c20', 
                        linestyle=style, lw=10)

    if save:
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
        
def cornerScan(interp: rbfInterpolator, 
               out: str='out/default') -> None:
    """
    Generates a corner plot with the 1D and 2D scan plots

    :param interp: The interpolator to use
    :type interp: rbfInterpolator
    :param out: The output dir, defaults to 'out/default'
    :type out: str, optional
    """
    # Setup
    pois = interp.pois
    bounds = interp.bounds
    fig = plt.figure(figsize=(45, 30))
    gs = GridSpec(len(pois), len(pois), hspace=0., wspace=0.)
    paddings = [50, 15, -15]
    tex_names = [r'$C_{Hg}\times 10^3$', r'$C_{Hw}\times 10^2$', r'$\Re{(C_{tg})}\times 10^1$']
    
    # Plot
    for i in range(len(pois)):
        for j in range(len(pois)):
            ax = fig.add_subplot(gs[i*len(pois) + j])
            if j > i:
                ax.axis('off')
                # Add legend
                if i == 0 and j == len(pois)-1:
                    ax.plot(1, label="Combine", color=colors[0], lw=10)
                    ax.plot(1, label=r"Combine $68\%$ CL", color=colors[0], lw=10, ls='--')
                    ax.plot(1, label=r"Combine $95\%$ CL", color=colors[0], lw=10, ls='-.')
                    
                    ax.plot(1, label="Interpolator", color=colors[1], lw=10)
                    ax.plot(1, label=r"Interpolator $68\%$ CL", color=colors[1], lw=10, ls='--')
                    ax.plot(1, label=r"Interpolator $95\%$ CL", color=colors[1], lw=10, ls='-.')
                    
                    ax.legend(loc='upper right', handlelength=2, prop={'size': 54})
            # Plot
            elif j == i:
                plotScan1D(pois[i], label='', ax=ax)
                ax.set_xlim((bounds[i][0]*1.075, bounds[i][1]*1.1))
                ax.set_yticks([])
                ax.tick_params(labelleft=False, left=False)
            else:
                plotScan2D((pois[j], pois[i]), label='', ax=ax)
                ax.set_xlim((bounds[j][0]*1.075, bounds[j][1]*1.1))
                ax.set_ylim((bounds[i][0]*1.075, bounds[i][1]*1.1))
            
            # Label
            if j == 0:
                ax.set_ylabel(tex_names[i], labelpad=paddings[i], fontsize=48)
            else:
                ax.tick_params(labelleft=False)
            if i == len(pois)-1:
                ax.set_xlabel(tex_names[j], fontsize=48)
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize=40)
    
    plt.savefig(os.path.join(out, f"cornerScan.pdf"), facecolor='white',
                bbox_inches='tight', dpi=250)

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
    # Initialise Combine1D interpolator
    data_test = Combine1D(data_config, poi).data
    xs = data_test[poi].to_numpy()
    ys = data_test['deltaNLL'].to_numpy()
    mask = np.where(2*ys <= 10)[0]
    
    # TODO vectorise
    # Compute differences to true value
    diffs = np.full(ys.shape, np.nan)
    scan_points = data_test[interp.pois]
    for i in range(len(scan_points)):
        diffs[i] = ys[i] - interp.evaluate(scan_points.iloc[i:i+1])
    outliers = np.argwhere(np.abs(diffs) > 0.5).flatten()
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.scatter(xs[mask], diffs[mask], label='test', s=100)
    ax.scatter(xs[outliers], np.clip(diffs[outliers], -0.5, 0.5),
               marker=7, c='r', zorder=10, s=200)
    ax.set_xlabel(poi)
    ax.set_ylabel(r"True - predicted")
    ax.set_ylim(-0.5, 0.5)
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
        
def getEdges(centers: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Get the bin edges from a list of bin centers

    :param centers: The bin centers
    :type centers: npt.NDArray[np.float32]
    :return: The bin edges
    :rtype: npt.NDArray[np.float32]
    """
    sorted_centers = np.sort(np.unique(centers))
    bin_widths = np.diff(sorted_centers)
    assert np.allclose(bin_widths, 
                       np.full(bin_widths.shape, bin_widths[0]))
    bin_width = bin_widths[0]
    edges = sorted_centers - 0.5*bin_width
    edges = np.append(edges, edges[-1]+bin_width)
    return edges

def plotDiff2D(pois: List[str], interp: rbfInterpolator, data_config: Data,
               out: str='out/default') -> None:
    """
    Plot the difference between the interpolated and true values using the\
        data from the 2D scans

    :param pois: The pair of pois to plot for
    :type pois: List[str]
    :param interp: The interpolator to use
    :type interp: rbfInterpolator
    :param data_config: Options for the data
    :type data_config: Data
    :param out: The output dir, defaults to 'out/default'
    :type out: str, optional
    """
    # Initialise Combine2D interpolator
    data_test = Combine2D(data_config, '_'.join(pois)).data
    xs = data_test[pois[0]].to_numpy()[1:]
    ys = data_test[pois[1]].to_numpy()[1:]
    zs = data_test['deltaNLL'].to_numpy()[1:]
    mask = np.where(2*zs <= 10)[0]
    
    # Get bin edges
    x_edges = getEdges(xs[mask])
    y_edges = getEdges(ys[mask])
    
    # TODO vectorise
    # Compute differences to true value
    diffs = np.full(zs.shape, np.nan)
    scan_points = data_test[interp.pois][1:]
    for i in range(len(scan_points)):
        diffs[i] = zs[i] - interp.evaluate(scan_points.iloc[i:i+1])

    # Plot with shifted weights so all weights are positive
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ws = diffs[mask]-diffs[mask].min()+1E-7
    norm = TwoSlopeNorm(vmin=-diffs[mask].min()-0.5, 
                        vmax=-diffs[mask].min()+0.5,
                        vcenter=-diffs[mask].min())
    h = ax.hist2d(xs[mask], ys[mask], weights=ws, 
                  cmap="bwr", cmin=1E-8, bins=[x_edges, y_edges], 
                  norm=norm)
    ax.scatter(xs[mask], ys[mask], c='k', marker='x', s=100)
    
    # Undo shift in colorbar
    cbar = fig.colorbar(h[3], ax=ax, extend='both')
    cbar.ax.set_yticks(np.arange(norm.vmin, norm.vmax+0.5, 0.5))
    ticks = (cbar.ax.get_yticks()+diffs[mask].min()).astype(str)
    ticks[0] = r'$\leq$' + ticks[0]
    ticks[-1] = r'$\geq$' + ticks[-1]
    cbar.ax.set_yticklabels(ticks)
    
    # Annotate
    ax.set_xlabel(pois[0])
    ax.set_ylabel(pois[1])
    
    # Save
    plt.savefig(os.path.join(out, f"{'_'.join(pois)}_diff.png"), 
                facecolor='white', bbox_inches='tight', dpi=125)
    
def plotAllDiff2D(interp: rbfInterpolator, data_config: Data,
                  out: str='out/default') -> None:
    """
    Plots the 2D differences for the specified POI pairs

    Args:
        pois (str): The pois to plot, pairwise
        label (str): The legend entries
        out (str, optional): The output dir. Defaults to 'out/default'.
    """
    poi_pairs = list(combinations(interp.pois, 2))
    for pair in poi_pairs:
        plotDiff2D(pair, interp, data_config, out=out)

"""
Handles all plotting for the models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
from scipy.optimize import OptimizeResult
from torch import nn
from typing import Tuple, Any, List
from torch.utils.data import DataLoader
from dataLoader import NLLDataset
from utils import Training, Data
hep.style.use("CMS")

eps = 1E-8

def plotLosses(train_losses: List[float], 
               test_losses: List[float], name: str) -> None:
    """
    Plot the training and testing losses per epoch

    Args:
        train_losses (List[float]): The training losses
        test_losses (List[float]): The testing losses
        name (str): The name of the output folder
    """
    # Logarithmic
    xs = np.arange(len(train_losses)) + 1
    _, ax = plt.subplots(1, 1, figsize=(16, 9))

    # Add training loss
    ax.semilogy(xs, train_losses, label="Training loss")
    ax.semilogy(xs, test_losses, label="Testing loss")
    
    # Label
    ax.legend(loc="upper right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    
    # Save
    plt.tight_layout()
    plt.savefig(f"out/{name}/lossLog.png", facecolor="white", 
                bbox_inches="tight", dpi=125)
    
    # Linear
    _, ax = plt.subplots(1, 1, figsize=(16, 9))

    # Add training loss
    ax.plot(xs, train_losses, label="Training loss")
    ax.plot(xs, test_losses, label="Testing loss")
    
    # Label
    ax.legend(loc="upper right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    
    # Save
    plt.tight_layout()
    plt.savefig(f"out/{name}/lossLinear.png", facecolor="white", 
                bbox_inches="tight", dpi=125)
    
def getDiffs(training_params: Training, data_params: Data, 
             model: nn.Sequential, loader: DataLoader[Any],
             size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the difference between predicted and true values for the model

    Args:
        training_params (Training): Options for training
        data_params (Data): Optiions for the data
        model (nn.Sequential): The model
        loader (DataLoader[Any]): The dataset to diff
        size (int): The size of the dataset to diff

    Raises:
        ValueError: Raised if mode isn't 'NLL' or 'likelihood'

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The difference in predicted and\
            true values and their corresponding xy coordinates

    """
    diffs = torch.ones((size, 1), dtype=torch.float32)
    xs = torch.ones((size, 2), dtype=torch.float32)
    batch_size = loader.batch_size or 0
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            pred = model(x)
            if data_params["mode"] == "likelihood":
                pred = max(pred, eps)
                pred = -np.log(pred)/training_params["temperature"]
                y = -np.log(y)/training_params["temperature"]
            elif data_params["mode"] == "NLL":
                pass
            else:
                raise ValueError("Mode must be 'NLL' or 'likelihood' " + 
                                f"not {data_params['mode']}")
            diffs[batch*batch_size:batch*batch_size+batch_size] = pred - y
            xs[batch*batch_size:batch*batch_size+batch_size] = x
    return diffs, xs

def plotDiffs(data_params: Data, training_params: Training, 
              model: nn.Sequential, loader: DataLoader[Any],
              model_min: float, name: str, suffix: str="All") -> None:
    """
    Plot the difference between the model prediction and the true value

    Args:
        data_params (Data): Options for the data
        training_params (Training): Options for training
        model (nn.Sequential): The model
        loader (DataLoader[Any]): The dataset to plot
        model_min (float): The best-fit value of the model
        name (str): The name of the output folder
        suffix (str, optional): Appended to the plot name. Defaults to "All".

    Raises:
        ValueError: Raised if mode isn't 'NLL' or 'likelihood'
    """
    if data_params["subtractbest"]:
        best_str = r" $ - \mathrm{best}$"
    else:
        best_str = ''
    
    # Get dataset size
    assert isinstance(loader.dataset, NLLDataset)
    size = len(loader.dataset)
    
    # Convert model minimum
    if data_params["mode"] == "likelihood":
        model_min = -np.log(-model_min)/training_params["temperature"]
    elif data_params["mode"] == "NLL":
        pass
    else:
        raise ValueError("Mode must be 'NLL' or 'likelihood' " + 
                         f"not {data_params['mode']}")
        
    # Get NLL_pred - NLL_true
    diffs, xys = getDiffs(training_params, data_params, model, loader, size)
    diffs = diffs.numpy().flatten()
    xys = xys.numpy()
    xs = xys[:, 0]
    ys = xys[:, 1]
        
    # Remin the network
    diffs -= model_min
    
    # Histogram the differences
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.hist(2*diffs)
    ax.set_xlabel(r"$2(\Delta\mathrm{NLL}_\mathrm{pred} -\Delta\mathrm{NLL}_\mathrm{true})$")
    ax.set_ylabel("Counts")
    plt.savefig(f"out/{name}/diffHist1D{suffix}.png", facecolor="white", 
                bbox_inches="tight", dpi=125)
    
    # Plot 2(NLL_pred - NLL_true)
    grid_size = (len(np.unique(xs)), len(np.unique(ys))) 
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    h = ax.hist2d(xs, ys, weights=np.abs(2*diffs), bins=grid_size, 
                  cmin=eps, norm=mcolors.Normalize(0, 10))
    ax.set_xlabel(r"$c_\mathrm{Hg}\times10^{3}$"+best_str)
    ax.set_ylabel(r"$c_\mathrm{HW}\times10^{2}$"+best_str)
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label(r"$2|\Delta\mathrm{NLL}_\mathrm{pred} -\Delta\mathrm{NLL}_\mathrm{true}|$")
    
    # Save
    plt.savefig(f"out/{name}/diffHist2D{suffix}.png", facecolor="white", 
                bbox_inches="tight", dpi=125)
    
def plotSurface(data_params: Data, training_params: Training, 
                model: nn.Sequential, res: OptimizeResult, 
                name: str) -> None:
    """
    Plot the NLL surface for a given model

    Args:
        data_params (Data): Options for the data
        training_params (Training): Options for the training
        model (nn.Sequential): The model
        res (OptimizeResult): The output from optimizing the model
        name (str): The name of the output folder

    Raises:
        ValueError: Raised if mode isn't 'NLL' or 'likelihood'
        ValueError: Raised if mode isn't 'NLL' or 'likelihood'
    """
    
    # TODO actually convert the values
    if data_params["subtractbest"]:
        best_str = r" $ - \mathrm{best}$"
    else:
        best_str = ''
    
    # Convert model minimum
    model_min = res['fun']
    if data_params["mode"] == "likelihood":
        model_min = -np.log(-model_min)/training_params["temperature"]
    elif data_params["mode"] == "NLL":
        pass
    else:
        raise ValueError("Mode must be 'NLL' or 'likelihood' " + 
                         f"not {data_params['mode']}")
    
    # Build grid
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.ones(X.shape).flatten()
    
    # Calculate values
    for i, (x, y) in enumerate(zip(X.flatten(), Y.flatten())):
        model.eval()
        coeffs_t = torch.from_numpy(np.array([x, y])).to(torch.float32)
        model_val = model(coeffs_t).detach().numpy().flatten()[0]
        if data_params["mode"] == "likelihood":
            model_val = max(model_val, eps)
            model_val = -np.log(model_val)/training_params["temperature"]
        elif data_params["mode"] == "NLL":
            pass
        else:
            raise ValueError("Mode must be 'NLL' or 'likelihood' " + 
                            f"not {data_params['mode']}")
        Z[i] = 2*(model_val - model_min)
        if(Z[i] < 0):
            print(i, x, y, model(coeffs_t).detach().numpy().flatten()[0],
                  model_min)
    
    # Reshape and plot
    Z = Z.reshape(X.shape)
    _, ax = plt.subplots(subplot_kw={"projection": "3d"},
                            figsize=(10, 10))
    ax.plot_surface(X, Y, Z, cmap="viridis", lw=0, antialiased=False,
                    norm=mcolors.AsinhNorm(vmin=Z.min()+1E-5,
                                           vmax=Z.max()))
    ax.set_xlabel(r"$c_\mathrm{Hg}\times10^{3}$" + best_str, labelpad=20)
    ax.set_ylabel(r"$c_\mathrm{HW}\times10^{2}$" + best_str, labelpad=20)
    ax.set_zlabel(r"$2\Delta\mathrm{NLL}$", labelpad=12)
    plt.savefig(f"out/{name}/surface.png", facecolor="white", dpi=125)

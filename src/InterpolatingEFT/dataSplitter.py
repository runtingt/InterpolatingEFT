"""
Splits the dataset int testing and training data, either randomly or uniformly
across a grid
"""

import torch
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from torch.utils.data import Subset, DataLoader
from dataLoader import toTorch, NLLDataset, loadData
from utils import Data, Training

NLLEntry = Tuple[torch.Tensor, torch.Tensor]

class Splitter(ABC):
    """
    Abstract base class for a generic data splitter

    Args:
        ABC (ABC): Helper class for abstract base classes
    """
    @abstractmethod
    def __init__(self, dataset: NLLDataset) -> None:
        """
        Initialises the splitter with a dataset

        Args:
            dataset (NLLDataset): The dataset to be split
        """
        super().__init__()
        self.dataset = dataset
    
    @abstractmethod
    def split(self, train_split: float) -> List[Subset[NLLDataset]]:
        """
        Split the dataset, reserving a fraction train_split for training

        Args:
            train_split (float): The fraction of data for training

        Returns:
            List[Subset]: The training and test datasets as subsets
        """
        # Range check
        assert train_split <= 1.0
        assert train_split >= 0.0
        return []
        
class RandomSplitter(Splitter):
    """
    Splits the dataset randomly

    Args:
        Splitter (Splitter): Abstract base splitter class
    """
    def __init__(self, dataset: NLLDataset) -> None:
        """
        Initialises the splitter with a dataset

        Args:
            dataset (NLLDataset): The dataset to be split
        """
        super().__init__(dataset)
    
    def split(self, train_split: float) -> List[Subset[NLLDataset]]:
        """
        Split the dataset, reserving a fraction train_split for training

        Args:
            train_split (float): The fraction of data for training

        Returns:
            List[Subset]: The training and test datasets as subsets
        """
        
        super().split(train_split)
        
        # Get indices
        idxs = np.arange(len(self.dataset))
        np.random.shuffle(idxs)

        # Split
        split_point = int(len(idxs)*train_split)
        idxs_train = list(idxs[:split_point])
        idxs_test = list(idxs[split_point:])
        
        return [Subset(self.dataset, idxs_train), Subset(self.dataset, idxs_test)]
    
class GridSplitter(Splitter):
    """
    Splits the dataset uniuformly across a grid

    Args:
        Splitter (Splitter): Abstract base splitter class
    """
    def __init__(self, dataset: NLLDataset) -> None:
        """
        Initialises the splitter with a dataset

        Args:
            dataset (NLLDataset): The dataset to be split
        """
        super().__init__(dataset)

    def _get_unique(self, 
                    arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Gets the unique elements along an axis

        Args:
            arr (npt.NDArray[np.float32]): The input array

        Returns:
            npt.NDArray[np.float32]: The unique elements
        """
        return np.unique(arr)

    def _get_axis_size(self, arr: npt.NDArray[np.float32]) -> int:
        """
        Gets the number of unique elements along an axis

        Args:
            arr (npt.NDArray[np.float32]): The array to get the size of

        Returns:
            int: The size of the axis
        """
        return len(self._get_unique(arr))

    def _get_points(self, size, num) -> npt.NDArray[np.int32]:
        """
        Gets num evenly-spaced points along an axis of length size

        Args:
            size (int): The length of the axis
            num (int): The number of points to get

        Returns:
            npt.NDArray[int]: The indices of the selected points
        """
        return np.round(np.linspace(0, size, num, endpoint=False)).astype(int)

    def split(self, train_split: float) -> List[Subset[NLLDataset]]:
        """
        Split the dataset, reserving a fraction train_split for training

        Args:
            train_split (float): The fraction of data for training

        Returns:
            List[Subset]: The training and test datasets as subsets
        """
        
        super().split(train_split)
        
        # To get a fraction 'x' of the overall dataset, 
        # we need x^(1/n) in each of the n dimensions
        ndims = self.dataset.X.shape[-1]
        train_split_per_dim = np.power(train_split, 1/ndims)
        
        # Get the grid size
        grid_size = np.apply_along_axis(self._get_axis_size, 0, 
                                        self.dataset.X.detach().numpy())
        
        # Get x^(1/n)*size evenly-spaced points in each axis
        # TODO vectorize
        chosen = np.zeros(ndims, dtype=np.ndarray)
        num_per_dim = np.round(grid_size*train_split_per_dim).astype(int)
        for i, (size, num) in enumerate(zip(grid_size, num_per_dim)):
            unique_idxs = self._get_points(size, num)
            unique = self._get_unique(self.dataset.X.detach().numpy()[:, i])            
            chosen[i] = unique[unique_idxs]
        
        # Get idxs for grid points from larger array, along each axis
        idxs = np.zeros(ndims, dtype=np.ndarray)
        for i in range(ndims):
            axis = self.dataset.X.detach().numpy()[:, i]
            for val in chosen[i]:
                idxs[i] = np.append(idxs[i], np.argwhere(axis == val))
                
        # Get train-test indices
        idxs_train = np.arange(0, self.dataset.X.shape[0])
        for i in range(ndims):
            idxs_train = np.intersect1d(idxs_train, idxs[i])
        idxs_test = np.setdiff1d(np.arange(0, self.dataset.X.shape[0]),
                                 idxs_train)
        
        # Report how close we got to target percentage
        # train_split_actual = len(idxs_train)/(len(idxs_train)+len(idxs_test))
        # print(f"Target was {train_split:.2%} training data,",
        #       f"actual was {train_split_actual:.2%}")
            
        return [Subset(self.dataset, list(idxs_train)), Subset(self.dataset, list(idxs_test))]
        
def dataToDict(dataset: NLLDataset) -> List[Dict[str, Any]]:
    """
    Converts the data to an array of dictionaries with format {x, y, val}.
    
    Args:
        dataset (NLLDataset): The dataset to convert

    Returns:
        List[Dict]: List with a dictionary for each data point
    """
    
    # Convert training dataset to dictionary
    data = []
    for elem_x, elem_y in zip(dataset.X, dataset.Y):
        d = {}
        assert isinstance(elem_x, torch.Tensor)
        assert isinstance(elem_y, torch.Tensor)
        for key, val in dataset.POIs.items():
            d[key] = elem_x.detach().numpy()[val]
            d[key] = elem_x.detach().numpy()[val]
        d["deltaNLL"] = elem_y.detach().numpy()[0]
        data.append(d)
    
    return data

def dataToFrame(dataset: NLLDataset) -> pd.DataFrame:
    """
    Converts data to a dataframe
    
    Args:
        dataset (NLLDataset): The dataset to convert

    Returns:
        pd.DataFrame: The dataset as a dataframe
    """
    
    # Build columns dictionary
    d = {}
    for poi, idx in dataset.POIs.items():
        d[poi] = dataset.X[:, idx].detach().numpy()
    d['deltaNLL'] = dataset.Y.detach().numpy().flatten()
    
    df = pd.DataFrame(data=d)
    return df

# def loadAll(data_params: Data, 
#             training_params: Training) -> DataLoader[NLLDataset]:
#     """
#     Loads the entire dataset

#     Args:
#         data_params (Data): Options specifying the data
#         training_params (Training): Options specifying the training

#     Raises:
#         ValueError: Raised if the data mode isn't 'NLL' or 'Likelihood'

#     Returns:
#         DataLoader[NLLDataset]: The entire dataset
#     """
#     # Grab data
#     X, Y = toTorch(data_params["file"])  
    
#     # Subtract best-fit vector
#     if data_params["subtractbest"]:
#         # Load best fit
#         best_fit = loadData(data_params["file"])[1]
#         coord = best_fit[["lchgXE3", "lchwXE2"]].iloc[0].to_numpy()
#         X -= coord 
    
#     # Convert Y
#     if data_params["mode"] == "likelihood":
#         Y = torch.exp(-training_params["temperature"]*Y)
#     elif data_params["mode"] == "NLL":
#         pass
#     else:
#         raise ValueError("Mode must be 'NLL' or 'likelihood' " + 
#                          f"not {data_params['mode']}")
#     dataset = NLLDataset(X, Y)
        
#     # Pass into loaders
#     loader = DataLoader(dataset, shuffle=False, 
#                         batch_size=len(X))

#     return loader

def loadAndSplit(file: str, data_config: Data, split: float,
                 include_best: bool=False):
    # Load
    poi_map = {poi: i for i, poi in enumerate(data_config["POIs"])}
    X, Y = toTorch(file, list(data_config["POIs"].keys()), include_best=include_best)
    dataset = NLLDataset(X, Y, poi_map)
    
    # Split
    if data_config["splitting"] == "grid":
        gs = GridSplitter(dataset)
        train, test = gs.split(split)
        return train, test
    else:
        raise NotImplementedError("Only grid splitting is implemented")

# if __name__ =="__main__":
#     # Grab data
#     X, Y = toTorch("data/scan.hgg_statonly.STXStoSMEFTExpandedLinearStatOnly"
#                    ".observed.nominal.lchgXE3_lchwXE2.root")
#     dataset = NLLDataset(X, Y)
    
#     # Test random
#     rs = RandomSplitter(dataset)
#     rs_split = rs.split(0.9)
#     print(list(map(len, rs_split)))
    
#     # Test grid
#     gs = GridSplitter(dataset)
#     gs_split = gs.split(0.25**2)
#     print(list(map(len, gs_split)))
    
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 10))
#     plt.scatter(X.detach().numpy()[gs_split[0].indices][:, 0],
#                 X.detach().numpy()[gs_split[0].indices][:, 1],
#                 c='k', s=100, label="Train")
#     plt.scatter(X.detach().numpy()[gs_split[1].indices][:, 0],
#                 X.detach().numpy()[gs_split[1].indices][:, 1],
#                 c='r', s=0.5, label="Test")
#     plt.legend(loc='upper right')
#     plt.savefig("../plots/grid_6.png", bbox_inches='tight')
    
#     # dataToDict(1)
    
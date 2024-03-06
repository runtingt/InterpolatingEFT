"""
Splits the dataset int testing and training data, either randomly or uniformly
across a grid
"""

import torch
import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from abc import ABC, abstractmethod
from torch.utils.data import Subset
from InterpolatingEFT.dataLoader import toTorch, NLLDataset
from InterpolatingEFT.utils import Data

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
        
def loadAndSplit(file: str, data_config: Data, split: float,
                 include_best: bool=False) -> Tuple[Subset[NLLDataset],
                                                    Subset[NLLDataset],
                                                    NLLDataset]:
    """
    Loads the dataset as a train/test split

    Args:
        file (str): The file to load the dataset from
        data_config (Data): Options for the data
        split (float): The train fraction
        include_best (bool, optional): Include the best fit point.\
            Defaults to False.

    Raises:
        NotImplementedError: Raised if an option other than 'grid' is specified

    Returns:
        Tuple[Subset[NLLDataset], Subset[NLLDataset], NLLDataset]: The train /\
            test split and the best fit point
    """
    # Load
    poi_map = {poi: i for i, poi in enumerate(data_config["POIs"])}
    X, Y = toTorch(file, list(data_config["POIs"].keys()), include_best=include_best)
    dataset = NLLDataset(X, Y, poi_map)
    
    # Split
    best = NLLDataset(torch.tensor(np.full(len(poi_map), np.inf)),
                      torch.tensor(np.inf), poi_map)
    if include_best:
        best = NLLDataset(*dataset[:1], poi_map)
        dataset = NLLDataset(*dataset[1:], poi_map)
    
    if data_config["splitting"] == "grid":
        gs = GridSplitter(dataset)
        train, test = gs.split(split)        
        return train, test, best
    else:
        raise NotImplementedError("Only grid splitting is implemented")    

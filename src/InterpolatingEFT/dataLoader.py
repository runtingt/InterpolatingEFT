"""
Loads data from .ROOT files into a custom torch dataset
"""

from __future__ import annotations
import torch
import uproot
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple, Any, Dict, Sequence

def loadData(filename: str, POIs: List[str],
             include_best: bool=False) -> pd.DataFrame:
    """
    Loads the data from a ROOT file into a pandas dataframe
    
    Args:
        filename (str): The path to the file to load
        POIs (List[str]): The pois to load the scans for
        include_best (bool): Include the best-fit point. Defaults to False

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataframes with the grid data and\
            best fit value
    """
    # Get dataframe
    file = uproot.open(filename)
    assert isinstance(file, uproot.ReadOnlyDirectory)
    limit = file["limit"]
    assert isinstance(limit, uproot.TTree)
    df = pd.DataFrame(limit.arrays(["deltaNLL"] + POIs, library="pd"))
    file.close()
    
    # Clean up data
    df = df.drop_duplicates()
    df = df[df["deltaNLL"] <= 1E5] # NOTE: Clip large values
    if not include_best: df = df[df["deltaNLL"] != 0]
    df = df.reset_index()
    
    return df

def toTorch(file: str, POIs: List[str],
             include_best: bool=False) -> List[torch.Tensor]:
    """
    Convert data from a ROOT file into torch datasets for input and
    target data

    Args:
        file (str): The path to the file to load
        POIs (List[str]): The pois to load the scans for
        include_best (bool): Include the best-fit point. Defaults to False

    Returns:
        List[torch.Tensor]: The input and target torch tensors in a list.
    """
    df = loadData(file, POIs, include_best=include_best)
    X = torch.tensor(df.iloc[:, 2:].to_numpy(), dtype=torch.float32)
    Y = torch.tensor(df.iloc[:, 1:2].to_numpy(), dtype=torch.float32)
    return [X, Y]

class NLLDataset(Dataset[Any]):
    """
    Torch-compatible dataset for the NLL data
    """
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, 
                 POIs: Dict[str, int]) -> None:
        """
        Initialises the dataset

        Args:
            X (torch.Tensor): The model input data
            Y (torch.Tensor): The model output data
            POIs (Dict[str]): Map of POIs to their column index
        """
        super().__init__()
        self.X = X.clone().detach().to(torch.float32)
        self.Y = Y.clone().detach().to(torch.float32)
        self.POIs = POIs
    def __len__(self) -> int:
        """
        Gets the length of the input data

        Returns:
            int: The length of the input data
        """
        return len(self.X)
    def __getitem__(self, 
                    index: int | List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets a single item from the dataset

        Args:
            index (int | List[int]): The indices of the item to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The retrieved item
        """
        return self.X[index], self.Y[index]
    def append(self, new: NLLDataset) -> None:
        if self.POIs != new.POIs:
            raise ValueError("Datsets must have the same POIs, not " +
                             f"{self.POIs} and {new.POIs}")
        else:
            self.X = torch.cat((self.X, new.X))
            self.Y = torch.cat((self.Y, new.Y))

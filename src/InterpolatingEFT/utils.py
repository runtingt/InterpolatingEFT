"""
Utilities for handling model configurations
"""

import os
import torch
import yaml
from typing import TypedDict, Callable, Dict, List

WeightedLoss = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]

class LearningRate(TypedDict):
    """
    Options for the learning rate

    Args:
        TypedDict (TypedDict): Base class for TypedDict
    """
    initial: float
    decay: float
    patience: int

class Training(TypedDict):
    """
    Options for training

    Args:
        TypedDict (TypedDict): Base class for TypedDict
    """
    seedmin: int
    seedmax: int
    epochs: int
    batchsize: int
    temperature: float
    learningrate: LearningRate
    
class Model(TypedDict):
    """
    Options for the model

    Args:
        TypedDict (TypedDict): Base class for TypedDict
    """
    activation: str
    
class Data(TypedDict):
    """
    Options for the data

    Args:
        TypedDict (TypedDict): Base class for TypedDict
    """
    channel: str
    model: str
    type: str
    attribute: str
    POIs: Dict[str, Dict[str, List[int]]]
    mode: str
    splitting: str
    fraction: float
    subtractbest: bool
    
class Config(TypedDict):
    """
    The configuration format

    Args:
        TypedDict (TypedDict): Base class for TypedDict
    """
    training: Training
    model: Model
    data: Data

class Accessories(TypedDict):
    """
    Format of the model accessories

    Args:
        TypedDict (TypedDict): Base class for TypedDict
    """
    loss_fn: WeightedLoss
    optim: torch.optim.Adam
    sched: torch.optim.lr_scheduler.ReduceLROnPlateau
    
def loadConfig(filename: str) -> Config:
    """
    Loads a configuation file into a Config object

    Args:
        filename (str): The path to the config file

    Returns:
        Config: The configuration options
    """
    with open(os.path.abspath(filename)) as f:
        config = yaml.safe_load(f)
    config = Config(config)
    return config

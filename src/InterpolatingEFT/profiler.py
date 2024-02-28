"""
Makes all profiled 1D and 2D scans
"""

import os
import pickle
import numpy as np
import tqdm
import logging
import concurrent.futures
from functools import partial
from itertools import combinations
from typing import List
from InterpolatingEFT.utils import Data
from InterpolatingEFT.interpolator import rbfInterpolator, combineInterpolator, Combine1D, Combine2D
from InterpolatingEFT.logger import TqdmToLogger

def profileCombine1D(data_config: Data, out: str='out/default') -> None:
    """
    Extracts the data from the 1D Combine scans for all POIs

    Args:
        data_config (Data): Options for the data
        out (str, optional): The out dir. Defaults to 'out/default'.
    """
    pois = list(data_config["POIs"].keys())
    for poi in pois:
        interp = Combine1D(data_config, poi)
            
        # Save
        d = {'x': interp.data[poi], 'y': interp.data['deltaNLL'], 'best': 0}
        with open(os.path.join(out, f"{poi}_combine.pkl"), "wb") as f:
            pickle.dump(d, f)
            
def profileCombine2D(data_config: Data, out: str='out/default') -> None:
    """
    Extracts the data from the 2D pairwise Combine scans for all POIs

    Args:
        data_config (Data): Options for the data
        out (str, optional): The out dir. Defaults to 'out/default'.
    """
    pois = list(data_config["POIs"].keys())
    poi_pairs = list(combinations(pois, 2))
    for pair in poi_pairs:
        pair_name = '_'.join(pair)
        interp = Combine2D(data_config, pair_name)
            
        # Save
        with open(os.path.join(out, f"{pair_name}_combine.pkl"), "wb") as f:
            pickle.dump(interp.contours, f)
            
def profileCombine(data_config: Data, out: str='out/default') -> None:
    """
    Extracts the data from the 1D and 2D Combine scans

    Args:
        data_config (Data): Options for the data
        out (str, optional): The out dir. Defaults to 'out/default'.
    """
    profileCombine1D(data_config, out)
    profileCombine2D(data_config, out)

def profile1D(interp: rbfInterpolator | combineInterpolator, poi: str, 
              num: int=50, out: str='out/default') -> None:
    """
    Get the 1D profiled scan for the specified POI, using an interpolator

    Args:
        interp (rbfInterpolator): The interpolator to use
        poi (str): The poi to scan
        num (int, optional): The number of scan points. Defaults to 50.
        out (str, optional): The out dir. Defaults to 'out/default'.
    """
    print(poi)
    # Get free keys
    free_keys = [key for key in interp.pois if key != poi]
    
    # Get bounds for fixed parameter
    bounds = interp.bounds[interp.pois.index(poi)]
    
    if isinstance(interp, rbfInterpolator):
        xs = np.linspace(bounds[0], bounds[1], num=num)
    elif isinstance(interp, combineInterpolator):
        xs = np.sort(interp.data_train[poi].unique())
    ys = []
    
    # Profile
    for x in xs:
        ys.append(interp.minimize(free_keys, fixed_vals={poi: [x]})['fun'])
        
    # Get overall minimum
    interp_min = interp.minimize(interp.pois, {})['fun']
    
    # Save
    d = {'x': xs, 'y': ys, 'best': interp_min}
    with open(os.path.join(out, f"{poi}_rbf.pkl"), "wb") as f:
        pickle.dump(d, f)
   
def profileAll1D(interp: rbfInterpolator | combineInterpolator,
                 pois: List[str], num: int=50, 
                 out: str='out/default') -> None:
    """
    Get the 1D profiled scan for all POIs, using an interpolator

    Args:
        interp (rbfInterpolator): The interpolator to use
        pois (str): The pois to scan
        num (int, optional): The number of scan points. Defaults to 50.
        out (str, optional): The out dir. Defaults to 'out/default'.
    """
    for poi in pois:
        profile1D(interp, poi, num=num, out=out)

def profile2D(interp: rbfInterpolator, num: int, 
              out: str, pois: List[str]) -> None:
    """
    Get the 2D pairwise profiled scan for the pois specified, \
        using an interpolator

    Args:
        interp (rbfInterpolator): The interpolator to use
        num (int, optional): The number of scan points. Defaults to 50.
        out (str, optional): The out dir. Defaults to 'out/default'.
        pois (str): The poi pair to scan
    """
    # Setup logging
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s',
                        filename=os.path.join(out, f'{pois[0]}_{pois[1]}.log'),
                        filemode='w')
    logger = logging.getLogger()
    assert isinstance(logger, logging.RootLogger)
    logger.setLevel(logging.INFO)
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    # Get free keys
    free_keys = list(set(interp.pois).symmetric_difference(set(pois)))
    
    # Get bounds for fixed parameters
    bounds = []
    for poi in pois:
        bounds.append(interp.bounds[interp.pois.index(poi)])
    
    # Profile
    x = np.linspace(bounds[0][0], bounds[0][1], num)
    y = np.linspace(bounds[1][0], bounds[1][1], num)
    X, Y = np.meshgrid(x, y)
    Z = np.full(X.shape, np.nan).flatten()
    for i, (x, y) in tqdm.tqdm(enumerate(zip(X.flatten(), Y.flatten())),
                               total=len(Z), file=tqdm_out, mininterval=10):
        res = interp.minimize(free_keys, fixed_vals={poi: [float(val)] for poi, val in zip(pois, [x, y])})
        Z[i] = res['fun']
        
    Z = Z.reshape(X.shape)
        
    # Get overall minimum
    interp_min = interp.minimize(interp.pois, {})['fun']
    
    # Save
    d = {'x': X, 'y': Y, 'z': Z, 'best': interp_min}
    with open(os.path.join(out, f"{pois[0]}_{pois[1]}_rbf.pkl"), "wb") as f:
        pickle.dump(d, f)
        
def profileAll2D(interp: rbfInterpolator, pois: List[str], 
                 num: int=10, out: str='out/default') -> None:
    """
    Get the 2D pairwise profiled scan for all POIs, using an interpolator

    Args:
        interp (rbfInterpolator): The interpolator to use
        pois (str): The pois to scan
        num (int, optional): The number of scan points. Defaults to 50.
        out (str, optional): The out dir. Defaults to 'out/default'.
    """
    poi_pairs = list(combinations(pois, 2))
    profile_part = partial(profile2D, interp, num, out)
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        executor.map(profile_part, poi_pairs)
    
"""
Runs the script
"""

import os
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from InterpolatingEFT.interpolator import rbfInterpolator, combineInterpolator
from InterpolatingEFT.profiler import profileCombine, profileAll1D, profileAll2D
from InterpolatingEFT.plotter import plotAllScan1D, plotAllScan2D, plotAllDiff1D, plotAllDiff2D, cornerScan
from InterpolatingEFT.utils import loadConfig
from InterpolatingEFT.toTable import toTable

if __name__ == "__main__":
    stime = datetime.now()
    os.chdir("..") # For testing only
    
    # Parse args
    parser = argparse.ArgumentParser("Interpolating EFT")
    parser.add_argument("--configfile", help="Path to .yaml config file",
                        required=True)
    args = vars(parser.parse_args())
    
    # Load config
    print("Loading config...")
    config  = loadConfig(args["configfile"])
    name = Path(args["configfile"]).stem
    outdir = f"out/{name}"
    os.makedirs(outdir, exist_ok=True)
    print("Done!")
    
    # Profile in 1D and 2D
    print("Loading Combine data...")
    profileCombine(config["data"], out=outdir)
    print("Done!")
    if config["data"]["interpolator"]["mode"] == "RBF":
        print("Initialising RBF...")
        interp = rbfInterpolator()
        interp.initialise(config["data"])
        print("Done!")
        
        print("Profiling 1D...")
        profileAll1D(interp, interp.pois, num=50, out=outdir)
        print("Done!")
        print("Profiling 2D (see logs for more)...")
        profileAll2D(interp, interp.pois, num=50, out=outdir)
        print("Done!")        
        
        # Plot
        print("Plotting profiled plots...")
        plotAllScan1D(interp.pois, 
                      [f"rbfSpline({len(interp.data_train)})"]*len(interp.pois),
                      out=outdir)
        plotAllScan2D(interp.pois, 
                      [f"rbfSpline({len(interp.data_train)})"]*len(interp.pois),
                      out=outdir)
        print("Done!")
        print("Plotting difference plots...")
        plotAllDiff1D(interp, config["data"], out=outdir)
        plotAllDiff2D(interp, config["data"], out=outdir)
        print("Done!")
        print("Plotting corner plots...")
        cornerScan(interp, out=outdir)
        print("Done!")
    else:
        raise NotImplementedError("Only RBF is implemented")
    
    # Add table
    toTable(args["configfile"], outdir)
    print(f"Finished! Total time: {datetime.now()-stime}")

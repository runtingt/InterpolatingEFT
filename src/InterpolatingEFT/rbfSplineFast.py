import sys
import numpy as np
import pandas as pd
import numpy.typing as npt

# -----------------
# Basis functions
# -----------------
class radialGauss():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.exp(-input)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return -self.evaluate(input)

class radialMultiQuad():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.sqrt(1+input)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 1/(2*self.evaluate(input))
    
class radialInverseMultiQuad():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.divide(1, np.sqrt(1+input))
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return -1/(2*np.power(1+input, 3/2))

class radialLinear():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.sqrt(input)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 1/(2*self.evaluate(input))

class radialCubic():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.power(input, 3/2)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 3*np.sqrt(input)/2
    
class radialQuintic():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.power(input, 5/2)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 5*np.power(input, 3/2)/2

class radialThinPlate():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.multiply(input, np.log(np.sqrt(input)))
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return (np.log(input)+1)/2
# -----------------

class rbfSplineFast:
    def __init__(self, ndim=1) -> None:
        self._ndim = ndim
        self._initialised = False
        self._radialFuncs = dict(
            [("gaussian", radialGauss),
             ("multiquadric", radialMultiQuad),
             ("inversemultiquadric", radialInverseMultiQuad),
             ("linear", radialLinear),
             ("cubic", radialCubic),
             ("quintic", radialQuintic),
             ("thinplate", radialThinPlate)
            ])

    def _initialise(self, input_data: pd.DataFrame, target: str, 
                    eps: float, rescaleAxis: bool) -> None:
        # Parse args
        self._input_data = input_data
        self._target_col = target
        self._input_pts = input_data.drop(target, axis="columns").to_numpy()
        self._eps = eps  
        self._rescaleAxis = rescaleAxis
        self._parameter_keys = list(input_data.columns)
        self._parameter_keys.remove(target)

        # Check number of basis points
        self._M = len(input_data)
        if self._M < 1 : 
            sys.exit("Error - At least one basis point is required")
        
        # Check dimensions
        if self._ndim!=len(self._parameter_keys): 
            sys.exit(f"Error - initialise given points with more dimensions " +
                     f"({len(self._parameter_keys)}) than ndim ({self._ndim})")

        # Get scalings by axis (column)
        self._axis_pts = np.power(self._M, 1./self._ndim)
        if self._rescaleAxis:
            self._scale = np.divide(self._axis_pts, 
                                    (np.max(self._input_pts, axis=0) -
                                     np.min(self._input_pts, axis=0)))
        else:
            self._scale = 1

        self.calculateWeights()

    def initialise(self, input_data: pd.DataFrame, target_col: str, 
                   radial_func: str="gaussian", eps: float=10.,
                   rescaleAxis: bool=True) -> None:
        # Get basis function and initialise
        try:
            self.radialFunc = self._radialFuncs[radial_func]()
        except KeyError:
            sys.exit(f"Error - function '{radial_func}' not in " +
                     f"'{list(self._radialFuncs.keys())}'")
        self._initialise(input_data, target_col, eps, rescaleAxis)
        
    def initialise_text(self, input_file: str, target_col, 
                        radial_func: str="gaussian", eps: float=10.,
                        rescaleAxis: bool=True) -> None:
        df = pd.read_csv(input_file, index_col=False, delimiter=' ')
        self.initialise(df,target_col,radial_func,eps,rescaleAxis)
        
    def diff(self, points1: npt.NDArray[np.float32],
             points2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Get diff between two sets of points, pairwise
        v = np.multiply(self._scale, (points1[:, np.newaxis, :] - 
                                      points2[np.newaxis, :, :]))
        return v    
    
    def diff2(self, points1: npt.NDArray[np.float32], 
              points2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Get squared diff between two sets of points, pairwise
        return np.power(self.diff(points1, points2), 2)
    
    def getDistFromSquare(self, point: npt.NDArray[np.float32]):
        # Get distance between a point and the basis points, per axis
        return self.diff2(point, self._input_pts)
    
    def getRadialArg(self, 
                     d2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Get arg to pass to basis functions
        return np.divide(d2, self._eps*self._eps)

    def grad_r2(self, point) -> npt.NDArray[np.float32]:
        # Calculates grad(|r|^2)
        return (2*self.diff(point, self._input_pts)*self._scale/(self._eps*self._eps))

    def evaluate(self, point: pd.DataFrame) -> float:
        # Check input is okay (can be turned off for perfomance)
        if not self._initialised:
            print("Error - must first initialise spline with set of points " + 
                  "before calling evaluate()") 
            return np.nan
        if not set(point.keys()) == set(self._parameter_keys): 
            print(f"Error - {point.keys()} must match {self._parameter_keys}")
            return np.nan
        
        # Evaluate spline at point
        point_arr = point.to_numpy()
        radial_arg = self.getRadialArg(np.sum(self.getDistFromSquare(point_arr), axis=-1))
        vals = self.radialFunc.evaluate(radial_arg).flatten()
        
        # Get val and grads
        weighted_vals = self._weights * vals
        ret_val = np.sum(weighted_vals)
        
        return ret_val.astype(float)
    
    def evaluate_grad(self, point: pd.DataFrame) -> npt.NDArray[np.float32]:
        # Check input is okay (can be turned off for perfomance)
        if not self._initialised:
            print("Error - must first initialise spline with set of points " + 
                  "before calling evaluate()") 
            return np.array(np.nan)
        if not set(point.keys()) == set(self._parameter_keys): 
            print(f"Error - {point.keys()} must match {self._parameter_keys}")
            return np.array(np.nan)

        # Evaluate spline at point
        point_arr = point.to_numpy()
        radial_arg = self.getRadialArg(self.getDistFromSquare(point_arr))
        delta_phi = np.linalg.norm(self.radialFunc.getDeltaPhi(radial_arg), axis=-1)
        np.set_printoptions(linewidth=200)
        grads = (self.grad_r2(point_arr) * delta_phi.reshape(1, self._M, 1))
        
        # Get val and grads
        weighted_grads = np.multiply(self._weights.reshape(1, self._M, 1), grads)
        ret_grad = np.sum(weighted_grads, axis=1)
        
        return ret_grad.astype(float)
        
    def calculateWeights(self) -> None: 
        # Solve interpolation matrix equation for weights
        inp = self._input_pts
        B = self._input_data[self._target_col].to_numpy()
        d2 = np.sum(self.diff2(inp, inp), axis=2)
        A = self.radialFunc.evaluate(self.getRadialArg(d2)) 
        np.fill_diagonal(A, 1)
    
        self._weights = np.linalg.solve(A, B)
        self._initialised = True
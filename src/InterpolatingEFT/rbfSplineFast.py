import sys
import numpy as np
from typing import Optional, List

class rbfSplineFast:
    def __init__(self,ndim=1):
        self._ndim = ndim
        self._initialised = False
        self.radialFuncs = dict([("gaussian", self.radialGauss),
                                 ("multiquadric", self.radialMultiQuad),
                                 ("inversemultiquadric", self.radialInverseMultiQuad),
                                 ("cubic", self.radialCubic)])
    
    def _initialise(self,input_data,target_col,eps,rescaleAxis):
        self._input_data = input_data
        self._target_col = target_col
        self._input_points = input_data.drop(target_col, 
                                             axis="columns").to_numpy()
        
        self._eps = eps  
        self._rescaleAxis = rescaleAxis

        self._M = len(input_data) # Num points
        if self._M < 1 : sys.exit("Error - rbf_spline must be initialised with at least one basis point")
        
        self._parameter_keys = list(input_data.columns)
        self._parameter_keys.remove(target_col)

        if self._ndim!=len(self._parameter_keys): 
            sys.exit("Error - initialise given points with more dimensions (%g) than ndim (%g)"%(len(self._parameter_keys),self._ndim))

        self._axis_pts = self._M**(1./self._ndim)
       
        self.calculateWeights()

    def initialise(self,input_points,target_col,radial_func="gaussian",eps=10,rescaleAxis=True):
        try:
            self.radialFunc = self.radialFuncs[radial_func]
        except KeyError:
            sys.exit("Error - function '%s' not in '%s'"%(radial_func, list(self.radialFuncs.keys())))
        self._initialise(input_points,target_col,eps,rescaleAxis)
    
    def diff2(self, points1, points2):
        # The interpolator must have been initialised on points2
        v = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        if self._rescaleAxis: v=self._axis_pts*v/(np.max(points2, axis=0) - np.min(points2, axis=0))
        return np.power(v, 2)

    def getDistSquare(self, col):
        return self.diff2(col, col)
        
    def getDistFromSquare(self, point, inp):
        dk2 = np.sum(self.diff2(point, inp), axis=-1).flatten()
        return dk2

    def getRadialArg(self, d2):
        return (d2/(self._eps*self._eps))

    def radialGauss(self,d2):
        return np.e**(-1*self.getRadialArg(d2))
    
    def radialMultiQuad(self,d2):
        return np.sqrt(1+self.getRadialArg(d2)) 
        
    def radialInverseMultiQuad(self, d2):
        return 1/self.radialMultiQuad(self.getRadialArg(d2))

    def radialCubic(self, d2):
        return np.power(self.getRadialArg(d2), 3/2)

    def evaluate(self,point):
        if not self._initialised:
            print("Error - must first initialise spline with set of points before calling evaluate()") 
            return np.nan
        if not set(point.keys())==set(self._parameter_keys): 
            print ("Error - must have same variable labels, you provided - ",point.keys(),", I only know about - ",self._parameter_keys)
            return np.nan
        vals = self.radialFunc(self.getDistFromSquare(point.to_numpy(), self._input_points))
        weighted_vals = self._weights * vals
        return sum(weighted_vals)

    def calculateWeights(self) : 
        inp = self._input_points
        B = self._input_data[self._target_col].to_numpy()
        d2 = np.sum(self.diff2(inp, inp), axis=2)
        A = self.radialFunc(d2) 
        np.fill_diagonal(A, 1)
    
        self._weights = np.linalg.solve(A,B)
        self._initialised=True


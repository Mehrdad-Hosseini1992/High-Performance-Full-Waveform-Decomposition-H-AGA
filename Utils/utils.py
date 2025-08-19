# utils.py
# --- (MODIFIED) ---
# The main fitting function now includes a 'baseline' and 'p' (shape) parameter for the generalized Gaussian.

import numpy as np

class N_Gaussian:
    @staticmethod
    def generalized_gaussian(x: np.ndarray, amp: float, pos: float, wid: float, p: float) -> np.ndarray:
        if wid <= 1e-6:
            return np.zeros_like(x)
        return amp * np.exp(-np.power(np.abs(x - pos) / (np.sqrt(2) * wid), p))

    @staticmethod
    def func(x: np.ndarray, *params) -> np.ndarray:
        """
        Calculates the sum of multiple generalized Gaussian curves plus a baseline offset.
        
        Args:
            x (np.ndarray): The independent variable array.
            *params: A flattened list [amp1, pos1, wid1, p1, ..., baseline].
        """
        # The last parameter is now the baseline
        baseline = params[-1]
        y = np.full_like(x, baseline, dtype=float)
        
        # Process the Gaussian components
        num_gaussians = (len(params) - 1) // 4
        for i in range(num_gaussians):
            amp, pos, wid, p = params[i*4 : (i+1)*4]
            if wid > 1e-6 and not np.isnan(wid):
                 y += N_Gaussian.generalized_gaussian(x, amp, pos, wid, p)
        return y
    
    @staticmethod
    def gaussian(x: np.ndarray, amp: float, pos: float, wid: float) -> np.ndarray:
        """Standard Gaussian (p=2 case of generalized Gaussian)"""
        if wid <= 1e-6:
            return np.zeros_like(x)
        return amp * np.exp(-((x - pos) ** 2) / (2 * wid ** 2))
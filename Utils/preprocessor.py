# Utils/preprocessor.py
# --- (DEFINITIVE FINAL VERSION) ---
# The `run` method now returns the calculated baseline array, which is essential
# for correctly repositioning the final fitted curve.

import numpy as np
import pywt
from scipy.signal import savgol_filter, medfilt
from scipy.stats import median_abs_deviation
from scipy import sparse
from scipy.sparse.linalg import spsolve

class Preprocessor:
    def __init__(self, n_sigma: float = 2.0):
        self.n_sigma = n_sigma

    def wavelet_denoise(self, signal: np.ndarray, wavelet='db4', level=None):
        if len(signal) < 20: return signal
        if level is None:
            level = min(pywt.dwt_max_level(len(signal), wavelet), 5)
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = median_abs_deviation(coeffs[-1], scale='normal')
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs_thresh, wavelet)
    
    def adaptive_median_filter(self, signal: np.ndarray, max_window=9):
        result = signal.copy()
        for i in range(len(signal)):
            for window in range(3, max_window + 1, 2):
                start = max(0, i - window // 2)
                end = min(len(signal), i + window // 2 + 1)
                window_data = signal[start:end]
                med = np.median(window_data)
                mad = median_abs_deviation(window_data, scale='normal')
                if not np.isfinite(mad) or mad < 1e-9: continue
                if abs(signal[i] - med) > 3 * mad:
                    result[i] = med
                    break
        return result
    
    def estimate_baseline(self, signal: np.ndarray):
        L = len(signal)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = 1e5 * D.dot(D.T)
        w = np.ones(L)
        y = signal
        for _ in range(10):
            W = sparse.diags(w, 0, shape=(L, L))
            Z = W + D
            baseline = spsolve(Z, w * y)
            w = 0.99 * (y > baseline) + 0.01 * (y <= baseline)
        return baseline
    
    def run(self, y_data: np.ndarray):
        y_despike = self.adaptive_median_filter(y_data)
        y_denoised = self.wavelet_denoise(y_despike)
        
        # --- MODIFICATION: Store the baseline to return it ---
        original_baseline = self.estimate_baseline(y_denoised)
        y_corrected = y_denoised - original_baseline
        # --------------------------------------------------
        
        is_inverted = abs(np.min(y_corrected)) > abs(np.max(y_corrected))
        if is_inverted:
            y_corrected *= -1
        
        noise_segment = y_corrected[:min(len(y_corrected), 50)]
        noise_std = median_abs_deviation(noise_segment, scale='normal')
        threshold = self.n_sigma * noise_std
        
        print(f"  - Preprocessing complete. Noise threshold set to: {threshold:.4f}")
        # Return the calculated baseline along with other results
        return y_corrected, threshold, is_inverted, original_baseline
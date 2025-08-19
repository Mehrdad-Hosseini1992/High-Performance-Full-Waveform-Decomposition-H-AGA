# Core/peak_detector.py
# --- (CORRECTED FOR ROBUSTNESS) ---
# Tuned find_peaks parameters to be less sensitive to noise and find only major peaks.

import numpy as np
from scipy.signal import find_peaks, peak_widths
from typing import List, Dict

class PeakDetector:
    def __init__(self, noise_threshold: float):
        self.noise_threshold = noise_threshold

    def _estimate_peak_widths(self, y_data: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        if len(peaks) == 0:
            return np.array([])
        widths_info = peak_widths(y_data, peaks, rel_height=0.5)
        fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
        sigmas = widths_info[0] * fwhm_to_sigma
        return sigmas

    def run(self, y_data: np.ndarray) -> List[Dict]:
        # --- MODIFICATION START: Making peak detection more robust ---
        
        # The peak height must be significantly above the noise.
        min_height = self.noise_threshold * 4.0

        # The prominence is crucial. A peak must stand out from its surroundings.
        # We set it to a fraction of the total signal height.
        min_prominence = (np.max(y_data) - self.noise_threshold) * 0.1

        # Peaks must be reasonably separated. 50 samples is a good starting point.
        min_distance = 50
        
        print(f"  - Using Peak Detection Parameters: height > {min_height:.4f}, prominence > {min_prominence:.4f}, distance > {min_distance}")

        peaks, _ = find_peaks(
            y_data,
            height=min_height,
            distance=min_distance,
            prominence=min_prominence
        )
        # --- MODIFICATION END ---

        if len(peaks) == 0:
            print("  - No prominent peaks found with the new robust parameters.")
            return []

        print(f"  - Found {len(peaks)} initial prominent peaks.")
        
        initial_params = []
        sigmas = self._estimate_peak_widths(y_data, peaks)

        for i, p_idx in enumerate(peaks):
            sigma = max(sigmas[i], 1.0) if i < len(sigmas) else 10.0
            
            initial_params.append({
                'A': y_data[p_idx], 
                't': float(p_idx), 
                'sigma': sigma, 
                'p': 2.0
            })
        
        initial_params.sort(key=lambda p: p['t'])
        print(f"  - Total initial components detected: {len(initial_params)}")
        return initial_params
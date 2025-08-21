
import numpy as np
import torch
from Utils.preprocessor import Preprocessor
from Core.peak_detector import PeakDetector
from optimization.aga_optimizer import AdaptiveGeneticOptimizer
from optimization.lm_optimizer import LMFitter
from Utils.utils import N_Gaussian

class WaveformDecomposer:
    def __init__(self, time_data: np.ndarray, amplitude_data: np.ndarray,
                 filename: str, channel_name: str, output_dir: str, n_sigma: float):
        self.x_data = time_data
        self.y_data = amplitude_data
        self.filename = filename
        self.channel_name = channel_name
        self.output_dir = output_dir
        self.n_sigma = n_sigma
        
        self.final_params = None
        self.final_fit = None
        self.r_squared = 0.0
        self.is_inverted = False
        self.y_processed = None
        self.original_baseline = None
        
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

    def run_decomposition(self) -> tuple:
        preprocessor = Preprocessor(n_sigma=self.n_sigma)
        self.y_processed, self.noise_threshold, self.is_inverted, self.original_baseline = preprocessor.run(self.y_data)

        peak_detector = PeakDetector(noise_threshold=self.noise_threshold)
        initial_params = peak_detector.run(self.y_processed)
        
        if not initial_params:
            return None, None, 0

        aga = AdaptiveGeneticOptimizer(self.x_data, self.y_processed, initial_params, generations=100)
        aga_params_flat = aga.optimize()

        if aga_params_flat is None:
            return None, None, 0
            
        aga_params_as_dicts = []
        for i in range(0, (len(aga_params_flat) - 1), 4):
            aga_params_as_dicts.append({
                'A': aga_params_flat[i], 't': aga_params_flat[i+1],
                'sigma': aga_params_flat[i+2], 'p': aga_params_flat[i+3]
            })

        lm_fitter = LMFitter(self.x_data, self.y_processed, aga_params_as_dicts)
        final_params_flat, _ = lm_fitter.optimize()
        
        if final_params_flat is None:
            self.final_params = aga_params_flat
        else:
            self.final_params = final_params_flat
        
        processed_fit_curve = N_Gaussian.func(self.x_data, *self.final_params)

        oriented_fit_curve = processed_fit_curve * -1.0 if self.is_inverted else processed_fit_curve
        self.final_fit = oriented_fit_curve + self.original_baseline

        self.r_squared = self._calculate_r_squared(self.y_data, self.final_fit)
        
        print(f"  -- Decomposition successful. Final R-squared: {self.r_squared:.4f}")
        
        return self.final_params, self.final_fit, self.r_squared

    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    def get_plotting_data(self) -> dict:
        """Prepares a dictionary of data for visualization."""
        final_components = []
        if self.final_fit is not None and self.final_params is not None and len(self.final_params) > 1:
            num_components = (len(self.final_params) - 1) // 4
            processed_baseline_val = self.final_params[-1]

            for i in range(num_components):
                amp, pos, wid, p = self.final_params[i*4:(i+1)*4]
                component = N_Gaussian.generalized_gaussian(self.x_data, amp, pos, wid, p)
                
                component_plot = component + processed_baseline_val
                
                if self.is_inverted:
                    component_plot *= -1.0
                
    
                # Add the original baseline to position the component correctly for the final plot
                final_components.append(component_plot + self.original_baseline)
       

        return {
            "x_data": self.x_data, "y_original": self.y_data, "final_fit": self.final_fit,
            "final_components": final_components, "r_squared": self.r_squared,
            "noise_threshold": self.noise_threshold if not self.is_inverted else -self.noise_threshold,
            "channel_name": self.channel_name
        }
        
    def get_final_params_as_list(self) -> list:

        return self.final_params.tolist() if self.final_params is not None else []

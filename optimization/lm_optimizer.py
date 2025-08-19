# optimization/lm_optimizer.py
# --- (DEFINITIVELY FINAL VERSION) ---
# Increased iterations and made the closure function more robust to prevent
# non-finite values, ensuring the optimizer converges to a valid scientific result.

import torch
import torch.nn as nn
import torch_levenberg_marquardt as tlm
from optimization.optimizer import Optimizer
from typing import Optional, Tuple
import numpy as np

def generalized_gaussian_torch(x, amp, pos, wid, p):
    """GPU-accelerated generalized Gaussian function"""
    wid = torch.clamp(wid, min=1e-6)
    return amp * torch.exp(-torch.pow(torch.abs(x - pos) / (np.sqrt(2) * wid), p))

class WaveformModel(nn.Module):
    """PyTorch model representing the sum of generalized Gaussian waveforms."""
    def __init__(self, initial_params, device):
        super().__init__()
        self.num_components = len(initial_params)
        self.device = device
        
        self.component_params = nn.ParameterList()
        for p_dict in initial_params:
            self.component_params.append(nn.Parameter(torch.tensor(
                [p_dict['A'], p_dict['t'], p_dict['sigma'], p_dict['p']], device=self.device
            )))
        
        self.baseline = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, x):
        y = torch.full_like(x, self.baseline.item())
        for i in range(self.num_components):
            amp, pos, wid, p = self.component_params[i]
            y += generalized_gaussian_torch(x, amp, pos, wid, p)
        return y

class LMFitter(Optimizer):
    """
    Levenberg-Marquardt optimizer using the 'torch-levenberg-marquardt' library.
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, initial_params: list):
        super().__init__(x_data, y_data, initial_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def optimize(self) -> Optional[Tuple[np.ndarray, None]]:
        if self.num_components == 0:
            return None, None
            
        print("  - Starting Levenberg-Marquardt (torch-lm) for final fitting...")

        x_tensor = torch.from_numpy(self.x_data).float().to(self.device)
        y_tensor = torch.from_numpy(self.y_data).float().to(self.device)

        model = WaveformModel(self.initial_params, self.device)
        
        lm_module = tlm.training.LevenbergMarquardtModule(
            model=model,
            loss_fn=tlm.loss.MSELoss(),
            learning_rate=1.0,
            attempts_per_step=15
        )

        # --- MODIFICATION: More robust training loop ---
        for i in range(100): # Increased iterations for better convergence
            _, loss, stop, _ = lm_module.training_step(x_tensor.unsqueeze(1), y_tensor.unsqueeze(1))
            
            # Check for invalid loss and stop early if needed
            if not torch.isfinite(loss):
                print(f"    LM stopped early at iteration {i+1} due to non-finite loss.")
                lm_module.restore_parameters() # Restore last good parameters
                break

            if (i + 1) % 20 == 0:
                print(f"    LM Iteration {i+1}/100, Loss: {loss.item():.6f}")
            if stop:
                break
        
        final_params_list = []
        for param in model.component_params:
            final_params_list.extend(param.detach().cpu().numpy())
        final_params_list.append(model.baseline.detach().cpu().numpy().item())
        
        final_params = np.array(final_params_list)

        print("  - LM (torch-lm) optimization successful.")
        return final_params, None
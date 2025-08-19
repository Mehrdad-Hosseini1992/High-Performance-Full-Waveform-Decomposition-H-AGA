# # aga_optimizer.py
# # GPU-OPTIMIZED VERSION

# import numpy as np
# import torch
# import torch.cuda as cuda
# from gadapt.ga import GA as GAdapt 
# from optimization.optimizer import Optimizer
# from Utils.utils import N_Gaussian
# from typing import Optional
# import warnings

# class AdaptiveGeneticOptimizer(Optimizer):
#     def __init__(self, x_data: np.ndarray, y_data: np.ndarray, initial_params: list, generations: int = 50):
#         super().__init__(x_data, y_data, initial_params)
#         self.generations = generations
        
#         # Check GPU availability
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if torch.cuda.is_available():
#             print(f"    Using GPU: {torch.cuda.get_device_name(0)}")
#             print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
#         else:
#             print("    GPU not available, using CPU")
        
#         # Pre-convert data to torch tensors on GPU
#         self.x_tensor = torch.from_numpy(x_data).float().to(self.device)
#         self.y_tensor = torch.from_numpy(y_data).float().to(self.device)

#     def _generalized_gaussian_torch(self, x, amp, pos, wid, p):
#         """GPU-accelerated generalized Gaussian function"""
#         if wid <= 1e-6:
#             return torch.zeros_like(x)
#         return amp * torch.exp(-torch.pow(torch.abs(x - pos) / (np.sqrt(2) * wid), p))

#     def _fitness_func(self, params):
#         try:
#             # Convert params to torch tensors on GPU
#             num_components = (len(params) - 1) // 4
            
#             # Initialize result tensor on GPU
#             y_predicted = torch.full_like(self.x_tensor, params[len(params) - 1])  # baseline
            
#             # Calculate all Gaussians on GPU
#             for i in range(num_components):
#                 base_idx = i * 4
#                 amp = params[base_idx]
#                 pos = params[base_idx + 1]
#                 wid = params[base_idx + 2]
#                 p = params[base_idx + 3]
                
#                 if wid > 1e-6:
#                     y_predicted += self._generalized_gaussian_torch(
#                         self.x_tensor, amp, pos, wid, p
#                     )
            
#             # Calculate SSE on GPU
#             sse = torch.sum((self.y_tensor - y_predicted) ** 2)
            
#             # Return as Python float
#             sse_value = sse.cpu().item() if torch.cuda.is_available() else sse.item()
            
#             if np.isnan(sse_value) or np.isinf(sse_value):
#                 return 1e10
                
#             return float(sse_value)
            
#         except Exception as e:
#             print(f"    Fitness function error: {str(e)}")
#             return 1e10

#     def optimize(self) -> Optional[np.ndarray]:
#         if self.num_components == 0:
#             return None
#         print("  - Starting Adaptive Genetic Algorithm (gadapt) for preliminary optimization...")

#         ga = GAdapt(cost_function=self._fitness_func, population_size=32)

#         # Add parameters with their respective bounds
#         time_range = self.x_data[-1] - self.x_data[0]
        
#         for i, p in enumerate(self.initial_params):
#             # Validate and clean parameters
#             amp = float(p['A']) if not np.isnan(p['A']) else 0.1
#             pos = float(p['t']) if not np.isnan(p['t']) else len(self.x_data) // 2
#             sigma = float(p['sigma']) if not np.isnan(p['sigma']) else 10.0
            
#             # Debug print
#             print(f"    Component {i}: A={amp:.4f}, t={pos:.1f}, sigma={sigma:.2f}")
            
#             # Amplitude - ensure positive values
#             amp_min = max(0.001, amp * 0.5)
#             amp_max = max(amp * 1.5, amp_min + 0.1)
#             amp_step = max((amp_max - amp_min) / 100, 0.001)
#             ga.add(amp_min, amp_max, amp_step)
            
#             # Position (time index) - ensure valid bounds
#             pos_min = max(0.0, pos - 50.0)
#             pos_max = min(float(len(self.x_data) - 1), pos + 50.0)
#             pos_step = max((pos_max - pos_min) / 100, 0.1)
#             ga.add(pos_min, pos_max, pos_step)
            
#             # Width (sigma) - ensure positive values
#             sigma_min = max(1.0, sigma * 0.5)
#             sigma_max = max(sigma * 1.5, sigma_min + 1.0)
#             sigma_step = max((sigma_max - sigma_min) / 100, 0.1)
#             ga.add(sigma_min, sigma_max, sigma_step)
            
#             # Shape parameter p
#             ga.add(1.0, 4.0, 0.1)

#         # Add the baseline parameter
#         baseline_samples = np.concatenate([self.y_data[:min(30, len(self.y_data))], 
#                                         self.y_data[-min(30, len(self.y_data)):]])
#         baseline_guess = float(np.mean(baseline_samples))
#         y_min = float(np.min(self.y_data))
#         y_max = float(np.max(self.y_data))
#         baseline_step = max((y_max - y_min) / 100, 0.001)
#         ga.add(y_min, y_max, baseline_step)

#         # Configure and run the optimization
#         ga.exit_check = 'avg_cost'
#         ga.max_attempt_no = 5
#         ga.number_of_generations = self.generations
        
#         try:
#             result = ga.execute()
            
#             # Check if optimization was successful
#             if not result.success:
#                 error_msg = result.message if hasattr(result, 'message') else 'Unknown error'
#                 print(f"  - AGA optimization failed: {error_msg}")
#                 if hasattr(result, 'messages'):
#                     for msg in result.messages:
#                         print(f"    {msg[0]}: {msg[1]}")
#                 return None
                
#             print(f"  - AGA (gadapt) finished. Final best cost: {result.min_cost:.4f}")
            
#             # Extract the optimized parameters
#             optimized_params = []
#             for key in sorted(result.result_values.keys()):
#                 optimized_params.append(float(result.result_values[key]))
                
#             return np.array(optimized_params)
            
#         except Exception as e:
#             print(f"  - AGA optimization error: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return None
#         finally:
#             # Clear GPU cache if used
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()


# optimization/aga_optimizer.py
# --- (FINAL CORRECTED GPU-NATIVE VERSION) ---
# Fixed the PyTorch in-place operation error by cloning the tensor before mutation.
# This version is fully GPU-native and resolves the previous performance and stability issues.

import numpy as np
import torch
from optimization.optimizer import Optimizer
from typing import Optional

class AdaptiveGeneticOptimizer(Optimizer):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, initial_params: list,
                 generations: int = 100, population_size: int = 64):
        super().__init__(x_data, y_data, initial_params)
        self.generations = generations
        self.population_size = population_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    AGA using device: {self.device}")

        self.x_tensor = torch.from_numpy(x_data).float().to(self.device)
        self.y_tensor = torch.from_numpy(y_data).float().to(self.device)
        self.param_bounds_lower = []
        self.param_bounds_upper = []
        self.noise_threshold = np.std(y_data[:min(len(y_data), 50)])

    def _initialize_population(self):
        num_params = len(self.param_bounds_lower)
        lower_bounds = torch.tensor(self.param_bounds_lower, device=self.device, dtype=torch.float32)
        upper_bounds = torch.tensor(self.param_bounds_upper, device=self.device, dtype=torch.float32)
        population = lower_bounds + (upper_bounds - lower_bounds) * torch.rand(
            self.population_size, num_params, device=self.device
        )
        return population

    def _fitness_func_batch(self, population: torch.Tensor):
        num_params = population.shape[1]
        num_components = (num_params - 1) // 4
        y_true_batch = self.y_tensor.unsqueeze(0).expand(self.population_size, -1)
        baselines = population[:, -1].unsqueeze(1)
        y_predicted_batch = baselines.expand(-1, self.x_tensor.shape[0]).clone()

        for i in range(num_components):
            amps, poss, wids, ps = population[:, i*4], population[:, i*4+1], population[:, i*4+2], population[:, i*4+3]
            x = self.x_tensor.unsqueeze(0)
            amp, pos, wid, p_shape = amps.unsqueeze(1), poss.unsqueeze(1), wids.unsqueeze(1).clamp(min=1e-6), ps.unsqueeze(1)
            component = amp * torch.exp(-torch.pow(torch.abs(x - pos) / (np.sqrt(2) * wid), p_shape))
            y_predicted_batch += component
            
        sse = torch.sum((y_true_batch - y_predicted_batch) ** 2, dim=1)
        return 1.0 / (sse + 1e-9)

    def optimize(self) -> Optional[np.ndarray]:
        if self.num_components == 0: return None
        print("  - Starting FULLY GPU-NATIVE Adaptive Genetic Algorithm...")

        for p in self.initial_params:
            amp, pos, sigma = float(p['A']), float(p['t']), float(p['sigma'])
            self.param_bounds_lower.extend([max(0.01, amp * 0.5), max(0, pos - 50), max(1.0, sigma * 0.5), 1.0])
            self.param_bounds_upper.extend([amp * 1.5, min(len(self.x_data)-1, pos + 50), sigma * 1.5, 4.0])
        
        baseline_guess = np.mean(self.y_data[:50])
        self.param_bounds_lower.append(baseline_guess - 2*self.noise_threshold)
        self.param_bounds_upper.append(baseline_guess + 2*self.noise_threshold)

        population = self._initialize_population()
        best_fitness_overall = -1
        best_solution_overall = None

        for gen in range(self.generations):
            fitness = self._fitness_func_batch(population)
            best_fitness_gen, best_idx_gen = torch.max(fitness, 0)
            if best_fitness_gen > best_fitness_overall:
                best_fitness_overall = best_fitness_gen
                best_solution_overall = population[best_idx_gen]

            if (gen + 1) % 20 == 0:
                cost = 1.0/best_fitness_overall.cpu().item() if best_fitness_overall > 0 else float('inf')
                print(f"    Generation {gen+1}/{self.generations}, Best Cost (SSE): {cost:.4f}")

            parent_indices = torch.randint(0, self.population_size, (self.population_size, 2), device=self.device)
            p1_fit, p2_fit = fitness[parent_indices[:, 0]], fitness[parent_indices[:, 1]]
            winner_indices = torch.where(p1_fit > p2_fit, parent_indices[:, 0], parent_indices[:, 1])
            parents = population[winner_indices]

            eta = 20.0
            rand = torch.rand_like(parents)
            beta = torch.where(rand <= 0.5, (2.0 * rand)**(1.0 / (eta + 1.0)), (1.0 / (2.0 * (1.0 - rand)))**(1.0 / (eta + 1.0)))
            
            p1, p2 = parents[::2], parents[1::2]
            if p1.shape[0] != p2.shape[0]: p1 = p1[:-1]
                
            offspring1 = 0.5 * ((1.0 + beta[:p1.shape[0]]) * p1 + (1.0 - beta[:p1.shape[0]]) * p2)
            offspring2 = 0.5 * ((1.0 - beta[:p1.shape[0]]) * p1 + (1.0 + beta[:p1.shape[0]]) * p2)
            
            # --- THIS IS THE FIX ---
            offspring1_mutated = offspring1.clone()
            # -----------------------
            
            mutation_prob, eta_mut = 0.1, 20.0
            do_mutate = torch.rand_like(offspring1) < mutation_prob
            mu = torch.rand_like(offspring1)
            delta = torch.where(mu <= 0.5, (2.0 * mu)**(1.0 / (eta_mut + 1.0)) - 1.0, 1.0 - (2.0 * (1.0 - mu))**(1.0 / (eta_mut + 1.0)))
            
            bounds_range = torch.tensor(self.param_bounds_upper, device=self.device) - torch.tensor(self.param_bounds_lower, device=self.device)
            offspring1_mutated += do_mutate * delta * bounds_range
            
            new_population = torch.cat((offspring1_mutated, offspring2), dim=0)
            population = new_population[:self.population_size - 1]
            population = torch.cat((population, best_solution_overall.unsqueeze(0)), dim=0)

        final_cost = 1.0/best_fitness_overall.cpu().item() if best_fitness_overall > 0 else float('inf')
        print(f"  - GPU-Native AGA finished. Final Best Cost (SSE): {final_cost:.4f}")
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        return best_solution_overall.cpu().numpy()
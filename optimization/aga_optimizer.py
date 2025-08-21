
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

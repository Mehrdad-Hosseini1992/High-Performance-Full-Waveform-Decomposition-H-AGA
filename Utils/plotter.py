
# Added logic to mark the detected peak of each Gaussian component with a red dot.

import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter:
    @staticmethod
    def create_and_save_debug_plot(original_filepath: str, channel_name: str, x_data: np.ndarray, y_smoothed: np.ndarray, adjusted_threshold: float, output_dir: str):
        # This function remains the same
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_data, y_smoothed, color='blue', linewidth=2, label='Processed Data for Peak Finding')
        ax.axhline(y=adjusted_threshold, color='r', linestyle='--', label=f'Peak Finding Threshold ({adjusted_threshold:.2f})')
        ax.set_title(f"DEBUG: Peak Detection Failed - {os.path.basename(original_filepath)} - {channel_name}")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Processed Amplitude (Baseline Corrected)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        base_filename = os.path.splitext(os.path.basename(original_filepath))[0]
        output_filename = os.path.join(output_dir, f"DEBUG_{base_filename}_{channel_name}.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"--- Debug plot saved to {output_filename} ---")

    @staticmethod
    def create_and_save_combined_plot(original_filepath: str, all_channels_data: list, output_dir: str):
        num_channels = len(all_channels_data)
        if num_channels == 0: return
        
        fig, axes = plt.subplots(num_channels, 1, figsize=(14, 7), squeeze=False)
        base_filename = os.path.splitext(os.path.basename(original_filepath))[0]
        fig.suptitle(f"Waveform Decomposition for {base_filename}", fontsize=16)
        
        for i, data in enumerate(all_channels_data):
            ax = axes[i, 0]
            ax.plot(data['x_data'], data['y_original'], color='grey', alpha=0.7, linewidth=1.5, label='Original Waveform')
            
            if data['final_fit'] is not None:
                ax.plot(data['x_data'], data['final_fit'], 'r--', linewidth=2, label=f"Final Fit (R²={data['r_squared']:.3f})")
            
            if 'final_components' in data and data['final_components']:
                baseline_level = np.mean(np.concatenate([
                    data['final_fit'][:30], data['final_fit'][-30:]
                ])) if data['final_fit'] is not None else 0
                
                for j, component in enumerate(data['final_components']):
                    ax.fill_between(data['x_data'], baseline_level, component, 
                                    alpha=0.4, label=f"Component {j+1}")
                
                    # Find the index of the minimum value (the peak) for this component
                    peak_index = np.argmin(component)
                    peak_time = data['x_data'][peak_index]
                    peak_amplitude = component[peak_index]
                    # Add a red dot marker. The 'zorder' ensures it's drawn on top.
                    ax.plot(peak_time, peak_amplitude, 'ro', markersize=5, zorder=10)
            
            ax.axhline(y=data['noise_threshold'], color='g', linestyle=':', label='Noise Threshold')
            ax.set_title(f"Channel: {data['channel_name']}")
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        output_filename = os.path.join(output_dir, f"{base_filename}_decomposition.png")
        plt.savefig(output_filename)
        plt.close(fig)

        print(f"--- Plot saved to {os.path.join(output_dir, os.path.basename(output_filename))} ---")

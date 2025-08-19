# main.py
# --- (OPTIMIZED) ---
# Added a command-line argument '--n_sigma' to allow for easy tuning
# of the noise threshold sensitivity.
# PARALLEL GPU-OPTIMIZED VERSION

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import glob
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from Core.decomposer import WaveformDecomposer
from Utils.data_loader import WaveformLoader
from Utils.plotter import Plotter

def check_gpu_status():
    """Check and print GPU status"""
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("GPU not available. Using CPU.")
        return False

def process_channel(channel_data):
    """Process a single channel (can be run in parallel)"""
    channel_name, waveform_data, filepath, output_dir, n_sigma = channel_data
    
    print(f"\n  -- Analyzing Channel: {channel_name} --")
    try:
        decomposer = WaveformDecomposer(
            time_data=waveform_data.get_time_data(),
            amplitude_data=waveform_data.get_channel_data(channel_name),
            filename=filepath,
            channel_name=channel_name,
            output_dir=output_dir,
            n_sigma=n_sigma
        )
        
        final_params, final_fit, r_squared = decomposer.run_decomposition()

        if final_params is not None and len(final_params) > 0:
            print(f"  -- Decomposition successful for {channel_name}. R-squared: {r_squared:.4f}")
            return decomposer.get_plotting_data()
        else:
            print(f"  -- [{os.path.basename(filepath)} - {channel_name}] No valid parameters were found.")
            return None

    except Exception as e:
        print(f"!!!!!! FAILED to process channel {channel_name}: {e} !!!!!!")
        return None

def process_single_file(filepath, output_dir, n_sigma, use_parallel=True):
    """
    Processes a single CSV file, decomposing each channel.
    Can use parallel processing for multiple channels.
    """
    print(f"\n--- Processing {os.path.basename(filepath)} ---")
    loader = WaveformLoader()
    waveform_data = loader.load_csv(filepath)

    if waveform_data is None or waveform_data.is_empty():
        print(f"!!!!!! SKIPPING {os.path.basename(filepath)} due to loading error or empty data. !!!!!!")
        return

    channels_to_process = waveform_data.get_data_channels()

    
    if not channels_to_process:
        # If no channels found, check if there's at least one channel
        all_channels = waveform_data.get_channel_names()
        if len(all_channels) >= 2 and all_channels[0] == 'Time':
            channels_to_process = all_channels[1:2]  # Process first channel after Time
        else:
            print(f"!!!!!! No data channels found in {os.path.basename(filepath)}. Skipping. !!!!!!")
            return

    all_channels_plot_data = []

    if use_parallel and len(channels_to_process) > 1:
        # Prepare data for parallel processing
        channel_data_list = [
            (channel_name, waveform_data, filepath, output_dir, n_sigma)
            for channel_name in channels_to_process
        ]
        
        # Use ThreadPoolExecutor for I/O-bound operations with GPU
        # (GPU operations are better with threads due to CUDA context)
        with ThreadPoolExecutor(max_workers=min(len(channels_to_process), 3)) as executor:
            results = list(executor.map(process_channel, channel_data_list))
        
        all_channels_plot_data = [r for r in results if r is not None]
    else:
        # Sequential processing
        for channel_name in channels_to_process:
            result = process_channel((channel_name, waveform_data, filepath, output_dir, n_sigma))
            if result is not None:
                all_channels_plot_data.append(result)

    if all_channels_plot_data:
        Plotter.create_and_save_combined_plot(filepath, all_channels_plot_data, output_dir)

def process_directory(input_dir, output_dir, n_sigma, use_parallel=True):
    """
    Processes all CSV files in a given directory.
    Can use parallel processing for multiple files.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    csv_files = glob.glob(os.path.join(input_dir, '*.csv')) + \
            glob.glob(os.path.join(input_dir, '*.txt')) + \
            glob.glob(os.path.join(input_dir, '*.dat'))
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(csv_files)} files. Results will be saved in '{output_dir}'")

    if use_parallel and len(csv_files) > 1:
        # Process multiple files in parallel
        process_func = partial(process_single_file, output_dir=output_dir, 
                              n_sigma=n_sigma, use_parallel=True)
        
        # Use ProcessPoolExecutor for CPU-bound operations
        with ProcessPoolExecutor(max_workers=min(len(csv_files), mp.cpu_count())) as executor:
            executor.map(process_func, csv_files)
    else:
        for filepath in csv_files:
            process_single_file(filepath, output_dir, n_sigma, use_parallel=True)

if __name__ == '__main__':
    # Check GPU status
    gpu_available = check_gpu_status()
    
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Full-Waveform Laser Decomposition based on the HAGA method (GPU-Optimized).")
    parser.add_argument('-i', '--input', 
                       default='C:/Users/mehrd/OneDrive/Desktop/Mitacs/Models/Dataset/TurbidityImpact/C2--Water_2M_SandReflection_21mLturbid_3V3Amp39V8SiPM_18ns15.csv',
                       help="Input directory containing CSV files or path to a single CSV file.")
    parser.add_argument('-o', '--output', 
                       default='C:/Users/mehrd/OneDrive/Desktop/Mitacs/Models/Dataset/TurbidityImpact/output/',
                       help="Output directory to save result plots.")
    parser.add_argument('--n_sigma', type=float, default=2.0,
                       help="Sigma multiplier for noise threshold calculation. Default: 2.0")
    parser.add_argument('--parallel', action='store_true', default=True,
                       help="Enable parallel processing for multiple channels/files")
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                       help="Disable parallel processing")
    parser.add_argument('--gpu', action='store_true', default=gpu_available,
                       help="Force GPU usage (if available)")
    parser.add_argument('--cpu', dest='gpu', action='store_false',
                       help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    # Set GPU usage based on arguments
    if not args.gpu or not gpu_available:
        # Disable GPU in PyTorch
        torch.cuda.is_available = lambda: False
        print("Processing will use CPU only.")
    else:
        print("Processing will use GPU acceleration.")
    
    print(f"Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    
    # Process files
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.n_sigma, use_parallel=args.parallel)
    elif os.path.isfile(args.input):
        os.makedirs(args.output, exist_ok=True)
        process_single_file(args.input, args.output, args.n_sigma, use_parallel=args.parallel)
    else:
        print(f"Error: Input '{args.input}' is not a valid directory or file.")

# To run this from your command line:
# python main.py --input /path/to/your/csv_files --output /path/to/save/results
#
# To use GPU acceleration (if available):
# python main.py --input ./input_data --output ./results --gpu
#
# To disable parallel processing:
# python main.py --input ./input_data --output ./results --no-parallel
#
# To force CPU usage:
# python main.py --input ./input_data --output ./results --cpu

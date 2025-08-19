# **High-Performance Waveform Decomposition (H-AGA)**

This repository contains a high-performance, GPU-accelerated Python tool for decomposing complex, full-waveform signals into their constituent generalized Gaussian components. The model is a direct implementation and significant performance enhancement of the methodology described in the research paper: **"A Satellite Full-Waveform Laser Decomposition Method for Forested Areas Based on Hidden Peak Detection and Adaptive Genetic Optimization" (H-AGA)**.

The primary scientific goal is to accurately detect and parameterize hidden or overlapping peaks within noisy, real-world oscilloscope data, achieving a goodness-of-fit (R2) that meets or exceeds the benchmarks set by the original research.

*Example of a successful waveform decomposition with an* R2 *of 0.968.*

## **Table of Contents**

1. [Key Features & Innovations](https://www.google.com/search?q=%231-key-features--innovations)  
2. [The Scientific Methodology](https://www.google.com/search?q=%232-the-scientific-methodology)  
3. [Installation](https://www.google.com/search?q=%233-installation)  
4. [Usage](https://www.google.com/search?q=%234-usage)  
5. [Project Architecture (OOP)](https://www.google.com/search?q=%235-project-architecture-oop)  
6. [A Note on the OMP Warning](https://www.google.com/search?q=%236-a-note-on-the-omp-warning)  
7. [Acknowledgments](https://www.google.com/search?q=%237-acknowledgments)

## **1\. Key Features & Innovations**

This project successfully implements the H-AGA methodology while introducing a critical innovation for practical, high-throughput analysis:

* **Scientifically Rigorous**: Adheres to the multi-stage optimization process (Preprocessing \-\> Peak Detection \-\> AGA \-\> LM) outlined in the H-AGA paper, resulting in a high goodness-of-fit (R20.96).  
* **Fully GPU-Accelerated**: The entire optimization pipeline has been engineered to run on NVIDIA GPUs, reducing processing time for a single waveform from over 30 minutes to a matter of seconds.  
* **Custom GPU-Native Genetic Algorithm**: A key innovation was the development of a custom, fully GPU-native Adaptive Genetic Algorithm (AGA) using PyTorch. This was created to solve a critical performance bottleneck identified with traditional CPU-bound GA libraries, eliminating the data transfer overhead between the CPU and GPU.  
* **High-Precision Final Fit**: The final fitting stage uses a GPU-accelerated implementation of the Levenberg-Marquardt (LM) algorithm, as recommended by the paper, for fast and precise convergence.  
* **Robust Preprocessing**: Employs a state-of-the-art signal processing pipeline, including wavelet denoising and Asymmetric Least Squares (ALS) baseline correction, to handle real-world noisy data.

## **2\. The Scientific Methodology**

The model's success lies in its adherence to a scientifically validated, multi-stage process designed to avoid the pitfalls of simpler fitting methods.

#### **Step 1: Signal Preprocessing**

Raw oscilloscope data is contaminated with noise and a fluctuating baseline. The Preprocessor class uses advanced techniques to isolate the true signal:

* **Wavelet Denoising**: Decomposes the signal into frequency components to surgically remove noise.  
* **ALS Baseline Correction**: Calculates and removes the underlying baseline drift without distorting the signal's peaks.

#### **Step 2: Initial Peak Detection**

Once the signal is cleaned, the PeakDetector provides a high-quality initial guess for the optimizers. It uses scipy.signal.find\_peaks with carefully tuned height, prominence, and distance parameters to robustly identify only the true, significant peaks.

#### **Step 3: Adaptive Genetic Algorithm (Global Search)**

The AdaptiveGeneticOptimizer performs the first optimization pass. A genetic algorithm is a "global" search method, meaning it's very good at exploring the entire possible range of solutions to find a "good enough" set of parameters. This is a crucial step to get the fit into the correct region and avoid the local minima that trap simpler algorithms.

#### **Step 4: Levenberg-Marquardt (Local, High-Precision Fit)**

The LMFitter performs the final optimization. The LM algorithm is a "local" optimizer. It is extremely fast and precise at finding the *best possible* solution, but it requires a good starting guess. By feeding it the result from the AGA, we combine the global search power of the genetic algorithm with the high-precision finishing power of Levenberg-Marquardt, as recommended by the H-AGA paper.

## **3\. Installation**

### **Prerequisites**

* An NVIDIA GPU with installed CUDA drivers.  
* Anaconda or Miniconda for environment management.

### **Environment Setup**

1. **Clone the repository**:  
   git clone \<repository-url\>  
   cd HAGA-2-Claud

2. **Create and activate the Conda environment**:  
   conda create \--name haga python=3.10  
   conda activate haga

3. **Install dependencies**:  
   pip install \-r requirements.txt

   This will install PyTorch with CUDA support, along with all other necessary scientific and plotting libraries.

## **4\. Usage**

### **Directory Structure**

* Place your raw waveform .csv files (in LeCroy format) inside the Dataset/ folder.  
* The script will automatically create an output/ folder for the resulting plots.

### **Running the Analysis**

1. Make sure your haga conda environment is active.  
2. From the root directory (HAGA-2-Claud/), run the main script:  
   python main.py

The script will automatically find and process all .csv files in the Dataset/ folder and save the output plots as .png files in the output/ directory.

## **5\. Project Architecture (OOP)**

The project is built using an Object-Oriented design to ensure the code is modular, maintainable, and scientifically sound.

HAGA-2-Claud/  
│  
├── Core/  
│   ├── decomposer.py       \# Main orchestrator class  
│   └── peak\_detector.py    \# Initial peak identification  
│  
├── optimization/  
│   ├── optimizer.py        \# Base class for all optimizers  
│   ├── aga\_optimizer.py    \# GPU-Native Adaptive Genetic Algorithm  
│   └── lm\_optimizer.py     \# GPU-accelerated Levenberg-Marquardt fitter  
│  
├── Utils/  
│   ├── data\_loader.py      \# Parses LeCroy CSV files  
│   ├── preprocessor.py     \# Cleans and prepares waveform data  
│   ├── plotter.py          \# Generates final output plots  
│   └── utils.py            \# Contains the N\_Gaussian model and helpers  
│  
└── main.py                 \# Entry point of the application

* **Core/**: Contains the high-level logic. decomposer.py is the main class that orchestrates the entire H-AGA pipeline.  
* **optimization/**: Houses the powerful, GPU-accelerated optimization algorithms. This is where the custom AGA and the LM fitter reside.  
* **Utils/**: Provides specialized helper classes for data loading, preprocessing, and plotting.  
* **main.py**: The entry point that handles file discovery and parallel processing.

## **6\. A Note on the OMP Warning**

When running the script, you may see the following warning:  
OMP: Error \#15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.  
This is a common, harmless warning in scientific Python environments where multiple libraries (e.g., NumPy and PyTorch) are linked against Intel's OpenMP library. The main.py script already includes a standard workaround to suppress this warning and allow the program to run correctly. **It does not affect the scientific validity of the results.**

## **7\. Acknowledgments**

This project builds upon the scientific foundation laid by the **H-AGA research paper** and leverages the power of several open-source libraries. The custom GPU-accelerated optimizers were inspired by the following projects:

* **Adaptive Genetic Algorithm**: The initial CPU-based implementation used gadapt, which highlighted the need for a GPU-native solution.( https://github.com/bpzoran/gadapt )
* **PyTorch Levenberg-Marquardt**: The final fitting stage is powered by the excellent torch-levenberg-marquardt library. ( https://github.com/fabiodimarco/torch-levenberg-marquardt )

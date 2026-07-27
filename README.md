## Megahertz Fourier Light Field Microscopy  

This repository contains all the code associated with our new device, Ultrafast Light field Tracking and Recording Apparatus for Surface Deformation Velocimetry (ULTRA-SDV), which is described in a manuscript currently under review. 

## System requirements

This package does not require specialized hardware. It was tested under the following conditions: 

| Component | Version |
|---|---|
| Operating System | Windows 11 (version 25H2) |
| Development Environment | Visual Studio Code 1.130.0 |
| Package Manager | Miniforge/Conda 26.5.3 |
| Platform | Windows 64-bit (`win-64`) |
| Python | 3.11.15 |

## Environment and Installation 

We recommend running this code using a conda environment. The repository and environment can be set up using the following steps. 

1. Clone repository

```
git clone git@github.com:clarebcook/FiLMScope.git
cd HighSpeedFLFM
```

2. Set up the environment
```
conda install conda-build
conda env create -f environment.yml
conda activate hsflfm_cpu
conda develop .
```

Note that ``conda-build`` must be installed in the base environment to use the ``develop`` command. The full installation should be fast, within a couple minutes depending on connection strength. 

## Downloading Data 
Our full dataset of trap-jaw ant snapping videos will be available prior to the publication of our manuscript. A partial dataset is temporarily available on Google Drive: 
https://drive.google.com/drive/folders/1ugZwc9GC4XBzgeqevQwpPMVN0A26MxRv?usp=drive_link

The partial dataset contains the processed 3D deformation results for all trap-jaw ant snapping videos mentioned in the manuscript, micro-CT scans needed for certain display and analysis scripts, and select raw videos needed to run the example processing scripts.  

The data should be downloaded and placed in a folder within the user's directory. **`home_directory` in `hsflfm/config.py` must then be set to this folder to successfully run the code.**

We will add additional information here on how to organize future datasets for use with this repository. 

## Navigating repository
This folder contains three subfolders of user-friendly scripts: 'calibration_scripts', 'processing_scripts', and 'analysis_scripts'. Each folder contains an individual README describing how to use the scripts. The underlying code package is inside the 'hsflfm' subfolder. When using our released dataset, the scripts in each subfolder are independent. For new datasets, calibration must be performed first, followed by processing and then analysis. 

'calibration_scripts' contains the necessary code for establishing the stereo relationship between the perspective views and only needs to be performed once for a given hardware setup (in practice, calibration was performed each day the device was used, to account for potential movement). 'processing_scripts' outputs the 3D motion vectors for tracked points, and 'analysis_scripts' contains example visualizations and analysis of the 3D results, including re-creating figures from our ULTRA-SDV manuscript. 

> **Suggested demo:** We recommend running 'processing_scripts/process_from_pre_alignment.ipynb' to assess processing speed, and the .ipynb files in 'analysis_scripts' to view results. These scripts can be run in any order. 



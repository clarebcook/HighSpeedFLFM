## Ultra-High Speed Fourier Light Field Mesoscope 

This will contain instructions for using the code repository associated withour new ultra-high speed fourier light field mesoscope for imaging snapping trap-jaw ants, currently in submission. 
 
## Environment and Installation 

We recommend running this code using a conda environment. The repository and environment can be set up using the following steps. 

1. Clone repository

```
git clone git@github.com:clarebcook/FiLMScope.git
cd HighSpeedFLFM
```

2. Set up the environment
```
conda env create -f environment.yml
conda activate hsflfm_cpu
conda develop .
```

Note that using `conda develop` requires `conda-build` to be installed. This can be done with: 
```
conda install conda-build
```

## Downloading Data 
A sample of our full dataset of trap-jaw ant snapping videos will be available prior to the publication of our manuscript. A partial dataset is temporarily available on Google Drive: 
https://drive.google.com/drive/folders/1ugZwc9GC4XBzgeqevQwpPMVN0A26MxRv?usp=drive_link


The data should be downloaded and placed in a folder within the user's directory. `home_directory` in `hsflfm/config.py` should then be set to this folder. 

We will add additional information here on organizing new datasets for use with this repository. 

## Navigating repository
This folder contains three main folders of scripts: 'calibration_scripts', 'processing_scripts', and 'analysis_scripts'. Each folder contains an individual README describing how to use the scripts. 
## Ultra-High Speed Fourier Light Field Mesoscope 

This will contain instructions for using the code repository associated withour new ultra-high speed fourier light field mesoscope for imaging snapping trap-jaw ants. 

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

Note that using `conda develop` requires install `conda-build`. This can be done with: 
```
conda install conda-build
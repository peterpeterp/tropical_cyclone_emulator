
# Readme

## Installation

These python scripts require the following python packages:

joblib==0.14.1
xarray==0.15.0
multiprocess==0.70.12.2
matplotlib==3.1.3
Shapely==1.6.4.post1
pandas==1.0.1
numpy==1.18.1
tqdm==4.43.0
seaborn==0.10.0
Cartopy==0.17.0
copulas==0.3.3
scipy==1.4.1
statsmodels==0.11.0
xesmf==0.3.0
MiniSom==2.2.3
addons==0.7
scikit_learn==0.24.2

## Usage

The script *_weather_pattern_class.py* contains a class that is used to define and analyse weather patterns.

In the jupyter notebook *0_prepare_ibTracks.ipynb* the Ibtracks data is loaded and stored in a format that is required by the following scripts and notebooks.

In the jupyter notebook *1_weather_patterns.ipynb* the weather patterns are calcualted.

The script *_emulator.py* contains a class that is used to build emulators and perform emulations. It also contains functions to analyse these emulations. The emulator class needs components that are stored in individual scripts in the folder *components*. The overarching emulator architecture is stored in *Emu.Emu0.py* while the components for storm formation, storm duration and storm intensities are stored in the folders *g*, *sL*, *wS* respectively.

In the jupyter notebook *2_emulations.ipynb* different emulators are build and emulations as well as validations of these emulations are performed.

In the jupyter notebook *3_counterFactual_scenarios.ipynb* counterfactual scenarios are emulated.

In the jupyter notebook *4_plot_diverses.ipynb* some of the main figures are plotted.

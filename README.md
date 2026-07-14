<img src="MagIE_Mags.png" alt="Magnetometer Map" width="400">

## Code useful for Irish magnetometers. Such as downloading from the website.

## Install Guide
### pip install
conda create -n magie python=3.12 # important to not use later python versions if building from pip

conda activate magie

pip install "git+https://github.com/magie-network/MAGIE.git@3.3.0"

For alert support with `pip`, install the `alerts` extra:

pip install "magie[alerts] @ git+https://github.com/magie-network/MAGIE.git@3.3.0"

This will install mastodon for mastodon alerts

For magnetometer monitor map support with `pip`, install the `monitor` extra:

pip install "magie[monitor] @ git+https://github.com/magie-network/MAGIE.git@3.3.0"

This will install Plotly and secsy for the magnetometer status map tools.

This package temporarily depends on a GitHub commit of geomagpy
because the circular import fix is not yet released on PyPI.

### Build from environment file (not as limited with python version)
Alternatively:

conda env create -f ./binder/environment.yml

conda activate magie

This creates a development environment with the required dependencies but
does not install the `magie` package itself. For local development after
cloning the repository, install the package manually if needed, for example:

pip install -e .

For local development with the monitor map dependencies, install the
`monitor` extra:

pip install -e ".[monitor]"

For alert posting support, use the alerts environment instead:

conda env create -f ./binder/environment_alerts.yml

conda activate magie-alerts

## Tutorials
In the notebook folder a set of notebooks can be found to demonstrate how to use the magie package.

The `FLO_live_data_wrapper.py` script in `notebooks/` downloads the latest raw X, Y and Z components from Florence Court variometer (FLO), supplied by the British Geological Survey (BGS). The script saves the most recent 24-hour FLO 1-second data in both tab-delimited MagIE format and IAGA-2002 format, and saves the latest 3-day timeseries as a PNG file. Contact BGS to obtain data download authorisation.

FLO is set up and operated by BGS. For more information, visit [SAGE variometer data page](https://intermagnet.bgs.ac.uk/research/SAGE/variometer_data.html?). Ownership and copyright of the FLO data is retained by UKRI.


# File Problems
- Cases of repeated time stamps with different measurement values for Dunsink and Armagh daily 1-second data files. The duplication of timestamps can be checked and fixed using the tutorial example in `Plot_data.ipynb` of the notebook
- Some files have lines only part written as if it broke part way through writing the file (current download code removes these lines but still grabs the remaining good parts of the file)
- There are varied cadence for some sites some have only 1-minute data (E.g. Valentia Observatory) others sometimes have 1-second
- 

# [To do list](todo.md)

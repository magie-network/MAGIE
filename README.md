<img src="MagIE_Mags.png" alt="Magnetometer Map" width="400">

## Code useful for Irish magnetometers. Such as downloading from the website.

## Install Guide
### pip install
conda create -n magie python=3.12 # important to not use later python versions if building from pip

conda activate magie

pip install "git+https://github.com/magie-network/MAGIE.git@3.5.1"

For alert support with `pip`, install the `alerts` extra:

pip install "magie[alerts] @ git+https://github.com/magie-network/MAGIE.git@3.5.1"

This will install mastodon for mastodon alerts

For magnetometer monitor map support with `pip`, install the `monitor` extra:

pip install "magie[monitor] @ git+https://github.com/magie-network/MAGIE.git@3.5.1"

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

The `FLO_live_data_wrapper.py` script in `notebooks/` downloads the latest raw X, Y and Z components from Florence Court variometer (FLO), supplied by the British Geological Survey (BGS). The script saves the most recent 24-hour FLO 1-second data in both tab-delimited MagIE format and IAGA-2002 format, and saves the latest 3-day timeseries as a PNG file. If individual daily IAGA-2002 files are provided instead, use `iaga2magie_xyzf` from `file_conversions.py` to convert data to MagIE txt file format. List of file Path for txt files can be obtained using `get_SAGE_filepaths`. Contact BGS to obtain data download authorisation.

FLO is set up and operated by BGS. For more information, visit [SAGE variometer data page](https://intermagnet.bgs.ac.uk/research/SAGE/variometer_data.html?). Ownership and copyright of the FLO data is retained by UKRI.


# File Problems
- Cases of repeated time stamps with different measurement values for Dunsink and Armagh.
- Some files have lines only part written as if it broke part way through writing the file (current download code removes these lines but still grabs the remaining good parts of the file)
- There are varied cadence for some sites some are only 1-minute others sometimes have 1-second
- 

# [To do list](todo.md)


## Contributing

For future package changes, follow [Package standards](PACKAGE_STANDARDS.md).

## Live LEMI dashboard

For a step-by-step walkthrough, see the [live display notebook](notebooks/live_magpy_display_tutorial.ipynb).

The dashboard is included with MAGIE as `magie.live_magpy_display` and reads
MagPyBin files using MagPy.
After installing MAGIE, start it with a local file or directory:

```bash
magie-live --local /path/to/data/
```

For an SSH source, specify a file or a directory and filename template:

```bash
magie-live --ssh user@host:/path/to/data/ --filename 'LEMI018_3_0001_{date}.bin' --station 'Dunsink LEMI'
```

`{date}` is replaced with the current UTC date in `YYYY-MM-DD` format on each
refresh, so a daily file rolls over at midnight. Use a template matching your
own instrument's filenames. For a fixed remote file, omit `--filename` and give
its full path to `--ssh`. SSH keys and host verification must already be set up
for non-interactive `scp`. No credentials are stored by the dashboard.

Local directory mode picks the lexicographically last `*.bin` file. To select
particular files, use `--filename` with a glob or a `{date}` template; for
example `--local /path/to/data --filename 'sensor_{date}.bin'`. A local file
can also contain `{date}` in its path. The original positional syntax remains
available: `python -m magie.live_magpy_display /path/to/file.bin` and
`python -m magie.live_magpy_display --remote host:/path/to/file.bin`. The
previous `magie.live_dashboard` module entry point remains available.

Open <http://127.0.0.1:8765/> on the same machine. The target refresh interval
is one second, but each refresh copies and reads the full file, so SSH and
parsing may take longer. Copies do not overlap. Use `--interval 30` to reduce
network and CPU load. The browser checks the monitor every second. Run
`magie-live --help` for all options.

The monitor retains up to 259,200 samples by default. With a `{date}`
template in the source or filename, it reads the files covering the last 72
hours and caches completed UTC days in memory. A rolling 72-hour window can
span four daily files. Missing historical files are reported and retried every
minute. Without a dated template, only the selected file is read. The sample
limit can shorten the history if the cadence is faster than one sample per
second; increase `--limit` as needed. Select a component card,
choose a time range, then drag a box on the plot to zoom both axes. Select Pan
and drag horizontally, or use Earlier and Later, to slide the time window.
Reset view returns to the selected time range and resumes following live data.
Move over the plot to inspect a sample. Use **Full screen** in the header for
a display layout adapted to the screen; press Escape or use the button again
to exit. Small, short or portrait screens use a scrollable layout so controls
and readings remain accessible. Both temperature channels and the
reported latitude and longitude are displayed. For TE/TF-labelled data, the
dashboard tentatively identifies TE as electronics and TF as the sensor; the
LEMI-018 documentation consulted does not explicitly define these labels.
The temperature gauges clamp visually to −20–50 °C; numeric values are not
clamped. An embedded Google map and a
link use the reported coordinates; the map requires internet access and sends
those coordinates to Google. A range cannot extend beyond the available files or the configured sample limit. Data age and copy/read failures are reported on
the page, which retains the last successful readings after a failed refresh.
At midnight a missing new file is reported until it appears.

The provisional K panel uses MagPy's operational live FMI approach: it filters
LEMI XYZ data, pads the future with missing values and applies the station's K9
threshold and longitude. It does not use the experimental anomaly workflow.
For Dunsink, `--station 'Dunsink LEMI'` selects the `dun_lemi` metadata; for
other stations use `--k-site` with a MAGIE station alias. Dated sources load
up to four preceding UTC days for FMI context in a separate background worker.
The K calculation refreshes approximately once a minute and may need 45 hours
of prior data; an interval is shown only with at least 80% sample coverage, and
the current interval requires at least 30 minutes. FMI may still return missing
values. The plot is provisional and assumes the input axes are appropriate for
K calculation; raw LEMI data are not anomaly-cleaned or rotated here.

By default the dashboard corrects the XYZ divisors in a temporary copy of the
MagPyBin header to `1` before reading. This addresses the incorrect `0.001`
divisor in the Dunsink files without changing the original or temperature
divisors. For files with correct header divisors, use `--preserve-xyz-divisors`.
XYZ are displayed in nT; H is `sqrt(X² + Y²)` and D is `atan2(Y, X)` in degrees.
The compass and D are relative to the input X axis: interpret them as geographic
declination only if the instrument's X and Y axes have been calibrated and
rotated to geographic north and east. This is a local monitoring tool; it does
not publish the dashboard or expose its server outside localhost.

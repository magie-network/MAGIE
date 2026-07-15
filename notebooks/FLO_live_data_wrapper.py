"""
Wrapper script that returns the latest 24 hours live
1-second data provided by BGS from a password-protected webpage:

    https://geomag.bgs.ac.uk/SpaceWeather/fl_24hrdata.out

and plots the latest Florence Court (FLO) raw variometer data
Script can be made to run on the cron every 30 minutes.
Latest 24-hr FLO data and plot gets updated every 30 minutes.

Setup
-----
Update <pwd_dir> to the Path where .env file is stored locally.
The .env file contains the user name and password provided by BGS
to access the webpage.

Update <base_dir> where each one-second daily files are stored in
<base_dir/year/mm/dd/txt/>

Plotting
--------
Plots the last three-days of data. Empty plot if no data available.
To change number of days since latest, edit
<duration = dt.timedelta(days=2)>

Guanren Wang 2026 Email: gwang1@tcd.ie
"""
import matplotlib.pyplot as plt
from pathlib import Path
from magie.Data_Download import get_SAGE_variometer, save_SAGE_data
from magie.Data_Download import save_SAGE2iaga2002
from magie.Data_Processing import get_SAGE_filepaths
from magie.Plotting_Tools import plot_variometer_data
from magie.utils import get_site_metadata

# change pw_dir to path folder where .env file is stored locally
pw_dir = Path(r'../notebooks/')

# change base_dir to path folder where daily variometer files are stored
base_dir = Path(r'../Data/')

df = get_SAGE_variometer(pw_dir, printHeader=False)

# save the online data to into <base_dir/year/mon/dd/txt/>
obs = "flo"
save_SAGE_data(df, base_dir, freq='1s', obs=obs, flag=99999.00)

"""
if "Missing file : ..."" statement is printed, uncomment to
import and call generate_missing_day function as below.
"""
# from magie.Data_Processing import generate_missing_day()
# generate_missing_day(base_dir, "flo20260516.txt", obs='flo')

"""
Plot txt file data.
Define start_time, end_time and duration if not using None. Uncomment below
and comment out "Plot latest 3 days only" block.
"""
# import datetime as dt
# duration = dt.timedelta(days=2)
# end_time = dt.datetime(2026, 7, 14, 23, 59, 59)
# start_time = (end_time - duration).replace(hour=0, minute=0, second=0)
# all_file_path, start_time, end_time = get_SAGE_filepaths(
#     base_dir, start_time=None, end_time=end_time, duration=duration
#     )

"""
Plot latest 3 days time series.
Comment this out if start_time and end_time are defined manually above
"""
all_file_path, start_time, end_time = get_SAGE_filepaths(base_dir)

# convert then save the daily file in <all_file_path> to IAGA-2002 format
save_SAGE2iaga2002(
    all_file_path, base_dir, obs, site_name="florence court", print_msg=False
    )

start = start_time.strftime('%Y-%m-%d')
end = end_time.strftime('%Y-%m-%d')

# plotting parameters
outfile_name = f"{start}_to_{end}.png"
comps = ["X", "Y", "Z"]
lat = get_site_metadata(obs)["geodetic_latitude"]
lon = get_site_metadata(obs)["geodetic_longitude"]
title = (
    f"Florence Court, Fermanagh, one-second fluxgate variometer "
    f"Lat: {lat}\u00b0 , Lon: {lon}\u00b0"
)
footer = "X, Y, Z data provided by BGS \u00A9 UKRI. For research use only."
# plot the latest <duration> days data for SAGE variometer
fig, ax = plot_variometer_data(
    start_time, end_time, obs, base_dir, title, outfile_name,
    footer, comps, print_msg=False, print_debug=False
    )
plt.show()

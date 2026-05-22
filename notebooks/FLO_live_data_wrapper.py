"""
Wrapper script that returns the latest 24 hours live
1-second data provided by BGS from a password-protected webpage:

    https://geomag.bgs.ac.uk/SpaceWeather/fl_24hrdata.out

and plots the latest Florence Court (FLO) raw variometer data
Script can be made to run on the cron every 30 minutes.
Latest 24-hr FLO data and plot gets updated every 30 minutes.

Setup:
------
Update <pwd_dir> to the Path where .env file is stored locally.
The .env file contains the user name and password provided by BGS
to access the webpage.

Update <base_dir> where each one-second daily files are stored in
<base_dir/year/mm/dd/txt/>

Plotting:
---------
Plots the last three-days of data. Empty plot if no data available.
To change number of days since latest, edit
<duration = dt.timedelta(days=2)>

Guanren Wang January 2026
Email: gwang1@tcd.ie
"""
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
from magie.Data_Download import get_SAGE_variometer, save_SAGE_data
from magie.Data_Download import save_SAGE2iaga2002
from magie.Plotting_Tools import plot_variometer_data

# change pw_dir to path folder where .env file is stored locally
pw_dir = Path(r'../notebooks/')

# change base_dir to path folder where daily variometer files are stored
base_dir = Path(r'../Data/')

df = get_SAGE_variometer(pw_dir, printHeader=False)

# save the online data to into <base_dir/year/mon/dd/txt/>
save_SAGE_data(df, base_dir, freq='1s', obs="flo", flag=99999.00)

# if "Missing file : ..."" statement is printed
# uncomment to import and call __generate_missing_day_ as below
# from magie.Data_Processing import generate_missing_day()
# generate_missing_day(base_dir, "flo20260516.txt", obs='flo')

# define start and end time for plotting
end_time = dt.datetime.now().replace(
    hour=23, minute=59, second=59, microsecond=0
    )

# define number of days of time series data plotted
duration = dt.timedelta(days=2)

start_time = (end_time - duration).replace(hour=0, minute=0, second=0)
start = start_time.strftime('%Y-%m-%d')
end = end_time.strftime('%Y-%m-%d')
obs = "flo"
all_file_path = []
day_iterator = start_time
while day_iterator <= end_time:
    day_file_path = (
        base_dir
        / day_iterator.strftime("%Y")
        / day_iterator.strftime("%m")
        / day_iterator.strftime("%d")
        / "txt"
    )

    if day_file_path.exists():
        all_file_path.append(day_file_path)

    day_iterator += dt.timedelta(days=1)

# convert then save the daily file in <all_file_path> to IAGA-2002 format
save_SAGE2iaga2002(
    all_file_path, base_dir, obs, site_name="florence court",
    print_msg=False, print_debug=False
    )

# PLOT txt file data
# plotting parameters
outfile_name = f"{start}_to_{end}.png"
comps = ["X", "Y", "Z"]
title = "Florence Court one-second fluxgate variometer, Fermanagh."
footer = "Data provided by BGS \u00A9 UKRI. For research use only."
# plot the latest <duration> days data for SAGE variometer
fig, ax = plot_variometer_data(
        start_time, end_time, obs, base_dir, title, outfile_name,
        footer, comps, print_msg=False, print_debug=False)
plt.show()

"""
Wrapper script that returns the latest 24 hours live 
1s Data provided by BGS by accessing from a 
password-protected webpage

https://geomag.bgs.ac.uk/SpaceWeather/fl_24hrdata.out

and plots the lastest Florence Court (FLO) raw variometer data
Script can be made to run on the cron every 30 minutes
Latest 24-hr FLO data and plot gets updated every 30 minutes

Update <base_dir> where each one-second day-file are stored in
<base_dir/year/mon/dd/txt/>

Plots the last three-days of data. Empty plot if no data.
To change number of days since latest, edit
<duration = dt.timedelta(days=2)>

Guanren Wang January 2026
Email: gwang1@tcd.ie
"""
import datetime as dt
from pathlib import Path
from magie.Data_Download import get_SAGE_variometer
from magie.Data_Download import save_SAGE_data
from magie.Plots import plot_variometer_data

# change base_dir to path folder where daily variometer files are stored
base_dir = Path(r'../../Data/')

# Data down function requires .env file in <src/magie> that stores the
# username and password provided by BGS
df = get_SAGE_variometer(printHeader=False)

# save the online data to file <base_dir>
# the script sorts file into <base_dir/year/mon/dd/txt/>
save_SAGE_data(df, base_dir, freq='1s', obs="flo", flag=99999.00)

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

    if day_file_path .exists():
        all_file_path.append(day_file_path)

    day_iterator += dt.timedelta(days=1)

outfile_name = f"{start}_to_{end}.png"
comps = ["X", "Y", "Z"]

# plot the latest <duration> days data for variometer
plot_variometer_data(
    start_time, end_time, obs, base_dir, outfile_name, comps
    )
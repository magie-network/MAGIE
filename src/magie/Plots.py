import numpy as np
from os import path as pth
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
import matplotlib.dates as mdates
from Data_Download import Download_MAGIE

def fix_missing_timestamps(df, site_col="Site"):
    """
        Creates a full second-by-second index from min to max.
        Missing rows get NaN for Bx/By/Bz, but Site is filled via ffill/bfill.
        Duplicate timestamps are fixed first in fix_duplicated_timestamps function.

        Parameters:
        df (pandas.DataFrame): data frame sorted and ensure it is datetime and sorted.

        Dependencies:
        function fix_duplicated_timestamps

        Returns:
        df_fill (pandas.DataFrame): chronologically-correct  gap-free DataFrame.
    """
    # Ensure datatime index is sorted
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    # Build full timeline between start to end timestamps
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="s")

    # Reindex so we get missing rows as missing value
    df_full = df.reindex(full_index)

    # fill Site column. Forward-fill then backward-fill to ensure all rows have correct Site.
    if site_col in df_full.columns:
        df_full[site_col] = df_full[site_col].ffill().bfill()

    return df_full


def fix_timestamp_duplicates(df):
    """
    Finds duplicated timestamps.
    Shifts first duplicate timestamp backward and keep second timestamp as is.
    None-duplicated timestamps (rows) left untouched
    Produces a chronologically-correct, duplicate-free DataFrame.

    Parameters:
    df (pandas.DataFrame): data frame sorted and ensure it is datetime and sorted.

    Dependencies:
    None.

    Returns:
    df_fill (pandas.DataFrame): chronologically-correct  gap-free DataFrame.
    """
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    # Identify duplicates (all but first occurrence)
    duplicates = df[df.index.duplicated(keep=False)]

    # If no duplicates, return original row
    if duplicates.empty:
        return df.copy()

    # New index starts as original
    new_index = df.index.to_list()

    # Process each duplicate group
    # initially all original timestamps are used
    used = set(df.index)
    # Group duplicates by timestamp
    groups = duplicates.groupby(level=0)

    for ts, group in groups:
        # get the positions of this timestamp in df
        positions = group.index.to_list()
        for occurence, ts_row in enumerate(group.itertuples(index=False, name=None)):
            # find integer location
            ts_original = positions[occurence]
            if occurence == 0:
                # First occurence, shift backward 1 seconds
                new_ts = ts_original - pd.Timedelta(seconds=1)
                # Ensure new_ts doesn't conflict with other timestamps
                while new_ts in used:
                    new_ts -= pd.Timedelta(seconds=1)
            elif occurence == 1:
                # Second occurence keep original timestamps_minute
                new_ts = ts_original
            else:
                # Third or later, shift forward until free
                new_ts = ts_original
                while new_ts in used:
                    new_ts += pd.Timedelta(seconds=1)

            # Use the index in the original index list
            idx_in_index_list = new_index.index(ts_original)
            new_index[idx_in_index_list] = pd.Timestamp(new_ts)
            used.add(new_ts)

    # Assign new index
    df_new = df.copy()
    df_new.index = pd.to_datetime(new_index)

    # Sort index to keep chronological order
    df_new = df_new.sort_index()

    return df_new


start_time = dt.datetime(2025, 11, 11, 0, 0, 0)
end_time =  dt.datetime(2025, 11, 12, 23, 59, 59)
np_start_time = np.datetime64(start_time, 's')
np_end_time = np.datetime64(end_time, 's')
# Download_MAGIE(np_start_time, np_end_time, sites=['dun', 'val'],
#                save_file_name='../../Data/dun_val2025_11_11_to_2025_11_12.hdf5')

file_path = "../../Data/dun_val2025_11_11_to_2025_11_12.hdf5"
df = pd.read_hdf(file_path, columns =['Date_UTC', 'Site', 'Bx', 'By', 'Bz'])

# Remove timezone after checking that it is UTC
df["Date_UTC"] = df["Date_UTC"].dt.tz_localize(None)

# assign individual dataframes for each site
df_dun = df[df['Site'].str.contains("dun")]
df_val = df[df['Site'].str.contains("val")]

# sort indexing in chronological order
df_dun = df_dun.set_index('Date_UTC')
df_val = df_val.set_index('Date_UTC')
df_val = df_val.sort_index()

# filter for start and end UTC times
df_dun = df_dun.loc[start_time:end_time]
df_val = df_val.loc[start_time:end_time]

# check for duplicates
dun_duplicates = df_dun.index[df_dun.index.duplicated()]

# fix duplicates
df_dun_duplicates_fixed = fix_timestamp_duplicates(df_dun)
df_dun_full = fix_missing_timestamps(df_dun_duplicates_fixed)

def quick_plot(df, start_time, end_time, comp):
    """
    Quick plot to zoom into specific time interval.

    Parameters:
    df (pandas.DataFrame): data frame sorted and ensure it is datetime and sorted.
    start_time (pandas.DataTime): yyyy-mm-dd HH:MM:SS
    end_time (pandas.DataTime): yyyy-mm-dd HH:MM:SS
    comp (string): either Bx, By, Bh, or Bz

    Dependencies:
    None

    Returns:
    xy plot
    """
    start_index = df.iloc[start_time:end_time]
    end_index = df.iloc[start_time:end_time]
    valid_comp = ['Bx', 'By', 'Bz']
    fig, ax=plt.subplots(figsize=(14, 10))
    ax.plot(df_dun_full.index[start_index:end_index], df_dun_full["Bx"][start_index:end_index])
    ax.set_xlim(df_dun_full.index[start_index], df_dun_full.index[end_index])
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0,1,24)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    # Tick parameters
    ax.tick_params(axis='x', which='minor', direction='in', labelsize=15, length=5)
    ax.tick_params(axis='x', which='major', direction='out', labelsize=15, length=20)
    ax.tick_params(axis='y', which='major', direction='in', labelsize=15, length=5)
    ax.set_xlabel('Time [UT]', fontsize=14)

    if comp not in valid_comp:
        raise ValueError(f'component {comp} not a valid column name in DataFrame{df}'. Must be one of {valid_comps})
    if comp == "Bx":
        ax.set_ylabel('$B_x$ [nT]', fontsize=14)
    elif comp == "By":
        ax.set_ylabel('$B_y$ [nT]', fontsize=14)
    elif comp == "Bz":
        ax.set_ylabel('$B_z$ [nT]', fontsize=14)
    elif comp == "Bh":
        ax.set_ylabel('$B_h$ [nT]', fontsize=14)
        

# quick plot
fig, ax=plt.subplots(figsize=(14, 10))
ax.plot(df_dun_full.index, df_dun_full["Bx"])
ax.set_xlim(df_dun_full.index[0], df_dun_full.index[-1])
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0,6,24)))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlabel('Time [UT]')
ax.set_ylabel('By [nT]')
# Tick parameters
ax.tick_params(axis='x', which='minor', direction='in', labelsize=15, length=5)
ax.tick_params(axis='x', which='major', direction='out', labelsize=15, length=20)
ax.tick_params(axis='y', which='minor', direction='in', labelsize=15, length=5)


# run these lines if start_time and end_time don't end on full-minute
# Find the last full minute at Dunsink
last_minute = df_dun_full.index[-1].floor('min')  # round down to nearest minute
# Only keep rows with timestamp < next minute so it is divisible by 60 s
df_dun_full = df_dun_full[df_dun_full.index <= last_minute]


# apply 61-point cosine filter to reduce one-second data to produce one-minute values
# only when there are no data gaps
window_size = 61  # 1-minute smoothing for 1-second data

def cosine_smooth(series, window):
    # Create a Hanning window
    w = np.hanning(window)
    w = w / w.sum()  # normalize to preserve amplitude
    # Convolve with the data
    return np.convolve(series, w, mode='valid')

"""
# apply smoothing
Bx_smooth = cosine_smooth(df_dun_full['Bx'].values, window_size)
By_smooth = cosine_smooth(df_dun_full['By'].values, window_size)
Bz_smooth = cosine_smooth(df_dun_full['Bz'].values, window_size)

# Because mode='valid' reduces length by window_size-1
Offset = window_size//2

# Downsample to one value per minute
Bx_minute = Bx_smooth[::60]
By_minute = By_smooth[::60]
Bz_minute = Bz_smooth[::60]

# Get minute timestamps aligned with smoothed values
# Timestamps: shift back by offset to align with start of minute
valid_index = df_dun.index[offset:-offset]
# Trim the index so it matches the "valid" convolution result
# valid_index = df_dun.index[window_size - 1:]  # N - (M - 1)
# want perfect alignment with original timestamps
# valid_index = df_dun.index[offset + offset : -offset + offset]  # shift both sides
timestamps_minute = valid_index[::60].floor('min')
"""
def one_minute_sampling(df):
    minute_mean = df.resample('min').mean()
    resampled_count = df.resample('min').count()
    # set a coverage threshold: require at least 45 valid seconds of data
    coverage_threshold = 45
    minute_mean[resampled_count < coverage_threshold] = np.nan
    # keep the count or fraction for diagonistics
    resampled = pd.DataFrame({
        'mean': minute_mean,
        'count_1s': resampled_count,
        'coverage_frac': resampled_count / 60.0
    })
    return minute_mean, resampled

Bx_minute, Bx_resampled = one_minute_sampling(df_dun_full['Bx'])
By_minute, Bz_resampled = one_minute_sampling(df_dun_full['By'])
Bz_minute, Bz_resampled = one_minute_sampling(df_dun_full['Bz'])
timestamps_minute = Bx_minute.index
# Create a new DataFrame for one-minute Dunsink values
df_dun_minute = pd.DataFrame({
    'Site': 'dun',
    'Bx': Bx_minute,
    'By': By_minute,
    'Bz': Bz_minute
}, index=timestamps_minute)

print(df_dun_minute.head())
print(df_dun_minute.index.min(), df_dun_minute.index.max())

# compute first 10 hours of mean in Bx, By and Bz on a geomagnetically quiet time pre-storm
minutes = 10*60
mean_dun_x = df_dun_minute['Bx'].iloc[:minutes].mean(skipna=True)
mean_val_x = df_val['Bx'].iloc[:minutes].mean()
mean_dun_y = df_dun_minute['By'].iloc[:minutes].mean(skipna=True)
mean_val_y = df_val['By'].iloc[:minutes].mean()
mean_dun_z = df_dun_minute['Bz'].iloc[:minutes].mean(skipna=True)
mean_val_z = df_val['Bz'].iloc[:minutes].mean()

# --- Create 3-panel subplot ---
# Small offset
offset_x, offset_y, offset_z = 1000, -2300, -180
text_offset_x, text_offset_y, text_offset_z = 1020, -2270, -150
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
components = ['Bx', 'By', 'Bz']
means_dun = [mean_dun_x, mean_dun_y, mean_dun_z]
means_val = [mean_val_x, mean_val_y, mean_val_z]
offsets = [offset_x, offset_y, offset_z]
text_offsets = [text_offset_x, text_offset_y, text_offset_z]
ylabels = ["$B_x$ (nT)", "$B_y$ (nT)", "$B_z$ (nT)"]


for i, ax in enumerate(axes):
    comp = components[i]

    # Plot data
    ax.plot(df_dun_minute.index, df_dun_minute[comp]+offsets[i], label='Dunsink', color='crimson', linewidth=1.5)
    ax.plot(df_val.index, df_val[comp], label='Valentia', color='blue', linewidth=1.5)
    ax.set_xlim(df_dun_minute.index[0], df_dun_minute.index[-1])

    # Add mean labels at start
    ax.text(df_dun_minute.index[50], means_dun[i]+text_offsets[i], f'{means_dun[i]:.0f}', color='crimson',
            va='bottom', ha='left', fontsize=16)
    ax.text(df_val.index[50], means_val[i], f'{means_val[i]:.0f}', color='blue',
            va='bottom', ha='left', fontsize=16)

    # Hide right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Remove y-axis ticks
    ax.yaxis.set_ticks([])

    # Set y-axis ticks at first point
    ax.set_yticks([df_dun_minute[comp].iloc[0]+offsets[i], df_val[comp].iloc[0]])
    # labeled with site names
    ax.set_yticklabels([df_dun_minute['Site'].iloc[0], df_val['Site'].iloc[0]], fontsize=17)
    
    # x-axis formatters (will be applied only on bottom panel, but safe to set for all)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0,12,24)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    # Draw vertical dashed lines at major x-ticks
    for tick in ax.get_xticks():
        ax.axvline(x=tick, color='black', linestyle='--', alpha=0.5, linewidth=2)

    # Tick parameters
    ax.tick_params(axis='x', which='minor', direction='in', labelsize=15, length=5)
    ax.tick_params(axis='x', which='major', direction='out', labelsize=16, length=20)

    # --- Draw vertical scale bar (400 nT) in top-right of each panel ---
    scale_length = 400
    ymin, ymax = ax.get_ylim()
    rel_height = scale_length / (ymax - ymin)
    x_pos, y_top = 0.95, 0.95
    ax.plot([x_pos, x_pos], [y_top-rel_height, y_top], transform=ax.transAxes, color='black', lw=2, clip_on=False)
    ax.text(x_pos+0.01, y_top-rel_height/2, '400 nT', transform=ax.transAxes,
            va='center', ha='left', fontsize=16, color='black')

    # set y-axis label
    ax.set_ylabel(ylabels[i], fontsize=18)


# Bottom panel gets the x-axis label
axes[-1].set_xlabel('Time UTC', fontsize=14)

plt.suptitle('Comparison of Dunsink and Valentia One-Minute Data', fontsize=16)
plt.tight_layout(rect=[0,0,1,0.96])
filename = 'Dun_Val_Nov11_halfday-12_Bxyz.png'
output_file_path = r'/Users/Guanren/Desktop/'
ff = pth.join(output_file_path, filename)
print("Plot {} is saved in {}".format(filename, output_file_path))
plt.savefig(ff, format="png")
plt.show()

# Ensure your index is datetime
# df_dun_minute.index = pd.to_datetime(df_dun_minute.index)

def compute_H(df_x, df_y):
    """
    Compute H = sqrt (X^2 + Y^2)
    """
    H = np.sqrt(df_x**2 + df_y**2)
    return H

# Append Bh column in data frames
df_dun_minute['Bh'] = compute_H(df_dun_minute['Bx'], df_dun_minute['By'])
df_val['Bh'] = compute_H(df_val['Bx'], df_val['By'])


# Compute derivative (difference between consecutive points)
# nT per minute
df_dun_minute['dBx_dt'] = df_dun_minute['Bx'].diff()
df_dun_minute['dBy_dt'] = df_dun_minute['By'].diff()
df_dun_minute['dBz_dt'] = df_dun_minute['Bz'].diff()
df_dun_minute['dBh_dt'] = df_dun_minute['Bh'].diff()
df_val['dBx_dt'] = df_val['Bx'].diff()
df_val['dBy_dt'] = df_val['By'].diff()
df_val['dBz_dt'] = df_val['Bz'].diff()
df_val['dBh_dt'] = df_val['Bh'].diff()

# --- Create 3-panel subplot ---
# --- Setup ---
offset = 90
text_offset = 92
scale_length = 50

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
components = ['dBx_dt', 'dBy_dt', 'dBh_dt', 'dBz_dt']
titles = ['$dB_x/dt$ (nT/min)', '$dB_y/dt$ (nT/min)', '$dB_h/dt$ (nT/min)', '$dB_z/dt$ (nT/min)']

for ax, comp, label in zip(axes, components, titles):
    # Build column names
    col = f'{comp}'

    # --- Plot data ---
    ax.plot(df_dun_minute.index, df_dun_minute[col] + offset, color='crimson', linewidth=1)
    ax.plot(df_val.index, df_val[col], color='blue', linewidth=1)

    # --- Set limits and style ---
    ax.set_xlim(df_dun_minute.index[0], df_dun_minute.index[-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # --- label zeros at start of each trace ---
    ax.text(df_dun_minute.index[50], text_offset, '0 nT/min', color='black',
            va='bottom', ha='left', fontsize=16)
    ax.text(df_val.index[50], 0, '0 nT/min', color='black',
            va='bottom', ha='left', fontsize=16)

    # --- Y-axis site labels ---
    y_ticks = [df_dun_minute[col].iloc[1] + offset, df_val[col].iloc[1]]
    y_labels = [df_dun_minute['Site'].iloc[0], df_val['Site'].iloc[0]]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=17)
    ax.tick_params(axis='y', which='both', length=10)

    # --- Vertical dashed grid lines at major ticks ---
    for tick in ax.get_xticks():
        ax.axvline(x=tick, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

    # --- Vertical scale bar ---
    x_bar = df_dun_minute.index[-180]
    y_bar_top = max(df_dun_minute[col].max() + offset, df_val[col].max())
    y_bar_bottom = y_bar_top - scale_length

    ax.plot([x_bar, x_bar], [y_bar_bottom, y_bar_top], color='black', lw=2)
    ax.text(x_bar + pd.Timedelta(minutes=10),
            y_bar_bottom + scale_length/2,
            f'{scale_length} nT',
            va='center', ha='left', fontsize=16, color='black')

    # --- Axis labeling ---
    ax.set_ylabel(label, fontsize=18)

# --- X-axis formatting (shared) ---
axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=(0,12,24)))
axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=3))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
axes[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

axes[-1].tick_params(axis='x', which='major', direction='out', length=15, labelsize=16)
axes[-1].tick_params(axis='x', which='minor', direction='in', length=5, labelsize=15)

axes[-1].set_xlabel('Time UTC', fontsize=14)

# --- Figure title ---
fig.suptitle('Comparison of Dunsink and Valentia One-Minute Time Derivatives', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
filename = 'Dun_Val_Nov11_halfday-12_dBdt.png'
output_file_path = r'/Users/Guanren/Desktop/'
ff = pth.join(output_file_path, filename)
print("Plot {} is saved in {}".format(filename, output_file_path))
plt.savefig(ff, format="png")
plt.show()






offset_dBxdt = 100
text_offset_dBxdt = 102
plt.figure(figsize=(12,5))

plt.plot(df_dun_minute.index, df_dun_minute['dBx_dt']+offset_dBxdt, label='Dunsink', color='crimson', linewidth=1)
plt.plot(df_val.index, df_val['dBx_dt'], label='Valentia', color='blue', linewidth=1)

# Access current axes
ax = plt.gca()
ax.set_xlim(df_dun_minute.index[0], df_dun_minute.index[-1])

# Add mean labels at start
ax.text(df_dun_minute.index[50], text_offset_dBxdt, '0', color='crimson',
        va='bottom', ha='left', fontsize=12)
ax.text(df_val.index[50], 0, '0', color='blue',
        va='bottom', ha='left', fontsize=12)

# Hide right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Remove y-axis ticks
ax.yaxis.set_ticks([])

# Set y-axis ticks at first point
ax.set_yticks([df_dun_minute['dBx_dt'].iloc[1]+offset_dBxdt, df_val['dBx_dt'].iloc[1]])
# labeled with site names
ax.set_yticklabels([df_dun_minute['Site'].iloc[0], df_val['Site'].iloc[0]])

# Draw vertical dashed lines at each major x tick
for tick in ax.get_xticks():
    ax.axvline(x=tick, color='black', linestyle='--', alpha=0.5, linewidth=2)

# x-axis formatter
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0,24,12)))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

# x-axis ticks to appear inside the plot
ax.tick_params(axis='x', which = 'minor', direction='in', labelsize=11, length=5)
ax.tick_params(axis='x', which = 'major', direction='out', labelsize=11, length=15)

# Get y-data range to convert nT to relative height
ymin, ymax = ax.get_ylim()
rel_height = scale_length / (ymax - ymin)

# --- Vertical scale bar (50 nT) in data coordinates ---
scale_length = 50
x_bar = df_dun_minute.index[-180]  # rightmost x
y_bar_top = max(df_dun_minute['dBx_dt'].max() + offset_dBxdt, df_val['dBx_dt'].max())
y_bar_bottom = y_bar_top - scale_length

ax.plot([x_bar, x_bar], [y_bar_bottom, y_bar_top], color='black', lw=2)
ax.text(x_bar + pd.Timedelta(minutes=10), y_bar_bottom + scale_length/2,
        '50 nT', va='center', ha='left', fontsize=12)

plt.title('Comparison of Dunsink and Valentia one-minute time derivative', fontsize=12)
plt.xlabel('Time UTC')
plt.ylabel('dBx/dt (nT/min)')
plt.tight_layout()
plt.show()


# Example data: suppose we have 4 sites, each with values over time
times = np.arange(0, 100)  # e.g., time points
site1 = np.sin(times / 10) + 0.0
site2 = np.cos(times / 10) + 2.0
site3 = np.sin(times / 5) + 4.0
site4 = np.cos(times / 5) + 6.0

plt.figure(figsize=(8,6))
plt.plot(times, site1, label='Site 1')
plt.plot(times, site2, label='Site 2')
plt.plot(times, site3, label='Site 3')
plt.plot(times, site4, label='Site 4')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value (offset)')
plt.title('Stacked traces (offset by constant)')
plt.show()


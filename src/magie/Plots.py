"""
Functions for plotting IAGA GIN & MagIE data
after they are downloaded.
Data download codes from Data_Downloads.py.
Data processing codes from Data_Processing.py.
Uses Fabio Crameri scientific colour scheme (cmcrameri)

Guanren Wang December 2025
"""
import numpy as np
import os
import pathlib
import datetime as dt
import pandas as pd
import cmcrameri as cmc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from os import path as pth
from magie.utils import enforce_types
from matplotlib.ticker import MaxNLocator
plt.style.use('tableau-colorblind10')


@enforce_types(
    df=pd.DataFrame,
    obs_plot_list=list,
    padding_fraction=float,
    component_list=list,
    means=dict,
    scale_length=(int, float),
    ylabels=list,
    title=str,
    file_name=str,
    output_file_path=pathlib.Path,
)
def stack_plot(df, obs_plot_list, padding_fraction, component_list,
               means, scale_length, ylabels, title, file_name,
               output_file_path):
    """
    --- Create 3-panel subplot of observatory timeseries ---
    a station with a naturally large X, Y and Z values
    ends up far away from the others. To deal with this we must:
    Step 1. subtract the absolute baseline - treated as mean_xyz[obs]
    in each component so all stations are near zero.
    Step 2. Apply equal vertical offsets
    y_adjusted = y_original - y_baseline + offset
    Draws a 500 nT scale bar at the top right corner of each panel.

    Parameters:
    -----------
    df: pd.DataFrame
        index timestamped as datetime object containing data with column header
        as OBSX, OBSY, OBSZ, etc
    obs_plot_list: list
        iaga three-letter observatory code in a comma separated list
    padding_fraction: float
        Used in equal-spacing offset for stacking based on maximum variation.
        Usually between 0.1 and 0.3 or 10-30% padding
    component_list: list
        strings of components in comma sepatated list
        E.g. ['X', 'Y, 'Z']
    means: dict
        dictionary of numpy.float values for mean in different components
    scale_length: float, int
        Scale bar in nT
    ylabels: list
        strings for y-axis label in a comma separated list.
    title: str
        plot title, can be '' if no title
    file_name: str
        name of file e.g. Mid-Latitude_Nov11-12_Bxyz.png
    output_file_path: pathlib.Path
        e.g. r'/Users/Name/Desktop/'
    Dependencies:
    -------------
    Function: means_calc
    Ensure your data frame index is datetime:
    df.index = pd.to_datetime(df.index)

    Returns:
    --------
    None
    """
    # Assign offset values per-observatory dynamically based on data range
    data_ranges = []
    for obs in obs_plot_list:
        obs_spreads = []
        for comp in component_list:
            col = f"{obs.upper()}{comp}"
            spread = df[col].max() - df[col].min()
            obs_spreads.append(spread)
        data_ranges.append(max(obs_spreads))

    # compute equal spacing offset for stacking, based on maximum variation
    equal_spacing = max(data_ranges) * padding_fraction
    print(f"Using vertical spacing of {equal_spacing:.0f} nT")

    y_offsets_per_obs = []
    # compute range per component
    for obs in obs_plot_list:
        obs = obs.upper()
        ranges_per_component = []
        for comp in ["X", "Y", "Z"]:
            comp = f"{obs}{comp}"
            obs_range = df[comp].max() - df[comp].min()
            ranges_per_component.append(obs_range)
        # use maximum range per component for spacing
        y_offsets_per_obs.append(max(ranges_per_component))

    N = len(obs_plot_list)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)

    for i, ax in enumerate(axes):
        # set x-axis limits
        ax.set_xlim(df.index[0], df.index[-1])
        # Prepare lists for y-ticks and labels
        ytick_positions = []
        ytick_labels = []

        for obs_index, obs in enumerate(obs_plot_list):
            if N < 5:
                colours = ["black", "crimson", "blue", "green"]
            else:
                colours = cmc.cm.batlowK(np.linspace(0, 0.7, N))

            colour = colours[obs_index]
            obs = obs.upper()
            comp = f"{obs}{component_list[i]}"
            # offset values by-observatory-per-component
            # first observatory in the list gets highest offset
            y_offset = (N - 1 - obs_index) * equal_spacing
            if comp and comp.endswith(("X", "Y", "Z", "H"))\
                    and len(comp) == len(obs) + 1:
                # Add mean labels at start of each time series trace
                mean_obs = means[obs][component_list[i]]
                print(obs, component_list[i], "mean is: ", mean_obs, "nT")
                # normalise each observatory by removing its mean
                y_val_normalised = df[comp] - mean_obs
                y_val_series = y_val_normalised + y_offset
                # label mean values in each component
                ax.text(
                    df.index[50],
                    y_offset,
                    f'{mean_obs:.0f} nT',
                    va='bottom', ha='left', fontsize=16
                    )
                first_y_val_normalised = df[comp].iloc[0] - mean_obs
                first_y_val = first_y_val_normalised + y_offset
            else:
                y_val_series = df[comp] + y_offset
                first_y_val = y_offset

            # plot time series
            ax.plot(
                df.index,
                y_val_series,
                label=obs, linewidth=1.2, color=colour
                )

            # y-tick location per observatory
            ytick_positions.append(first_y_val)
            ytick_labels.append(obs)

        # Set and label y-ticks after plotting all observatory time series
        ax.yaxis.set_ticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=17)

        # Hide right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # x-axis formatters (will be applied only on bottom panel,
        # but safe to set for all)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12, 24)))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        # Draw vertical dashed lines at major x-ticks
        for tick in ax.get_xticks():
            ax.axvline(
                x=tick, color='black', linestyle='--',
                alpha=0.5, linewidth=1.5
                )

        # Tick parameters
        ax.tick_params(
            axis='x', which='minor', direction='in',
            labelsize=15, length=5
            )
        ax.tick_params(
            axis='x', which='major', direction='out',
            labelsize=16, length=20
            )
        # --- Draw vertical scale bar in nT on the top-right of each panel ---
        ymin, ymax = ax.get_ylim()
        rel_height = scale_length / (ymax - ymin)
        x_pos, y_top = 0.95, 0.98
        ax.plot(
            [x_pos, x_pos], [y_top-rel_height, y_top],
            transform=ax.transAxes, color='black', lw=2, clip_on=False
            )
        ax.text(
            x_pos+0.01, y_top-rel_height/2, f'{scale_length} nT',
            transform=ax.transAxes,
            va='center', ha='left', fontsize=16, color='black'
            )
        # set y-axis label
        ax.set_ylabel(ylabels[i], fontsize=16)

    # Bottom panel gets the x-axis label
    axes[-1].set_xlabel('Time UTC', fontsize=14)
    # set title
    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    # save file
    ff = pth.join(output_file_path, file_name)
    print("Plot {} is saved in {}".format(file_name, output_file_path))
    plt.savefig(ff, format="png")
    plt.show()
    """
    end program
    """


@enforce_types(
    start_time=dt.datetime,
    end_time=dt.datetime,
    obs=str,
    base_dir=pathlib.Path,
    outfile=str,
    comps=(list, tuple),
)
def plot_variometer_data(start_time, end_time, obs, base_dir,
                         outfile, comps=["X", "Y", "Z"]):
    """
    Program to plot timeseries of raw variometer data for up to four components

    Parameters:
    -----------
    start_time: datetime.datetime
        datetime.datetime(2026, 4, 29, 0, 0) or
        pd.Timestamp("2026-04-29 00:00:00")
    end_time: datetime.datetime
        datetime.datetime(2026, 5, 1, 23, 59, 59) or
        pd.Timestamp("2026-05-01 23:59:59")
    obs: str
        iaga three-letter observatory code
    base_dir: pathlib.Path
        Base folder where daily iaga-2002 day files live.
        Presently we assume directory structure is in the form of:
        base_dir/year/mon/dd/txt/
    outfile: str
        File name for figure
    comps: list or tuple
        specify components to plot defaults to ['X','Y','Z']
        E.g. ['X'], ['X','Y'], ['X','Y','Z'] or ['X','Y','Z', 'F']

    Dependencies:
    -------
    Function: plot_xyzf

    Raises:
    -------
    FileNotFoundError

    Returns:
    --------
    Saves figure as png file in base_dir/year/mon/dd/png/
    """

    df_list = []
    date_range = pd.date_range(start_time, end_time, freq="D")
    for day in date_range:
        date_str = day.strftime("%Y%m%d")
        year = day.strftime("%Y")
        mon = day.strftime("%m")
        dd = day.strftime("%d")
        daily_dir = pth.join(base_dir, year, mon, dd, "txt")
        fname = pth.join(daily_dir, f"{obs}{date_str}.txt")
        daily_dir_fig = pth.join(base_dir, year, mon, dd, "png")
        os.makedirs(daily_dir_fig, exist_ok=True)
        try:
            df = pd.read_csv(
                fname, sep=r"\s+",
                parse_dates=["Date & Time"],
                index_col="Date & Time",
                na_values=99999.00
                )
            df_list.append(df)
        except FileNotFoundError:
            print(f"Missing file: {fname} in {daily_dir}")

    obs = obs.upper()
    old_col_names = ["Bx", "By", "Bz", "Bf"]
    new_col_names = [f"{obs}{x}" for x in ["X", "Y", "Z", "F"]]

    combined_df = pd.concat(df_list)
    combined_df.rename(
        columns=dict(zip(old_col_names, new_col_names)),
        inplace=True
        )
    combined_df.drop(columns=["Index#"], inplace=True)
    title = f"{obs} raw one-second fluxgate data provided by BGS \u00A9 UKRI"
    # File saved in the latest daily file path
    ff = pth.join(daily_dir_fig, outfile)
    plot_xyzf(combined_df, obs, start_time, end_time, title, ff, comps)
    """
    end program
    """


@enforce_types(
    df=pd.DataFrame,
    obs=str,
    start_time=dt.datetime,
    end_time=dt.datetime,
    title=str,
    outfile=str,
    comps=(list, tuple),
    print_msg=bool,
)
def plot_xyzf(df, obs, start_time, end_time, title, outfile, comps,
              print_msg=False):
    """
    Quick plot to zoom into specific time interval for one component.

    Parameters:
    -----------
    df: pd.DataFrame
        data frame sorted and ensure it is in datetime.
    obs: str
        iaga three-letter observatory code
    start_time: datetime.datetime
        datetime.datetime(2026, 4, 29, 0, 0) or
        pd.Timestamp("2026-04-29 00:00:00")
    end_time: datetime.datetime
        datetime.datetime(2026, 5, 1, 23, 59, 59) or
        pd.Timestamp("2026-05-01 23:59:59")
    title: str
        usually f"{ob} raw one-second fluxgate data provided by BGS (c) UKRI"
    out_file: str
        file path or name of output file for plot figure
    comps: list or tuple
        Components to plot
        E.g. ['X'], ['X','Y'], ['X','Y','Z'] or ['X','Y','Z', 'F']
    print_msg: boolean optional
        to print Plot is saved in "output directory path/file name" message

    Raises:
    -------
    ValueError
    KeyError

    Returns:
    --------
    Saves figure as png
    """
    df_window = df.loc[start_time:end_time]
    valid_comps = {'X', 'Y', 'Z', 'F'}
    # capitalise elements in comps if they aren't capitalised
    comps = [x.upper() for x in comps]

    invalid = [x for x in comps if x not in valid_comps]
    if invalid:
        raise ValueError(
            f"invalid component(s): {invalid}"
            f"Valid options are {sorted(valid_comps)}."
        )

    # build column names, check if requested components in comps exits in df
    cols = [f"{obs.upper()}{c}"for c in comps]
    existing_cols = [c for c in cols if c in df.columns]
    missing_cols = [c for c in cols if c not in df.columns]

    if not existing_cols:
        raise KeyError(
            f"None of the columns in comps exists: '{cols}'. "
            f"Available columns: {list(df.columns)}"
        )

    if missing_cols:
        warnings.warn(
            "The following requested columns are missing and will be skipped:"
            f"{missing_cols}"
        )

    n = len(comps)
    fig, ax = plt.subplots(
        nrows=n, ncols=1, sharex=True, figsize=(14, 3.5*n)
        )

    if n == 1:
        ax = [ax]

    # Plot using the window
    for axis, col, comp in zip(ax, cols, comps):
        axis.plot(
            df_window.index, df_window[col], color='k', label=comp, linewidth=1
            )
        # Force exactly 4 y-axis ticks
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        subscript = comp.lower() if comp != 'F' else "F"
        axis.set_ylabel(rf"$B_{subscript}$ [nT]", fontsize=14)
        # y-axis tick labels (numbers along the axis)
        axis.tick_params(
            axis='y', which='major', direction='in', labelsize=15, length=10
            )

    ax[-1].set_xlim(df_window.index[0], df_window.index[-1])

    # X-axis ticks
    ax[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12)))
    ax[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
    ax[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    ax[-1].xaxis.set_major_locator(mdates.DayLocator())
    for axis in ax:
        axis.grid(
            True, which="major", axis="x", linestyle="-",
            alpha=1, linewidth=1.5
            )
        axis.grid(
            True, which="minor", axis="both", linestyle="-",
            alpha=0.5, linewidth=1.5
            )

    # X-tick parameters
    ax[-1].tick_params(
        axis='x', which='minor', direction='in', labelsize=15, length=10
        )
    ax[-1].tick_params(
        axis='x', which='major', direction='out', labelsize=15, length=20
        )
    ax[-1].set_xlabel("Time (UT)", fontsize=14)
    # set title
    plot_title = f"{title}"
    ax[0].set_title(plot_title, fontsize=14)
    # save file as png
    if print_msg is True:
        print(f"Saving figure to: {outfile}")

    plt.savefig(outfile, format="png")
    plt.show()
    """
    end program
    """

"""
Functions for processing MagIE data after they are downloaded
using functions download_magie.
Function calls are not included here.
November 2025 - June 2026
Guanren Wang (gwang1@tcd.ie)
"""
import datetime as dt
import numpy as np
import pandas as pd
import pathlib
import warnings
from pathlib import Path
from magie.Data_Download import daily_file_template
from magie.file_conversions import _infer_iaga_interval_type
from magie.utils import enforce_types


@enforce_types(df=pd.DataFrame, obs=str)
def calc_minute_derivatives(df, obs):
    """
    Compute derivative (difference between consecutive points)
    for existing X, Y, Z and H component columns in data frame
    per 60s.

    Parameters
    ----------
    df: pandas.DataFrame
        index timestamped as datetime object
    obs: str
        iaga three-letter observatory code

    Dependencies
    ------------
    Function: compute_H first if you need H column.

    Returns
    --------
    df: pandas.DataFrame
        with time derivative columns in nT/minute appended
    """
    ob = obs.upper()
    dt = df.index. to_series().diff().dt.total_seconds()
    for comp in ["X", "Y", "Z", "H"]:
        col = f"{ob}{comp}"
        if col in df.columns:
            df[f"{ob}d{comp}dt"] = df[col].diff() / dt

    return df


@enforce_types(df=pd.DataFrame, obs=str, gap_threshold=int)
def calc_second_derivatives(df, obs, gap_threshold=120):
    """
    Compute derivative (difference between consecutive points)
    for existing X, Y, Z and H component columns in data frame.
    Assuming constant timestamp spacing with/without gaps.
    Computes difference between consecutive timestamps then
    within time gap threshold.
    When handling gaps in data:
    dB/dt = (B_t0 - B_ti) / gap threshold e.g. 120 s

    Parameters
    ----------
    df: pandas.DataFrame
        index timestamped as datetime object
    obs: str
        IAGA three-letter observatory code
    gap_threshold: int
        optional. Time gap in seconds.

    Dependencies
    ------------
    Function: compute_H first if you need H column.

    Returns
    --------
    df: pandas.DataFrame
        with time derivative columns in nT/second appended
    """
    ob = obs.upper()
    dt = df.index.to_series().diff().dt.total_seconds()
    for comp in ['X', 'Y', 'Z', 'H']:
        col = f"{ob}{comp}"
        if col in df.columns:
            dBdt_sec = df[col].diff() / dt
            dBdt_sec[dt > gap_threshold] = float("nan")
            new_col = f"{ob}d{comp}dt"
            df[new_col] = dBdt_sec

    return df


@enforce_types(df=pd.DataFrame, obs=str)
def compute_H(df, obs):
    """
    Compute H = sqrt(X^2 + Y^2)
    Checks whether X Y and H columns exist before computing H

    Parameters
    ----------
    df: pandas.DataFrame
        index timestamped as datetime object
    obs: str
        IAGA three-letter observatory code

    Dependencies
    ------------
    Ensure your index is datetime
    df.index = pd.to_datetime(df.index)

    Returns
    -------
    df: pandas.DataFrame
        input DataFrame with col_H appended as a new column
    """
    col_X = f"{obs.upper()}X"
    col_Y = f"{obs.upper()}Y"
    col_H = f"{obs.upper()}H"
    for col in [col_X, col_Y]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    if col_H not in df.columns:
        df[col_H] = np.sqrt(df[col_X]**2 + df[col_Y]**2)

    return df


@enforce_types(
        df=pd.DataFrame,
        obs=str,
        hrs=(int, float),
        start_time=(dt.datetime, type(None)),
        )
def means_calc(df, obs, hrs, start_time=None):
    """
    Computes mean of OBSX, OBSY, OBSZ and OBSH within
    a user selected time-window for one-minute or one-second data
    Function uses a clock time-based approach.
    df_quiet represents geomagnetically quiet time duration pre-storm.

    Parameters
    ----------
    df: pandas.DataFrame
        index timestamped as datetime object
    obs: str
        IAGA three-letter observatory code
    hrs: int, float
        either integer hours or decimal hours
    start_time: dt.datetime, None
        Beginning of quiet time window.
        Defaults to first time in DataFrame.

    Returns
    -------
    quiet_mean: dict
        Dictionary of mean values keyed by component.
    """
    if start_time is None:
        start_time = df.index[0]

    end_time = start_time + dt.timedelta(hours=hrs)
    df_quiet = df.loc[start_time:end_time]

    quiet_mean = {}
    ob = obs.upper()
    for key in ['X', 'Y', 'Z', 'H']:
        col = f"{ob}{key}"
        if col in df.columns:
            quiet_mean[key] = df_quiet[col].mean(skipna=True)
            coverage = df_quiet[col].count() / len(df_quiet)
            if coverage < 0.9:
                warnings.warn(
                    f"Data coverage for {col} in the selected"
                    f"{hrs} hour quiet period is < 90%."
                    )

    return quiet_mean


@enforce_types(series=np.ndarray, window=int)
def cosine_smooth(series, window=61):
    """
    Apply 61-point cosine filter to reduce one-second data to produce
    one-minute values. Only when there are no data gaps

    Parameters
    ----------
    series: numpy.ndarray
        one-second data. Index timestamped as datetime object
    window: int
        61-point filter (approx 1-minute) on middle sample


    Returns
    --------
    series: numpy.series

    Dependencies
    -------------
    Functions: fix_missing_timestamps
    """
    # Create a Hanning window
    w = np.hanning(window)
    w = w / w.sum()  # normalize to preserve amplitude
    # Convolve with the data
    return np.convolve(series, w, mode='valid')


@enforce_types(data=(pd.Series, pd.DataFrame))
def one_minute_sampling(data):
    """
    Sample one-second MagIE data to produce one-minute data.

    Parameters
    ----------
    data: pandas.Series, pandas.DataFrame
        one-second data. Index timestamped as datetime object

    Dependencies
    ------------
    Function: fix_missing_timestamps

    Returns
    -------
    df: pandas.DataFrame
    """
    minute_mean = data.resample('min').mean()
    resampled_count = data.resample('min').count()
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


def fix_missing_timestamps(df, site_col="Site"):
    """
    Creates a full second-by-second index from min to max.
    Missing rows get NaN for Bx/By/Bz, but Site is filled via
    ffill/bfill.
    Duplicate timestamps are fixed first in fix_duplicated_timestamps.

    Parameters
    ----------
    df: pandas.DataFrame
        data frame sorted and ensure it is datetime.
    site_col: str, optional

    Dependencies
    ------------
        function fix_timestamp_duplicates

    Returns
    -------
    df_full: pandasDataFrame
        Gaps are timestamped as missing data
    """
    # Ensure datatime index is sorted
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    # Build full timeline between start to end timestamps
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="s")

    # Reindex so we get missing rows as missing value
    df_full = df.reindex(full_index)

    # fill Site column. Forward-fill then backward-fill to ensure all rows
    # have correct Site.
    if site_col in df_full.columns:
        df_full[site_col] = df_full[site_col].ffill().bfill()

    return df_full


@enforce_types(df=pd.DataFrame)
def fix_timestamp_duplicates(df):
    """
    Finds duplicated timestamps.
    Shifts first duplicate timestamp backward and keep second timestamp as is.
    None-duplicated timestamps (rows) left untouched
    Produces a chronologically-correct, duplicate-free DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame
        data frame sorted and ensure it is datetime.

    Returns
    -------
    df_new: pandas.DataFrame
        chronologically-correct gap-free DataFrame.
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
        for occurence, ts_row in enumerate(group.itertuples(index=False,
                                                            name=None)):
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


@enforce_types(
        all_file_path=list,
        obs=str,
        flag=float,
        sampling=(str, type(None)),
        print_debug=bool,
               )
def load_iaga2002(all_file_path, obs, flag=99999.0,
                  sampling=None, print_debug=False
                  ):
    """
    Load saved IAGA-2002 from all_file_path into one concatenated DataFrame.

    Parameters
    ----------
    all_file_path: list of pathlib.Path
        List of directory path of MagIE txt files as returned by
        get_SAGE_filepaths().
    obs: str
        Lower case three-letter IAGA observatory code.
    flag: float
        Usually 99999.0 depending on IAGA file content. Defaults to 99999.0.
    sampling: str, optional
        File extension of the desired sampling interval e.g. 'sec', 'min'.
        If None, infers from the first file found in iaga_dir.
    print_debug: bool, optional
        Print debugging information including search patterns, directories
        and glob results. Defaults to False.

    Returns
    -------
    df: pandas.DataFrame
        Concatenated DataFrame with DatetimeIndex. Contains magnetic field
        data for consecutive days depending on number of day files listed
        in all_file_path.

    Raises
    ------
    ValueError
        if no IAGA-2002 files are found to load.

    Dependencies
    ------------
    get_SAGE_filepaths, read_IAGA2002, _infer_iaga_interval_type

    Example
    -------

    """
    dfs = []
    for path in all_file_path:
        dd = path.parent.name
        mm = path.parent.parent.name
        year = path.parent.parent.parent.name
        iaga_dir = Path(path.parent, "iaga2002")
        pattern = f"{obs.lower()}{year}{mm}{dd}*"
        file_list = list(iaga_dir.glob(pattern))
        if print_debug:
            print(f"Searching for: {pattern}")
            print(f"In directory: {iaga_dir}")
            if iaga_dir.iterdir() is True:
                print(f"Files found: {list(iaga_dir.iterdir())}")
            else:
                print(f"Directory does not exist: {iaga_dir}")
            print(f"Glob result: {file_list}")

        if not file_list:
            warnings.warn(f"Warning: NO IAGA-2002 file found for stem "
                          f"{pattern}, skipping."
                          )
            continue

        for candidate in file_list:
            df_daily = read_IAGA2002(
                dir=iaga_dir, fname=candidate.name, print_header=False
                )
            _, ext = _infer_iaga_interval_type(df_daily.index)
            if sampling is None:
                # lock-in sampling from first inferred value
                sampling = ext
                if print_debug:
                    print(f"Inferred sampling interval: {ext}")
            if ext != sampling:
                print(f"Warning: Skipping {candidate.name}, "
                      f"sampling '{ext}' does not match target '{sampling}'.")
                continue
            dfs.append(df_daily)

    if not dfs:
        raise ValueError("No IAGA-2002 files found to load")

    df = pd.concat(dfs)
    df = df.replace(flag, np.nan)
    return df


@enforce_types(
        base_dir=pathlib.Path,
        start_time=(dt.datetime, type(None)),
        end_time=(dt.datetime, type(None)),
        duration=dt.timedelta,
        print_debug=bool,
        )
def get_SAGE_filepaths(base_dir, start_time=None, end_time=None,
                       duration=dt.timedelta(days=2), print_debug=False):
    """
    Retrieve daily txt directory for BGS' XYZ files in MagIE format within a
    defined time range via variable duration.

    Parameters
    ----------
    base_dir: pathlib.Path
        Base directory where nested year/mm/dd/txt/ live.
    start_time: dt.datetime, optional
        Start of the time range. Defaults to 2 full days
        before end_time at midnight (00:00:00).
    end_time: dt.datetime, optional
        End of the time range. Defaults to today at 23:59:59.
    duration: dt.timedelta, optional
        Duration of the time range if start_time is not provided.
        Defaults to 2 days (after today).
    print_debug: bool, optional
        Print each day's checked path to confirm they exists.
        Defaults to False.

    Returns
    -------
    all_file_path: list of pathlib.Path
        List of existing file paths within specified time range.
    start_time: dt.datetime
        The resolved start of the time range.
    end_time: dt.datetime
        The resolved end of the time range.

    Raises
    ------
    ValueError
        If no files are found within the specified time range.

    Examples
    --------
    >>> all_file_path, start_time, end_time = get_SAGE_filepaths(base_dir)
    >>> all_file_path, start_time, end_time = get_SAGE_filepaths(
            base_dir, duration=dt.timedelta(days=2)
            )
    """
    if end_time is None:
        end_time = dt.datetime.now().replace(
            hour=23, minute=59, second=59, microsecond=0
            )
    if start_time is None:
        start_time = (end_time - duration).replace(hour=0, minute=0, second=0)

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
        exists = day_file_path.exists()
        if print_debug:
            print(f"Checking: {day_file_path} -> exists: {exists}")
        if exists:
            all_file_path.append(day_file_path)

        day_iterator += dt.timedelta(days=1)
    if not all_file_path:
        raise ValueError(f"No files found between {start_time.date()} and "
                         f"{end_time.date()} in {base_dir}.")
    return all_file_path, start_time, end_time


@enforce_types(obs=str, dir=pathlib.Path, fname=str)
def iaga2magie_xyzf(obs, dir, fname):
    """
    Convert an IAGA-2002 formatted file to MagIE data format.

    Reads an IAGA-2002 file with observatory-specific columns
    OBSX, OBSY, OBSZ, OBSF then renames these columns to generic MagIE
    column names, drops the DOY column, and adds a sequential index column.

    Parameters
    ----------
    obs: str
        Three-letter observatory code.
    dir: pathlib.Path
        Path folder where daily iaga-2002 day files live.
    fname: str
        File of the IAGA file:
        FLO20260116.sec or FLO20260116vsec.sec.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with DateTimeIndex as "Date & Time", and columns:
        Index#, Bx, B, Bz, Bf.

    Raises
    ------
    KeyError:
        If expected IAGA three-letter code not found in file,
        suggest a mismatch between obs and the file's column headers.

    Example
    -------
    >>> iaga2magie_xyzf("flo", iaga_dir, "flo20260911vsec.sec")
    """
    df = read_IAGA2002(dir, fname)
    df.drop(columns=["DOY"], inplace=True)
    df['Index#'] = range(1, len(df) + 1)
    ob = obs.upper()
    old_col_names = [f"{ob}{x}" for x in ("X", "Y", "Z", "F")]
    missing_cols = [col for col in old_col_names if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Expected columns {missing_cols} not found in {fname}. "
            f"Check obs code '{obs}' matches {fname}'s column headers."
            )
    new_col_names = ("Bx", "By", "Bz", "Bf")

    df.rename(columns=dict(zip(old_col_names, new_col_names)), inplace=True)
    cols = ('Index#', "Bx", "By", "Bz", "Bf")
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    return df


@enforce_types(dir=pathlib.Path, fname=str, print_header=bool)
def read_IAGA2002(dir, fname, print_header=False):
    """
    Read in a text file in IAGA-2002 format, defined in
    https://www.ncei.noaa.gov/services/world-data-system/v-dat-working-group/iaga-2002-data-exchange-format,
    whilst keeping flag values (99999.00).

    Parameters
    ----------
    dir: pathlib.Path
        Path folder where daily IAGA-2002 day files live
    fname: str
        File name of the IAGA-2002 file e.g. 'val20260116vsec.sec'.
    print_header: bool, optional
        Prints the IAGA-2002 file mandatory file header records.
        Defaults to False.

    Returns
    -------
    df: pandas.DataFrame
        Data are under 5 columns  DOY OBSX OBSY OBSZ OBSF with a
        DateTime index, where OBS is the three-letter observatory code.
    """
    file = Path(dir, fname)
    headerData = []
    with open(file) as f:
        for i, line in enumerate(f):
            headerData.append(line.rstrip())
            if line.lstrip().startswith("#"):
                continue

            if "DATE" in line:
                header = line.lstrip().rstrip("|").split()
                headerLine = i
                break
    if print_header:
        print('Mandatory file header records')
        for i, line in enumerate(headerData):
            print(f"{i}: [{line}]")

    df = pd.read_csv(
        file, sep=r"\s+",
        skiprows=headerLine + 1,
        header=None,
        names=header
        )
    df["Date & Time"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
    df.set_index("Date & Time", inplace=True)
    df.drop(columns=["DATE", "TIME", "|"], inplace=True, errors="ignore")
    return df


@enforce_types(
        base_dir=str,
        fname=str,
        obs=str,
        freq=str,
        flag=(int, float),
        print_msg=bool,
        )
def generate_missing_day(
        base_dir, fname, obs='flo', freq='1s', flag=99999.00, print_msg=False
        ):
    """
    Fills missing daily variometer file with flagged missing data,
    output saved in a nested file structure of the form
    base_dir/yyyy/mm/dd/txt/. Out file in MagIE tab-delimited format.

    Parameters
    ----------
    base_dir: str
        Path folder to base directory where daily variometer files are
        stored, actual files live in its sub-folders.
    fname: str
        Name of the file to create inside base_dir E.g. flo20260116.txt.
    obs: str, optional
        Three-letter lowercase observatory code.
        Defaults to "flo" BGS variometer.
    freq: str, optional
        Sampling frequency. Defaults to "1s" Florence Court (FLO) data.
        Use "1min" for per-minute frequency and "1h" for per-hour.
    flag: float, optional
        indicates missing data either 99999.0 or 99999.00
        depending on component
    print_msg : bool, optional
        Print save confirmation messages. Defaults to False.

    Returns
    -------
    None
        Saves day file with missing data under formatted columns
        to base_dir/yyyy/mm/dd/txt/, either to be later replaced with
        real data or remain flagged.

    Raises
    ------
    ValueError
        f"{fname} not in the format of {obs}YYYYMMDD.txt".

    Examples
    --------
    >>> generate_missing_day(outputDir, "flo20260126.txt")
    Generates placeholder file for 2026-01-26.
    """
    base_dir = Path(base_dir)
    stem = Path(fname).stem

    valid = (
        fname.startswith(obs)
        and fname.endswith('txt')
        and len(stem[len(obs):]) == 8
        and fname[len(obs):-4].isdigit()
    )

    if not valid:
        raise ValueError(
            f"{fname} not in the format of {obs}YYYYMMDD.txt"
        )

    date_str = stem[len(obs):]
    date = dt.datetime.strptime(date_str, "%Y%m%d").date()
    target_dir = (
        base_dir / date.strftime("%Y") / date.strftime("%m")
        / date.strftime("%d") / "txt"
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    out_path = target_dir / fname

    if not out_path.exists():
        df = daily_file_template(date, freq=freq, flag=flag)
        with open(out_path, 'w') as f:
            f.write("Date & Time\tIndex#\tBx\tBy\tBz\n")
            for i, (ind, row) in enumerate(df.iterrows(), start=1):
                timestamp_str = pd.Timestamp(ind).strftime("%d/%m/%Y %H:%M:%S")
                f.write(
                    f"{timestamp_str}\t{i}\t"
                    f"{row['Bx']:.2f}\t{row['By']:.2f}\t{row['Bz']:.2f}\n"
                )
        if print_msg:
            print(f"Saved/updated: {out_path.name} in {target_dir}")


if __name__ == '__main__':
    obs = "val"
    iaga_dir = Path(r'../../Data/')  # where the IAGA-2002 file is saved
    magie_dir = Path(r'../../Data/')  # where you want the magie file saved
    fname = "OBSYYYYMMDDvsec.sec"  # input IAGA-2002 file
    out_name = "obsYYYYMMDD.txt"  # MagIE file name after iaga2magie_xyzf()
    out_file = Path(magie_dir, out_name)
    df_out = iaga2magie_xyzf(obs, iaga_dir, fname)
    # save df_out DataFrame into tab-delimited MagIE format
    with open(out_file, 'w') as f:
        f.write("Date & Time\tIndex#\tBx\tBy\tBz\n")
        for i, (ind, row) in enumerate(df_out.iterrows(), start=1):
            dt_str = pd.Timestamp(ind).strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"{dt_str}\t{i}\t"
                f"{row['Bx']:.2f}\t{row['By']:.2f}\t{row['Bz']:.2f}\n"
                )

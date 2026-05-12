"""
Functions for processing MagIE data after they are downloaded
using functions download_magie.
Function calls are not included here.
November 2025 - May 2026
Guanren Wang (gwang1@tcd.ie)
"""
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
from magie.Data_Download import daily_file_template
from magie.utils import enforce_types


@enforce_types(df=pd.DataFrame, obs=str)
def calc_minute_derivatives(df, obs):
    """
    Compute derivative (difference between consecutive points)
    for existing X, Y, Z and H component columns in data frame
    per 60s.

    Parameters:
    -----------
    df: pandas.DataFrame
        index timestamped as datetime object
    obs: str
        iaga three-letter observatory code

    Dependencies:
    -------------
    Function: compute_H first if you need H column.

    Returns:
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

    Parameters:
    -----------
    df: pandas.DataFrame
        index timestamped as datetime object
    obs: str
        iaga three-letter observatory code
    gap_threshold: int
        optional. Time gap in seconds.

    Dependencies:
    -------------
    Function: compute_H first if you need H column.

    Returns:
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

    Parameters:
    -----------
    df: pandas.DataFrame
        index timestamped as datetime object
    obs: str
        iaga three-letter observatory code

    Dependencies:
    -------------
    Ensure your index is datetime
    df.index = pd.to_datetime(df.index)

    Returns:
    --------
    df: pandas.DataFrame
        H appended as list
    """
    col_X = f"{obs.upper()}{"X"}"
    col_Y = f"{obs.upper()}{"Y"}"
    col_H = f"{obs.upper()}{"H"}"
    df_x = df[col_X]
    df_y = df[col_Y]
    H = np.sqrt(df_x**2 + df_y**2)
    df[col_H] = H

    return df


@enforce_types(df=pd.DataFrame, obs=str, hours=(int, float), freq=str)
def means_calc(df, obs, hours, freq):
    """
    Computes mean of OBSX, OBSY, OBSZ, OBSH within an integer hour time-window
    for either one-minute or one-second data.
    Time-window represents geomagnetically quiet time pre-storm.

    Parameters:
    -----------
    df: pandas.DataFrame
        index timestamped as datetime object
    obs: str
        iaga three-letter observatory code
    hours: int, float
        either integer hours or decimal hours
    freq: str
        either "1min" or "1s"
    Returns:
    --------
    quiet_mean: dict
        Dictionary of mean values keyed by component
    """
    if freq == "1min":
        samples = int(hours*60)
    else:
        samples = int(hours*3600)

    quiet_mean = {}
    ob = obs.upper()
    for key in ['X', 'Y', 'Z', 'H']:
        col = f"{ob}{key}"
        if col in df.columns:
            quiet_mean[key] = df[col].iloc[:samples].mean(skipna=True)

    return quiet_mean


@enforce_types(series=np.ndarray, window=int)
def cosine_smooth(series, window=61):
    """
    Apply 61-point cosine filter to reduce one-second data to produce
    one-minute values. Only when there are no data gaps

    Parameters:
    -----------
    series: numpy.ndarray
        one-second data. Index timestamped as datetime object
    window: int
        61-point filter (approx 1-minute) on middle sample

    Dependencies:
    -------------
    Functions: fix_missing_timestamps

    Returns:
    --------
    series: numpy.series
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

    Parameters:
    -----------
    data: pandas.Series, pandas.DataFrame
        one-second data. Index timestamped as datetime object

    Dependencies:
    -------------
    Function: fix_missing_timestamps

    Returns:
    --------
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

    Parameters:
    df: pandas.DataFrame
        data frame sorted and ensure it is datetime.
    site_col: str, optional

    Dependencies:
        function fix_timestamp_duplicates

    Returns:
    df_full: pandas.DataFrame
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

    Parameters:
    -----------
    df: pandas.DataFrame
        data frame sorted and ensure it is datetime.

    Returns:
    --------
    df_new: pandas.DataFrame
        chronologically-correct  gap-free DataFrame.
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


@enforce_types(dir=pathlib.Path, df=pd.DataFrame)
def read_IAGA2002(dir, fname):
    """
    Program to read in text files in IAGA-2002 format defined in
    https://www.ncei.noaa.gov/services/world-data-system/v-dat-working-group/iaga-2002-data-exchange-format
    whilst keeping flag values 99999.00
    Example of daily file's header and comments (#) at Florence Court
    Format                 IAGA-2002                                    |
    Source of Data         British Geological Survey (BGS)              |
    Station Name           Florence Court                               |
    IAGA code              FLO                                          |
    Geodetic Latitude      54.25                                        |
    Geodetic Longitude     352.27                                       |
    Elevation              87                                           |
    Reported               XYZF                                         |
    Sensor Orientation     XYZF                                         |
    Digital Sampling       1-second                                     |
    Data Interval Type     1-second                                     |
    Data type              variation                                    |
    # Data file created on 17-01-2026 at 01:09:59                       |
    #                                                                   |
    # This data file was created by the BGS geomagnetic data processing |
    # software running under the Linux operating system. D and I are    |
    # reported in angular units of minutes of arc, E,H,X,Y,Z and F      |
    # are reported in nanotesla. Missing data are denoted by 99999.00   |
    #                                                                   |
    # CONDITIONS OF USE: For scientific/academic studies only. Please   |
    # acknowledge BGS in any publications, conference presentations     |
    # and posters that make use of this data. For all other applications|
    # please contact the Geomagnetism team of BGS, Edinburgh.           |
    # Contact details are available from                                |
    # http://www.geomag.bgs.ac.uk/contactus/staff.html                  |
    #                                                                   |
    DATE       TIME         DOY     FLOX      FLOY      FLOZ      FLOF  |

    Parameters:
    -----------
    dir: pathlib.Path
        Path folder where daily iaga-2002 day files live

    obs: str optional
        Three observatory code.

    Returns:
    --------
    df: pandas.DataFrame
        Data are under 5 columns from iaga file.
        DATE and TIME columns set to index.
        DATE TIME DOY OBSX OBSY OBSZ OBSF
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
    df.drop(columns=["DATE", "TIME", "|"], inplace=True)

    return df


@enforce_types(obs=str, dir=pathlib.Path, df=str)
def iaga2magie_xyzf(obs, dir, infile):
    """
    This program converts IAGA data columns
    DATE       TIME         DOY     OBSX      OBSY      OBSZ      OBSF
    To MagIE data columns
    Data & Time Index# Bx By Bz Bf

    Input IAGA-2002 file format defined in
    https://www.ncei.noaa.gov/services/world-data-system/v-dat-working-group/iaga-2002-data-exchange-format                    |

    BGS provides us raw data in iaga-2002 format, we read in their data
    then save that data into file format in
    https://data.magie.ie/

    Parameters:
    -----------
    obs: str optional
        Three observatory code

    dir: pathlib.Path
        Path folder where daily iaga-2002 day files live

    infile: str
        File in from Dir e.g. FLO202609116.sec

    Dependencies:
    -------------
    Function: read_IAGA2002

    Returns:
    --------
    df: pandas.DataFrame
        Data are under 5 columns from iaga file.
        DATE and TIME columns set to index.
        DATE TIME DOY OBSX OBSY OBSZ OBSF
    """
    dir = Path(r'../Data/')
    df = read_IAGA2002(dir, infile)
    df['Index#'] = range(1, len(df) + 1)
    df.drop(columns=["DOY"], inplace=True)
    ob = obs.upper()
    old_col_names = [f"{ob}{x}" for x in ("X", "Y", "Z", "F")]
    new_col_names = ("Bx", "By", "Bz", "Bf")
    df.rename(
        columns=dict(zip(old_col_names, new_col_names)),
        inplace=True
        )
    df.index = pd.to_datetime(df.index, format="%Y/%m/%d %H:%M:%S")
    cols = ('Index#', "Bx", "By", "Bz", "Bf")
    df = df.reindex(columns=[c for c in cols if c in df.columns])

    return df


@enforce_types(
        output_dir=str,
        current_file_name=(str, pathlib.Path),
        obs=str,
        freq=str,
        flag=(int, float),
        print_msg=bool,
        )
def generate_missing_days(
        output_dir, current_file_name, obs='flo', freq='1s',
        flag=99999.00, print_msg=False
        ):
    """
    Fills missing daily variometer files between the previous file on disk
    and the latestFile.

    Parameters:
    -----------
    output_dir: str
        Path folder where daily variometer files live

    current_file_name: str
        Name to current file in the output_dir
        E.g. flo20260116.txt; if the most recent variometer data
        and daily file is generated on 16-01-2026.

    freq: str optional
        defaults to one-second Florence Court (FLO) data.
        Use "1min" for per minute freqeuncy and "1h" for per hour

    obs: str optional
        Three letters lowercase default to "flo" BGS variometer

    flag: float optional
        indicates missing data either 99999.0 or 99999.00
        depending on component.

    Raises:
    -------
    FileNotFoundError

    Returns:
    --------
    Saves day file with missing data under formatted columns
    until they are replaced with real data or remain flagged

    Usage:
    --------
        generate_missing_days(outputDir, "flo20260126.txt")
        if current day with data is 2026-01-26
    """
    output_dir = Path(output_dir)
    current_file = output_dir / current_file_name

    if not current_file.exists():
        raise FileNotFoundError(f"Latest file not found: {current_file}")

    existing_files = sorted(
        f for f in output_dir.glob(f"{obs}????????.txt")
        if f.stem[len(obs):].isdigit()
    )

    # find the most recent file before latestFile
    # no previous files then nothing to fill
    prev_files = [x for x in existing_files if x < current_file]
    if not prev_files:
        return

    prev_file = prev_files[-1]
    prev_day = pd.to_datetime(prev_file.stem[len(obs):], format="%Y%m%d")
    curr_day = pd.to_datetime(current_file.stem[len(obs):], format="%Y%m%d")

    for missing_days in pd.date_range(
        prev_day + pd.Timedelta(days=1),
        curr_day - pd.Timedelta(days=1),
        freq="D"
    ):
        fname = output_dir / f"{obs}{missing_days.strftime('%Y%m%d')}.txt"

        if not fname.exists():
            df = daily_file_template(missing_days, freq=freq, flag=flag)
            df.index = df.index.strftime("%Y/%m/%d %H:%M:%S")
            df.to_csv(fname, sep=" ", index=True, float_format="%.2f")
            if print_msg:
                print(f"Saved/updated: {fname.name} in {output_dir}")


if __name__ == '__main__':
    # Where the IAGA-2002 file is saved
    data_dir = Path(r'../Data/')
    # ASCII file in IAGA-2002 format, v stands for variometer
    file = "OBSYYYYMMDDv.sec"
    # Name the MagIE textfile after iaga2magie_xyzf() conversion
    out_name = "obsYYYYMMDD.txt"
    out_file = Path(dir, out_name)
    df_out = iaga2magie_xyzf("OBS", dir, file)
    df_out.to_csv(out_file, sep=" ", index=True, float_format="%.2f")

import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import json
import warnings
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from magie.utils import enforce_types, get_asset_path, validinput


#For loading packaged JSON resources
import importlib.resources as importlib_resources


@enforce_types()
def _load_default_site_thresholds():
    """
    Load the packaged ``site_thresholds.json`` that lives alongside this module.

    This assumes ``site_thresholds.json`` is inside the same Python package
    (e.g. src/magie/site_thresholds.json) and declared in pyproject.toml as
    package data.
    """
    pkg = __package__ or "magie"
    try:
        # Modern API (Python 3.9+)
        json_path = importlib_resources.files(pkg).joinpath("site_thresholds.json")
        with json_path.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, AttributeError):
        # Fallback for older importlib.resources which may not have files()
        try:
            with importlib_resources.open_text(pkg, "site_thresholds.json") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Default site_thresholds.json not found inside the "
                f"'{pkg}' package. Either ensure it is installed as "
                "package data or pass site_thresholds explicitly."
            ) from e



""" Core K-index calculation and smoothing functions """
# All functions are type-checked via enforce_types to catch misuse early.
# The pipeline follows: kindex (raw K) -> FMI smoothing -> spline subtraction
# -> provisional K -> smoothed/final K; helpers handle padding, streaming, and filtering.
@enforce_types(
    df=pd.DataFrame,
    reference_thresholds=(np.ndarray, list, tuple, pd.Series),
    k9=(int, float, np.number),
)
def kindex(df, reference_thresholds=np.array([0, 5, 10, 20, 40, 70, 120, 200, 330, 500]), k9=570):
    """
    Compute 3-hourly K-index values from 1-minute (or higher cadence) Bx/By data.

    The maximum range within each 3-hour window is compared against scaled
    reference thresholds to assign integer K bins.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['Bx', 'By'] with a DatetimeIndex.
    reference_thresholds : array-like
        Quiet to storm thresholds, will be scaled by ``k9/500``.
    k9 : float
        Local K9 threshold used to scale ``reference_thresholds`` (default 570).

    Returns
    -------
    pandas.DataFrame
        Columns ``max_var``, ``counts`` and ``K_index`` indexed every 3 hours.

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01', periods=6, freq='h')
    >>> df = pd.DataFrame({'Bx': [1, 3, -2, 0, 1, 2], 'By': [0, 1, 2, 1, 0, -1]}, index=idx)
    >>> kindex(df).iloc[0].K_index
    0
    """
    # Compute per-3h range of horizontal components and count of samples
    df_new= pd.DataFrame()
    df_new['max_var']= (df.loc[:, ['Bx', 'By']].resample('3h').max()-df.loc[:, ['Bx', 'By']].resample('3h').min()).max(axis=1)
    df_new['counts']= df.loc[:, ['Bx']].resample('3h').count().values
    nans= df_new.max_var.isnull()
    # Scale canonical thresholds by local K9 and digitize the observed range
    thresh = reference_thresholds * k9/500.0
    df_new['K_index']= np.digitize(df_new.max_var, thresh)-1
    df_new.loc[nans, 'K_index']= np.nan
    return df_new
@enforce_types(
    df_min=pd.DataFrame,
    df_k=pd.DataFrame,
    decimals=int,
)
def fmi_smoothed_df_vectorized(df_min, df_k, decimals=6):
    """
    Adaptive smoothing of minute-resolution magnetic field data
    using K-index and time-of-day dependent windows.

    Parameters
    ----------
    df_min : DataFrame
        Minute-resolution data with datetime index and columns ['Bx','By','Bz']
    df_k : DataFrame
        3-hourly K index data with datetime index and column ['K']
    decimals : int
        Number of decimal places to round in the final output (default=6)

    Returns
    -------
    DataFrame
        Hourly smoothed Bx, By, Bz, index at hh:30

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01', periods=120, freq='min')
    >>> df_min = pd.DataFrame({'Bx': np.sin(np.arange(len(idx))), 'By': np.cos(np.arange(len(idx)))}, index=idx)
    >>> df_k = kindex(df_min[['Bx', 'By']])
    >>> fmi_smoothed_df_vectorized(df_min, df_k).head(1).columns.tolist()
    ['Bx', 'By', 'Bz']
    """

    # Ensure time order for both series
    df_min = df_min.sort_index()
    df_k = df_k.sort_index()

    if df_min.empty:
        return pd.DataFrame(columns=['Bx', 'By', 'Bz'])

    # --- 1. extra_time pattern (minutes) ---
    extra_time = np.array([
        120, 120, 120,  60,  60,  60,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,  60,  60,  60, 120, 120, 120
    ])

    # --- 2. Hour grid + mid-times ---
    start_day = df_min.index[0].normalize()
    end_time = df_min.index[-1]

    hours = pd.date_range(start=start_day, end=end_time, freq='h')
    if hours.empty:
        return pd.DataFrame(columns=['Bx', 'By', 'Bz'])

    mid_times = hours + pd.Timedelta(minutes=30)

    # --- 3. Expand K to hourly ---
    k_hourly = df_k['K_index'].reindex(hours, method='ffill')
    k_hourly = k_hourly.fillna(0.0)
    n = k_hourly.pow(3.3).to_numpy()  # legacy non-linear scaling of window length

    # --- 4. time-of-day adjustment ---
    m = extra_time[mid_times.hour.to_numpy()]  # minutes

    # half-window size
    half_window_minutes = 30.0 + n + m  # base 30 min plus storm- and time-weighting
    half_window = pd.to_timedelta(half_window_minutes, unit='m')

    window_start = mid_times - half_window
    window_end   = mid_times + half_window

    # --- 5. cumulative sums over minute data ---
    minute_index = df_min.index.view('int64')  # nanoseconds

    bx = df_min['Bx'].to_numpy()
    by = df_min['By'].to_numpy()
    bz = df_min['Bz'].to_numpy() if 'Bz' in df_min.columns else np.zeros_like(bx)

    # Treat NaNs as zeros for the purpose of SUMS (legacy code had no NaNs)
    bx_vals = np.nan_to_num(bx, nan=0.0)
    by_vals = np.nan_to_num(by, nan=0.0)
    bz_vals = np.nan_to_num(bz, nan=0.0)

    # Cumulative sums for sums
    cBx = np.concatenate(([0.0], np.cumsum(bx_vals)))
    cBy = np.concatenate(([0.0], np.cumsum(by_vals)))
    cBz = np.concatenate(([0.0], np.cumsum(bz_vals)))

    # --- 6. searchsorted for window boundaries ---
    start_ns = window_start.view('int64')
    end_ns   = window_end.view('int64')

    start_pos = np.searchsorted(minute_index, start_ns, side='left')
    end_pos   = np.searchsorted(minute_index, end_ns,   side='right')

    # SUMS from cumulative arrays
    sumBx = cBx[end_pos] - cBx[start_pos]
    sumBy = cBy[end_pos] - cBy[start_pos]
    sumBz = cBz[end_pos] - cBz[start_pos]

    # --- 7. counts that match legacy semantics ---
    # Valid sample = Bx and By both non-NaN (legacy code had no NaNs)
    valid = df_min[['Bx', 'By']].notna().all(axis=1).to_numpy()

    valid_int = valid.astype(np.int64)
    cCount = np.concatenate(([0], np.cumsum(valid_int)))

    # counts = number of valid samples inside window
    counts = (cCount[end_pos] - cCount[start_pos]).astype(float)

    # Means
    with np.errstate(invalid='ignore', divide='ignore'):
        meanBx = sumBx / counts
        meanBy = sumBy / counts
        meanBz = sumBz / counts


    # --- 7. Build result + drop empty windows ---
    result = pd.DataFrame({
        'Bx': meanBx,
        'By': meanBy,
        'Bz': meanBz,
    }, index=mid_times)

    # --- 8. Final rounding (NEW) ---
    result = result.round(decimals)

    return result


@enforce_types(
    df_original=pd.DataFrame,
    df_fmi_smoothed=pd.DataFrame,
    order=int,
    oversample_factor=int,
)
def spline_subtract(df_original, df_fmi_smoothed, order=3, oversample_factor=10):
    """
    Reproduce the original smoothed() + subtracted() behaviour:

    1. For each of Bx, By:
       - Fit a spline on the FMI-smoothed hourly series.
       - Evaluate spline on a dense uniform grid.
       - Linearly interpolate that dense curve to minute times.
       - Subtract from original minute data.

    Parameters
    ----------
    df_original : DataFrame
        Minute data with DatetimeIndex and columns ['Bx','By'].
    df_fmi_smoothed : DataFrame
        FMI-smoothed data (hourly) with DatetimeIndex and ['Bx','By'].
    order : int
        Spline order k (as in original; default 3).
    oversample_factor : int
        How much denser than the hourly grid to sample (original used 10).

    Returns
    -------
    DataFrame
        Residuals at minute resolution, columns ['Bx','By'].

    Examples
    --------
    >>> t = pd.date_range('2024-01-01', periods=60, freq='min')
    >>> df_orig = pd.DataFrame({'Bx': np.sin(np.linspace(0, 2*np.pi, 60)), 'By': 0}, index=t)
    >>> hourly = df_orig.resample('h').mean()
    >>> residuals = spline_subtract(df_orig, hourly)
    >>> residuals.shape[0]
    60
    """

    # Align both inputs chronologically
    df_original = df_original.sort_index()
    df_fmi_smoothed = df_fmi_smoothed.sort_index()

    # Use a local time scale: seconds from the start of the original series
    t0_ns = df_original.index[0].value
    t_min = (df_original.index.view('int64') - t0_ns) / 1e9   # seconds
    t_smooth = (df_fmi_smoothed.index.view('int64') - t0_ns) / 1e9

    result = pd.DataFrame(index=df_original.index)

    for col in ['Bx', 'By']:
        y = df_fmi_smoothed[col].to_numpy()

        # Drop NaNs from the smoothed series before building the spline
        mask = ~np.isnan(y)
        x_clean = t_smooth[mask]
        y_clean = y[mask]

        # If not enough points for a spline, just return original (no subtraction)
        if len(x_clean) <= order:
            result[col] = df_original[col]
            continue

        # Build spline on the clean FMI-smoothed curve
        spline = InterpolatedUnivariateSpline(x_clean, y_clean, k=order)

        # Dense grid xi as in original smoothed(): linspace over [x0, xN]
        xi = np.linspace(x_clean[0], x_clean[-1], oversample_factor * len(x_clean))
        yi = spline(xi)

        # Now linearly interpolate this dense smooth curve to minute times
        smooth_at_minute = np.interp(t_min, xi, yi)

        # Subtract smoothed from original to get residual
        result[col] = df_original[col].to_numpy() - smooth_at_minute

    return result


""""Helper functions for K-index calculation and file handling"""
@enforce_types(
    df_site=pd.DataFrame,
    pad_freq=str,
)
def data_padding(
    df_site,
    block_start=None,
    block_end=None,
    pad_freq='1min',
):
    """
    Pad a single-site time series to a regular grid before K-index calculation.

    Parameters
    ----------
    df_site : pandas.DataFrame
        Input data with a DatetimeIndex and at least ['Bx', 'By']; no 'Site' column.
    block_start : pandas.Timestamp, optional
        Inclusive start of the padded grid. Defaults to the earliest timestamp in df_site.
    block_end : pandas.Timestamp, optional
        Exclusive end of the padded grid. Defaults to one minute past the latest timestamp.
    pad_freq : str
        Pandas offset string for the desired grid (default '1min').

    Returns
    -------
    pandas.DataFrame
        Reindexed DataFrame on the regular grid with gaps filled by NaN.

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01 00:00', periods=3, freq='2min')
    >>> padded = data_padding(pd.DataFrame({'Bx': [1, 2, 3], 'By': [0, 0, 0]}, index=idx))
    >>> len(padded)
    5
    """

    df_site = df_site.sort_index()  # enforce chronological order
    if not isinstance(df_site.index, pd.DatetimeIndex):
        raise ValueError("df_site must have a DatetimeIndex")

    if block_start is None:
        block_start = df_site.index.min()  # earliest sample
    if block_end is None:
        # end is exclusive; make it one minute beyond last sample
        block_end = df_site.index.max() + pd.Timedelta(minutes=1)

    # Full grid [block_start, block_end) at pad_freq
    full_index = pd.date_range(
        start=block_start,
        end=block_end - pd.Timedelta(minutes=1),
        freq=pad_freq,
    )

    # Reindex inserts NaNs for gaps; downstream functions handle them
    df_padded = df_site.reindex(full_index)
    # Preserve the index name if it was set before
    df_padded.index.name = df_site.index.name
    return df_padded

@enforce_types(
    file=str,
    site_thresholds=(dict, int, float, np.number),
    inkey=str,
    outfile=str,
    outkey=str,
    hr3_chunks=int,
    site_code=str,
    use_mag_filter=bool,
)
def _provisional_kindexhdf(
        file,
        site_thresholds,
        inkey='main',
        outfile='./Provisional_K_index.hdf5',
        outkey='main',
        hr3_chunks=1000,
        site_code='dun',
        use_mag_filter=True,
        **kindex_kwargs,
):
    """
    Compute provisional K from an HDF5 file in time chunks of hr3_chunks*3 hours,
    writing results to `outfile`/`outkey`.

    Works whether `file` and `outfile` are the same or different.

    Parameters
    ----------
    file : str
        Path to the input HDF5 file containing magnetic field data.
    site_thresholds : dict or scalar
        Mapping of site code to k9 threshold, or single k9 value for all sites.
    inkey : str
        HDF5 key to read input data from.
    outfile : str
        HDF5 file path to write provisional K results.
    outkey : str
        HDF5 key under which results are stored.
    hr3_chunks : int
        Number of 3-hour bins per processing chunk (controls memory use).
    site_code : str
        Site code used in single-site files without a 'Site' column.
    use_mag_filter : bool, optional
        If True (default), apply the legacy mag_filter to raw data
        BEFORE resampling to 1-minute:
          * For multi-site data (Site column present), apply per site,
            skipping Valentia (site codes starting with 'val').
          * For single-site data, controlled by `site_code`, skip if
            `site_code` starts with 'val'.
    **kindex_kwargs :
        Extra arguments forwarded to ``kindex`` (e.g., custom thresholds).

    Returns
    -------
    str
        Path to the outfile written.
    """

    # Work out time range and chunk boundaries
    data_min, data_max = _get_time_range(file, inkey, date_col='Date_UTC')
    start_np = np.datetime64(data_min)
    end_np   = np.datetime64(data_max + np.timedelta64(1, 'h'))
    step     = np.timedelta64(hr3_chunks * 3, 'h')

    time_chunks = np.arange(start_np, end_np, step)

    in_path  = os.path.abspath(file)
    out_path = os.path.abspath(outfile)
    same_io  = (in_path == out_path)  # True when reading and writing same HDF

    # Make sure we don't mutate caller's kwargs
    kindex_kwargs = kindex_kwargs.copy()

    # Check existing outkey
    if os.path.isfile(out_path):
        with pd.HDFStore(out_path, mode='a') as store:
            keys = [k.lstrip('/') for k in store.keys()]
            if outkey in keys:
                if not validinput(
                    f"File {outfile} already has key '{outkey}'. "
                    f"Append to pre-existing key? (y/n)",
                    'y', 'n'
                ):
                    raise FileExistsError(
                        'Please either provide a new save file or key path using "outfile" or "outkey" argument '
                        'or delete existing key and rerun.'
                    )
                else:
                    warnings.warn(
                        f'Appending to pre-existing file and key: {outfile} {outkey}',
                        UserWarning
                    )

    # Open outfile store once for the whole loop
    with pd.HDFStore(out_path, mode='a') as out_store:
        for start in tqdm(time_chunks,
                          desc="Processing Provisional K chunks",
                          total=len(time_chunks)):

            end  = start + step
            bs   = pd.Timestamp(start)
            be   = pd.Timestamp(end)

            where = [
                f"Date_UTC >= '{bs.isoformat()}'",
                f"Date_UTC < '{be.isoformat()}'",
            ]

            # Read chunk either from the same store (if same file) or via pd.read_hdf
            if same_io:
                chunk = out_store.select(inkey, where=where)
            else:
                chunk = pd.read_hdf(file, inkey, where=where)

            if chunk.empty:
                continue

            if 'Date_UTC' in chunk.columns:
                chunk.set_index('Date_UTC', inplace=True)

            chunk = chunk.dropna(subset=['Bx', 'By'])
            if chunk.empty:
                continue

            # ---- Multi-site case ----
            if 'Site' in chunk.columns:
                sites = np.unique(chunk.Site.values)
                dfs = []

                for site_code_local in sites:
                    site_str = str(site_code_local)
                    site3    = site_str[:3].lower()

                    # set k9
                    if isinstance(site_thresholds, dict):
                        if site_code_local not in site_thresholds:
                            raise KeyError(f"Site '{site_code_local}' not in site_thresholds")
                        kindex_kwargs['k9'] = site_thresholds[site_code_local]
                    else:
                        kindex_kwargs['k9'] = site_thresholds

                    # extract raw per-site data
                    # Extract raw per-site data
                    df_site = chunk.loc[chunk.Site == site_code_local].drop(columns=['Site']).copy()

                    # Apply mag_filter on raw (1-sec) data, skip Valentia
                    if use_mag_filter and not site3.startswith('val'):
                        df_site = mag_filter(df_site)

                    # 1-min padding + K
                    # Pad to 1-minute grid and compute provisional K
                    df_site = data_padding(
                        df_site.resample('1min').mean(),
                        block_start=bs,
                        block_end=be
                    )
                    df_site = kindex(df_site, **kindex_kwargs)
                    df_site['Site'] = site_code_local

                    dfs.append(df_site)

                if not dfs:
                    continue

                df_out = pd.concat(dfs)

            # ---- Single-site case ----
            else:
                if isinstance(site_thresholds, dict):
                    if site_code not in site_thresholds:
                        raise KeyError(f"Site '{site_code}' not in site_thresholds")
                    kindex_kwargs['k9'] = site_thresholds[site_code]
                else:
                    kindex_kwargs['k9'] = site_thresholds

                df_chunk = chunk.copy()

                # Apply mag_filter on raw single-site data, skip Valentia
                if use_mag_filter and not str(site_code).lower().startswith('val'):
                    df_chunk = mag_filter(df_chunk)

                df_chunk = data_padding(
                    df_chunk.resample('1min').mean(),
                    block_start=bs,
                    block_end=be
                )
                df_out = kindex(df_chunk, **kindex_kwargs)
                df_out['Site'] = site_code[:3].lower()

            df_out.reset_index(inplace=True)
            df_out.sort_values('Date_UTC', inplace=True)

            out_store.append(outkey, df_out, format='t', data_columns=True)

    return outfile


@enforce_types(
    df_or_path=(pd.DataFrame, str),
    reference_thresholds=(np.ndarray, list, tuple, pd.Series),
    site_thresholds=(dict, str, type(None)),
    site_code=str,
    use_mag_filter=bool,
)
def provisional_k(
        df_or_path,
        reference_thresholds=np.array([0, 5, 10, 20, 40, 70, 120, 200, 330, 500]),
        site_thresholds=None,
        site_code='dun',
        use_mag_filter=True,
        **kindex_hdf_kwargs
):
    """
    Compute preliminary K for a DataFrame or a small file.

    Parameters
    ----------
    df_or_path : DataFrame or str
        DataFrame with Bx/By(/Bz) and DatetimeIndex, or path to
        .hdf5/.h5/.csv/.txt.
    site_thresholds : dict or str or None
        - dict: {site_code: k9} mapping (e.g. {'dun': 570, 'arm': 630})
        - str : path to a JSON file containing such a dict
        - None: load the packaged ``site_thresholds.json`` that is shipped
          with the magie package.
    site_code : str
        Used in single-site mode when there is no 'Site' column.
    use_mag_filter : bool, optional
        If True (default), apply mag_filter to raw data BEFORE resampling
        to 1-minute:
            * Multi-site: per-site, skipping Valentia (site_code startswith 'val').
            * Single-site: skip if `site_code` startswith 'val'.

        For purely synthetic or already-minute-averaged data, set
        `use_mag_filter=False` to avoid over-filtering.
    """
    # --- Normalise site_thresholds (None / str / dict) ---
    if site_thresholds is None:
        # Load packaged default JSON
        site_thresholds = _load_default_site_thresholds()
    elif isinstance(site_thresholds, str):
        # User-supplied path to a JSON file
        with open(site_thresholds, 'r') as f:
            site_thresholds = json.load(f)

    # If it's a path, load DataFrame
    if isinstance(df_or_path, str):
        path = df_or_path
        ext = os.path.splitext(path)[1].lower()

        if ext in ('.hdf5', '.h5'):
            # Delegate to HDF-streaming version (which also supports use_mag_filter)
            return _provisional_kindexhdf(
                df_or_path,
                reference_thresholds=reference_thresholds,
                site_thresholds=site_thresholds,
                site_code=site_code,
                use_mag_filter=use_mag_filter,
                **kindex_hdf_kwargs,
            )
        elif ext in ('.csv', '.txt'):
            df = pd.read_csv(path, parse_dates=['Date_UTC'])
            df.set_index('Date_UTC', inplace=True)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        df = df_or_path.copy()
        if 'Date_UTC' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['Date_UTC'] = pd.to_datetime(df['Date_UTC'])
            df.set_index('Date_UTC', inplace=True)

    # Basic cleaning
    df.dropna(subset=['Bx', 'By'], inplace=True)
    if df.empty:
        return pd.DataFrame(columns=['Date_UTC', 'K_index'])

    # Optional mag_filter on raw cadence data to suppress spikes
    if use_mag_filter:
        if 'Site' in df.columns:
            # per-site, skip Valentia
            filtered = []
            for sc, grp in df.groupby('Site'):
                sc_str = str(sc).lower()
                if sc_str.startswith('val'):
                    filtered.append(grp)
                else:
                    filtered.append(mag_filter(grp))
            df = pd.concat(filtered).sort_index()
        else:
            # single-site, skip if site_code is Valentia
            if not str(site_code).lower().startswith('val'):
                df = mag_filter(df)

    # ---- Multi-site case ----
    if 'Site' in df.columns:
        site_codes = np.unique(df['Site'].values)
        results = []

        for sc in site_codes:
            sc_str = str(sc)
            if sc_str not in site_thresholds:
                raise KeyError(f"Site '{sc_str}' not found in site_thresholds")

            k9 = site_thresholds[sc_str]
            bs = df.index.min().normalize()
            be = df.index.max().normalize() + pd.Timedelta(days=1)

            df_site = df.loc[df['Site'] == sc].drop(columns=['Site'])

            k_df = kindex(
                data_padding(
                    df_site.resample('1min').mean(),
                    block_start=bs,
                    block_end=be
                ),
                reference_thresholds=reference_thresholds,
                k9=k9,
            )
            k_df['Site'] = sc_str
            results.append(k_df)

        return pd.concat(results).sort_index().reset_index()

    # ---- Single-site case ----
    if isinstance(site_thresholds, dict):
        if site_code not in site_thresholds:
            raise KeyError(f"Site '{site_code}' not found in site_thresholds")
        k9 = site_thresholds[site_code]
    else:
        k9 = site_thresholds

    bs = df.index.min().normalize()
    be = df.index.max().normalize() + pd.Timedelta(days=1)

    return kindex(
        data_padding(
            df.resample('1min').mean(),
            block_start=bs,
            block_end=be,
        ),
        reference_thresholds=reference_thresholds,
        k9=k9,
    ).reset_index()



@enforce_types(
    hdf_path=str,
    key=str,
    date_col=str,
)
def _get_time_range(hdf_path, key, date_col='Date_UTC'):
    """
    Return the normalized min/max timestamps for a given HDF5 key.

    Parameters
    ----------
    hdf_path : str
        Path to the HDF5 file.
    key : str
        Dataset key to inspect.
    date_col : str
        Column name holding timestamps if not using the index (default 'Date_UTC').

    Returns
    -------
    tuple
        (t_min, t_max_plus_day) as pandas.Timestamp objects, normalized to day boundaries.

    Notes
    -----
    Assumes the HDF table is sorted by ``date_col`` or index.
    """
    with pd.HDFStore(hdf_path, mode='r') as store:
        storer = store.get_storer(key)
        nrows = storer.nrows

        first = store.select(key, start=0, stop=1)
        last = store.select(key, start=nrows-1, stop=nrows)

    if date_col in first.columns:
        t_min = pd.to_datetime(first[date_col].iloc[0])
        t_max = pd.to_datetime(last[date_col].iloc[0])
    else:
        t_min = pd.to_datetime(first.index[0])
        t_max = pd.to_datetime(last.index[0])

    return t_min.normalize(), t_max.normalize() + pd.Timedelta(days=1)
@enforce_types(
    data_file=str,
    k_file=(str, bool),
    inkey=str,
    kinkey=str,
    outfile=str,
    outkey=str,
    four_day_multiple=int,
    date_col=str,
    provisional_k_kwargs=(dict, type(None)),
)
def _stream_finalised_k_to_hdf(
        data_file,
        k_file=False,
        inkey='main',
        kinkey='main',
        outfile='./Finalised_K_index.hdf5',
        outkey='main',
        four_day_multiple=1,         # 1 -> 4 days, 4 -> 16 days, etc.
        date_col='Date_UTC',
        provisional_k_kwargs=None,
        **K_kwargs,
):
    """
    Stream over large datasets to compute finalised K in rolling 4-day windows.

    Parameters
    ----------
    data_file : str
        HDF5 file containing magnetic field data.
    k_file : str or bool
        HDF5 file containing provisional K; if False, it will be generated on the fly.
    inkey : str
        Key for magnetic data inside ``data_file``.
    kinkey : str
        Key for provisional K inside ``k_file``.
    outfile : str
        Destination HDF5 path for finalised K output.
    outkey : str
        Key under which finalised K is appended in ``outfile``.
    four_day_multiple : int
        Multiplier for 4-day processing blocks (1 -> 4 days, 2 -> 8 days, etc.).
    date_col : str
        Name of the datetime column if not using the index.
    provisional_k_kwargs : dict, optional
        Arguments to pass to ``_provisional_kindexhdf`` when building k_file.
    **K_kwargs :
        Forwarded to ``_run_finalised_k_pipeline`` (e.g., site thresholds).

    Returns
    -------
    str
        Path to the outfile with appended finalised K results.

    Examples
    --------
    >>> # For quick checks, pass small demo files; function streams large inputs.
    >>> _stream_finalised_k_to_hdf('demo.h5', k_file='demo.h5', inkey='main', kinkey='k', outfile='out.h5')  # doctest: +SKIP
    """
    if provisional_k_kwargs is None:
        provisional_k_kwargs = {}

    # If no k_file provided, build one from data_file
    if not k_file:
        k_file = _provisional_kindexhdf(data_file, **provisional_k_kwargs)

    # Window / block logic
    W = pd.Timedelta(days=4)
    L = W * four_day_multiple
    block_step = L - W + pd.Timedelta(days=1)

    # Time ranges
    data_min, data_max = _get_time_range(data_file, inkey, date_col=date_col)
    k_min, k_max = _get_time_range(k_file, kinkey, date_col=date_col)

    global_start = max(data_min, k_min)
    global_end = min(data_max, k_max)

    if global_start + W > global_end:
        raise ValueError("Not enough overlapping data for at least one 4-day window.")

    # Precompute block start times for progress bar
    block_starts = []
    current = global_start
    while current + W <= global_end:
        block_starts.append(current)
        current += block_step

    if not block_starts:
        return outfile

    # Normalise paths
    data_path = os.path.abspath(data_file)
    k_path    = os.path.abspath(k_file)
    out_path  = os.path.abspath(outfile)

    same_data_out = (data_path == out_path)
    same_k_out    = (k_path == out_path)

    # Check/ask about existing outkey (if outfile already exists)
    if os.path.isfile(out_path):
        with pd.HDFStore(out_path, mode='a') as store:
            keys = [k.lstrip('/') for k in store.keys()]
            if outkey in keys:
                if not validinput(
                    f"File {outfile} already has key '{outkey}'. "
                    f"Append to pre-existing key? (y/n)",
                    'y', 'n'
                ):
                    raise FileExistsError(
                        "Please either provide a new save file or key path using "
                        "'outfile' or 'outkey' arguments, or delete the existing "
                        "key and rerun."
                    )
                else:
                    warnings.warn(
                        f'Appending to pre-existing file and key: {outfile} {outkey}',
                        UserWarning
                    )
    # Re-open outfile store for the full processing loop
    with pd.HDFStore(out_path, mode='a') as out_store:
        for block_start in tqdm(
            block_starts,
            desc="Processing Finalised K blocks",
            total=len(block_starts)
        ):
            block_end = min(block_start + L, global_end)

            bs = pd.Timestamp(block_start)
            be = pd.Timestamp(block_end)

            # HDF time-based query (assumes date_col is a data column)
            where = [
                f"{date_col} >= '{bs.isoformat()}'",
                f"{date_col} <= '{be.isoformat()}'",
            ]

            # --- Read data_block ---
            if same_data_out:
                # data_file is the same as outfile
                data_block = out_store.select(inkey, where=where)
            else:
                data_block = pd.read_hdf(data_file, inkey, where=where)

            # --- Read k_block ---
            if same_k_out:
                # k_file is the same as outfile
                k_block = out_store.select(kinkey, where=where)
            else:
                k_block = pd.read_hdf(k_file, kinkey, where=where)

            if data_block.empty or k_block.empty:
                continue

            # Ensure datetime index
            if date_col in data_block.columns:
                data_block[date_col] = pd.to_datetime(data_block[date_col])
                data_block = data_block.set_index(date_col)
            if date_col in k_block.columns:
                k_block[date_col] = pd.to_datetime(k_block[date_col])
                k_block = k_block.set_index(date_col)

            data_block = data_block.sort_index()
            k_block = k_block.sort_index()


            # Let _run_finalised_k_pipeline do the rolling + "day 3" selection
            block_result = _run_finalised_k_pipeline(
                data_block,
                data_k=k_block,
                **K_kwargs,
            )
            if not block_result.empty:
                out_store.append(outkey, block_result,
                                 format='t', data_columns=True)

    return outfile

@enforce_types(
    data=pd.DataFrame,
    data_k=(pd.DataFrame, type(None)),
    window_days=(int, pd.Timedelta),
    step_days=(int, pd.Timedelta),
    keep_day_offset=int,
    n_jobs=int,
    provisional_k_kwargs=(dict, type(None)),
)
def _run_finalised_k_pipeline(
        data,
        data_k=None,
        window_days=4,
        step_days=1,
        keep_day_offset=2,        # 0=day1, 1=day2, 2=day3
        n_jobs=1,
        provisional_k_kwargs=None,
        **K_kwargs,
):
    """
    Apply smooth_kindex on rolling 4-day windows (by default) with 1-day steps,
    and return ONLY the 'day 3' part of each smoothed window,
    concatenated over all windows.

    Parameters
    ----------
    data : pandas.DataFrame
        Magnetic data with a DatetimeIndex (and optional 'Site' column).
    data_k : pandas.DataFrame or None
        Provisional K aligned to ``data``; if None, computed via ``provisional_k``.
    window_days : int or pandas.Timedelta
        Length of each rolling window (default 4 days).
    step_days : int or pandas.Timedelta
        Step between window starts (default 1 day).
    keep_day_offset : int
        Which day of each window to keep (0-based; default 2 keeps day 3).
    n_jobs : int
        Number of worker processes (1 runs sequentially).
    provisional_k_kwargs : dict, optional
        Extra args passed to ``provisional_k`` when ``data_k`` is None.
    **K_kwargs :
        Forwarded to ``smooth_kindex`` / ``kindex`` (e.g., site thresholds).

    Returns
    -------
    pandas.DataFrame
        Concatenated finalised K over all kept-day slices, sorted by time.

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01', periods=24*4*2, freq='h')
    >>> df = pd.DataFrame({'Bx': np.sin(np.arange(len(idx))), 'By': np.cos(np.arange(len(idx)))}, index=idx)
    >>> provisional = provisional_k(df, site_thresholds={'dun': 570}, site_code='dun')
    >>> _run_finalised_k_pipeline(df, provisional, keep_day_offset=1, site_code='dun').empty
    False
    """

    if provisional_k_kwargs is None:
        provisional_k_kwargs = {}
    if 'Date_UTC' in data.columns:
        data.set_index('Date_UTC', inplace=True)
    if not 'site_code' in K_kwargs:
        K_kwargs.setdefault('site_code', 'dun')  # default site for single-site flows
    # Normalise data/index
    data = data.sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("`data` must have a DatetimeIndex")
    if data.empty:
        return data_k.iloc[0:0]  # empty result with same type
    # Compute provisional K if none provided
    if data_k is None:
        data_k = provisional_k(data, **provisional_k_kwargs)
    if 'Date_UTC' in data_k.columns:
        data_k.set_index('Date_UTC', inplace=True)
    data_k = data_k.sort_index()

    # Convert window/step to Timedelta
    if not isinstance(window_days, pd.Timedelta):
        window_days = pd.Timedelta(days=window_days)
    if not isinstance(step_days, pd.Timedelta):
        step_days = pd.Timedelta(days=step_days)

    start = data.index.min().normalize()
    end = data.index.max().normalize()+pd.Timedelta(1, 'D')

    # Prepare window start times
    window_starts = []
    current = start
    while current + window_days < end:
        window_starts.append(current)
        current += step_days

    if not window_starts:
        return data_k.iloc[0:0]

    # Helper for a single window (top-level-ish for pickling)
    def _process_one_window(w_start):
        """
        Process a single rolling window: smooth K and extract the kept day slice.

        Parameters
        ----------
        w_start : pandas.Timestamp
            Window start time; window_end is derived via ``window_days``.

        Returns
        -------
        pandas.DataFrame or None
            Finalised K for the kept day of this window, or None if empty.
        """
        w_end = w_start + window_days

        d_win = data.loc[(data.index >= w_start) & (data.index < w_end)]
        k_win = data_k.loc[(data_k.index >= w_start) & (data_k.index < w_end)]

        if d_win.empty or k_win.empty:
            return None
        
        # Smooth K on this window
        k_smoothed = smooth_kindex(d_win, k_win, block_start=w_start, block_end= w_end, **K_kwargs)

        # # Select "day 3" (or whatever offset) of this window
        keep_start = w_start + keep_day_offset * step_days
        keep_end = keep_start + step_days

        mask = (k_smoothed.index >= keep_start) & (k_smoothed.index < keep_end)
        out = k_smoothed.loc[mask]
        if out.empty:
            return None
        return out

    results = []

    if n_jobs == 1 or len(window_starts) == 1:
        # Sequential
        for ws in window_starts:
            out = _process_one_window(ws)
            if out is not None:
                results.append(out)
    else:
        # Parallel; note: data & data_k must be pickled to workers
        # so don't make windows huge *and* n_jobs huge.
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_process_one_window, ws) for ws in window_starts]
            for fut in futures:
                out = fut.result()
                if out is not None:
                    results.append(out)

    if not results:
        return data_k.iloc[0:0]

    return pd.concat(results).sort_index()

@enforce_types(
    df_or_path=(pd.DataFrame, str),
    k_data_or_path=(pd.DataFrame, str, type(None)),
    reference_thresholds=(np.ndarray, list, tuple, pd.Series),
    site_thresholds=(dict, str, type(None)),
    site_code=(str, type(None)),
    provisional_kwargs=(dict, type(None)),
    final_kwargs=(dict, type(None)),
    use_mag_filter=bool,
)
def finalised_k(
        df_or_path,
        k_data_or_path=None,
        reference_thresholds=np.array([0, 5, 10, 20, 40, 70, 120, 200, 330, 500]),
        site_thresholds=None,
        site_code='dun',
        provisional_kwargs=None,
        final_kwargs=None,
        use_mag_filter=True,
):
    """
    Compute final (smoothed) K-index for a DataFrame or a file.

    Parameters
    ----------
    site_thresholds : dict or str or None
        - dict: {site_code: k9} mapping
        - str : path to JSON with such a mapping
        - None: load the packaged ``site_thresholds.json``.
    use_mag_filter : bool, optional
        If True (default), mag_filter is applied in the provisional
        step on raw data (see provisional_k). Set to False for synthetic
        or already-minute-averaged tests.
    """

    # --- Normalise site_thresholds (None / str / dict) ---
    if site_thresholds is None:
        site_thresholds = _load_default_site_thresholds()
    elif isinstance(site_thresholds, str):
        with open(site_thresholds, 'r') as f:
            site_thresholds = json.load(f)

    provisional_kwargs = provisional_kwargs or {}
    final_kwargs = final_kwargs or {}

    # Ensure thresholds and filter flag go to BOTH stages:
    provisional_kwargs.setdefault('reference_thresholds', reference_thresholds)
    provisional_kwargs.setdefault('site_thresholds', site_thresholds)
    provisional_kwargs.setdefault('use_mag_filter', use_mag_filter)
    if site_code is not None:
        provisional_kwargs.setdefault('site_code', site_code)

    final_kwargs.setdefault('reference_thresholds', reference_thresholds)
    final_kwargs.setdefault('site_thresholds', site_thresholds)
    final_kwargs.setdefault('use_mag_filter', use_mag_filter)
    if site_code is not None:
        final_kwargs.setdefault('site_code', site_code)

    # If it's a path, load DataFrame
    if isinstance(df_or_path, str):
        path = df_or_path
        ext = os.path.splitext(path)[1].lower()

        if k_data_or_path is None:
            k_data_or_path = provisional_k(df_or_path, **provisional_kwargs)

        if ext in ('.hdf5', '.h5'):
            return _stream_finalised_k_to_hdf(
                df_or_path,
                k_data_or_path,
                **final_kwargs
            )
        elif ext in ('.csv', '.txt'):
            df = pd.read_csv(path, parse_dates=['Date_UTC'])
            df.set_index('Date_UTC', inplace=True)
            k_df = pd.read_csv(k_data_or_path, parse_dates=['Date_UTC'])
            k_df.set_index('Date_UTC', inplace=True)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        df = df_or_path.copy()
        if k_data_or_path is None:
            k_df = provisional_k(df, **provisional_kwargs)
        else:
            k_df = k_data_or_path.copy()

        if 'Date_UTC' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['Date_UTC'] = pd.to_datetime(df['Date_UTC'])
            df.set_index('Date_UTC', inplace=True)
        if 'Date_UTC' in k_df.columns and not isinstance(k_df.index, pd.DatetimeIndex):
            k_df['Date_UTC'] = pd.to_datetime(k_df['Date_UTC'])
            k_df.set_index('Date_UTC', inplace=True)

    # Single site, site name provided in argument
    if isinstance(site_thresholds, dict):
        if site_code not in site_thresholds:
            raise KeyError(f"Site '{site_code}' not found in site_thresholds")

    return _run_finalised_k_pipeline(df, k_df, **final_kwargs)

@enforce_types(
    df=pd.DataFrame,
    df_k=pd.DataFrame,
    block_start=(pd.Timestamp, type(None)),
    block_end=(pd.Timestamp, type(None)),
)
def smooth_kindex(df, df_k, block_start=None, block_end=None, use_mag_filter=True, **K_kwargs):
    """
    Apply the FMI smoothing + spline subtraction twice and recompute K-index.

    Handles both single-site and multi-site dataframes, padding to full minute
    grids before smoothing and using per-site thresholds when provided.

    Parameters
    ----------
    df : pandas.DataFrame
        Magnetic data with columns ['Bx', 'By'] and optional 'Site'.
    df_k : pandas.DataFrame
        Provisional K values aligned to ``df``.
    block_start, block_end : Timestamp, optional
        Bounds used for padding when constructing minute grids.
    use_mag_filter : bool
        If True, apply ``mag_filter`` to raw data.
    **K_kwargs :
        Passed through to ``kindex`` (e.g., ``k9`` or ``site_code``).

    Returns
    -------
    pandas.DataFrame
        Smoothed K-index values aligned to the padded timeline.

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01', periods=120, freq='min')
    >>> df = pd.DataFrame({'Bx': np.sin(np.arange(len(idx))), 'By': np.cos(np.arange(len(idx)))}, index=idx)
    >>> df_k = provisional_k(df, site_thresholds={'dun': 570}, site_code='dun')
    >>> smooth_kindex(df, df_k, site_thresholds={'dun': 570}, site_code='dun').empty
    False
    """
    df= df.copy()
    if 'Date_UTC' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['Date_UTC'] = pd.to_datetime(df['Date_UTC'])
        df.set_index('Date_UTC', inplace=True)
    if 'Date_UTC' in df_k.columns and not isinstance(df_k.index, pd.DatetimeIndex):
        df_k['Date_UTC'] = pd.to_datetime(df_k['Date_UTC'])
        df_k.set_index('Date_UTC', inplace=True)
    K_kwargs= K_kwargs.copy()
    df_K= []
    if 'site_thresholds' in K_kwargs:
        site_thresholds= K_kwargs.pop('site_thresholds')
    else:
        site_thresholds= False
    if 'Site' in df.columns:
        K_kwargs.pop('site_code')
        for site in df.Site.unique():
            if use_mag_filter and site!='val':
                df_site= mag_filter(df.loc[df.Site==site].drop(columns=['Site']))
            else:
                df_site= df.loc[df.Site==site].drop(columns=['Site'])
            df_site = data_padding(df_site.\
                                   dropna(subset=['Bx', 'By']).resample('1min').mean().copy(),
                block_start=block_start,
                block_end=block_end,
                pad_freq='1min',
            )
            df_k_site= df_k.loc[df_k.Site==site].drop(columns=['Site'])
            if site_thresholds and isinstance(site_thresholds, dict):
                K_kwargs.update({'k9':site_thresholds[site]})
            elif site_thresholds:
                K_kwargs.update({'k9':site_thresholds})
            for i in range(2):
                df_sub= fmi_smoothed_df_vectorized(df_site, df_k_site)
                df_sub= spline_subtract(df_site, df_sub, 3)
                df_k_site= kindex(df_sub, **K_kwargs)
            df_K.append(df_k_site)
    else:
        if site_thresholds and isinstance(site_thresholds, dict):
            site= K_kwargs.pop('site_code')
            K_kwargs.update({'k9':site_thresholds[site]})
        elif site_thresholds:
            K_kwargs.update({'k9':site_thresholds})
        if use_mag_filter and site!='val':
            df= mag_filter(df.dropna(subset=['Bx', 'By']))
        else:
            df= df.dropna(subset=['Bx', 'By'])
        # Use data padding to ensure full time range is covered
        df= data_padding(df.resample('1min').mean().copy(),
                        block_start=block_start,
                        block_end=block_end,
                        pad_freq='1min',
                    )
        for i in range(2):
            df_sub= fmi_smoothed_df_vectorized(df, df_k)
            df_sub= spline_subtract(df, df_sub, 3)
            df_k= kindex(df_sub, **K_kwargs)
        df_K.append(df_k)
    return pd.concat(df_K)

@enforce_types(
    df=pd.DataFrame,
    cols=(tuple, list),
    window=int,
    threshold=(int, float, np.number),
)
def mag_filter(
    df,
    cols=("Bx", "By", "Bz"),
    window=60,
    threshold=10.0,
    fill_value=np.nan,
):
    """
    Filter out short, spiky disturbances (e.g. cars) based on the rate of change
    of the magnetic field.

    For each non-overlapping block of `window` samples:
      - Compute dF[i] = |dBx| + |dBy| + |dBz| between successive samples
      - If max(dF) in that block > threshold, mark the whole block as "bad"
      - Bad blocks get `fill_value` in the specified columns

    This reproduces the logic of the original mag_filter, but:
      - works on the entire DataFrame
      - is vectorized
      - returns a filtered DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least the columns in `cols`.
    cols : tuple of str
        Component columns to filter (default: ("Bx", "By", "Bz")).
    window : int
        Block size in number of samples (default: 60).
    threshold : float
        If max(dF) in a block exceeds this, that block is marked bad.
    fill_value : float or np.nan
        Value to assign to bad samples in `cols`. Use np.nan to integrate
        cleanly with later dropna/resample logic.

    Returns
    -------
    df_filtered : pandas.DataFrame
        Same as input, but with spikes replaced by fill_value in `cols`.

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01', periods=120, freq='min')
    >>> df = pd.DataFrame({'Bx': np.zeros(len(idx)), 'By': np.zeros(len(idx)), 'Bz': np.zeros(len(idx))}, index=idx)
    >>> mag_filter(df, window=30, threshold=0.1).equals(df)
    True
    """

    df = df.copy()  # avoid mutating caller data

    # 1. Compute dF = |dBx| + |dBy| + |dBz|
    dF = df[list(cols)].diff().abs().sum(axis=1).to_numpy()
    # First difference is undefined; treat as 0 (won't trigger anything alone)
    if len(dF) > 0:
        dF[0] = 0.0

    n = len(df)
    mask_bad = np.zeros(n, dtype=bool)

    # 2. Process non-overlapping blocks of length `window`
    #    Original code effectively skipped the tail; we mimic that.
    last_full = (n // window) * window   # up to but not including tail
    for start in range(0, last_full, window):
        end = start + window
        if end > n:
            break

        # max dF over this block
        if np.nanmax(dF[start:end]) > threshold:
            mask_bad[start:end] = True

    # 3. Apply fill_value to bad samples
    if np.isnan(fill_value):
        df.loc[mask_bad, cols] = np.nan
    else:
        df.loc[mask_bad, cols] = fill_value

    return df
@enforce_types(
    date=(pd.Timestamp, np.datetime64, str),
    site_code=str,
    path_prefix=str,
)
def _get_live(date, site_code, path_prefix='https://data.magie.ie/'):
    """
    Download a single day's live TXT data for a site from data.magie.ie.

    Parameters
    ----------
    date : datetime-like
        Date to download.
    site_code : str
        Three-letter site code (e.g. ``'dun'``).

    Returns
    -------
    pandas.DataFrame
        Raw downloaded measurements with a ``Site`` column.

    Examples
    --------
    >>> _get_live(pd.Timestamp('2024-01-02'), 'dun')  # doctest: +SKIP
    """
    import tempfile
    from magie.Data_Download import download
    from pathlib import Path
    if path_prefix.startswith('https'):
        url_prefix = path_prefix
        if isinstance(date, pd.Timestamp):
            date= date.to_numpy()
        date = date.astype('datetime64[D]').astype(str).split('-')
        url = url_prefix + '{}/{}/{}/txt/'.format(*date)
        filename = site_code + '{}{}{}.txt'.format(*date)
        with tempfile.TemporaryDirectory(prefix="live_mags_download") as tmpdir:
            download(f'{url}{filename}', tmpdir +'/'+ filename)  # network fetch to local cwd
            columns = ['Date_UTC', 'Index', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG', 'TE', 'Volts']
            drop_index = columns.copy()
            drop_index[1] = 'Site'
            df = pd.read_csv(tmpdir +'/'+ filename, delimiter='\t', 
                                names=columns, 
                                skiprows=1, parse_dates=['Date_UTC'], dayfirst=True, index_col=False).replace(99.99999e3, np.nan)
            df['Site'] = [site_code] * len(df)
    else:
        date = date.astype('datetime64[D]').astype(str).split('-')
        folder = path_prefix + '{}/{}/{}/txt/'.format(*date)
        filename = site_code + '{}{}{}.txt'.format(*date)
        columns = ['Date_UTC', 'Index', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG', 'TE', 'Volts']

        df = pd.read_csv(folder+filename, delimiter='\t',
                    names=columns,
                    skiprows=1, parse_dates=['Date_UTC'], dayfirst=True, index_col=False).replace(99.99999e3, np.nan)
        df['Date_UTC'] = pd.to_datetime(df['Date_UTC'], format='mixed')
        df['Site'] = [site_code] * len(df)
    return df
@enforce_types(
    now_time=(pd.Timestamp, np.datetime64, str),
    site_code=str,
    filter=bool,
    path_prefix=str,
)
def live_k(now_time, site_code, filter=True, path_prefix='https://data.magie.ie/', **kwargs):
    """
    Fetch the past 3 days of live data for ``site_code`` and return smoothed K.

    Parameters
    ----------
    now_time : datetime-like
        Reference time; data are fetched from ``now_time - 3 days`` onward.
    site_code : str
        Three-letter site code to download.
    **kwargs :
        Passed to ``provisional_k`` and ``smooth_kindex``.

    Returns
    -------
    pandas.DataFrame
        Smoothed K-index values covering the last two full days.

    Examples
    --------
    >>> live_k(pd.Timestamp('2024-01-03'), 'dun')  # doctest: +SKIP
    """

    start_time = pd.Timestamp(now_time).floor('1D')-pd.Timedelta(3, 'D')
    end_time = pd.Timestamp(now_time).ceil('1D')
    if filter:
        df= mag_filter(pd.concat([_get_live(date, site_code, path_prefix=path_prefix)\
            for date in np.arange(start_time, end_time, np.timedelta64(1, 'D'))]))
    else:
        df= pd.concat([_get_live(date, site_code, path_prefix=path_prefix)\
            for date in np.arange(start_time, end_time, np.timedelta64(1, 'D'))])
    # return df
    # Trim to requested window and run provisional + smoothed K
    df= df.loc[(df.Date_UTC>=start_time)&(df.Date_UTC<=end_time)]
    # return df
    df_k= provisional_k(df, site_code=site_code, **kwargs)
    # return df_k
    df=smooth_kindex(df, df_k, site_code=site_code,  **kwargs)
    return df.loc[(df.index>=start_time.floor('1D'))&(df.index<=end_time)]
#%%
@enforce_types(
    K_data=pd.DataFrame,
)
def plot_k(K_data):
    """
    Plot K-index values as colored 3-hour bars with a qualitative legend.

    Parameters
    ----------
    K_data : pandas.DataFrame
        Must include ``K_index`` and a datetime index or ``Date_UTC`` column.

    Returns
    -------
    tuple
        ``(fig, ax, cax)`` Matplotlib objects for further customization.

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01', periods=4, freq='3h')
    >>> df = pd.DataFrame({'K_index': [1, 2, 3, 4]}, index=idx)
    >>> fig, ax, cax = plot_K(df)
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    K_data = K_data.copy()  # avoid mutating caller data
    # K_data["K_index"] = K_data["K_index"].replace(0, 0.2)  # lift zeros for visibility
    fig= plt.figure(figsize=(30, 15))
    gs= fig.add_gridspec(2, 1, height_ratios=[1, .1], hspace=0.2)
    ax= fig.add_subplot(gs[0, 0])
    cax= fig.add_subplot(gs[1, 0])

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # 1. Define the colours for each category (Quiet -> Severe Storm)

    K_cmap = ListedColormap([
        'blue',      # Quiet: 0-1
        'cyan',      # Unsettled: 2-3
        'green',     # Active: 4
        'orange',    # Minor Storm: 5
        'red',       # Major Storm: 6-7
        'magenta'    # Severe Storm: 8-9
    ])

    # 2. Define the boundaries between categories
    bounds = [0, 2, 4, 5, 6, 8, 10]   # right edge is exclusive by default
    norm = BoundaryNorm(bounds, K_cmap.N)
    if 'Date_UTC' in K_data.columns:
        K_data.set_index('Date_UTC', inplace=True)

    # Three-hour bars colored by category
    bars= ax.bar(K_data.index, K_data.K_index, edgecolor='black', width=np.timedelta64(3, 'h'),
           color=K_cmap(norm(K_data.K_index)), align='edge', lw=2, zorder=4)
    for bar in bars:
        x = bar.get_x()
        w = bar.get_width()
        y = bar.get_height()
        ax.hlines(
            y,
            x,
            x + w,
            colors=bar.get_facecolor(),
            linewidth=5,
            zorder=3
        )
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=K_cmap), cax=cax, orientation='horizontal',
                # ticks=[1, 3, 4.5, 5.5, 7, 9.0],
                )
    cbar.ax.set_xticks([])
    ax.set_yticks(range(0, 10))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, which='both', linestyle='--', alpha=1, lw=1)
    for x, t in zip([1, 3, 4.5, 5.5, 7, 9.0], ['Quiet\n0-1', 'Unsettled\n2-3', 'Active\n4', 'Minor Storm\n5', 'Major Storm\n6-7', 'Severe Storm\n8-9']):
        cax.text(x, .5, t, fontsize=20, verticalalignment='center', ha='center', bbox=dict(facecolor='white', alpha=0.5))
    import matplotlib.dates as mdates
    # # --- Major ticks every day ---
    # ax.xaxis.set_major_locator(mdates.DayLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # or '%d %b'

    # # --- Minor ticks every 3 hours ---
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))

    # --- Major ticks: days ---
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b-%Y'))

    # --- Minor ticks: hours ---
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

    # Styling
    ax.tick_params(axis='x', which='major', labelsize=25)
    ax.tick_params(axis='y', which='major', labelsize=25)

    ax.tick_params(axis='x', which='minor', labelsize=25)
    # ax.minorticks_on(axis='x')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(K_data.index.values.astype('datetime64[D]').min(), K_data.index.values.astype('datetime64[D]').max()+np.timedelta64(1, 'D'))
    ax.set_ylim(-.05, 9.2)
    with get_asset_path("MagIE-logo.png") as logo_path:
        logo = mpimg.imread(logo_path)
    logo[..., :3] = 1.0 - logo[..., :3]  # invert RGB, keep alpha
    imagebox = OffsetImage(logo, zoom=0.3)
    ab = AnnotationBbox(
        imagebox,
        (0.78, 0.6),            # bottom-right in figure coords
        xycoords="figure fraction",
        frameon=False,
        box_alignment=(1, 0),
    )
    logo= fig.add_artist(ab)

    return fig, ax, cax
#%%
@enforce_types(
    K_data=pd.DataFrame,
)
def plot_k_plotly(K_data: pd.DataFrame):
    """
    Plot K-index values as interactive 3-hour bars using Plotly.

    Parameters
    ----------
    K_data : pandas.DataFrame
        Table containing K-index values and their timestamps.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure for the supplied K-index series.
    """

    def add_3hour_gridlines(fig, start, end):
        """
        Draw vertical gridlines every 3 hours (matplotlib-style).
        """
        times = pd.date_range(start=start, end=end, freq="3h", tz="UTC")

        for t in times:
            fig.add_vline(
                x=t,
                line_width=1,
                line_color="rgba(0,0,0,0.18)",  # light gridline
                layer="below",                 # behind bars
            )

    import plotly.graph_objects as go
    df = K_data.copy()

    # Ensure datetime index (UTC)
    if "Date_UTC" in df.columns:
        df["Date_UTC"] = pd.to_datetime(df["Date_UTC"], utc=True, errors="coerce")
        df = df.set_index("Date_UTC")
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    df = df.sort_index()
    df = df[df.index.notna()]

    # Lift zeros for visibility (as in your mpl code)
    df["K_index"] = df["K_index"].replace(0, 0.2)

    # Color categories (match mpl)
    def cat_color(k: float) -> str:
        if 0 <= k <= 1:
            return "blue"
        if 2 <= k <= 3:
            return "cyan"
        if 4 <= k < 5:
            return "green"
        if 5 <= k < 6:
            return "orange"
        if 6 <= k <= 7:
            return "red"
        return "magenta"

    bar_colors = df["K_index"].apply(cat_color)

    # Bar width = 3 hours in ms
    width_ms = 3 * 60 * 60 * 1000

    # X range to day boundaries (like mpl)
    day_min = df.index.min().floor("D")
    day_max = df.index.max().floor("D") + pd.Timedelta(days=1)

    # Daily ticks only (labels)
    days = pd.date_range(day_min, day_max, freq="D", tz="UTC")
    day_tickvals = list(days)
    day_ticktext = [d.strftime("%Y-%m-%d") for d in days]

    fig = go.Figure()
    fig.update_layout(
        width=1200,   # pixels
        height=600,   # pixels
    )
    width_hours = 3
    half_width = pd.Timedelta(hours=width_hours / 2)
    width_ms = width_hours * 60 * 60 * 1000

    x_left_edges = df.index                      # like matplotlib align='edge'
    x_centers = x_left_edges + half_width        # plotly wants centers

    fig.add_trace(
        go.Bar(
            x=x_centers,
            y=df["K_index"],
            width=width_ms,
            marker=dict(color=bar_colors, line=dict(color="black", width=1)),
        )
    )


    # --- Layout: reserve bottom space for legend strip
    fig.update_layout(
        barmode="overlay",
        bargap=0,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=520,
        margin=dict(l=70, r=20, t=20, b=140),  # big bottom margin for legend strip
    )

    # --- Y axis like mpl
    fig.update_yaxes(
        range=[0, 9],
        tickmode="array",
        tickvals=list(range(0, 10)),
        showgrid=False,
        gridwidth=1,
        zeroline=False,
        title=None,
    )

    # --- X axis: daily labels only, but keep 3-hour gridlines
    # Trick: set dtick to 3h for gridlines, but override ticks with daily tickvals/ticktext.
    fig.update_xaxes(
        range=[day_min, day_max],
        showgrid=True,
        dtick=3 * 60 * 60 * 1000,           # 3-hour grid spacing
        tickmode="array",
        tickvals=day_tickvals,              # ONLY show daily labels
        ticktext=day_ticktext,
        tickangle=0,
        title=None,
        zeroline=False,
    )

    # Make grid look a bit more like mpl dashed (Plotly only supports solid lines,
    # but lighter grid helps approximate)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.18)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.18)")

    # -----------------------------
    # Legend strip (like your cax)
    # Draw a colored bar under the plot with boxed labels
    # Use "paper" coordinates so it stays anchored.
    # -----------------------------

    legend_segments = [
        ("Quiet<br>0-1",        "blue",    0.00, 0.1667),
        ("Unsettled<br>2-3",    "cyan",    0.1667, 0.3334),
        ("Active<br>4",         "green",   0.3334, 0.5001),
        ("Minor Storm<br>5",    "orange",  0.5001, 0.6668),
        ("Major Storm<br>6-7",  "red",     0.6668, 0.8335),
        ("Severe Storm<br>8-9", "magenta", 0.8335, 1.00),
    ]

    # Legend strip vertical placement (paper coords)
    y0, y1 = -0.31, -0.18
    y_mid = (y0 + y1) / 2

    for label, color, x0, x1 in legend_segments:
        # colored block
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color="rgba(0,0,0,0.35)", width=1),
            fillcolor=color,
            layer="below",
        )

        # centered label
        fig.add_annotation(
            xref="paper", yref="paper",
            x=(x0 + x1) / 2,
            y=y_mid,
            text=label,
            showarrow=False,

            # critical alignment controls
            xanchor="center",
            yanchor="middle",
            align="center",

            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.55)",
            bordercolor="rgba(0,0,0,0.35)",
            borderwidth=1,

            # optional pixel nudges (tweak if needed)
            xshift=0,
            yshift=0,
        )
    add_3hour_gridlines(fig, day_min, day_max)
    fig.update_yaxes(showgrid=True)
    fig.update_yaxes(
        title_text="K Index (0–9)",
        title_font=dict(size=30),
        tickfont=dict(size=20)
    )

    fig.update_xaxes(
        title_text="Universal Time",
        title_font=dict(size=30),
        tickfont=dict(size=20),
        title_standoff=0,
    )


    return fig

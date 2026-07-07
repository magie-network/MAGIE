import os
import re
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from glob import glob
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from magpy.core import activity as act
from magpy.stream import DataStream, read, join_streams
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from magie.file_conversions import (
    _format_iaga_component_series,
    _iaga_comment_record,
    _iaga_filename,
    _iaga_header_record,
    _infer_iaga_interval_type,
    _normalise_iaga_numeric,
    magie2iaga2002,
)
from magie.utils import enforce_types, get_asset_path, get_site_metadata, tqdm_joblib


def _as_utc_naive_timestamp(value):
    """
    Convert timezone-aware timestamps to UTC while preserving naive UTC inputs.
    """

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def _utc_day(value):
    """
    Return the UTC archive day for a datetime-like value.
    """

    return _as_utc_naive_timestamp(value).floor("1D")


def _date_tokens(date):
    return _utc_day(date).strftime("%Y-%m-%d").split("-")


def _path_prefix_join(path_prefix, *parts):
    if path_prefix.startswith("http"):
        return path_prefix.rstrip("/") + "/" + "/".join(parts)
    return str(Path(path_prefix).joinpath(*parts))


def _read_text_source(path_or_url):
    if path_or_url.startswith("http"):
        with urlopen(path_or_url) as response:
            return response.read().decode("utf-8")
    return Path(path_or_url).read_text(encoding="utf-8")


def _iaga_file_candidates(date, site_code, path_prefix):
    year, month, day = _date_tokens(date)
    yyyymmdd = f"{year}{month}{day}"
    folder = _path_prefix_join(path_prefix, year, month, day, "iaga2002")
    if path_prefix.startswith("http"):
        return [
            f"{folder}/{site_code}{yyyymmdd}psec.sec",
            f"{folder}/{site_code}{yyyymmdd}pmin.min",
        ]

    pattern = str(Path(folder) / f"{site_code}{yyyymmdd}*")
    return sorted(glob(pattern))


def _get_iaga_path(date, site_code, path_prefix):
    candidates = _iaga_file_candidates(date, site_code, path_prefix)
    if not candidates:
        raise FileNotFoundError(
            f"No IAGA files found for {site_code} on "
            f"{pd.Timestamp(date).date()} under {path_prefix!r}."
        )
    return candidates[0]


def _empty_padding_stream(start_time, site_code, sampling_step_seconds=60):
    start_time = pd.Timestamp(start_time)
    if sampling_step_seconds is None or sampling_step_seconds <= 0:
        sampling_step_seconds = 60

    times = pd.date_range(
        start=start_time,
        end=start_time + pd.Timedelta(days=2) - pd.to_timedelta(sampling_step_seconds, unit="s"),
        freq=pd.to_timedelta(sampling_step_seconds, unit="s"),
    )

    array = [np.asarray([]) for _ in DataStream().KEYLIST]
    array[0] = times.to_numpy()
    for key in ("x", "y", "z", "f"):
        array[DataStream().KEYLIST.index(key)] = np.full(len(times), np.nan)

    return DataStream(
        header={
            "DataSamplingRate": sampling_step_seconds,
            "StationID": site_code.upper(),
            "col-x": "X",
            "col-y": "Y",
            "col-z": "Z",
            "col-f": "F",
        },
        ndarray=np.asarray(array, dtype=object),
    )

@enforce_types(
    iaga_text=str,
    label=str,
)
def _read_iaga_header_value(iaga_text, label):
    """
    Extract one labeled value from an IAGA-2002 header block.

    Parameters
    ----------
    iaga_text : str
        Full IAGA-2002 text content.
    label : str
        Header label to match.

    Returns
    -------
    str or None
        The stripped header value when found, otherwise ``None``.
    """
    prefix = f" {label:<23}"
    for line in iaga_text.splitlines():
        if line.startswith(prefix):
            return line[24:69].rstrip(" |")
    return None


@enforce_types(
    iaga_text=str,
)
def _sampling_step_seconds_from_header(iaga_text):
    """
    Infer the sampling cadence in seconds from IAGA-2002 header text.

    Parameters
    ----------
    iaga_text : str
        Full IAGA-2002 text content.

    Returns
    -------
    float or None
        Sampling step in seconds parsed from the header, or ``None`` when
        the cadence cannot be inferred.
    """
    # Prefer Digital Sampling, then fall back to Data Interval Type
    value = _read_iaga_header_value(iaga_text, "Digital Sampling")
    if not value:
        value = _read_iaga_header_value(iaga_text, "Data Interval Type")
    if not value:
        return None

    value = value.strip().lower()

    m = re.match(r"(\d+(?:\.\d+)?)\s*[- ]?(millisecond|second|minute|hour|day)s?", value)
    if not m:
        return None

    n = float(m.group(1))
    unit = m.group(2)

    factors = {
        "millisecond": 1e-3,
        "second": 1.0,
        "minute": 60.0,
        "hour": 3600.0,
        "day": 86400.0,
    }
    return n * factors[unit]

@enforce_types(
    start_time=(pd.Timestamp, np.datetime64, str),
    source_of_data=str,
    station_name=str,
    iaga_code=str,
    geodetic_latitude=(int, float, str, type(None), np.number),
    geodetic_longitude=(int, float, str, type(None), np.number),
    elevation=(int, float, str, type(None), np.number),
    reported=str,
    sensor_orientation=str,
    digital_sampling=(str, type(None)),
    data_interval_type=(str, type(None)),
    data_type=str,
    publication_date=(str, type(None)),
    comments=(list, tuple, type(None)),
    sampling_step_seconds=(int, float, type(None), np.number),
)
def build_empty_iaga_window(
    start_time,
    source_of_data="MagIE",
    station_name="Unknown station",
    iaga_code="XXX",
    geodetic_latitude=None,
    geodetic_longitude=None,
    elevation=None,
    reported="XYZF",
    sensor_orientation="XYZF",
    digital_sampling=None,
    data_interval_type=None,
    data_type="Provisional",
    publication_date=None,
    comments=None,
    sampling_step_seconds=60,
):
    """
    Create an in-memory IAGA-2002 file filled with missing values from
    start_time to start_time + 1 day.

    Returns
    -------
    tuple[str, str]
        The full IAGA-2002 file contents and a recommended output filename.
    """
    start_time = pd.Timestamp(start_time)

    if sampling_step_seconds is None:
        sampling_step_seconds = 60

    if comments is None:
        comments = [
            "H = squareroot(X*X + Y*Y), D = atan2(Y, X), I = atan2(Z, H)",
        ]

    end_time = start_time + pd.Timedelta(days=2)
    times = pd.date_range(
        start=start_time,
        end=end_time - pd.to_timedelta(sampling_step_seconds, unit="s"),
        freq=pd.to_timedelta(sampling_step_seconds, unit="s"),
    )

    file = pd.DataFrame({"Date_UTC": times})

    # empty magnetic channels
    for col in ["Bx", "By", "Bz", "TFG"]:
        file[col] = np.nan

    interval_label, interval_extension = _infer_iaga_interval_type(file["Date_UTC"])

    if digital_sampling is None:
        if np.isclose(sampling_step_seconds, 1.0):
            digital_sampling = "1 second"
        else:
            digital_sampling = interval_label.split(" ", 1)[0]

    if data_interval_type is None:
        data_interval_type = interval_label

    code = iaga_code.upper()[:3] if iaga_code else "XXX"

    headers = [
        _iaga_header_record("Format", "IAGA-2002"),
        _iaga_header_record("Source of Data", source_of_data),
        _iaga_header_record("Station Name", station_name),
        _iaga_header_record("IAGA CODE", code),
        _iaga_header_record("Geodetic Latitude", _normalise_iaga_numeric(geodetic_latitude)),
        _iaga_header_record("Geodetic Longitude", _normalise_iaga_numeric(geodetic_longitude)),
        _iaga_header_record("Elevation", _normalise_iaga_numeric(elevation, precision=0)),
        _iaga_header_record("Reported", reported),
        _iaga_header_record("Sensor Orientation", sensor_orientation),
        _iaga_header_record("Digital Sampling", digital_sampling),
        _iaga_header_record("Data Interval Type", data_interval_type),
        _iaga_header_record("Data Type", data_type),
    ]

    if publication_date is not None:
        headers.append(_iaga_header_record("Publication date", publication_date))

    headers.extend(_iaga_comment_record(comment) for comment in comments)

    data_columns = [f"{code}{component}" for component in reported]
    data_header = f"{'DATE':<10} {'TIME':<12} {'DOY':>3} {' '.join(f'{column:>9}' for column in data_columns)}"
    lines = [*headers, f"{data_header[:69]:<69}|"]

    date_str = file["Date_UTC"].dt.strftime("%Y-%m-%d")
    time_str = file["Date_UTC"].dt.strftime("%H:%M:%S.%f").str[:-3]
    doy = file["Date_UTC"].dt.dayofyear.astype(str).str.zfill(3)

    component_map = {
        "X": file["Bx"],
        "Y": file["By"],
        "Z": file["Bz"],
        "F": file["TFG"],
    }

    formatted_components = []
    for component in reported:
        series = component_map.get(component, pd.Series(np.nan, index=file.index))
        formatted_components.append(
            _format_iaga_component_series(series, missing_value=999999.00)
        )

    data_lines = date_str + " " + time_str + " " + doy
    if formatted_components:
        component_block = formatted_components[0]
        for formatted in formatted_components[1:]:
            component_block = component_block + " " + formatted
        data_lines = data_lines + " " + component_block

    lines.extend(data_lines.str.slice(0, 70).str.pad(70, side="right").tolist())

    filename = _iaga_filename(code, file["Date_UTC"].iloc[0], data_type, interval_extension)
    return "\n".join(lines) + "\n", filename[:3]+'_empty'+filename[3:]

@enforce_types(
    date=(pd.Timestamp, np.datetime64, str),
    site_code=str,
    path_prefix=str,
    file_format=str,
)
def _get_live(date, site_code, path_prefix='https://data.magie.ie/', file_format="iaga2002"):
    """
    Load one day's live data for a site.

    Parameters
    ----------
    date : datetime-like
        Date to load.
    site_code : str
        Three-letter site code (e.g. ``'dun'``).
    path_prefix : str, optional
        Archive root. For local paths this is expected to contain
        ``YYYY/MM/DD/{txt,iaga2002}/`` directories.
    file_format : {"iaga2002", "txt"}, optional
        ``"iaga2002"`` reads the persistent IAGA-2002 file. ``"txt"`` reads
        the legacy TXT file and returns converted IAGA-2002 text.

    Returns
    -------
    tuple[str, str]
        IAGA-2002 file contents and filename.

    Examples
    --------
    >>> _get_live(pd.Timestamp('2024-01-02'), 'dun')  # doctest: +SKIP
    """
    file_format = file_format.lower()
    if file_format in {"iaga", "iaga2002"}:
        for iaga_path in _iaga_file_candidates(date, site_code, path_prefix):
            try:
                return _read_text_source(iaga_path), Path(iaga_path).name
            except (FileNotFoundError, HTTPError, URLError):
                continue
        raise FileNotFoundError(
            f"No IAGA files found for {site_code} on "
            f"{pd.Timestamp(date).date()} under {path_prefix!r}."
        )
    if file_format not in {"txt", "legacy"}:
        raise ValueError("file_format must be 'iaga2002' or 'txt'.")

    if path_prefix.startswith('https'):
        url_prefix = path_prefix
        date = _date_tokens(date)
        url = _path_prefix_join(url_prefix, *date, "txt") + "/"
        filename = site_code + '{}{}{}.txt'.format(*date)
        columns = ['Date_UTC', 'Index', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG', 'TE', 'Volts']
        drop_index = columns.copy()
        drop_index[1] = 'Site'
        df = pd.read_csv(f'{url}{filename}', delimiter='\t',
                            names=columns,
                            skiprows=1, parse_dates=['Date_UTC'], dayfirst=True, index_col=False).replace(99.99999e3, np.nan)
        df['Site'] = [site_code] * len(df)
        df = df[drop_index]
        df = pd.concat([pd.DataFrame(columns=df.columns,
                                    data=[[pd.Timestamp(df.Date_UTC.min().to_numpy().astype('datetime64[D]').astype('datetime64[ns]')),
                                            site_code] + [np.nan] * (len(df.columns) - 2)]), df])
        df= magie2iaga2002(df, site_code)
    else:
        date = _date_tokens(date)
        folder = _path_prefix_join(path_prefix, *date, "txt") + "/"
        filename = site_code + '{}{}{}.txt'.format(*date)
        columns = ['Date_UTC', 'Index', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG', 'TE', 'Volts']
        drop_index = columns.copy()
        drop_index[1] = 'Site'
        df = pd.read_csv(folder+filename, delimiter='\t',
                    names=columns,
                    skiprows=1, parse_dates=['Date_UTC'], dayfirst=True, index_col=False).replace(99.99999e3, np.nan)
        df['Date_UTC'] = pd.to_datetime(df['Date_UTC'], format='mixed')
        df['Site'] = [site_code] * len(df)
        df = df[drop_index]
        df = pd.concat([pd.DataFrame(columns=df.columns,
                                     data=[[pd.Timestamp(df.Date_UTC.min().to_numpy().astype('datetime64[D]').astype('datetime64[ns]')),
                                            site_code] + [np.nan] * (len(df.columns) - 2)]), df])
        df= magie2iaga2002(df, site_code)
    return df

@enforce_types(
    now_time=(pd.Timestamp, np.datetime64, str),
    site_code=str,
    path_prefix=str,
    file_format=str,
)
def live_k(now_time, site_code, path_prefix='https://data.magie.ie/', site_metadata=None, file_format="iaga2002"):
    """
    Fetch the past 3 days of live data for ``site_code`` and return smoothed K.

    Parameters
    ----------
    now_time : datetime-like
        Reference time; data are fetched from ``now_time - 3 days`` onward.
    site_code : str
        Three-letter site code to download.
    file_format : {"iaga2002", "txt"}, optional
        ``"iaga2002"`` reads existing persistent IAGA files. ``"txt"`` reads
        legacy TXT files, converts them, saves persistent IAGA files, then reads
        those IAGA files.
    **kwargs :
        Passed to ``provisional_k`` and ``smooth_kindex``.

    Returns
    -------
    pandas.DataFrame
        Smoothed K-index values covering the last two full days.

    Examples
    --------
    >>> live_k(pd.Timestamp('2024-01-03'), 'dun')
    """
    now_time = _as_utc_naive_timestamp(now_time)
    start_time = now_time.floor('1D')-pd.Timedelta(4, 'D')
    end_time = now_time.floor('1D') + pd.Timedelta(1, 'D')
    if path_prefix.startswith("http"):
        raise ValueError(
            "live_k now expects a local archive path_prefix when "
            "file_format='iaga2002'. Run the live IAGA updater first, then "
            "read from the local iaga2002 archive."
        )
    data = DataStream()
    counter = 0
    padding_sampling_step_seconds = None
    for date in np.arange(start_time, end_time, np.timedelta64(1, 'D')):
        try:
            if file_format.lower() in {"txt", "legacy"}:
                iaga_text, filename = _get_live(
                    date,
                    site_code,
                    path_prefix=path_prefix,
                    file_format="txt",
                )
                padding_sampling_step_seconds = _sampling_step_seconds_from_header(iaga_text)
                output_dir = Path(_path_prefix_join(path_prefix, *_date_tokens(date), "iaga2002"))
                output_dir.mkdir(parents=True, exist_ok=True)
                iaga_path = output_dir / filename
                iaga_path.write_text(iaga_text, encoding="utf-8")
            else:
                iaga_path = Path(_get_iaga_path(date, site_code, path_prefix))
        except FileNotFoundError as e:
            print(f"File not found for date {date}: {e}")
            continue

        stream = read(str(iaga_path))
        if padding_sampling_step_seconds is None:
            sampling_rate = stream.samplingrate()
            if sampling_rate and sampling_rate > 0:
                padding_sampling_step_seconds = sampling_rate
        data = join_streams(data, stream)
        counter += 1

    if not counter:
        raise FileNotFoundError(
            f"No live data files found for site {site_code!r} between "
            f"{start_time.date()} and {(end_time - pd.Timedelta(days=1)).date()}"
        )
    data = join_streams(
        data,
        _empty_padding_stream(
            now_time,
            site_code,
            sampling_step_seconds=padding_sampling_step_seconds,
        ),
    )
    data= data.filter()
    if site_metadata is None:
        site_metadata= get_site_metadata(site_code)
        
    data = act.K_fmi(data, K9_limit=site_metadata['k9_threshold'], longitude=site_metadata['geodetic_longitude'], step_size=60)
    data['var1']= np.where(data['var1']>=0, data['var1'], np.nan)
    return data.trim(starttime=(start_time + pd.Timedelta(45, 'h')).to_numpy(), endtime=(now_time+pd.Timedelta(1, 'D')).floor('1D').to_numpy())

@enforce_types(
    K_data=(DataStream, pd.DataFrame),
    logo_path=(str, Path, type(None)),
    auto_xlim=bool,
    colorbar=bool,
    show_logo=bool,
)
def plot_k(K_data, logo_path=None, auto_xlim=True, colorbar=True, show_logo=False):
    """
    Plot K-index values as colored 3-hour bars with a qualitative legend.

    Parameters
    ----------
    K_data : magpy.stream.DataStream or pandas.DataFrame
        K-index data to plot. Must provide ``time`` values and ``var1`` K-index
        values, as returned by the MagPy K-index calculation.
    logo_path : str or pathlib.Path, optional
        Path to the logo image to be displayed on the plot. When omitted, the
        packaged ``MagIE-logo.png`` asset is used if ``show_logo`` is true.
    auto_xlim : bool, default True
        Whether to automatically set the x-axis limits.
    colorbar : bool, default True
        Whether to add the horizontal qualitative K-index color legend.
    show_logo : bool, default False
        Whether to add the MagIE logo overlay to the figure.

    Returns
    -------
    tuple
        ``(fig, ax, cax)`` when ``colorbar`` is true, otherwise ``(fig, ax)``.

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01', periods=4, freq='3h')
    >>> df = pd.DataFrame({'time': idx, 'var1': [1, 2, 3, 4]})
    >>> fig, ax, cax = plot_k(df)
    """
    K_data = K_data.copy()  # avoid mutating caller data
    # K_data["K_index"] = K_data["K_index"].replace(0, 0.2)  # lift zeros for visibility
    fig= plt.figure(figsize=(900/96, 400/96))
    if colorbar:
        gs= fig.add_gridspec(2, 1, height_ratios=[1, .1], hspace=0.2)
        cax= fig.add_subplot(gs[1, 0])
    else:
        gs= fig.add_gridspec(1, 1)
    ax= fig.add_subplot(gs[0, 0])

    # Map each K-index range to the operational category colours.
    K_cmap = ListedColormap([
        'blue',      # Quiet: 0-1
        'cyan',      # Unsettled: 2-3
        'green',     # Active: 4
        'orange',    # Minor Storm: 5
        'red',       # Major Storm: 6-7
        'magenta'    # Severe Storm: 8-9
    ])

    # Boundaries group K values into the same ranges shown in the legend.
    bounds = [0, 2, 4, 5, 6, 8, 10]   # right edge is exclusive by default
    norm = BoundaryNorm(bounds, K_cmap.N)

    # Draw each three-hour K bin as a colored bar, aligned to its start time.
    bars= ax.bar(np.array(K_data['time']).astype('datetime64[ns]'), K_data['var1'], edgecolor='black', width=np.timedelta64(3, 'h'),
           color=K_cmap(norm(K_data['var1'])), align='edge', lw=2, zorder=4)
    for bar in bars:
        x = bar.get_x()
        w = bar.get_width()
        y = bar.get_height()
        if y==0:
            ax.hlines(
                y+.01,
                x,
                x + w,
                colors=bar.get_facecolor(),
                linewidth=2,
                zorder=3
            )

    ax.set_yticks(range(0, 10))
    # Reset grid state before applying separate x/y grid styles.
    ax.grid(False)

    # Vertical grid lines mark six-hour minor ticks.
    ax.grid(True, which='minor', axis='x', linestyle='--', alpha=1, lw=1)

    # Horizontal grid lines track the integer K-index values.
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=1, lw=1)
    if colorbar:
        cbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=K_cmap), cax=cax, orientation='horizontal',
                    # ticks=[1, 3, 4.5, 5.5, 7, 9.0],
                    )
        cbar.ax.set_xticks([])
        for x, t in zip([1, 3, 4.5, 5.5, 7, 9.0], ['Quiet\n0-1', 'Unsettled\n2-3', 'Active\n4', 'Minor Storm\n5', 'Major Storm\n6-7', 'Severe Storm\n8-9']):
            cax.text(x, .5, t, fontsize=20, verticalalignment='center', ha='center', bbox=dict(facecolor='white', alpha=0.5))

    ax.xaxis.set_major_locator(
        mdates.HourLocator(byhour=[11])  # noon
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%d-%b-%Y')
    )

    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', which='major', labelrotation=0, pad=15)
    ax.tick_params(axis='x', which='minor', labelrotation=0, pad=2)
    # ax.minorticks_on(axis='x')
    # ax.spines[['top', 'right']].set_visible(False)
    xmin, xmax= np.array(K_data['time']).astype('datetime64[D]').min()+np.timedelta64(1, 'D'), np.array(K_data['time']).astype('datetime64[D]').max()+np.timedelta64(1, 'D')
    if auto_xlim :
        ax.set_xlim(np.array(K_data['time']).astype('datetime64[D]').min()+np.timedelta64(1, 'D'), np.array(K_data['time']).astype('datetime64[D]').max()+np.timedelta64(1, 'D'))
    # Emphasize day boundaries over the six-hour minor grid.
    for t in np.arange(xmin, xmax-np.timedelta64(1, 'D'), np.timedelta64(1, 'D')) +np.timedelta64(1, 'D'):
        ax.axvline(t, color='black', zorder=10)
    ax.set_ylim(-.05, 9.2)
    if show_logo:
        if logo_path is None:
            with get_asset_path("MagIE-logo.png") as default_logo_path:
                logo = mpimg.imread(default_logo_path)
        else:
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
    if colorbar:
        return fig, ax, cax
    else:
        return fig, ax

@enforce_types(data=DataStream, column=str)
def _datastream_column_to_array(data, column):
    """Return a datastream column as a NumPy array, or ``None`` when unavailable."""
    try:
        values = data[column]
    except Exception:
        return None

    if values is None:
        return None

    try:
        array = np.asarray(values)
    except Exception:
        return None

    if array.size == 0:
        return None
    return array


@enforce_types(data=DataStream, site_code=str, time=(str, pd.Timestamp, np.datetime64))
def _require_valid_k_window(data, site_code, time):
    """
    Validate that a datastream contains enough data to compute daily K values.

    Parameters
    ----------
    data : DataStream
        MagPy datastream containing timestamps and magnetic components.
    site_code : str
        Site identifier used in validation errors.
    time : str or pandas.Timestamp or numpy.datetime64
        Target day to validate.

    Raises
    ------
    ValueError
        If the datastream is missing timestamps, lacks three days of coverage,
        or has no valid magnetic data for the target day.
    """
    day = _utc_day(time)

    times = _datastream_column_to_array(data, "time")
    if times is None:
        raise ValueError(
            f"Missing timestamps for site '{site_code}' on {day.date()}."
        )

    time_index = pd.to_datetime(times)
    valid_times = time_index[~pd.isna(time_index)]
    if len(valid_times) == 0:
        raise ValueError(
            f"No valid timestamps found for site '{site_code}' on {day.date()}."
        )

    distinct_days = pd.Index(valid_times.floor("1D").unique())
    if len(distinct_days) < 3:
        raise ValueError(
            f"Datastream is too short; need three full days for site '{site_code}' "
            f"on {day.date()}."
        )

    day_mask = valid_times.floor("1D") == day
    if not np.any(day_mask):
        raise ValueError(
            f"No valid data found for site '{site_code}' on {day.date()}."
        )

    component_names = ("x", "y", "z", "h", "d", "f")
    has_valid_component_data = False
    for column in component_names:
        values = _datastream_column_to_array(data, column)
        if values is None or len(values) != len(time_index):
            continue

        numeric = pd.to_numeric(values, errors="coerce")
        if np.isfinite(numeric[day_mask]).any():
            has_valid_component_data = True
            break

    if not has_valid_component_data:
        raise ValueError(
            f"No valid magnetic component data found for site '{site_code}' on {day.date()}."
        )


@enforce_types(
    time=(str, pd.Timestamp, np.datetime64),
    site_code=str,
    archive_path_builder=Callable,
    site_metadata=(dict, type(None)),
)
def daily_K(
    time,
    site_code,
    archive_path_builder=lambda date: "./magnetometer_archive/{}/{}/{}/iaga2002/".format(*date),
    site_metadata=None,
):
    """
    Compute daily K values for one site and one UTC day.

    Parameters
    ----------
    time : str or pandas.Timestamp or numpy.datetime64
        Day to process.
    site_code : str
        Site identifier used in archive filenames and metadata lookup.
    archive_path_builder : collections.abc.Callable, optional
        Function that builds an archive directory from ``[YYYY, MM, DD]`` tokens.
    site_metadata : dict or None, optional
        Pre-resolved site metadata. When omitted it is looked up from ``site_code``.

    Returns
    -------
    object
        MagPy datastream trimmed to the requested day with computed K values.

    Raises
    ------
    FileNotFoundError
        If the archive does not contain enough daily files.
    ValueError
        If the datastream cannot produce valid K values for the requested day.
    """
    timestamp = _as_utc_naive_timestamp(time)
    day = timestamp.floor("1D")
    start_time = timestamp.floor("1D") - pd.Timedelta(2, "D")
    end_time = timestamp.ceil("1D") + pd.Timedelta(3, "D")
    counter = 0

    data = DataStream()
    for date in np.arange(start_time, end_time, pd.Timedelta(1, "D")):
        date_str = str(date.tolist())[:10].split("-")
        archive_path = glob(
            archive_path_builder(date_str) + site_code + "{}{}{}*".format(*date_str)
        )
        if len(archive_path) == 0:
            continue

        counter += 1
        day_stream = read(os.path.abspath(archive_path[0]))
        data = join_streams(data, day_stream)

    if not counter:
        raise FileNotFoundError(
            f"No files found for site '{site_code}' in the date range "
            f"{start_time} to {end_time}."
        )
    elif counter < 3:
        raise FileNotFoundError(
            f"Datastream is too short; need three full days for site '{site_code}' "
            f"on {day.date()}."
        )

    data = data.filter()
    _require_valid_k_window(data, site_code=site_code, time=time)
    if site_metadata is None:
        site_metadata = get_site_metadata(site_code)

    k_fmi_stdout = StringIO()
    k_fmi_stderr = StringIO()
    with redirect_stdout(k_fmi_stdout), redirect_stderr(k_fmi_stderr):
        data = act.K_fmi(
            data,
            K9_limit=site_metadata["k9_threshold"],
            longitude=site_metadata["geodetic_longitude"],
            step_size=60,
        )

    k_fmi_messages = "\n".join(
        message.strip()
        for message in [k_fmi_stdout.getvalue(), k_fmi_stderr.getvalue()]
        if message.strip()
    )
    if k_fmi_messages:
        raise ValueError(
            f"K_fmi failed for site '{site_code}' on {day.date()}: {k_fmi_messages}"
        )

    data["var1"] = data["var1"]
    data["var1"] = np.where(data["var1"] >= 0, data["var1"], np.nan)
    data = data.trim(
        starttime=day.to_numpy(),
        endtime=day.to_numpy() + np.timedelta64(1, "D"),
    )
    if not len(data):
        raise ValueError(
            f"No valid K index data found for site '{site_code}' on "
            f"{day.date()}."
        )
    return data


@enforce_types(
    exc=BaseException,
    site_code=str,
    date=(str, pd.Timestamp, np.datetime64),
)
def _build_daily_k_error_record(exc, site_code, date):
    """Build a structured error record including the traceback origin."""
    date = _utc_day(date)
    frames = traceback.extract_tb(exc.__traceback__)
    origin = frames[-1] if frames else None

    error = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "site": site_code,
        "date": date.strftime("%Y-%m-%d"),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "error_file": origin.filename if origin else "",
        "error_line": origin.lineno if origin else "",
        "error_function": origin.name if origin else "",
    }

    if origin is not None:
        error["message"] = (
            f"{error['message']} "
            f"(at {Path(origin.filename).name}:{origin.lineno} in {origin.name})"
        )

    return error


@enforce_types(error_log_path=(str, Path), errors=list)
def _append_daily_k_errors(error_log_path, errors):
    """Append captured daily K processing errors to a tab-delimited log file."""
    error_log = Path(error_log_path)
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8") as log_file:
        for error in errors:
            log_file.write(
                "{timestamp}\t{site}\t{date}\t{error_type}\t{error_file}\t{error_line}\t{error_function}\t{message}\n".format(**error)
            )


@enforce_types(output_file=(str, Path))
def _save_daily_k_csv(kvals, output_file):
    """Write computed K values to a CSV file with ``time`` and ``K_index`` columns."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(kvals["time"]),
            "K_index": kvals["var1"],
        }
    )
    df.to_csv(output_file, index=False)


@enforce_types(
    date=(str, pd.Timestamp, np.datetime64),
    site_code=str,
    archive_path_builder=Callable,
    output_path_builder=Callable,
    site_metadata=(dict, type(None)),
)
def _run_daily_k_for_date(
    date,
    site_code,
    archive_path_builder,
    output_path_builder,
    site_metadata,
):
    """Run daily K generation for one day and save the resulting CSV file."""
    kvals = daily_K(
        time=date,
        site_code=site_code,
        archive_path_builder=archive_path_builder,
        site_metadata=site_metadata,
    )
    date = _utc_day(date)
    date_tokens = date.strftime("%Y-%m-%d").split("-")
    output_dir = Path(output_path_builder(date_tokens))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{site_code}{date:%Y%m%d}.csv"
    _save_daily_k_csv(kvals, output_file)
    return output_file


@enforce_types(
    date=(str, pd.Timestamp, np.datetime64),
    site_code=str,
    archive_path_builder=Callable,
    output_path_builder=Callable,
    site_metadata=(dict, type(None)),
)
def _run_daily_k_for_date_with_error_capture(
    date,
    site_code,
    archive_path_builder,
    output_path_builder,
    site_metadata,
):
    """Run daily K generation for one day and return either an output path or a structured error."""
    date = _utc_day(date)
    try:
        output_file = _run_daily_k_for_date(
            date=date,
            site_code=site_code,
            archive_path_builder=archive_path_builder,
            output_path_builder=output_path_builder,
            site_metadata=site_metadata,
        )
        return {
            "date": date,
            "output_file": output_file,
            "error": None,
        }
    except Exception as exc:
        return {
            "date": date,
            "output_file": None,
            "error": _build_daily_k_error_record(exc, site_code=site_code, date=date),
        }


@enforce_types(
    site_code=str,
    archive_path_builder=Callable,
    output_path_builder=Callable,
    start=(str, pd.Timestamp, np.datetime64),
    end=(str, pd.Timestamp, np.datetime64),
    site_metadata=(dict, type(None)),
    max_workers=(int, type(None)),
    error_log_path=(str, Path, type(None)),
)
def daily_K_full_archive(
    site_code,
    archive_path_builder=lambda date: "./magnetometer_archive/{}/{}/{}/iaga2002/".format(
        *date
    ),
    output_path_builder=lambda date: "./magnetometer_archive/{}/{}/{}/k_index/".format(
        *date
    ),
    start="2025-01-01",
    end="2026-01-01",
    site_metadata=None,
    max_workers=None,
    error_log_path="live_scripts/daily_k_errors.log",
):
    """
    Compute daily K CSV outputs for every day in a date range.

    Parameters
    ----------
    site_code : str
        Site identifier used for archive lookup and output filenames.
    archive_path_builder : collections.abc.Callable, optional
        Function that builds an archive directory from ``[YYYY, MM, DD]`` tokens.
    output_path_builder : collections.abc.Callable, optional
        Function that builds an output directory from ``[YYYY, MM, DD]`` tokens.
    start : str or pandas.Timestamp or numpy.datetime64, optional
        Inclusive start date.
    end : str or pandas.Timestamp or numpy.datetime64, optional
        Exclusive end date.
    site_metadata : dict or None, optional
        Pre-resolved site metadata. When omitted it is looked up from ``site_code``.
    max_workers : int or None, optional
        Number of parallel workers. ``None`` picks a bounded default.
    error_log_path : str or pathlib.Path or None, optional
        File that receives one line per failed day. ``None`` disables logging.

    Returns
    -------
    tuple[list, list]
        Successful outputs as ``(date, output_file)`` pairs and captured errors.

    Raises
    ------
    ValueError
        If ``end`` is not later than ``start``.
    """
    start = _utc_day(start)
    end = _utc_day(end)
    if end <= start:
        raise ValueError("'end' must be later than 'start'.")

    if site_metadata is None:
        site_metadata = get_site_metadata(site_code)

    dates = pd.date_range(start=start, end=end - pd.Timedelta(days=1), freq="1D")
    if len(dates) == 0:
        return [], []

    if error_log_path is not None:
        error_log = Path(error_log_path)
        error_log.parent.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        max_workers = min(32, len(dates), max(1, os.cpu_count() or 1))
    else:
        max_workers = max(1, min(int(max_workers), len(dates)))

    results = []
    errors = []

    if max_workers == 1:
        for date in dates:
            try:
                results.append(
                    (
                        pd.Timestamp(date),
                        _run_daily_k_for_date(
                            date=date,
                            site_code=site_code,
                            archive_path_builder=archive_path_builder,
                            output_path_builder=output_path_builder,
                            site_metadata=site_metadata,
                        ),
                    )
                )
            except Exception as exc:
                error = _build_daily_k_error_record(exc, site_code=site_code, date=date)
                errors.append(error)
                if error_log_path is not None:
                    _append_daily_k_errors(error_log_path, [error])
    else:
        with tqdm_joblib(
            total=len(dates),
            desc_prefix=f"Computing daily K for {site_code}",
            unit="day",
        ):
            job_results = Parallel(n_jobs=max_workers, prefer='processes', backend="loky")(
                delayed(_run_daily_k_for_date_with_error_capture)(
                    date,
                    site_code,
                    archive_path_builder,
                    output_path_builder,
                    site_metadata,
                )
                for date in dates
            )

        for job_result in job_results:
            if job_result["error"] is None:
                results.append((job_result["date"], job_result["output_file"]))
            else:
                errors.append(job_result["error"])
                if error_log_path is not None:
                    _append_daily_k_errors(error_log_path, [job_result["error"]])

    results.sort(key=lambda item: item[0])
    errors.sort(key=lambda item: item["date"])

    return results, errors
def daily_K_plots_full_archive(
    site_code,
    archive_path_builder=lambda date: "./magnetometer_archive/{}/{}/{}/k_index/".format(
        *date
    ),
    output_path_builder=lambda date: "./magnetometer_archive/{}/{}/{}/png/".format(
        *date
    ),
    start="2025-01-01",
    end="2026-01-01",
    site_metadata=None,
    max_workers=None,
    error_log_path="live_scripts/daily_k_plot_errors.log",
    logo_path=None,
):
    """
    Save daily K plots for every day in a date range.

    Parameters
    ----------
    site_code : str
        Site identifier used for archive lookup and output filenames.
    archive_path_builder : collections.abc.Callable, optional
        Function that builds an archive directory from ``[YYYY, MM, DD]`` tokens.
    output_path_builder : collections.abc.Callable, optional
        Function that builds an output directory from ``[YYYY, MM, DD]`` tokens.
    start : str or pandas.Timestamp or numpy.datetime64, optional
        Inclusive start date.
    end : str or pandas.Timestamp or numpy.datetime64, optional
        Exclusive end date.
    site_metadata : dict or None, optional
        Pre-resolved site metadata. When omitted it is looked up from ``site_code``.
    max_workers : int or None, optional
        Number of parallel workers. ``None`` picks a bounded default.
    error_log_path : str or pathlib.Path or None, optional
        File that receives one line per failed day. ``None`` disables logging.

    Returns
    -------
    tuple[list, list]
        Successful outputs as ``(date, output_file)`` pairs and captured errors.

    Raises
    ------
    ValueError
        If ``end`` is not later than ``start``.
    """
    start = _utc_day(start)
    end = _utc_day(end)
    if end <= start:
        raise ValueError("'end' must be later than 'start'.")

    if site_metadata is None:
        site_metadata = get_site_metadata(site_code)

    dates = pd.date_range(start=start, end=end - pd.Timedelta(days=1), freq="1D")
    if len(dates) == 0:
        return [], []

    if error_log_path is not None:
        error_log = Path(error_log_path)
        error_log.parent.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        max_workers = min(32, len(dates), max(1, os.cpu_count() or 1))
    else:
        max_workers = max(1, min(int(max_workers), len(dates)))

    results = []
    errors = []

    def process_date(date):
        try:
            files = [os.path.join(archive_path_builder(date_.strftime("%Y-%m-%d").split("-")), f"{site_code}{date_:%Y%m%d}.csv") for date_ in [date - pd.Timedelta(days=3), date - pd.Timedelta(days=2), date - pd.Timedelta(days=1), date]]
            files = [file for file in files if os.path.isfile(file)]
            if not len(files):
                raise FileNotFoundError(f"No K CSV files found for {site_code} or dates in {[date - pd.Timedelta(days=3), date - pd.Timedelta(days=2), date - pd.Timedelta(days=1), date]}.")
            
            K= pd.concat(pd.read_csv(file, parse_dates=["time"]) for file in files)
            K.rename(columns={'K_index': 'var1'}, inplace=True)
            
            fig, ax = plot_k(K, colorbar=False, show_logo=False, auto_xlim=False)
            ax.set_xlim(date - pd.Timedelta(days=2), date + pd.Timedelta(days=1))
            met = get_site_metadata(site_code)
            fig.suptitle(f"MagIE {met['station_name']} Local K Index", y=.95)
            ax.set_ylabel('K Index')
            fig.set_dpi(96)
            fig.canvas.draw()

            output_dir = Path(output_path_builder(date.strftime("%Y-%m-%d").split("-")))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{site_code}{date:%Y%m%d}_kindex_magpy.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return {
                "date": date,
                "output_file": output_file,
                "error": None,
            }
        except Exception as exc:
            return {
                "date": date,
                "output_file": None,
                "error": _build_daily_k_error_record(exc, site_code=site_code, date=date),
            }

    if max_workers == 1:
        for date in dates:
            result = process_date(date)
            if result["error"] is None:
                results.append((result["date"], result["output_file"]))
            else:
                errors.append(result["error"])
                if error_log_path is not None:
                    _append_daily_k_errors(error_log_path, [result["error"]])
    else:
        with tqdm_joblib(
            total=len(dates),
            desc_prefix=f"Generating daily K plots for {site_code}",
            unit="day",
        ):
            job_results = Parallel(n_jobs=max_workers, prefer='processes', backend="loky")(
                delayed(process_date)(date) for date in dates
            )

        for result in job_results:
            if result["error"] is None:
                results.append((result["date"], result["output_file"]))
            else:
                errors.append(result["error"])
                if error_log_path is not None:
                    _append_daily_k_errors(error_log_path, [result["error"]])
    results.sort(key=lambda item: item[0])
    errors.sort(key=lambda item: item["date"])

    return results, errors

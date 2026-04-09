import pandas as pd
from magie.utils import enforce_types
import numpy as np
from magie.file_conversions import (magie2iaga2002, _iaga_header_record,
                                    _infer_iaga_interval_type, _normalise_iaga_numeric,
                                    _iaga_comment_record, _format_iaga_component_series, _iaga_filename)
from magpy.stream import read, DataStream

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
    import re
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


import pandas as pd
import numpy as np

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

    end_time = start_time + pd.Timedelta(days=1)
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
    return "\n".join(lines) + "\n", filename

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
            df = pd.concat([pd.DataFrame(columns=df.columns,
                                        data=[[pd.Timestamp(df.Date_UTC.min().to_numpy().astype('datetime64[D]').astype('datetime64[ns]')),
                                                site_code] + [np.nan] * (len(df.columns) - 2)]), df])
            df= magie2iaga2002(df, 'dun')
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
        df = pd.concat([pd.DataFrame(columns=df.columns,
                                     data=[[pd.Timestamp(df.Date_UTC.min().to_numpy().astype('datetime64[D]').astype('datetime64[ns]')),
                                            site_code] + [np.nan] * (len(df.columns) - 2)]), df])
        df= magie2iaga2002(df, 'dun')
    return df

@enforce_types(
    now_time=(pd.Timestamp, np.datetime64, str),
    site_code=str,
    path_prefix=str,
)
def live_k(now_time, site_code, path_prefix='https://data.magie.ie/', site_metadata=None):
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
    import tempfile
    from magie.utils import get_site_metadata
    from magpy.core import activity as act
    import os


    start_time = pd.Timestamp(now_time).floor('1D')-pd.Timedelta(4, 'D')
    end_time = pd.Timestamp(now_time).ceil('1D')
    with tempfile.TemporaryDirectory(prefix="live_mags_download") as tmpdir:
        for date in np.arange(start_time, end_time, np.timedelta64(1, 'D')):
            data, filename=_get_live(date, site_code, path_prefix=path_prefix)
            with open(tmpdir +'/'+ filename, 'w') as file:
                file.write(data)
        data, filename= build_empty_iaga_window(end_time,
                        iaga_code=site_code, sampling_step_seconds=_sampling_step_seconds_from_header(data))
        with open(f"{tmpdir}/{filename}", 'w') as file:
            file.write(data)
        data= read(f"{tmpdir}/*{filename.split('.')[-1]}")
    data= data.filter()
    if site_metadata is None:
        site_metadata= get_site_metadata(site_code)

        
    data = act.K_fmi(data, K9_limit=site_metadata['k9_threshold'], longitude=site_metadata['geodetic_longitude'], step_size=60)
    # data= data.trim(starttime=str(start_time.date()), endtime=str(end_time.date()))
    data['var1']= np.where(data['var1']>=0, data['var1'], np.nan)
    return data.trim(starttime=(start_time + pd.Timedelta(1, 'D')).to_numpy(), endtime=end_time.to_numpy())

@enforce_types(
    K_data=DataStream,
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

    # Three-hour bars colored by category
    bars= ax.bar(K_data['time'].astype('datetime64[ns]'), K_data['var1'], edgecolor='black', width=np.timedelta64(3, 'h'),
           color=K_cmap(norm(K_data['var1'])), align='edge', lw=2, zorder=4)
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
    ax.set_xlim(K_data['time'].astype('datetime64[D]').min(), K_data['time'].astype('datetime64[D]').max()+np.timedelta64(1, 'D'))
    ax.set_ylim(-.05, 9.2)
    logo = mpimg.imread("../MagIE-logo.png")
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

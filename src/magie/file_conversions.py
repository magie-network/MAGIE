import pandas as pd
import numpy as np
from magie.utils import enforce_types, get_site_metadata


def _iaga_header_record(label, value):
    """Build a 70-character IAGA-2002 header record."""
    return f"{f' {label:<23}{value}'[:69]:<69}|"


def _iaga_comment_record(comment):
    """Build a 70-character IAGA-2002 comment record."""
    return f"{f' #{comment}'[:69]:<69}|"


def _normalise_iaga_numeric(value, precision=3):
    """Format optional numeric header values for IAGA-2002 headers."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.{precision}f}"


def _infer_iaga_interval_type(times):
    """Infer the IAGA interval label and recommended file extension."""
    step = _infer_iaga_step_seconds(times)
    if step is None:
        return "Unknown", "sec"

    if np.isclose(step, 1.0):
        return "1-second instantaneous", "sec"
    if np.isclose(step, 60.0):
        return "1-minute (00:00 - 00:59)", "min"
    if np.isclose(step, 3600.0):
        return "1-hour (00 - 59)", "hor"
    if step < 1.0:
        milliseconds = int(round(step * 1000))
        return f"{milliseconds} millisecond (instantaneous values)", "sec"
    if step < 60.0:
        return f"{step:g}-second instantaneous", "sec"
    if step < 3600.0:
        minutes = step / 60.0
        return f"{minutes:g}-minute", "min"
    if step < 86400.0:
        hours = step / 3600.0
        return f"{hours:g}-hour", "hor"
    return f"{step / 86400.0:g}-day", "day"


def _infer_iaga_step_seconds(times):
    """Infer the base sampling step in seconds from the smallest positive delta."""
    if len(times) < 2:
        return None

    deltas = times.sort_values().diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return None

    return float(deltas.min())


def _regularize_iaga_times(file, step_seconds=None):
    """Reindex onto a complete regular time grid so missing timestamps are explicit."""
    if step_seconds is None:
        step_seconds = _infer_iaga_step_seconds(file["Date_UTC"])
    if step_seconds is None:
        return file

    file = file.drop_duplicates(subset="Date_UTC", keep="first")
    full_index = pd.date_range(
        start=file["Date_UTC"].iloc[0],
        end=file["Date_UTC"].iloc[-1],
        freq=pd.to_timedelta(step_seconds, unit="s"),
    )

    return (
        file.set_index("Date_UTC")
        .reindex(full_index)
        .rename_axis("Date_UTC")
        .reset_index()
    )


def _iaga_type_code(data_type):
    """Map a human-readable IAGA data type to its filename code."""
    normalised = data_type.strip().lower()
    if normalised.startswith("definitive"):
        return "d"
    if normalised.startswith("quasi"):
        return "q"
    if normalised.startswith("variation"):
        return "v"
    return "p"


def _iaga_filename(iaga_code, start_time, data_type, interval_extension):
    """Build a recommended IAGA-2002 filename."""
    code = iaga_code.lower()[:3] if iaga_code else "xxx"
    type_code = _iaga_type_code(data_type)

    if interval_extension in {"sec", "min"}:
        stem = f"{code}{start_time:%Y%m%d}{type_code}{interval_extension}"
    elif interval_extension == "hor":
        stem = f"{code}{start_time:%Y%m}{type_code}{interval_extension}"
    elif interval_extension == "day":
        stem = f"{code}{start_time:%Y}{type_code}{interval_extension}"
    else:
        stem = f"{code}{start_time:%Y}{type_code}mon"
        interval_extension = "mon"

    return f"{stem}.{interval_extension}"


def _format_iaga_component(value, missing_value=999999.00):
    """Format one magnetic component using the IAGA-2002 F9.2 layout."""
    if pd.isna(value):
        value = missing_value
    return f"{float(value):9.2f}"


def _format_iaga_component_series(values, missing_value=999999.00):
    """Format a component column using the IAGA-2002 F9.2 layout."""
    return values.fillna(missing_value).map("{:9.2f}".format)


def _derived_total_field(file):
    """Return observed total field or derive it from X, Y, Z when needed."""
    if "TFG" in file.columns and file["TFG"].notna().any():
        return file["TFG"], True

    required = [col for col in ["Bx", "By", "Bz"] if col in file.columns]
    if len(required) != 3:
        return pd.Series(np.nan, index=file.index), False

    magnitude = np.sqrt(file["Bx"] ** 2 + file["By"] ** 2 + file["Bz"] ** 2)
    return magnitude.where(file[["Bx", "By", "Bz"]].notna().all(axis=1)), False


@enforce_types(file=(str, pd.DataFrame), site=str)
def eziemag2magie_legacy(file, site='dun'):
    """
    Convert EZIE-MAG data into the legacy MagIE text export format.

    Parameters
    ----------
    file : str or pandas.DataFrame
        CSV path or DataFrame containing EZIE-MAG data with an ``iso_time`` column.
    site : str, optional
        Site prefix used when constructing the output filename.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        A DataFrame formatted for the legacy export and the suggested output filename.
    """
    if isinstance(file, str):
        file = pd.read_csv(file, parse_dates=['iso_time'])
    file["iso_time"] = pd.to_datetime(file["iso_time"], format="mixed")
    site = site + '_' + file['stid'].iloc[0]
    rename_dict = {
        'iso_time': ';Date & Time',
        'BxnT': 'Bx',
        'BynT': 'By',
        'BznT': 'Bz',}
    file.rename(columns=rename_dict, inplace=True)
    file['Index#'] = range(len(file))
    # Legacy format uses a sentinel for missing values.
    file[file.isnull()] = 99999.99

    filename = f"{site}{''.join(file[';Date & Time'].dt.date.astype(str).iloc[0].split('-'))}.txt"
    file[';Date & Time'] = file[';Date & Time'].dt.strftime('%d/%m/%Y %H:%M:%S')
    return file[[';Date & Time', 'Index#', 'Bx', 'By', 'Bz']], filename


@enforce_types(file=(str, pd.DataFrame), site=str)
def eziemag2magie(file, site='dun'):
    """
    Convert EZIE-MAG data into the standard MagIE DataFrame format.

    Parameters
    ----------
    file : str or pandas.DataFrame
        CSV path or DataFrame containing EZIE-MAG data with ``iso_time`` and magnetics.
    site : str, optional
        Site prefix used to build the ``Site`` column.

    Returns
    -------
    pandas.DataFrame
        A DataFrame in MagIE standard column order.
    """
    if isinstance(file, str):
        file = pd.read_csv(file, parse_dates=['iso_time'])
    file["iso_time"] = pd.to_datetime(file["iso_time"], format="mixed")

    columns = ['Date_UTC', 'Site', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG', 'TE', 'Volts']

    rename_dict = {
        'iso_time': 'Date_UTC',
        'stid': 'Site',
        'BxnT': 'Bx',
        'BynT': 'By',
        'BznT': 'Bz',
        'ctemp': 'TE',
    }

    file = file.rename(columns=rename_dict)
    file['Site'] = site + '_' + file['Site']

    # Ensure all required columns exist (fill missing with 0)
    for col in columns:
        if col not in file.columns:
            file[col] = np.nan

    # Reorder columns to match expected output format
    file = file[columns]
    return file


@enforce_types(file=(str, pd.DataFrame))
def magie2magie_legacy(file):
    """
    Convert a MagIE-standard dataset to the legacy MagIE export format.

    Parameters
    ----------
    file : str or pandas.DataFrame
        CSV path or DataFrame containing MagIE data with ``Date_UTC`` and ``Site``.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        A DataFrame formatted for the legacy export and the suggested output filename.
    """
    if isinstance(file, str):
        file = pd.read_csv(file, parse_dates=['Date_UTC'])
    rename_dict = {
        'Date_UTC': ';Date & Time',}
    file = file.rename(columns=rename_dict)
    file['Index#'] = range(len(file))
    columns = [
        col for col in [';Date & Time', 'Index#', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG', 'TE', 'Volts']
        if col in file.columns and file[col].notnull().sum()
    ]
    # Legacy format uses a sentinel for missing values.
    file[file.isnull()] = 99999.99
    site = file['Site'].iloc[0]
    filename = f"{site}{''.join(file[';Date & Time'].dt.date.astype(str).iloc[0].split('-'))}.txt"
    file[';Date & Time'] = file[';Date & Time'].dt.strftime('%d/%m/%Y %H:%M:%S')
    # columns= [';Date & Time', 'Index#', 'Bx', 'By', 'Bz']
    return file[columns], filename


@enforce_types(
    file=(str, pd.DataFrame),
    source_of_data=str,
    station_name=str,
    iaga_code=str,
    geodetic_latitude=(int, float, str, type(None)),
    geodetic_longitude=(int, float, str, type(None)),
    elevation=(int, float, str, type(None)),
    reported=str,
    sensor_orientation=str,
    digital_sampling=(str, type(None)),
    data_interval_type=(str, type(None)),
    data_type=str,
    publication_date=(str, type(None)),
    sampling_step_seconds=(int, float, type(None)),
)
def magie2iaga2002(
    file,
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
    sampling_step_seconds=None,
):
    """
    Convert a MagIE-standard dataset to an IAGA-2002 text export.

    Parameters
    ----------
    file : str or pandas.DataFrame
        CSV path or DataFrame containing MagIE data with ``Date_UTC`` and magnetic fields.
    source_of_data : str, optional
        Institute or project responsible for the data.
    station_name : str, optional
        Human-readable station name written into the IAGA header.
    iaga_code : str, optional
        Three-letter observatory code used in the header and column names.
    geodetic_latitude, geodetic_longitude, elevation : float, str, or None, optional
        Station metadata written into the mandatory IAGA header fields.
    reported : str, optional
        Reported field-element code. Defaults to ``XYZF``.
    sensor_orientation : str, optional
        Physical sensor orientation written into the header.
    digital_sampling : str or None, optional
        Sampling description for the header. Inferred from cadence when omitted.
    data_interval_type : str or None, optional
        Interval description for the header. Inferred from cadence when omitted.
    data_type : str, optional
        IAGA data type label such as ``Provisional`` or ``Definitive``.
    publication_date : str or None, optional
        Optional publication date in ``YYYY-MM-DD`` format.
    comments : sequence[str] or None, optional
        Extra IAGA comment records to append before the data header.
    sampling_step_seconds : float or None, optional
        Explicit sampling cadence in seconds. When omitted, the cadence is
        inferred from the timestamps in ``Date_UTC``.

    Returns
    -------
    tuple[str, str]
        The full IAGA-2002 file contents and a recommended output filename.
    """
    if isinstance(file, str):
        file = pd.read_csv(file, parse_dates=['Date_UTC'])

    file = file.copy()
    file["Date_UTC"] = pd.to_datetime(file["Date_UTC"], format="mixed")
    file = file.sort_values("Date_UTC").reset_index(drop=True)
    file = _regularize_iaga_times(file, step_seconds=sampling_step_seconds)

    site_metadata = None
    if "Site" in file.columns and file["Site"].notna().any():
        site_metadata = get_site_metadata(file["Site"].iloc[0], longitude_style="signed")

    if station_name == "Unknown station":
        if site_metadata is not None:
            station_name = site_metadata["station_name"]
        elif "Site" in file.columns and file["Site"].notna().any():
            station_name = str(file["Site"].iloc[0])

    if iaga_code == "XXX":
        if site_metadata is not None:
            iaga_code = site_metadata["iaga_code"]
        elif "Site" in file.columns and file["Site"].notna().any():
            site_code = "".join(ch for ch in str(file["Site"].iloc[0]).upper() if ch.isalpha())
            if site_code:
                iaga_code = site_code[:3]

    if geodetic_latitude is None and site_metadata is not None:
        geodetic_latitude = site_metadata["geodetic_latitude"]
    if geodetic_longitude is None and site_metadata is not None:
        geodetic_longitude = site_metadata["geodetic_longitude"]

    interval_label, interval_extension = _infer_iaga_interval_type(file["Date_UTC"])
    if digital_sampling is None:
        digital_sampling = "1 second" if interval_extension == "sec" else interval_label.split(" ", 1)[0]
    if data_interval_type is None:
        data_interval_type = interval_label

    if comments is None:
        comments = [
            "H = squareroot(X*X + Y*Y), D = atan2(Y, X), I = atan2(Z, H)",
        ]

    total_field, has_observed_f = _derived_total_field(file)
    component_map = {
        "X": file["Bx"] if "Bx" in file.columns else pd.Series(np.nan, index=file.index),
        "Y": file["By"] if "By" in file.columns else pd.Series(np.nan, index=file.index),
        "Z": file["Bz"] if "Bz" in file.columns else pd.Series(np.nan, index=file.index),
        "F": total_field,
    }

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

    timestamps = file["Date_UTC"]
    date_str = timestamps.dt.strftime("%Y-%m-%d")
    time_str = timestamps.dt.strftime("%H:%M:%S.%f").str[:-3]
    doy = timestamps.dt.dayofyear.astype(str).str.zfill(3)

    formatted_components = []
    for component in reported:
        series = component_map.get(component)
        if series is None:
            series = pd.Series(np.nan, index=file.index)
        missing_value = 999999.00
        formatted_components.append(_format_iaga_component_series(series, missing_value=missing_value))

    data_lines = date_str + " " + time_str + " " + doy
    if formatted_components:
        component_block = formatted_components[0]
        for formatted in formatted_components[1:]:
            component_block = component_block + " " + formatted
        data_lines = data_lines + " " + component_block

    lines.extend(data_lines.str.slice(0, 70).str.pad(70, side="right").tolist())

    filename = _iaga_filename(code, file["Date_UTC"].iloc[0], data_type, interval_extension)
    return "\n".join(lines) + "\n", filename

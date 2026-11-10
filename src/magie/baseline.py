"""Retrieve historical HDZ reference levels and apply them to MagPy streams.

The Valentia live reader preserves the legacy declination scale by default.
Baseline application centres each UTC day independently and returns XYZ with
full-precision derived F, suitable for the package's MagPy IAGA writer.
"""

from collections.abc import Mapping
from datetime import date, datetime
from io import StringIO
from os import PathLike
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests
from magpy.stream import DataStream

from magie.utils import enforce_types


INTERMAGNET_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"

# Match the package's path contracts and accept explicit calendar values,
# including NumPy/Pandas timestamps. Numeric epochs are intentionally ambiguous.
_PATH_TYPES = (str, PathLike)
_TIME_TYPES = (str, date, datetime, pd.Timestamp, np.datetime64)


@enforce_types(
    time=_TIME_TYPES,
    site=str,
)
def get_intermagnet_baseline(time, site):
    """
    Retrieve median HDZ levels for the seven UTC days 16 through 10 days ago.

    Parameters
    ----------
    time : str, datetime.date, datetime.datetime, pandas.Timestamp or numpy.datetime64
        Reference date, rather than the date of a single measurement file.
        Naive values are treated as UTC; aware values are converted to UTC
        before selecting the calendar day. For example, 28 October selects
        12 through 18 October, inclusive.
    site : str
        Three-letter IAGA observatory code, e.g. 'VAL'. Whitespace is stripped
        and letters are normalized to uppercase.

    Returns
    -------
    dict[str, float]
        Median levels under keys 'H', 'D' and 'Z', rounded to two decimals.
        H and Z are in nT; D is in degrees, converted from IAGA arcminutes.

    Raises
    ------
    TypeError
        An argument does not match the documented type.
    ValueError
        The reference date or site is invalid, or the response has missing
        HDZ columns, duplicate/out-of-window timestamps, or no finite sample
        for a component on any requested day.
    requests.RequestException
        The HTTPS request fails, times out or returns an unsuccessful status.

    Notes
    -----
    Requests reported (variation) minute data in HDZF orientation. The
    ten-day delay preserves the legacy allowance for publication lag; it is
    not a guarantee of availability. Medians use all usable measurements in
    the window, rather than daily medians. The IAGA 99999 marker and nonfinite
    values are excluded. Requiring one sample per component per day does not
    establish full-day completeness. No local cache is read or written.

    Examples
    --------
    >>> reference = get_intermagnet_baseline('2026-10-28', 'VAL')  # doctest: +SKIP
    """
    site = site.strip().upper()
    if len(site) != 3 or not site.isascii() or not site.isalpha():
        raise ValueError("site must be a three-letter IAGA observatory code")
    # Select UTC calendar days so local offsets cannot change the archive window.
    reference = pd.Timestamp(time)
    if pd.isna(reference):
        raise ValueError("time must be a valid reference date")
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    else:
        reference = reference.tz_convert("UTC")
    start = reference.normalize() - pd.Timedelta(days=16)
    end = start + pd.Timedelta(days=7)

    # Request the complete window once, with an explicit orientation and timeout.
    response = requests.get(
        INTERMAGNET_URL,
        params={
            "Request": "GetData",
            "format": "iaga2002",
            "observatoryIagaCode": site,
            "samplesPerDay": "Minute",
            "dataStartDate": start.strftime("%Y-%m-%d"),
            "dataDuration": 7,
            "publicationState": "reported",
            "orientation": "HDZF",
        },
        timeout=60,
    )
    response.raise_for_status()

    # IAGA metadata precedes the DATE/TIME column header and varies in length.
    lines = response.text.splitlines()
    header = next(
        (i for i, line in enumerate(lines) if line.split()[:2] == ["DATE", "TIME"]),
        None,
    )
    if header is None:
        raise ValueError(f"INTERMAGNET returned no IAGA-2002 data for {site}")
    data = pd.read_csv(StringIO("\n".join(lines[header:])), sep=r"\s+")
    columns = [f"{site}{component}" for component in ("H", "D", "Z")]
    if not set(columns).issubset(data.columns):
        raise ValueError(f"INTERMAGNET response is missing H/D/Z columns for {site}")

    timestamps = pd.to_datetime(data["DATE"] + " " + data["TIME"], utc=True)
    if not ((timestamps >= start) & (timestamps < end)).all():
        raise ValueError("INTERMAGNET returned measurements outside the requested window")
    if timestamps.duplicated().any():
        raise ValueError("INTERMAGNET returned duplicate measurement timestamps")

    values = data[columns].apply(pd.to_numeric, errors="raise")
    # IAGA-2002 uses 99999.00 for missing measurements.
    values = values.mask((values == 99999.0) | ~np.isfinite(values))
    # Reject an empty or partially unavailable window rather than silently
    # calculating a reference from fewer calendar days.
    expected_days = pd.date_range(start, periods=7, freq="D")
    daily_counts = values.groupby(timestamps.dt.normalize()).count()
    if not (daily_counts.reindex(expected_days, fill_value=0) > 0).all().all():
        raise ValueError(
            f"INTERMAGNET has insufficient H/D/Z data for {site} "
            "in the requested seven days"
        )

    medians = values.median()
    return {
        "H": round(float(medians[f"{site}H"]), 2),
        "D": round(float(medians[f"{site}D"]) / 60, 2),
        "Z": round(float(medians[f"{site}Z"]), 2),
    }


@enforce_types(
    path=_PATH_TYPES,
    date=_TIME_TYPES,
    declination_unit=(str, type(None)),
)
def read_val_realtime(path, date, *, declination_unit="legacy"):
    """
    Read Valentia's hour/minute/HDZ live feed into a MagPy DataStream.

    Parameters
    ----------
    path : str or os.PathLike
        Text file with header ``%hh %mm H[nT] D[unit] Z[nT]`` and five
        whitespace-separated numeric columns: hour, minute, H, D and Z.
    date : str, datetime.date, datetime.datetime, pandas.Timestamp or numpy.datetime64
        Observation day, required because the file contains no date. Naive
        values are UTC; aware values are converted to UTC before selecting
        the day. Neither the clock nor file modification time is consulted.
    declination_unit : str or None, default 'legacy'
        D scale: 'legacy' divides raw D by 360, reproducing
        val_baseline_adder.py; 'deg' leaves angles unchanged; 'arcmin' divides
        by 60; 'arcsec' divides by 3600. 'nT' retains transverse variation in
        nT. None selects the unit written in the file header.

    Returns
    -------
    magpy.stream.DataStream
        Minute-cadence VAL variation data with UTC timestamps, H and Z in nT,
        and NaN placeholders for F. Angular inputs use HDZF with D in degrees.
        Transverse nT inputs use HEZF with E in nT; apply the baseline before
        treating that column as angular D. Rows and gaps are preserved, and
        missing/nonfinite measurements are represented by NaN.

    Raises
    ------
    TypeError
        An argument does not match the documented type.
    ValueError
        The header, column count, date or D scale is invalid, or hours/minutes
        are nonintegral, out of range or duplicated.
    OSError
        The source file cannot be opened or read.

    Notes
    -----
    The legacy comment describes arcseconds but its divisor is 360, not 3600.
    The default preserves that numerical behaviour without asserting physical
    units. Header-based interpretation is opt-in because the supplied feed
    labels D as nT. No baseline is downloaded or applied by this reader.

    Examples
    --------
    >>> raw = read_val_realtime('val_mag_realtime.txt', '2026-10-28')  # doctest: +SKIP
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig") as source:
        header = source.readline().strip()
        match = re.fullmatch(r"%hh\s+%mm\s+H\[nT\]\s+D\[([^\]]+)\]\s+Z\[nT\]", header)
        if match is None:
            raise ValueError("Expected Valentia header: %hh %mm H[nT] D[unit] Z[nT]")
        rows = np.loadtxt(source, ndmin=2)
    if rows.size == 0 or rows.shape[1] != 5:
        raise ValueError("Valentia realtime data must contain five columns and at least one row")
    unit = match.group(1) if declination_unit is None else declination_unit
    if unit not in {"legacy", "nT", "deg", "arcmin", "arcsec"}:
        raise ValueError("declination_unit must be legacy, nT, deg, arcmin or arcsec")

    # An explicit day avoids misdating a copied snapshot or a feed near midnight.
    day = pd.Timestamp(date)
    if pd.isna(day):
        raise ValueError("date must specify a valid observation day")
    day = (day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")).normalize()
    clock = rows[:, :2]
    if (not np.isfinite(clock).all() or (clock != np.floor(clock)).any()
            or (clock < 0).any() or (clock[:, 0] > 23).any() or (clock[:, 1] > 59).any()):
        raise ValueError("Hour/minute values must be integers in 00:00 through 23:59")
    minutes = clock[:, 0].astype(int) * 60 + clock[:, 1].astype(int)
    if len(np.unique(minutes)) != len(minutes):
        raise ValueError("Valentia realtime data contains duplicate timestamps")
    times = day.tz_localize(None).to_datetime64().astype("datetime64[us]") + minutes * np.timedelta64(1, "m")
    # Mask raw sentinels before scaling; otherwise D=99999 would become an angle.
    values = rows[:, 2:].copy()
    values[(values == 99999.0) | ~np.isfinite(values)] = np.nan
    divisors = {"legacy": 360., "arcmin": 60., "arcsec": 3600.}
    if unit in divisors:
        values[:, 1] /= divisors[unit]

    # MagPy stores H/E or H/D in x/y; metadata distinguishes field from angle.
    transverse = unit == "nT"
    array = [np.asarray([]) for _ in DataStream().KEYLIST]
    for key, column in {"time": times.astype(object), "x": values[:, 0],
                        "y": values[:, 1], "z": values[:, 2],
                        "f": np.full(len(times), np.nan)}.items():
        array[DataStream().KEYLIST.index(key)] = column
    return DataStream(header={
        "StationIAGAcode": "VAL", "StationName": "Valentia",
        "DataComponents": "HEZF" if transverse else "HDZF",
        "DataType": "variation", "DataSamplingRate": 60.,
        "col-x": "H", "col-y": "E" if transverse else "D", "col-z": "Z", "col-f": "F",
        "unit-col-x": "nT", "unit-col-y": "nT" if transverse else "deg",
        "unit-col-z": "nT", "unit-col-f": "nT",
        "ValRealtimeDUnit": unit,
    }, ndarray=np.asarray(array, dtype=object))


@enforce_types(
    data=DataStream,
    baseline=Mapping,
)
def apply_intermagnet_baseline(data, baseline):
    """
    Replace daily HDZ mean levels with a reference baseline and return XYZ.

    Parameters
    ----------
    data : magpy.stream.DataStream
        Nonempty stream with UTC timestamps and aligned x/y/z columns.
        DataComponents must identify XYZ, HDZ or HEZ, optionally suffixed by
        F (derived total field) or S (independent scalar field). HDZ requires
        unit-col-y='deg'; HEZ requires unit-col-y='nT'. XYZ fields are in nT.
    baseline : collections.abc.Mapping
        Reference levels with keys 'H', 'D' and 'Z', as returned by
        get_intermagnet_baseline. Values must convert to finite float scalars;
        H and Z are in nT and D is in degrees. H must be positive for HEZ.
        The supplied reference is used for every UTC day in the input.

    Returns
    -------
    magpy.stream.DataStream
        Independent XYZ copy with nT component units, baseline metadata and
        the original timestamps, row count and unrelated columns. A populated
        derived F column is recalculated from full-precision XYZ. Independent
        scalar S is preserved. Existing gaps remain gaps; no interpolation,
        file reading/writing or network requests occur.

    Raises
    ------
    TypeError
        An argument does not match the documented type.
    ValueError
        The baseline, component metadata, units or timestamps are invalid;
        a required column has the wrong length; or an HDZ component has no
        usable samples anywhere in the stream.

    Notes
    -----
    XYZ is converted to HDZ before processing. HEZ uses the small-angle
    relation D_degrees = degrees(E_nT / baseline_H_nT), following the
    INTERMAGNET variation relation E = H_reference * D_rad. See
    https://tech-man.intermagnet.org/stable/appendices/dataformats.html.
    MagPy HDZ already uses degrees; the reader's legacy /360 scaling is not
    repeated here.

    Each UTC day and component uses ``value - finite_mean + baseline``.
    Changing the input sample window changes the mean and thus every corrected
    value in that day. Declinations are unwrapped chronologically before
    centring. The 99999 marker and nonfinite values are excluded independently
    for each component. A day without usable values retains its gaps, provided
    that component has usable values elsewhere in the stream.

    Values are not rounded. In particular, F is derived before IAGA output
    rounding, unlike the legacy text-to-IAGA path that derived F from rounded
    XYZ. Applying a reference level does not promote reported data to definitive
    observatory data.

    Examples
    --------
    >>> reference = {'H': 19588.70, 'D': -4.64, 'Z': 44900.90}
    >>> corrected = apply_intermagnet_baseline(raw, reference)  # doctest: +SKIP
    """
    try:
        offsets = np.asarray([baseline[key] for key in ("H", "D", "Z")], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("baseline must contain numeric H, D and Z values") from exc
    if offsets.shape != (3,) or not np.isfinite(offsets).all():
        raise ValueError("baseline H, D and Z must be finite scalars")

    components = data.header.get("DataComponents", "").upper()
    if components not in {"XYZ", "XYZF", "XYZS", "HDZ", "HDZF", "HDZS", "HEZ", "HEZF", "HEZS"}:
        raise ValueError("DataComponents must identify XYZ, HDZ or HEZ measurements")
    if components.startswith("HDZ") and data.header.get("unit-col-y") != "deg":
        raise ValueError("HDZ declination must use MagPy degrees (unit-col-y='deg')")
    if components.startswith("HEZ"):
        if data.header.get("unit-col-y") != "nT":
            raise ValueError("HEZ transverse variation must use nT")
        if offsets[0] <= 0:
            raise ValueError("baseline H must be positive to convert transverse nT to degrees")

    times = np.asarray(data["time"], dtype="datetime64[us]")
    if times.ndim != 1 or not len(times) or np.isnat(times).any():
        raise ValueError("data must contain nonempty, valid UTC timestamps")
    # Mask sentinels before coordinate conversion and keep caller-owned data intact.
    corrected = data.copy()
    for key in ("x", "y", "z"):
        values = np.asarray(data[key], dtype=float).copy()
        if values.shape != times.shape:
            raise ValueError("x, y and z must share the DataStream time axis")
        values[(values == 99999.0) | ~np.isfinite(values)] = np.nan
        corrected[key] = values
    if components.startswith("XYZ"):
        corrected = corrected.xyz2hdz()
    elif components.startswith("HEZ"):
        corrected["y"] = np.rad2deg(np.asarray(corrected["y"], dtype=float) / offsets[0])
        corrected.header.update({
            "DataComponents": components.replace("HEZ", "HDZ"),
            "col-y": "D", "unit-col-y": "deg",
            "BaselineDConversion": "degrees(E_nT / baseline_H_nT)",
        })

    # Mirror the legacy daily-file centring when multiple days share one stream.
    days = times.astype("datetime64[D]")
    for key, offset in zip(("x", "y", "z"), offsets):
        values = np.asarray(corrected[key], dtype=float).copy()
        if not np.isfinite(values).any():
            raise ValueError(f"No usable {key} component samples for baseline application")
        for day in np.unique(days):
            valid = (days == day) & np.isfinite(values)
            if not valid.any():
                continue
            centred = values[valid]
            if key == "y":
                # Work chronologically even if the supplied stream is unsorted.
                order = np.argsort(times[valid])
                unwrapped = np.rad2deg(np.unwrap(np.deg2rad(centred[order])))
                centred = centred.copy()
                centred[order] = unwrapped
            values[valid] = centred - centred.mean() + offset
        corrected[key] = values

    corrected = corrected.hdz2xyz()
    # Derive F from full-precision XYZ; scalar S is a separate measurement.
    if components.endswith("F") and len(data["f"]):
        if np.asarray(data["f"]).shape != times.shape:
            raise ValueError("f must share the DataStream time axis")
        corrected["f"] = np.sqrt(sum(
            np.asarray(corrected[key], dtype=float) ** 2 for key in ("x", "y", "z")
        ))
        corrected.header.update({"col-f": "F", "unit-col-f": "nT"})
    corrected.header.update({
        "BaselineH": float(offsets[0]),
        "BaselineD": float(offsets[1]),
        "BaselineZ": float(offsets[2]),
        "BaselineMethod": "daily HDZ mean removal followed by reference baseline addition",
    })
    return corrected

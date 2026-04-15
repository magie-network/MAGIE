import functools
from datetime import datetime

import numpy as np
import pandas as pd

from magie.utils import enforce_types


class ArgumentError(Exception):
    """
    Error raised when date or filename arguments cannot be interpreted.
    """
    pass


@enforce_types(func=type(lambda: None))
def dates2npdate(func):
    """
    Wrap a function so its first ``date`` argument is normalized to ``numpy.datetime64``.

    Parameters
    ----------
    func : collections.abc.Callable
        Function whose first positional argument is a date-like value.

    Returns
    -------
    collections.abc.Callable
        Wrapped function that receives ``numpy.datetime64`` input.

    Raises
    ------
    ArgumentError
        If the provided date type is not supported.
    """
    @functools.wraps(func)
    def wrapper(date, *args):
        if isinstance(date, np.datetime64):
            pass
        elif isinstance(date, pd.Timestamp):
            date = date.to_numpy()
        elif isinstance(date, datetime):
            date = np.datetime64(date)
        else:
            raise ArgumentError(
                f"date type yet not known, update code or change format. Current type: {type(date)}"
            )
        return func(date, *args)

    return wrapper


@dates2npdate
@enforce_types(date=(np.datetime64, pd.Timestamp, datetime))
def date2filename(date):
    """
    Convert a date-like value to the underscore-delimited filename timestamp format.

    Parameters
    ----------
    date : numpy.datetime64 or pandas.Timestamp or datetime.datetime
        Date-time value to encode.

    Returns
    -------
    str
        Timestamp string in ``YYYY_MM_DD_HH_MM_SS``-style format.
    """
    import re
    return "_".join(
        "_".join(re.split("   |-|:", f'{date.astype("datetime64[m]").tolist()}')).split(" ")
    )


@enforce_types(filename=str)
def filename2date(filename):
    """
    Parse an underscore-delimited timestamp string into ``numpy.datetime64``.

    Parameters
    ----------
    filename : str
        Timestamp string in ``YYYY_MM_DD_HH_MM_SS``-style format.

    Returns
    -------
    numpy.datetime64
        Parsed timestamp value.
    """
    date = np.empty(3).astype(str)
    time = np.empty(3).astype(str)
    date[0], date[1], date[2], time[0], time[1], time[2] = filename.split("_")
    return np.datetime64("-".join(date) + "T" + ":".join(time))

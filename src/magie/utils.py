import inspect
import functools
import os
import json
from collections.abc import Callable
from tqdm import tqdm
import importlib.resources as importlib_resources


SITE_METADATA = {
    "dunsink": {
        "station_name": "Dunsink",
        "iaga_code": "DUN",
        "site_code": "dun",
        "geodetic_latitude": 53.38,
        "geodetic_longitude": -6.34,
        "k9_threshold": 570,
    },
    "dunsink EZIE Mag": {
        "station_name": "Dunsink EZIE Mag",
        "iaga_code": "DUN",
        "site_code": "dun",
        "geodetic_latitude": 53.38,
        "geodetic_longitude": -6.34,
        "k9_threshold": 570,
    },
    "valentia": {
        "station_name": "Valentia",
        "iaga_code": "VAL",
        "site_code": "val",
        "geodetic_latitude": 51.94,
        "geodetic_longitude": -10.24,
        "k9_threshold": 480,
    },
    "armagh": {
        "station_name": "Armagh",
        "iaga_code": "ARM",
        "site_code": "arm",
        "geodetic_latitude": 54.34,
        "geodetic_longitude": -6.66,
        "k9_threshold": 630,
    },
}

SITE_ALIASES = {
    "dun": "dunsink",
    "dunsink": "dunsink",
    "dun_eziemag": "dunsink EZIE Mag",
    "dunsink_eziemag": "dunsink EZIE Mag",
    "val": "valentia",
    "valentia": "valentia",
    "arm": "armagh",
    "armagh": "armagh",
}

@functools.lru_cache(maxsize=1)
def _load_site_thresholds():
    """Load packaged site K9 thresholds once and cache them."""
    pkg = __package__ or "magie"
    try:
        json_path = importlib_resources.files(pkg).joinpath("site_thresholds.json")
        with json_path.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, AttributeError):
        with importlib_resources.open_text(pkg, "site_thresholds.json") as f:
            return json.load(f)
        
def normalise_site_name(site):
    """Map a site label such as ``dun_test`` or ``Valentia`` to a known site key."""
    if site is None:
        return None

    label = str(site).strip().lower()
    if not label:
        return None

    for separator in ("_", "-", " "):
        label = label.split(separator, 1)[0]

    return SITE_ALIASES.get(label)


def get_site_metadata(site, longitude_style="signed"):
    """
    Return known metadata for a site.

    Parameters
    ----------
    site : str
        Site label or alias.
    longitude_style : {"signed", "360"}, optional
        Longitude convention for the returned metadata.
    """
    site_key = normalise_site_name(site)
    if site_key is None:
        return None

    metadata = SITE_METADATA[site_key].copy()
    if longitude_style == "360":
        metadata["geodetic_longitude"] = metadata["geodetic_longitude"] % 360
    elif longitude_style != "signed":
        raise ValueError(f"Unknown longitude style: {longitude_style}")
    metadata["latitude"] = metadata["geodetic_latitude"]
    metadata["longitude"] = metadata["geodetic_longitude"]
    metadata["site_key"] = site_key
    return metadata





def enforce_types(**type_map):
    """
    Lightweight runtime argument type checking.

    Usage:
        @enforce_types(path=str, site_code=str, opts=(dict, type(None)))
        def func(path, site_code, opts=None, **kwargs):
            ...

    Each key is the parameter name; each value is either a single type
    or a tuple of allowed types.
    """
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            for name, expected in type_map.items():
                if name not in bound.arguments:
                    continue

                value = bound.arguments[name]
                # Allow None when explicitly included via type(None)
                if isinstance(expected, tuple):
                    ok = isinstance(value, expected)
                    expected_names = ", ".join(t.__name__ for t in expected)
                else:
                    ok = isinstance(value, expected)
                    expected_names = expected.__name__

                if not ok:
                    raise TypeError(
                        f"Argument '{name}' to {func.__name__}() must be instance of "
                        f"{expected_names}, got {type(value).__name__}"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator

@enforce_types(
    inputstr=str,
    positive_answer=str,
    negative_answer=str,
)
def validinput(inputstr, positive_answer, negative_answer):
    """
    Ask for a yes/no style response and enforce one of two allowed answers.

    Parameters
    ----------
    inputstr : str
        Prompt shown to the user.
    positive_answer : str
        Accepted value that maps to True.
    negative_answer : str
        Accepted value that maps to False.

    Returns
    -------
    bool
        True for ``positive_answer``, False for ``negative_answer``.

    Examples
    --------
    >>> validinput('Continue?', 'y', 'n')  # doctest: +SKIP
    True
    """
    answer= input(inputstr+'\n')
    if answer==positive_answer:
        return True
    elif answer== negative_answer:
        return False
    else:
        print('Invalid response should be either '+ str(positive_answer)+ ' or ' +str(negative_answer))
        return validinput(inputstr, positive_answer, negative_answer)


@enforce_types(
    root=str,
    func=Callable,
    followlinks=bool,
    post_func=(Callable, type(None)),
    endings=(str, tuple, list, type(None)),
    show_progress=bool,
)
def apply_to_files(root, func, followlinks=False, post_func=None, endings=None, show_progress=True):
    """
    Walk a directory tree and apply a function to every file path found.

    Parameters
    ----------
    root : str
        Root directory to traverse.
    func : collections.abc.Callable
        Function that accepts a file path and returns a value.
    followlinks : bool, optional
        Whether to follow symbolic links when walking the tree.
    post_func : collections.abc.Callable or None, optional
        Optional function applied to the output of ``func`` for each file.
    endings : str or tuple or list, optional
        Only include files whose names end with one of these suffixes.
    show_progress : bool, optional
        Whether to display a progress bar using ``tqdm``.

    Returns
    -------
    list
        List of results from calling ``func`` (and ``post_func`` if provided)
        on each file path.
    """
    results = []
    if isinstance(endings, str):
        endings = (endings,)
    if isinstance(endings, list):
        endings = tuple(endings)

    file_paths = []
    for dirpath, _, filenames in os.walk(root, followlinks=followlinks):
        for name in filenames:
            if endings and not name.endswith(endings):
                continue
            file_paths.append(os.path.join(dirpath, name))
    file_paths.sort()

    # Apply the provided function to every file path in the tree.
    iterable = tqdm(file_paths, desc="Processing files", unit="file") if show_progress else file_paths
    for path in iterable:
        result = func(path)
        if post_func is not None:
            result = post_func(result, path)
        results.append(result)
    return results

import pandas as pd
import numpy as np
from magie.utils import enforce_types


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

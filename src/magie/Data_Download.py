# Imports
import numpy as np
import requests
from datetime import datetime as dt
import os
import pandas as pd
import time
# Handling import errors for GitHub repositories
def validinput(inputstr, positive_answer, negative_answer):
    answer = input(inputstr+'\n')
    if answer == positive_answer:
        return True
    elif answer == negative_answer:
        return False
    else:
        print('Invalid response should be either ' + str(positive_answer) + ' or ' + str(negative_answer))
        return validinput(inputstr, positive_answer, negative_answer)

from magie.Filename_tools import date2filename
import warnings

# Creates a tqdm-based progressbar if available; otherwise a non-functioning one (identity)
from tqdm import tqdm

def progressbar(iterable, max_value=None, **kwargs):
    """
    Wraps tqdm to keep the same API as the old `progressbar` package.

    Parameters
    ----------
    iterable : iterable
        The iterable to wrap.
    max_value : int, optional
        The total number of iterations (mapped to tqdm's `total` argument).
    **kwargs :
        Additional keyword arguments passed to tqdm.
    """
    if max_value is not None:
        return tqdm(iterable, total=max_value, **kwargs)
    else:
        return tqdm(iterable, **kwargs)

from pandas.errors import ParserError
from urllib.request import urlretrieve
import sys

def download_progress_hook(count, block_size, total_size):
    """
    Report hook to display a progress bar for downloading.

    :param count: Current block number being downloaded.
    :param block_size: Size of each block (in bytes).
    :param total_size: Total size of the file (in bytes).
    """
    # Calculate percentage of the download
    downloaded_size = count * block_size
    percentage = min(100, downloaded_size * 100 / total_size)

    # Create a simple progress bar
    progress_bar = f"\rDownloading: {percentage:.2f}% [{downloaded_size}/{total_size} bytes]"

    # Update the progress on the same line
    sys.stdout.write(progress_bar)
    sys.stdout.flush()

    # When download is complete
    if downloaded_size >= total_size:
        print("\nDownload complete!")

def download(url, file_name):
    try:
        return urlretrieve(url, file_name, reporthook=download_progress_hook)
    except ConnectionError:
        time.sleep(1)
        return urlretrieve(url, file_name, reporthook=download_progress_hook)

def exists_check(url, filename):
    try:
        return requests.get(f"{url}{filename}").status_code
    except requests.exceptions.ConnectionError:
        time.sleep(1)
        return exists_check(url, filename)

def Download_MAGIE(start, end, sites=['arm', 'dun', 'val', 'bir'], save_file_name=False):
    """
    Downloads MAGIE data for specified sites and date range, and saves it to a file.

    Parameters:
    start (numpy.datetime64): The start date of the data to be downloaded.
    end (numpy.datetime64): The end date of the data to be downloaded.
    sites (list of str, optional): List of site codes to download data from. Defaults to ['arm', 'dun', 'val'].
    save_file_name (str or bool, optional): Filename to save the downloaded data. If False, a default name will be generated. Defaults to False.

    Raises:
    ValueError: If no functioning url is found for years from 2000 to now and thus earliest and latest year for data coverage cannot be found
    FileExistsError: If the specified save file already exists and the user chooses not to append to it.
    
    Returns:
    str: The name of the file where the data is saved.
    """
    url_prefix = 'https://data.magie.ie/'
    
    # Check for the earliest available year
    min_year = 0
    for i in range(2000, dt.now().year, 1):
        if requests.get(f"{url_prefix}{i}/").status_code < 400:
            min_year = i
            break
    
    # Raise an error if no year is found
    if not min_year:
        raise ValueError(f'Unable to find the first year available in the range 2000-{dt.now().year}')
    
    # Check for the latest available year
    max_year = 0
    for i in range(dt.now().year, 2000, -1):
        if requests.get(f"{url_prefix}{i}/").status_code < 400:
            max_year = i
            break
    
    # Raise an error if no year is found
    if not max_year:
        raise ValueError(f'Unable to find the last year available in the range 2000-{dt.now().year}')
    
    # Adjust the start date if it is earlier than the first available year
    if start < np.datetime64(f'{min_year}-01-01'):
        warnings.warn(
            f'Start time is less than the first year available from site: {min_year}. '
            'The download will begin at the first year available.'
        )
        start = np.datetime64(f'{min_year}-01-01')
    
    # Adjust the end date if it is later than the last available year
    if end > np.datetime64(f'{max_year}-12-31'):
        warnings.warn(
            f'End time is greater than the last year available from site: {max_year}. '
            'The download will end at the last year available.'
        )
        end = np.datetime64(f'{max_year}-12-31')
    
    # Create a default filename if none is provided
    if not save_file_name:
        save_file_name = '_'.join(sites) + f'{date2filename(start)[:-9]}_to_{date2filename(end)[:-9]}.hdf5'
    
    # Check if the save file already exists
    if os.path.isfile(save_file_name):
        if not validinput(f'save file: {save_file_name} already exists. Append to pre-existing file? (y/n)', 'y', 'n'):
            raise FileExistsError(
                'Please either provide a new save file path using "save_file_name" argument '
                'or delete existing file and rerun'
            )
        else:
            warnings.warn(f'Appending to pre-existing file: {save_file_name}', UserWarning)
    
    # Define column names for the data
    columns = ['Date_UTC', 'Index', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG', 'TE', 'Volts']
    drop_index = columns.copy()
    drop_index[1] = 'Site'
    
    # Construct an array of dates to download
    dates = np.array(
        [start + np.timedelta64(i, 'D') for i in range((end - start).astype('timedelta64[D]').astype(int) + 1)]
    )
    
    # Loop through each date and download the data (using tqdm-backed progressbar)
    for date in progressbar(dates, max_value=len(dates)):
        date = date.astype('datetime64[D]').astype(str).split('-')
        url = url_prefix + '{}/{}/{}/txt/'.format(*date)
        # loop through each site
        for site in sites:
            filename = site + '{}{}{}.txt'.format(*date)
            
            # Check if the file exists on the server
            if exists_check(url, filename) >= 400:
                warnings.warn(f'File not found for site= {site} on ' + '{}-{}-{}'.format(*date[::-1]))
                continue
            # Download the file using wget
            download(f'{url}{filename}', filename)
            # File bug handling for empty tabs appearing on some lines
            try:
                # Try to read the file into a DataFrame
                file = pd.read_csv(
                    filename,
                    delimiter='\t',
                    names=columns,
                    skiprows=1,
                    parse_dates=['Date_UTC'],
                    dayfirst=True,
                    index_col=False
                ).replace(99.99999e3, np.nan)
                file['Site'] = [site] * len(file)
            except ParserError:
                # Handle ParserError by modifying the file content and re-reading it
                with open(filename, mode='r') as F:
                    f = F.read()
                new_f = '\n'.join(f.split('\t\n'))
                with open(filename, mode='w') as F:
                    F.write(new_f)
                
                if os.path.isfile('bad_files.txt'):
                    with open('bad_files.txt', 'r') as f:
                        bad_files = f.read()
                        f.close()
                else:
                    bad_files = ''
                
                bad_files += 'parser error:' + filename + '\n'
                
                with open('bad_files.txt', 'w') as f:
                    f.write(bad_files)
                    f.close()
                file = pd.read_csv(
                    filename,
                    delimiter='\t',
                    names=columns,
                    skiprows=1,
                    parse_dates=['Date_UTC'],
                    dayfirst=True,
                    index_col=False
                ).replace(99.99999e3, np.nan)
                file['Site'] = [site] * len(file)
            
            # File bug handling when last line of the file contains incomplete data points
            # Check for object data types and handle them appropriately
            if 'O' in [file[col].dtype for col in file.columns[:-1]]:
                if os.path.isfile('bad_files.txt'):
                    with open('bad_files.txt', 'r') as f:
                        bad_files = f.read()
                        f.close()
                else:
                    bad_files = ''
                
                bad_files += filename + '\n'
                
                with open('bad_files.txt', 'w') as f:
                    f.write(bad_files)
                    f.close()
                file = file.loc[0:len(file)-2]
                
                if file['Date_UTC'].dtype == 'O':
                    file['Date_UTC'] = pd.to_datetime(file.Date_UTC, dayfirst=True)
                
                for column in columns[2:]:
                    if file[column].dtype == 'O':
                        file[column] = file[column].astype('float64')
            
            # Save the data to an HDF5 file using specified columns
            # (the index column is removed as it is meaningless in a merged file)
            file[drop_index].to_hdf(
                save_file_name,
                key='main',
                mode='a',
                append=True,
                format='t',
                data_columns=True
            )
            
            # Remove the downloaded file to save space
            os.remove(filename)
    return save_file_name

if __name__ == '__main__':
    Download_MAGIE(np.datetime64('2022-01-01T00:00'), np.datetime64('2025-01-01T00:00'))

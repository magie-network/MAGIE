# Imports
import numpy as np
import requests
from datetime import datetime as dt
import os
import pandas as pd
import time
from urllib.parse import urlencode
from io import StringIO
# Handling import errors for GitHub repositories
def validinput(inputstr, positive_answer, negative_answer):
    answer= input(inputstr+'\n')
    if answer==positive_answer:
        return True
    elif answer== negative_answer:
        return False
    else:
        print('Invalid response should be either' + str(positive_answer) + ' or ' + str(negative_answer))
        return validinput(inputstr, positive_answer, negative_answer)
import numpy as np
class ArgumentError(Exception):
    """
    Custom exception class to handle argument-related errors.
    Raised when the function receives invalid input.
    """
    pass

def dates2npdate(func):
    def wrapper(date, *args):
        import pandas as pd
        from datetime import datetime
        if isinstance(date, np.datetime64):
            pass
        elif isinstance(date, pd.Timestamp):
            date=date.to_numpy()
        elif isinstance(date, datetime):
            date=np.datetime64(date)
        else:
            raise ArgumentError(f'date type yet not known, update code or change format. Current type: {type(date)}')
        return func(date, *args)
    return wrapper
@dates2npdate
def date2filename(date):
    import re
    return '_'.join('_'.join(re.split('   |-|:', f'{date.astype("datetime64[m]").tolist()}')).split(' '))
def filename2date(filename):
    date=np.empty(3).astype(str)
    time=np.empty(3).astype(str)
    date[0], date[1], date[2], time[0], time[1], time[2]= filename.split('_')
    return np.datetime64('-'.join(date)+'T'+':'.join(time))
import warnings
#Creates non functioning progressbar if the import of the progressbar package is not possible
try:
    from progressbar import progressbar
except ImportError:
    def progressbar(*args, **kwargs):
        return args[0]
from pandas.errors import ParserError
from platform import system
# If Linux download uses wget
if 'Linux' in system():
    def download(url, filename):
        return os.system(f'wget {url}')
# If not Linux use urllib.request.urlretrieve
else:
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
        warnings.warn(f'Start time is less than the first year available from site: {min_year}. '
                      'The download will begin at the first year available.')
        start = np.datetime64(f'{min_year}-01-01')
 
    # Adjust the end date if it is later than the last available year
    if end > np.datetime64(f'{max_year}-12-31'):
        warnings.warn(f'End time is greater than the last year available from site: {max_year}. '
                      'The download will end at the last year available.')
        end = np.datetime64(f'{max_year}-12-31')
  
    # Create a default filename if none is provided
    if not save_file_name:
        save_file_name = '_'.join(sites) + f'{date2filename(start)[:-9]}_to_{date2filename(end)[:-9]}.hdf5'
  
    # Check if the save file already exists
    if os.path.isfile(save_file_name):
        if not validinput(f'save file: {save_file_name} already exists. Append to pre-existing file? (y/n)', 'y', 'n'):
            raise FileExistsError('Please either provide a new save file path using "save_file_name" argument '
                                  'or delete existing file and rerun')
        else:
            warnings.warn(f'Appending to pre-existing file: {save_file_name}', UserWarning)
  
    # Define column names for the data
    columns = ['Date_UTC', 'Index', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG', 'TE', 'Volts']
    drop_index = columns.copy()
    drop_index[1] = 'Site'
    
    # Construct an array of dates to download
    dates = np.array([start + np.timedelta64(i, 'D') for i in range((end - start).astype('timedelta64[D]').astype(int) + 1)])
    
    # Loop through each date and download the data
    for date in progressbar(dates, max_value=len(dates)):
        date = date.astype('datetime64[D]').astype(str).split('-')
        url = url_prefix + '{}/{}/{}/txt/'.format(*date)
        # loop through each site
        for site in sites:
            filename = site + '{}{}{}.txt'.format(*date)
            
            # Check if the file exists on the server
            if requests.get(f"{url}{filename}").status_code >= 400:
                warnings.warn(f'File not found for site= {site} on ' + '{}-{}-{}'.format(*date[::-1]))
                continue
            # Download the file using wget
            download(f'{url}{filename}', filename)
            # File bug handling for empty tabs appearing on some lines
            try:
                # Try to read the file into a DataFrame
                file = pd.read_csv(filename, delimiter='\t', 
                                   names=columns, 
                                   skiprows=1, parse_dates=['Date_UTC'], dayfirst=True, index_col=False).replace(99.99999e3, np.nan)
                file['Site'] = [site] * len(file)

                # Ensure timestamps are UTC-aware and second-precision
                file['Date_UTC'] = pd.to_datetime(file['Date_UTC'], utc=True).dt.floor('s')
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                if start_ts.tzinfo is None:
                    start_ts = start_ts.tz_localize('UTC')
                if end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize('UTC')
                start_ts = start_ts.floor('s')
                end_ts = end_ts.floor('s')

                # ⏱ Filter to requested time range (hour/min precision)
                file = file[(file['Date_UTC'] >= start_ts) & (file['Date_UTC'] <= end_ts)]

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
                file = pd.read_csv(filename, delimiter='\t', 
                                   names=columns, 
                                   skiprows=1, parse_dates=['Date_UTC'], dayfirst=True, index_col=False).replace(99.99999e3, np.nan)
                file['Site'] = [site] * len(file)

                # Ensure timestamps are UTC-aware and second-precision
                file['Date_UTC'] = pd.to_datetime(file['Date_UTC'], utc=True).dt.floor('s')
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                if start_ts.tzinfo is None:
                    start_ts = start_ts.tz_localize('UTC')
                if end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize('UTC')
                start_ts = start_ts.floor('s')
                end_ts = end_ts.floor('s')

                # Apply filter to remove one second
                file = file[(file['Date_UTC'] >= start_ts) & (file['Date_UTC'] <= end_ts)]
       
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
          
            # Save the data to an HDF5 file using specified columns (the index column is removed as it is meaningless in a merged file)
            file[drop_index].to_hdf(save_file_name, key='main', mode='a', append=True, format='t', data_columns=True)
           
            # Remove the downloaded file to save space
            os.remove(filename)
    return save_file_name


def get_GIN_data(dataStartDate, iagaSites, dataDuration, orientation,
                 samples="Minute", dataFormat="iaga2002", state="best-avail"):
    """
    Get magnetic data using http from Edinburgh GIN for a list of
    observatories over continous days and saves iaga2002 format data 
    to a dataframe without saving to file locally.
    API webservice details are in https://imag-data.bgs.ac.uk/GIN_V1/

    Example API call:
    https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&observatoryIagaCode=VAL&samplesPerDay=Minute
    &dataStartDate=2024-12-31&dataDuration=1&publicationState=best-avail&Format=iaga2002&orientation=XYZF&recordTermination=UNIX
    File name in the format of val20241122dmin.min for definitive (d)
    state where 20241122 is the start date.
    File may contain multiple days of data in one file if dataDuration > 1.

    Parameters:
    -----------
    dataStartDate: datetime.datetime
        A date in the form yyyy-mm-dd. E.g. start = dt.datetime(2025, 11, 23)
    iagaSites: list of str
        List of 3-letter observatory codes to download data from.
        E.g. iagaSites = ['esk', 'ngk', 'val']
    dataDuration: int
        Integer number of days including start date. >= 1 day.
        Duration can cross calender years.
    orienation" str
        Native - use the orientation the data provider sent the data in.
        F calculated from vector data:
        XYZF - cartesian coordinate system for vector data.
        HDZF - cylindrical coordinate system for vector data.
        DIFF - spherical coordinate system for vector data.
        F from an independent instrument:
        XYZS - cartesian coordinate system for vector data.
        HDZS - cylindrical coordinate system for vector data.
        DIFS - spherical coordinate system for vector data.
    samples: str
        default is "Minute" (1440 rows/day).
        "Second" (86400 rows/day) values usually 99999.00.
    dataFormat: str, optional
        default to "iaga2002". Others are "json",  "html" and "wdc".
    state: str optional
        default to "best-avail".
        "best-avail" - the best data available
        "definitive" - definitive data
        "quasi-def" - quasi-definitive data
        "adjusted" - provisional (also called adjusted) data
        "reported" - variometer (also called reported) data

    Raises:
    -------
        RuntimeError in try-except loop for url validity
        HTTP status code
        200	The request was completed successfully.
        301	A resource has been moved.
        400	Bad request. Returned where the request parameters are incorrect.
        404	Requested data is not available.
        500	Server encountered an error trying to process the request.

    Returns:
    --------
        df_sites (pandas.DataFrame): Data content of GIN file.

    """
    urlBase = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
    # dataFrame of IAGA observatory data in iagaSites
    dfSites = pd.DataFrame()
    dfs = []
    # loop through each observatory
    for i, obs in enumerate(iagaSites):
        obs = obs.upper() if obs.islower() else obs
        params = {
                 "Request": "GetData",
                 "observatoryIagaCode": obs,
                 "samplesPerDay": samples,
                 "dataStartDate": str(dataStartDate),
                 "dataDuration": str(dataDuration),
                 "publicationState": state,
                 "Format": dataFormat,
                 "orientation": orientation,
                 "&recordTermination": "UNIX"}
        urlQuery = urlencode(params)

        url = urlBase + '?' + urlQuery
        resp = requests.get(url)

        # stops code continuing if url is bad
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"HTTP status code {resp.status_code} \n\
                               for {resp.url}") from e

        print(f'{dataDuration} day(s) of {dataStartDate} data for {obs} is \n\
        getting extracted from:\n{url}')

        data = resp.text
        dataLines = data.splitlines()

        headerData = []
        columnHeader = []
        numericLines = []
        afterDataFlag = False
        for line in dataLines:
            lineStrip = line.replace("|", "").strip()
            if afterDataFlag:
                numericLines.append(lineStrip)
                # skip rest of this if-loop body
                continue
            # if afterDataFlag is True then we have already seen DATE row
            if lineStrip.lstrip().startswith('DATE'):
                columnHeader.append(lineStrip)
                afterDataFlag = True
            else:
                headerData.append(lineStrip)

        print('Mandatory file header records')
        for i, line in enumerate(headerData):
            print(f"{i}: [{line}]")

        colNames = columnHeader[0].split()
        print(f'Column header for {obs} is: {colNames}')
        # parse_date is deprecated so not used
        df = pd.read_csv(StringIO("\n".join(numericLines)), sep=r'\s+',
                         names=colNames)
        df['DATE_TIME'] = pd.to_datetime(df.iloc[:, 0] +
                                         ' ' + df.iloc[:, 1])
        df.drop(columns=[df.columns[0], df.columns[1]], inplace=True)
        df.set_index('DATE_TIME', inplace=True, drop=True)
        dfs.append(df)
    dfSites = pd.concat(dfs, axis=1)

    return dfSites


if __name__ == '__main__':
    Download_MAGIE(np.datetime64('2022-01-01T00:00'),
                   np.datetime64('2025-01-01T00:00'))

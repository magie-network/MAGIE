# Imports
import numpy as np
import requests
from datetime import datetime as dt
import os
import pandas as pd
import time
from urllib.parse import urlencode
from io import StringIO
from magie.Filename_tools import date2filename
import warnings
import sys
from urllib.request import urlretrieve
from pandas.errors import ParserError
from magie.utils import validinput, enforce_types
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
import datetime


@enforce_types(
    max_value=(int, type(None)),
)
# Creates a tqdm-based progressbar
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


@enforce_types(
    count=int,
    block_size=int,
    total_size=int,
)
def download_progress_hook(count, block_size, total_size):
    """
    Report hook to display a progress bar for downloading.

    :param count: Current block number being downloaded.
    :param block_size: Size of each block (in bytes).
    :param total_size: Total size of the file (in bytes).
    """
    # Calculate percentage of the download
    downloaded_size = count * block_size
    if total_size != 0:
        percentage = min(100, downloaded_size * 100 / total_size)
    else:
        percentage = 0

    # Create a simple progress bar
    progress_bar = f"\rDownloading: {percentage:.2f}% [{downloaded_size}/{total_size} bytes]"

    # Update the progress on the same line
    sys.stdout.write(progress_bar)
    sys.stdout.flush()

    # When download is complete
    if downloaded_size >= total_size:
        print("\nDownload complete!")


@enforce_types(
    url=str,
    file_name=str,
)
def download(url, file_name):
    """
    Downloads the file at url and saves it to file_name.

    :param url: url of the file to download.
    :param file_name: file_name and path of where to save the downloaded file.
    """
    try:
        return urlretrieve(url, file_name, reporthook=download_progress_hook)
    # If could not connect to file due to connection error we wait one second and retry.
    # This will solve issues where servers have recieved too many requests from user
    except ConnectionError:
        time.sleep(1)
        return urlretrieve(url, file_name, reporthook=download_progress_hook)


@enforce_types(
    url=str,
    filename=str,
)
def exists_check(url, filename):
    """
    Checks if the url exists .

    :param url: url of the file to download.
    :param file_name: file_name and path of where to save the downloaded file.
    """
    try:
        return requests.get(f"{url}{filename}").status_code
    except requests.exceptions.ConnectionError:
        time.sleep(1)
        return exists_check(url, filename)


@enforce_types(
    start=np.datetime64,
    end=np.datetime64,
    sites=list,
    save_file_name=(str, bool),
)
def download_magie(start, end, sites=['arm', 'dun', 'val', 'bir'], save_file_name=False):
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


@enforce_types(
    dataStartDate=dt,
    iagaSites=list,
    dataDuration=int,
    orientation=str,
    samples=str,
    dataFormat=str,
    state=str,
    print_progress=bool,
)
def get_GIN_data(dataStartDate, iagaSites, dataDuration, orientation,
                 samples="Minute", dataFormat="iaga2002", state="best-avail",
                 print_progress=True):
    """
    Get magnetic data using http from Edinburgh GIN for a list of
    observatories over continous days and saves iaga2002 format data
    to a dataframe without saving to file locally.
    API webservice details are in https://imag-data.bgs.ac.uk/GIN_V1/

    Example API call:
    https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&
    observatoryIagaCode=VAL&samplesPerDay=Minute&dataStartDate=2024-12-31&
    dataDuration=1&publicationState=best-avail&Format=iaga2002&
    orientation=XYZF&recordTermination=UNIX

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
    print_progress: bool optional
        default: True
        set to False to quiet print statements

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
        if print_progress:
            print(f'{dataDuration} day(s) of {dataStartDate} data for {obs} \n\
        is getting extracted from:\n{url}')

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
        if print_progress:
            print('Mandatory file header records')
        for i, line in enumerate(headerData):
            print(f"{i}: [{line}]")

        colNames = columnHeader[0].split()
        if print_progress:
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


@enforce_types(printHeader=bool)
def get_SAGE_variometer(printHeader=False):
    """
    Returns the latest 24 hours live 1s Data from Florence Court (FLO)
    British Geological Survey's SWIMMR Activities in Ground Effects (SAGE)
    geomagnetic variometer from a password-protected website.
    Data are updated every 30 minutes

    Login credentials are listed in separate rows in a .env file.
    username=url_user_name
    password=url_password

    Magneic measurements provided for X, Y and Z components in nT
    Orientation is approximately in North, South, East directions
    Latitude: 54.25, Longitude: 352.27
    All times are UT
    Data Provided by the British Geological Survey
    Data are provided as is. It may contain unnatural disturbances and
    periods of missing data due to battery voltage dropping below cut-off
    contact: spaceweather@bgs.ac.uk

    Convert columns to the format available on data.magie.ie as follows:
    Data & Time Index# Bx By Bz

    Parameters:
    -----------
    printHeader:boolean optional

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
        df:pandas
            Live data from BGS Geomagnetic variometer at FLO

    Usage:
    --------
        df = get_SAGE_variometer()
    """
    url = "https://geomag.bgs.ac.uk/SpaceWeather/fl_24hrdata.out"
    # fetch the username and password from the .env file stored
    # in the same path as this script i.e. <src/magie/>
    load_dotenv()
    username = os.getenv("username")
    password = os.getenv("password")

    try:
        resp = requests.get(url, auth=(username, password))
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP status code {resp.status_code} \n\
                            for {resp.url}") from e

    data = resp.text
    dataLines = data.splitlines()

    headerLines = []
    columnHeader = []
    numericLines = []
    isNumericSection = False

    for line in dataLines:
        # removes leading space
        stripped = line.lstrip()

        if isNumericSection:
            numericLines.append(line)
            continue

        if stripped.startswith('Date'):
            # append column header
            columnHeader.append(line)
            # if isNumericSection = True then program has already seen row
            # beginning with "Date"
            isNumericSection = True
        else:
            # numeric data appended here
            headerLines.append(line)

    if printHeader is True:
        print('File Header Records')
        for i, line in enumerate(headerLines):
            print(f"{i}: [{line}]")

    colNames = columnHeader[0].split()
    df = pd.read_csv(
        StringIO("\n".join(numericLines)), sep=r'\s+', names=colNames
        )
    df['Date & Time'] = pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1])
    df['Date & Time'] = df['Date & Time'].dt.strftime("%Y/%m/%d %H:%M:%S")
    df.drop(columns=[df.columns[0], df.columns[1]], inplace=True)
    df.set_index('Date & Time', inplace=True, drop=True)
    df['Index#'] = range(1, len(df) + 1)
    df.rename(columns={"X": "Bx", "Y": "By", "Z": "Bz"}, inplace=True)
    df.index = pd.to_datetime(df.index, format="%Y/%m/%d %H:%M:%S")
    df = df.reindex(columns=['Index#', "Bx", "By", "Bz"])

    return df


@enforce_types(
    day=datetime.date,
    freq=str,
    flag=(int, float),
)
def daily_file_template(day, freq="1s", flag=99999.00):
    """
    Create a full-day DataFrame filled with flag values
    Columns are defined as those of data.magie.ie as follows:
    Data & Time Index# Bx By Bz

    Parameters:
    -----------
    day: datetime.date

    freq: str optional
        defaults to one-second BGS Geomagnetic Variometer Data
        Use "1min" for per minute freqeuncy and "1h" for per hour

    flag: float optional
        indicates missing data either 99999.0 or 99999.00
        depending on component.

    Returns:
    --------
    template: pandas.DataFrame
        Name file header. Pre-populate full day flag values in
        Bx, By and Bz.
     """
    idx = pd.date_range(start=pd.Timestamp(day),
                        end=pd.Timestamp(day) +
                        pd.Timedelta(days=1) -
                        pd.Timedelta(freq),
                        freq=freq
                        )

    n = len(idx)
    template = pd.DataFrame(index=idx)
    template.index.name = "Date & Time"
    template["Index#"] = np.arange(1, n + 1)
    template["Bx"] = flag
    template["By"] = flag
    template["Bz"] = flag
    # only data columns set to float
    template[["Bx", "By", "Bz"]] = template[["Bx", "By", "Bz"]].astype(float)

    return template


@enforce_types(
    df=pd.DataFrame,
    baseDir=str,
    freq=str,
    obs=str,
    flag=(int, float),
    printHeader=bool,
)
def save_SAGE_data(df, baseDir, freq='1s', obs="flo", flag=99999.00,
                   printHeader=False):
    """
    Allocate real data from dataframe and replace the flagged values in
    the day-files. Extracts final timestamp of input dataframe.

    DataFrame with datetime index spanning 24-hours, crossing midnight
    Saves daily files as floYYYYMMDD.txt.
    Each daily file is prepopulated with a full day's timestamps
    And have flag values=99999.0 for Bx, By, Bz
    When real data exists, they overwrite the flagged values
    Gaps are preserved automatically.

    Parameters:
    -----------
    baseDir: str
        base directory for daily variometer files,
        actual files live in its sub-folders

    df (pandas.DataFrame): data indexed with timestamps
        in %Y/%m/%d %H:%M:%S format; E.g: "2026/01/09 12:30:00"
        Columns in the order of: ['Index#', "Bx", "By", "Bz"]

    freq: str optional
        defaults to one-second Florence Court (FLO) data part of SAGE.
        Use "1min" for per minute freqeuncy and "1h" for per hour

    obs: str optional
        Three letters lowercase default to "flo" BGS variometer

    flag: float optional
        indicates missing data either 99999.0 or 99999.00
        depending on component.

    printHeader: boolean optional

    Raises:
    -------
    FileNotFoundError

    Dependencies:
    -------
        Calls daily_file_template function to populate a DataFrame of
        flagged values
        get_SAGE_variometer function stores real BGS data to input df

    Returns:
    --------
        None
    Usage:
    --------
        save_BGS_data(df, outputDir, freq='1s', obs="flo", flag=99999.00)
    """
    if not baseDir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {baseDir}"
            )

    cols = ["Bx", "By", "Bz"]
    for day, oneDayData in df.groupby(df.index.date):
        day = pd.Timestamp(day)
        dataStr = day.strftime("%Y%m%d")
        fname = f"{obs}{dataStr}.txt"
        # Extract date
        dateStr = Path(fname).stem.replace(f"{obs}", "")
        dateObj = dt.strptime(dateStr, "%Y%m%d")

        # Build directory
        targetDir = baseDir / dateObj.strftime("%Y") \
            / dateObj.strftime("%m") \
            / dateObj.strftime("%d") \
            / "txt"

        targetDir.mkdir(parents=True, exist_ok=True)

        # Save file there
        filePath = targetDir / fname

        if filePath.exists():
            dfOneDay = pd.read_csv(
                filePath, sep=r"\s+", index_col=0, parse_dates=True
                )
        else:
            dfOneDay = daily_file_template(day, freq=freq, flag=flag)

        src = oneDayData[cols].copy()
        # creates boolean-mask of rows in src[col] that are non-NaN nor flagged
        # valid rows are True for rows in src, should overwrite dfOneDay
        for col in cols:
            valid = ~src[col].isna() & (src[col] != flag)
            # intersection() only keep timestamps in src that exist in dfOneDay
            matchedIndex = src.index[valid].intersection(dfOneDay.index)
            dfOneDay.loc[matchedIndex, col] = src.loc[matchedIndex, col]

        dfOut = dfOneDay.copy()
        # ensure numeric columns are floats
        dfOut[cols] = dfOut[cols].astype(float)
        dfOut.index = pd.to_datetime(dfOut.index)

        # saves file space-delimited
        dfOut.to_csv(filePath, sep=" ", index=True, float_format="%.2f")
        if printHeader is True:
            print(f"Saved/updated: {filePath.name}")


if __name__ == '__main__':
    download_magie(np.datetime64('2022-01-01T00:00'),
                   np.datetime64('2025-01-01T00:00'))

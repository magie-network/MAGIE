import numpy as np
import requests
from datetime import datetime as dt
import os
import pandas as pd
from File_tools.Filename_tools import date2filename
from General_Tools.user_input_tools import validinput
import warnings
from progressbar import progressbar
from pandas.errors import ParserError
def Download_MAGIE(start, end, sites=['arm', 'dun', 'val'], save_file_name=False):
    url_prefix= 'https://data.magie.ie/'
    #check for earliest year
    min_year=0
    for i in range(2000, dt.now().year, 1):
        if requests.get(f"{url_prefix}{i}/").status_code<400:
            min_year= i
            break
    # Check if first year was found
    if not min_year:
        raise ValueError(f'Unable to find the first year available in the range 2000-{dt.now().year}')
    #check for latest year
    max_year=0
    for i in range(dt.now().year, 2000, -1):
        if requests.get(f"{url_prefix}{i}/").status_code<400:
            max_year= i
            break
    # Check if last year was found
    if not max_year:
        raise ValueError(f'Unable to find the last year available in the range 2000-{dt.now().year}')
    # Check if start date is valid
    if start<np.datetime64(f'{min_year}-01-01'):
        warnings.warn(f'start time is less than the first year available from site: {min_year}.'\
                            'The download will begin at the first year available')
        start= np.datetime64(f'{min_year}-01-01')
    # Check if end date is valid
    if end>np.datetime64(f'{max_year}-12-31'):
        warnings.warn(f'end time is greater than the last year available from site: {max_year}.'\
                            'The download will end at the last year available')
        end= np.datetime64(f'{max_year}-12-31')
    # Create Filename if not specified
    if not save_file_name:
        save_file_name= '_'.join(sites)+f'{date2filename(start)[:-9]}_to_{date2filename(end)[:-9]}.hdf5'
    if os.path.isfile(save_file_name):
        if not validinput('save file: {save_file_name} already exists append to pre-existing file? (y/n)', 
                      'y', 'n'):
            raise FileExistsError('Please either provide a new save file path using "save_file_name" argument'\
                                  'or delete existing file and rerun')
        else:
            warnings.warn(f'Appending to pre-existing file: {save_file_name}', UserWarning)
    columns= ['Date_UTC', 'Index', 'Bx', 'By', 'Bz', 'E1', 'E2', 'E3', 'E4', 'TFG','TE', 'Volts']
    drop_index= columns.copy()
    # drop_index.remove('Index')
    drop_index[1]='Site'
    # Construct array of dates to download
    dates= np.array([start + np.timedelta64(i, 'D') for i in \
                     range((end-start).astype('timedelta64[D]').astype(int)+1)])
    for date in progressbar(dates,max_value=len(dates)):
        date= date.astype('datetime64[D]').astype(str).split('-')
        url= url_prefix + '{}/{}/{}/txt/'.format(*date)
        for site in sites:
            filename= site+'{}{}{}.txt'.format(*date)
            if requests.get(f"{url}{filename}").status_code>=400:
                warnings.warn(f'File not found for site= {site} on '+'{}-{}-{}'.format(*date[::-1]))
                continue
            os.system(f'wget {url}{filename}')
            try:
                file= pd.read_csv(filename, delimiter='\t', 
                                names=columns, 
                                skiprows=1, parse_dates=['Date_UTC'], dayfirst=True, index_col=False).replace(99.99999e3, np.nan)
                file['Site']= [site]*len(file)
            except ParserError:
                F= open(filename, mode='r')
                f= F.read()
                new_f='\n'.join(f.split('\t\n'))
                F.close()
                F= open(filename, mode='w')
                F.write(new_f)
                F.close()
                if os.path.isfile('badfiles.txt'):
                    with open('bad_files.txt', 'r') as f:
                        bad_files= f.read()
                        f.close()
                else:
                    bad_files=''
                bad_files+='parser error:'+ filename+'\n'
                with open('bad_files.txt', 'w') as f:
                    f.write(bad_files)
                    f.close()
                file= pd.read_csv(filename, delimiter='\t', 
                                names=columns, 
                                skiprows=1, parse_dates=['Date_UTC'], dayfirst=True, index_col=False).replace(99.99999e3, np.nan)
                file['Site']= [site]*len(file)
            if 'O' in [file[col].dtype for col in file.columns[:-1]]:
                if os.path.isfile('bad_files.txt'):
                    with open('bad_files.txt', 'r') as f:
                        bad_files= f.read()
                        f.close()
                else:
                    bad_files=''
                bad_files+=filename+'\n'
                with open('bad_files.txt', 'w') as f:
                    f.write(bad_files)
                    f.close()
                file= file.loc[0:len(file)-2]
                if file['Date_UTC'].dtype=='O':
                    file['Date_UTC']= pd.to_datetime(file.Date_UTC, dayfirst=True)
                for column in columns[2:]:
                    if file[column].dtype=='O':
                        file[column]= file[column].astype('float64')
            file[drop_index].to_hdf(save_file_name, key='main',mode='a',append=True,format='t', data_columns=True)
            os.remove(filename)
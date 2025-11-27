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
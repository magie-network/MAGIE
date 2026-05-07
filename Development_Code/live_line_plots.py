import matplotlib.pyplot as plt
from magie.file_conversions import magie_legacy2iaga2002
import pandas as pd
from magie.k_index_magpy import _get_live
import numpy as np
from tempfile import TemporaryDirectory
from magpy.stream import read
from magie.utils import get_site_metadata
from magie.Plotting_Tools import plot_BxByBz, plot_dH



plt.style.use('tableau-colorblind10')


path_prefix= '/home/simon/Documents/magnetometer_archive/'
now_time = pd.Timestamp('2026-04-06').floor('1D')
now_time = pd.Timestamp.now()


site_code= 'val'


with TemporaryDirectory(prefix="live_mags_download") as tmpdir:
    for date in np.arange(now_time-pd.Timedelta('3D'), now_time+pd.Timedelta('1D'), np.timedelta64(1, 'D')):
        data, filename=_get_live(date, site_code)
        # , path_prefix=path_prefix)
        with open(tmpdir +'/'+ filename, 'w') as file:
            file.write(data)

    data= read(f"{tmpdir}/*{filename.split('.')[-1]}")
    # data= data.filter()


fig, _, _, _= plot_BxByBz(data)
met= get_site_metadata(site_code)
fig.suptitle(f"{met['station_name']} 3-Day Bx, By, Bz", size=60, y=.95)

fig, _, _, _= plot_dH(data)
met= get_site_metadata(site_code)
fig.suptitle(f"{met['station_name']} 3-Day D, H, dH", size=60, y=.95)

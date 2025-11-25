"""
Author: Sean Blake, Trinity College Dublin
Editor: John Malone Leigh, Dublin Institute for Advanced Studies


Date: January 2015
Date Edited: September 2019


Email: blakese@tcd.ie, jmalonel@tcd.ie


The purpose of this script is to calculate k indices using modules from k_index_module_pre_houdini.py.


The k-indices (and other plots) are then created and then saved to the DIAS server Houdini.


The script is divided into 3 sections: 
o The normal k-index section,
o the Valentia k-index section and
o the email alert section


The normal k-index section creates plots for Armagh and Birr.


The Valentia k-index section creates plots for Valentia. 
Valentia data is treated differently as we only have baseline subtracted H, D and Z data.


The email alert section reads the values for k-indices from the previous sections and sends the emails,
if a storm is present.
"""
import numpy as np
import datetime
import os
import pandas as pd

#each module needed to run functions in k_index_pre_houdini
from time import strptime
from datetime import timedelta
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
import matplotlib.font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

"""

Author: Sean Blake, Trinity College Dublin
Editor: John Malone Leigh, Dublin Institute for Advanced Studies


Date: January 2015
Date Edited: September 2019


Email: blakese@tcd.ie, jmalonel@tcd.ie


The purpose of this module is to provide functions which will facilitate the calculation of k-indices using the FMI method, as well as the time derivate mag fields. See http://swans.meteo.be/sites/default/files/documentation/TN-RMI-2010-01_K-LOGIC.pdf for more details.


Modules Edited: Send_emailz, do_other_plots, do_k_plots


Modules Added: nan_helper, create_folder and archive_maker


Script is upgraded to work for python 3


With the functions in this module, the FMI method can be used as follows:


1) Get minutely time, bx, by arrays using minutely()



2) Get initial K-indices using k_index_func()



3) Use these with initial_smooth()



4) Smooth the initial_smooth



5) Subtract this from minutely data



6) Get second K-indices from this subtracted data



7) Repeat steps 3-6 to get third set of K-indices



These are the final K-Indices needed. Phew!

"""

from numpy import arange

import numpy

import numpy as np

import datetime

import os

import math

from time import strptime

from scipy.interpolate import InterpolatedUnivariateSpline




import matplotlib.dates as mdates

from matplotlib.ticker import AutoMinorLocator



from matplotlib import pyplot as plt

import matplotlib.font_manager



##########################################################################

##########################################################################

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]





def time2float(x):

    """converts datetime to float, so that interpolation/smoothing can

       be performed"""

    if (type(x) == numpy.ndarray) or (type(x) == list):

        emptyarray = []

        for i in x:

            z = (i - datetime.datetime(1970, 1, 1, 0)).total_seconds()

            emptyarray.append(z)

        emptyarray = numpy.array([emptyarray])

        return emptyarray[0]

    else:

        return (x - datetime.datetime(1970, 1, 1, 0)).total_seconds()



##########################################################################

##########################################################################



def float2time(x):

    """converts array back to datetime so that it can be plotted with time

       on the axis"""

    if (type(x) == numpy.ndarray) or (type(x) == list):

        emptyarray = []

        for i in x:

            z = datetime.datetime.utcfromtimestamp(i)

            emptyarray.append(z)

        emptyarray = numpy.array([emptyarray])

        return emptyarray[0]

    else:

        return datetime.datetime.utcfromtimestamp(x)



##########################################################################

##########################################################################



def timedatez(date, timez):

    """creating 'timedate' array for date specified, from date + time columns"""

    timedate = [] 

    for i in arange(0, len(date)):

        a = date[i] + timez[i]

        c = datetime.datetime(*strptime(a, "%d/%m/%Y%H:%M:%S")[0:6])    

        timedate.append(c)

    return timedate



##########################################################################

##########################################################################



def data_read(saveaddress):

    """ Reads the data in from RTSO file. If the file has not finished a row, 

        it copies the complete rows to a new file, and then reads the data from 

        there.



        Returns date, time, bx, by, bz"""



    try:

        datez, timez = np.loadtxt(saveaddress, dtype = str, usecols = (0, 1), 

                        unpack = True, skiprows = 1)    # getting date from file 

        bx, by, bz = np.loadtxt(saveaddress, usecols = (3, 4, 5), unpack = True,

                        skiprows = 1)       # getting everything else



    except (IndexError, ValueError):

        datez, timez = np.loadtxt(saveaddress, dtype = str, usecols = (0, 1), 

                        unpack = True, skiprows = 2)    # getting date from file

        bx, by, bz = np.loadtxt(saveaddress, usecols = (3, 4, 5), unpack = True,

                         skiprows = 3)          # getting everything else



    return (datez, timez, bx, by, bz)



##########################################################################

##########################################################################
def mag_filter(Bx,By,Bz):
    """
    Filters out noise from cars
    """
    if len(Bx)==len(By) and len(Bx)==len(Bz):
        dF=[]#rate of change of full field
        for i in range(0,len(Bx)-1):

            dF.append(abs(Bx[i+1]-Bx[i])+abs(By[i+1]-By[i])+abs(Bz[i+1]-Bz[i]))
        Bxnew=[]
        Bynew=[]
        print('length',len(dF))
        Bznew=[]
        for j in range(0,len(dF)-60,60):
            if max(dF[j:j+60])>10:
                array=np.full(shape=60,fill_value=99999.99,dtype=np.float)
                for k in array:
                    Bxnew.append(k)
                    Bynew.append(k)
                    Bznew.append(k)
            else:
                for l in range(0,60):
                    Bxnew.append(Bx[j+l])
                    Bynew.append(By[j+l])
                    Bznew.append(Bz[j+l])

    else:
        print('Error in Mag Filter')
    return Bxnew,Bynew,Bznew


def minute_bin(timedate_float, bx, by, bz, n):
    """Bin 1-second magnetic field data into minute means over n days.

    Parameters
    ----------
    timedate_float : array-like
        Unix timestamps (seconds since 1970-01-01) for each 1-second sample.
    bx, by, bz : array-like
        Magnetic field components at 1-second cadence.
    n : int
        Number of days spanned by the data (used to size the accumulator).

    Returns
    -------
    minute_time : list
        Unix timestamps at 1-minute cadence (seconds).
    minute_bx, minute_by, minute_bz : list
        Minute-mean Bx, By, Bz values corresponding to `minute_time`.
    """



    # Gets the start of the day in seconds

    day_seconds = int(timedate_float[0])-int(timedate_float[0])%(24*3600)



    # Creates array of minutes

    minutes = arange(0, n * 1440)

    minutes = (minutes * 60) + day_seconds



    # master is numpy array with columns for bx, by, bz, count and times

    master = np.zeros((n*1440, 5))

    master[:,-1] = minutes



    # loop over times

    for i, v in enumerate(timedate_float):

        # check which master row it belongs to

        index = int((v - day_seconds)/60) #- 1

        # add to each column

        try:

            master[index][3] += 1

            master[index][0] += bx[i]

            master[index][1] += by[i]

            master[index][2] += bz[i]

        except:

            continue



    # now make empty arrays which will be filled

    minute_bx, minute_by, minute_bz, minute_time = [], [], [], []

    for i, v in enumerate(master):

        if v[3] == 0:   # if count = 0, ignore

            continue

        else:           # otherwise, add average to respective array

            minute_bx.append(v[0]/v[3])

            minute_by.append(v[1]/v[3])

            minute_bz.append(v[2]/v[3])

            minute_time.append(v[4])

    

    return minute_time, minute_bx, minute_by, minute_bz



##########################################################################

##########################################################################



def clean2(minute_time, minute_bx, minute_by, minute_bz, sigma_multiple, n):
    """Sigma-clip outliers in hour-long chunks of minute data.

    This routine iteratively removes points that deviate more than
    `sigma_multiple` standard deviations from the mean within each hour
    of data, for all three components simultaneously.

    Note
    ----
    The comment below notes this function is currently unused in the
    operational pipeline, but it is kept here for possible future use.
    """

    #This module is not used, but left for possible future use

    days = int(minute_time[0])/(24*3600)

    day_start = days*24*3600        # start of day in seconds



    # lists to be filled

    bx_chunks, by_chunks, bz_chunks, time_chunks = [], [], [], []



    first_hour = (int(minute_time[0])%(24*3600))/(3600)



    start = 0

    # next code block breaks inputs into hour-long sub-lists

    for i in arange(first_hour, 72, 1):     # loop over 72 hours

        for j in arange(start, len(minute_time), 1):    # loop over data

    

            # if data element is in the next hour, we record its index (j)

            if (minute_time[j] - day_start)/(60*60) > i+1:  



                # get slice of hour for time, bx, by and bz

                time_chunks.append(minute_time[start:j])

                bx_chunks.append(minute_bx[start:j])

                by_chunks.append(minute_by[start:j])

                bz_chunks.append(minute_bz[start:j])



                start = j

                break



    for k in range(n):      # loop over n times

        # loop over bx, by, bz

        for arrayz in (bx_chunks, by_chunks, bz_chunks):

            for index1, value1 in enumerate(arrayz):



                sigma = numpy.std(value1)   # get sigma value for hour chunk

                meanz =np.mean(value1)        # get mean value for hour chunk
                





                for index2, value2 in enumerate(value1[::-1]):

                    # if value > x sigma away from std, delete

                    if abs(value2 - meanz) >= sigma * sigma_multiple:

                        count += 1

                

                        length = len(value1)

                        # delete entries

                        del bx_chunks[index1][length- index2 -1]

                        del by_chunks[index1][length- index2 -1]

                        del bz_chunks[index1][length- index2 -1]

                        del time_chunks[index1][length- index2 -1]



    # finally, collapse 2d lists into 1-d

    time_out = [item for sublist in time_chunks for item1 in time_chunks]

    bx_out = [item for sublist in bx_chunks for item1 in time_chunks]

    by_out = [item for sublist in by_chunks for item1 in time_chunks]

    bz_out = [item for sublist in bz_chunks for item1 in time_chunks]



    return time_out, bx_out, by_out, bz_out



##########################################################################

##########################################################################



def k_index(minute_time, minute_bx, minute_by, k9):
    """Compute local K-indices from 1-minute horizontal magnetic variations.

    The algorithm:
        * Splits the minute data into 3-hour UT blocks.
        * For each block, computes peak-to-peak variation in Bx and By.
        * Uses site-specific k9 value (`k9`) to scale the Niemegk thresholds.
        * Assigns a K-index (0–9, with legacy 0.25 used as a special flag).

    Parameters
    ----------
    minute_time : array-like
        Unix timestamps at 1-minute cadence.
    minute_bx, minute_by : array-like
        Minute means of the horizontal components.
    k9 : float
        Site-specific reference value corresponding to K=9.

    Returns
    -------
    k_index : list of float
        K-value per 3-hour bin (may include 0.25 for very quiet bins).
    timestamp : list of float
        Unix timestamps of the start of each 3-hour bin.
    order : list of int
        1-based index of the 3-hour bin within the day (1–8).
    """



    # lists to be populated

    timestamp, variation, k_index, order = [], [], [], []



    # start of the day in seconds

    day_seconds = int(minute_time[0])-int(minute_time[0])%(24*3600)



    #loop over minute_array and sort them according to 3-hour block

    start = 0

    hour_block1 = int((minute_time[0] - day_seconds)/(3*60*60))

    for index, value in enumerate(minute_time):

        hour_block2 = int((value - day_seconds)/(3*60*60))



        # if hr1 and hr2 and not equal, we have entered a new 3-hr block

        if hour_block2 != hour_block1:

            try:
               
                varx = max(minute_bx[start:index-1]) - min(minute_bx[start:index-1])

                vary = max(minute_by[start:index-1]) - min(minute_by[start:index-1])

    

                # append max variation for that block

                variation.append(max(varx, vary))   

                timestamp.append(day_seconds + (hour_block1*3*60*60))

                order.append(hour_block1+1)

    

                hour_block1 = hour_block2

                start = index

            except:
                
                continue



    # add last entry

    varx = max(minute_bx[start:-1]) - min(minute_bx[start:-1])

    vary = max(minute_by[start:-1]) - min(minute_by[start:-1])
    #print(variation)
    variation.append(max(varx, vary))
    
    #print(variation)
    timestamp.append(day_seconds + (hour_block1*3*60*60))

    order.append(hour_block1+1)



    # now to use these variations to calculate the k-index value

    niemegk = numpy.array([500, 330, 200, 120, 70, 40, 20, 10, 5, 0])   # reference

    thresh = niemegk * k9/500.0 


    #print(variation)
    k_index = []        # k_index list to be populated

    for i in variation:

        for index, j in enumerate(thresh):

            if i >= j:

                z = 9-index

                if z == 0:

                    z = 0.25
                    
                    if i <0.5: #this is to cut out 99999(invalid) data, most are slighly >0 due to fmi_smoothing
                        z=0

                k_index.append(z)

                break



    return k_index, timestamp, order



##########################################################################

##########################################################################



def fmi_smoothed(minute_time, minute_bx, minute_by, minute_time_prev, 

              minute_bx_prev, minute_by_prev, k_index, k_time):
    """Compute FMI-style smoothed baseline using current and previous day data.

    This version replicates the original Finnish Meteorological Institute
    smoothing method, where:
        * 1-hour blocks are expanded by a number of minutes that depends
          on both the local hour (lookup table) and the K-index value.
        * Data is averaged over these extended intervals, using both the
          current and previous day when needed.
        * The output is a slowly varying baseline used for subsequent
          subtraction steps.

    Parameters
    ----------
    minute_time, minute_bx, minute_by : array-like
        Current-day minute means and their timestamps.
    minute_time_prev, minute_bx_prev, minute_by_prev : array-like
        Previous-day minute means and timestamps.
    k_index : list of float
        K values per 3-hour bin for the current day.
    k_time : list of int
        Integer hour indices associated with the K values.

    Returns
    -------
    smoothed_time : list
        Timestamps (seconds) of the smoothed baseline points.
    smoothed_bx, smoothed_by : list
        Smoothed baseline values for Bx and By.
    """



    days = int(minute_time[0])/(24*3600)

    day_seconds = days*24*3600      # start of day in seconds

    data_start = minute_time[0]

    data_end = minute_time[-1]



    # extra minutes, depending on hour  #m value

    extra_time = [120, 120, 120, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

                          60, 60, 60, 120, 120, 120]



    indices = []

    # loop over minute_array and sort them according to hour block

    start = 0       # index which 

    hour_block1 = int((minute_time[0] - day_seconds)/(60*60))

    for index, value in enumerate(minute_time):

        hour_block2 = int((value - day_seconds)/(60*60))



        # if hr1 and hr2 and not equal, we have entered a new hr block

        if hour_block2 != hour_block1:

            x = (hour_block1)%24    # find what hour of the day it is

            m = extra_time[x]       # get extra minutes from extra_time



            y = k_time.index((hour_block1)/3 + 1)   # find corresponding k-index

            n = k_index[y] ** 3.3       # get extra minutes for k-index

            nm = int(n + m)



            start_time = minute_time[start] - (nm * 60) # index of start of hour

            end_time = minute_time[index] + (nm * 60)   # index of end of hour



            indices.append([start,index, nm, start_time, end_time, hour_block1])

            hour_block1 = hour_block2   # move on to new hour

            start = index               # change index



    # add last index manually and values as above

    x = (hour_block1)%24

    m = extra_time[x]



    y = k_time.index((hour_block1)/3 + 1)

    n = k_index[y] ** 3.3

    nm = int(n + m)



    start_time = minute_time[start] - (nm * 60)

    end_time = minute_time[index] + (nm * 60)



    indices.append([start, len(minute_time), 

                    nm, start_time, end_time, hour_block1])



    # now to add the data

    smoothed_time, smoothed_bx, smoothed_by = [], [], []



    # add point at half-eleven from previous day

    index_start = len(minute_time_prev) - 70



    # get index of prev time where it is > 11:30 of day before

    while minute_time[index_start] < (day_seconds - (30*60)):

        index_start += 1



    # add time to list, mean bx and by values for specified time.

    smoothed_time.append(day_seconds - (30*60)) 

    smoothed_bx.append(numpy.mean(minute_bx_prev[index_start:]))    

    smoothed_by.append(numpy.mean(minute_by_prev[index_start:]))



    for index1, value1 in enumerate(indices): # loop over indices

        start, end, nm, start_time, end_time, hour = value1



        try:

            if start_time < data_start: 

                # if start time is before start of data

                # go on to previous data

    

                # index_start = len(prev_data) - extra time

                index_start = len(minute_time_prev) - nm    

                while minute_time_prev[index_start] < start_time:

                    index_start += 1

                # loop until minute_time_prev[index_start] > start_time

            

                # now do the same for the end data

                index_end = end + nm

                while minute_time[index_end] > end_time:

                    index_end -= 1

                

                # add mean of magnetic data

                smoothed_bx.append(numpy.mean(minute_bx_prev[index_start:] 

                                    + minute_bx[start:index_end]))

                smoothed_by.append(numpy.mean(minute_by_prev[index_start:] 

                                    + minute_by[start:index_end]))



                # add time to list

                smoothed_time.append(((hour+0.5)*(3600))+ day_seconds)  

    

            else:

                index_start = start - nm

                while minute_time[index_start] < start_time:

                    index_start += 1

            

                index_end = end + nm

                if index_end < len(minute_time):

                    while minute_time[index_end] > end_time:

                        index_end -= 1

    

                else:

                    index_end = len(minute_time)

    

                smoothed_bx.append(numpy.mean(minute_bx[index_start:index_end]))

                smoothed_by.append(numpy.mean(minute_by[index_start:index_end]))



                # add time to list

                smoothed_time.append(((hour+0.5)*(3600))+ day_seconds)  

    

                a = ((hour+0.5)*(3600))+ day_seconds



        except:

            continue

    return smoothed_time, smoothed_bx, smoothed_by





##########################################################################

##########################################################################



def fmi_smoothed2(minute_time, minute_bx, minute_by, k_index, k_time):
    """Alternative FMI-style smoothing using a matrix accumulator.

    This implementation:
        * Builds a per-hour table (`master`) with time windows extended by
          a combination of fixed hour-dependent padding and K-dependent
          padding (K**3.3).
        * Accumulates Bx and By samples that fall into each hour’s window.
        * Produces a baseline by averaging within each hour window and
          interpolating over any remaining gaps.

    Parameters
    ----------
    minute_time, minute_bx, minute_by : array-like
        Minute means and their timestamps.
    k_index : list of float
        K values per 3-hour bin (with 0.25 treated as very quiet).
    k_time : list of int
        Integer hour indices associated with the K values.

    Returns
    -------
    clean_time : list
        Hourly timestamps used in the smoothed baseline.
    smx, smy : list
        Smoothed Bx and By values at `clean_time`.
    """

    extra_time = [120, 120, 120, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

                              60, 60, 60, 120, 120, 120]





    start_time = minute_time[0]

    start_day = start_time - start_time%(24*60*60)

    end_time = minute_time[-1]



    # number of hours to look at:

    hour_number = int((end_time - start_day)/3600) + 1



    hour_indices =[[] for i in range(hour_number)]



    k_index_thing, k_time_thing = [], []

    for i, j in list(zip(k_index, k_time)):

        if i == 0.25:

            k_index_thing.extend((0, 0, 0))

        else:

            k_index_thing.extend((i, i, i))



        l = j*3 

        k_time_thing.extend((l-3, l-2, l-1))



    master = np.zeros((hour_number, 7))

    blank =[]

    for i in range(hour_number):

        try:

            k = k_index_thing[k_time_thing.index(i)]

            n = k**3.3

        except:

            n = 0

        m = extra_time[i%24]



        master[i][0] = i    # hour number

    



        avg_time =start_day + (i*60*60) + (30*60) # half an hour time

        blank.append(avg_time)



        master[i][1] = avg_time



        start_time = avg_time - (30*60) - ((n + m)*60)

        master[i][2] = start_time



        end_time = avg_time + (30*60) + ((n + m)*60)

        master[i][3] = end_time





    for index, value in enumerate(minute_time):



        for i, v in enumerate(master):

            start = master[i][2]

            end = master[i][3]



            if start <= value <= end:

                master[i][4] += minute_bx[index]

                master[i][5] += minute_by[index]

                master[i][6] += 1

            



    nz = (master == 0).sum(1)

    q = master[nz == 0, :]



    smoothed_time = master[:,1]

    smoothed_bx = master[:,4]/master[:,6]

    smoothed_by = master[:,5]/master[:,6]



    clean_time, clean_bx, clean_by = [], [], []

    for t, x, y in zip(smoothed_time,smoothed_bx, smoothed_by):

        if (math.isnan(x) == False) and (math.isnan(y) == False):

            clean_time.append(t)

            clean_bx.append(x)

            clean_by.append(y)





    smt, smx, smy = [], [], []

    for i in clean_time:

         x = np.interp(i, clean_time, clean_bx)

         y = np.interp(i, clean_time, clean_by)

         

         smx.append(x)

         smy.append(y)

         



    return clean_time, smx, smy





##########################################################################

##########################################################################

def smoothed(time_array, y_value, z):   # z = smoothing order <=5

    """ Smooths the rough Sr curve """

    x = time_array

    y = y_value

    xi = np.linspace(x[0], x[-1], 10*len(x))

    ius = InterpolatedUnivariateSpline(x, y, k=z)

    yi = ius(xi)

    return xi, yi



##########################################################################

##########################################################################



def subtracted(minute_time, smooth_time, minute_bx, smooth_bx, minute_by, 

               smooth_by):

    """ Subtracts smooth data from original minute array"""

    subtracted_bx, subtracted_by = [], []

    for index, value in enumerate(minute_time):

        x = numpy.interp(value, smooth_time, smooth_bx)

        y = numpy.interp(value, smooth_time, smooth_by)

        subtracted_bx.append(minute_bx[index]-x)

        subtracted_by.append(minute_by[index]-y)



    return subtracted_bx, subtracted_by



##########################################################################

##########################################################################



def colored(k_index, barlist):

    """Colours yer k-index plots so it looks nice.



    EXAMPLE:

    barlist = ax3.bar(k_timestamps, k_index, width = 0.124)

    colored(k_index)"""

    for i in arange(0, len(k_index), 1):

        if k_index[i] >= 8:

            barlist[i].set_color('deeppink')

            barlist[i].set_edgecolor('k')

            continue

        if k_index[i] >= 6:

            barlist[i].set_color('r')

            barlist[i].set_edgecolor('k')           

            continue

        if k_index[i] >= 5:

            barlist[i].set_color('orange')

            barlist[i].set_edgecolor('k')           

            continue

        if k_index[i] >= 4:

            barlist[i].set_color('g')

            barlist[i].set_edgecolor('k')           

            continue

        if k_index[i] >= 2:

            barlist[i].set_color('c')

            barlist[i].set_edgecolor('k')           

            continue

        if k_index[i] >= 0:

            barlist[i].set_color('b')

            barlist[i].set_edgecolor('k')           

            continue



##########################################################################

##########################################################################



def slope_refined(x, y):    #want it to return y array
    """Compute refined time derivative on a 1-minute grid.

    For each new point spaced 60 seconds apart, the derivative is
    approximated using a simple finite difference:

        dY/dt ≈ (Y(t) - Y(t - 60 s)) / 1.0

    Parameters
    ----------
    x : array-like
        Original time axis in seconds (Unix time).
    y : array-like
        Series values sampled at `x`.

    Returns
    -------
    xx : list
        New time axis at 60-second spacing.
    yy : list
        Estimated derivative values at each point of `xx`.
    """

    xx, yy = [], []

    new_x = arange(x[0], x[-1], 60)

    for i in new_x:

        y2 = numpy.interp(i, x, y)

        y1 = numpy.interp(i-60, x, y)

        deriv = (y2 - y1) / 1.0

        xx.append(i)

        yy.append(deriv)

    return xx, yy



##########################################################################

##########################################################################



def do_k_plots(k_index3, k_time3, k_timestamp3, minute_time, sitefull, save_address,save_address2):
    """Generate and save 3-day K-index bar plots for a given site.

    This reproduces the legacy plotting style used on the DIAS Houdini
    server, including:
        * 3 consecutive days on the x-axis (UT).
        * Color-coded bars for K-index levels.
        * Site- and institute-specific footer text.
        * A timestamp indicating when the plot was last updated.

    Parameters
    ----------
    k_index3 : list of float
        Final-stage K indices (one per 3-hour bin).
    k_time3 : list
        Integer 3-hour block index (not directly used in plotting).
    k_timestamp3 : list of float
        Unix timestamps corresponding to each K value.
    minute_time : array-like
        Minute-level timestamps (used to establish the 3-day window).
    sitefull : str
        Full site name (e.g. "Armagh", "Valentia").
    save_address : str
        Base path for saving current live plot PNG.
    save_address2 : str
        Base path for saving archived plot PNG.
    """

    # 3-Day K-Index Plot
    

    nowz = datetime.datetime.utcnow() #houdini needs to read utc time
    
    year_str=str(nowz.year)
    month_str="%02d" %(nowz.month)
    day_str="%02d" %(nowz.day)
    date_str= day_str+'/'+month_str+'/'+year_str
    
    #timestamp_string = "Plot updated {}/{}/{} {} UT".format(nowz.day, nowz.month, nowz.year, str(nowz)[10:19]) 
    timestamp_string = "Plot updated "+str(date_str)+ " "+ str(nowz)[10:19] + " UT"


    end = float2time(minute_time[-1]- (minute_time[-1]%(24*3600)) + (24*3600))

    start = end -datetime.timedelta(3)

    middle1 = end - datetime.timedelta(2)

    middle2 = end - datetime.timedelta(1)



    print ("START: ", start)

    print ("MIDDLE1: ", middle1)

    print ("MIDDLE2: ", middle2)

    print ("END: ", end)

 

#   start = minute_time[0] - minute_time[0]%(24*3600)

#   middle1 = float2time(start + (1*24*3600))

#   middle2 = float2time(start + (2*24*3600))

#   end = float2time(start + (3*24*3600))

#   start = float2time(start)

    

    

    plt.clf()
    plt.style.use('classic')
    fig1 = plt.figure(1)

    plt.subplots_adjust(bottom = 0.2, left = 0.07, right = 0.94)

    

    a = plt.subplot(111)

    bartimenew=[]
    for m in range(0,len(k_timestamp3)): #Adjusting figures to centre bar plots, 5400=1h 30m in sec
        bartime2=k_timestamp3[m]+5400
        bartimenew.append(bartime2)
    #print(len(k_timestamp3))
    #print(len(k_index3))
    #print(bartimenew)
    #print(k_index3)
    bartime=float2time(bartimenew)
    

        
        
    barlist = a.bar(bartime, k_index3, width = 0.124)

    colored(k_index3, barlist)

    

    plt.xlabel('{}-{}-{}                         {}-{}-{}                         {}-{}-{}'.format(start.day, start.strftime("%B")[0:3], start.year, middle1.day, middle1.strftime("%B")[0:3], middle1.year, middle2.day, middle2.strftime("%B")[0:3], middle2.year), fontsize = 14)

    plt.ylabel("K-Index", fontsize = 14)

    plt.title(str(sitefull)+" 3-Day Local K-Index", fontsize = 16)

    

    plt.xlim([start, end])

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,6)))

    plt.axvline(middle1, color = 'k', lw = 1)

    plt.axvline(middle2, color = 'k', lw = 1)

    

    plt.ylim([0, 9])

    plt.grid(True)

    

    plt.figtext(0.65, 0.02, timestamp_string, fontsize = 11, style='italic')



    

    fig1.set_size_inches(9, 4)
    
    if sitefull=='Armagh':
        plt.figtext(0.02, 0.02, "TCD/DIAS/ARM", fontsize = 11, style = 'italic')
        
    elif sitefull=='Valentia':
        plt.figtext(0.02, 0.02, "TCD/DIAS/Met Eireann", fontsize=11, style='italic')
        
    else:
        plt.figtext(0.02, 0.02, "TCD/DIAS", fontsize = 11, style = 'italic')
    
    #date_str= "%d%02d%02d" %(nowz.year,nowz.month,nowz.day)    

    #fig1.savefig("C:\\Users\\john_\\Documents\\MagIE\\MagIE_all_files\\K_Indices\\"+str(site_l)+str(middle2str)+"_kindex.png")
    fig1.savefig(str(save_address)+'_kindex.png')
    fig1.savefig(str(save_address2)+'_kindex.png')
    #fig1.savefig('/mnt/data.magie.ie/magnetometer_archive/'+year_str+'/'+month_str+'/'+day_str+'/png/'+str(site_l)+'_kindex'+year_str+month_str+day_str+'_kindex.png')
    

def do_other_plots(minute_time, minute_bx, minute_by, minute_bz,sitefull,save_address,save_address2):
    """Generate and save 3-day D/H/dHdt and Bx/By/Bz plots.

    These plots:
        * Compute H and declination D from Bx/By, then dH/dt using
          `slope_refined`.
        * Plot D, H, and dH/dt in a stacked figure.
        * Plot Bx, By, Bz in a second stacked figure.
        * Use the same 3-day window and DIAS/TCD footer style as the
          K-index plots.

    Parameters
    ----------
    minute_time : array-like
        Minute timestamps (seconds).
    minute_bx, minute_by, minute_bz : array-like
        Minute-mean field components.
    sitefull : str
        Full site name (e.g. "Armagh", "Valentia").
    save_address : str
        Base path for saving current live plot PNGs.
    save_address2 : str
        Base path for saving archived plot PNGs.
    """

    nowz = datetime.datetime.utcnow() #houdini needs to read utc time
    #print(nowz)
    year_str=str(nowz.year)
    month_str="%02d" %(nowz.month)
    day_str="%02d" %(nowz.day)
    date_str= day_str+'/'+month_str+'/'+year_str
    #timestamp_string = "Plot updated {}/{}/{} {} UT".format(nowz.day, nowz.month, nowz.year, str(nowz)[10:19]) 

    timestamp_string = "Plot updated "+str(date_str)+ " "+ str(nowz)[10:19] + " UT"

 

    end = float2time(minute_time[-1] - (minute_time[-1]%(24*3600)) + (24*3600))

    start = end -datetime.timedelta(3)

    middle1 = end - datetime.timedelta(2)

    middle2 = end - datetime.timedelta(1)



    print ("START: ", start)

    print ("MIDDLE1: ", middle1)

    print ("MIDDLE2: ", middle2)

    print ("END: ", end)

 

#   start = minute_time[0] - minute_time[0]%(24*3600)

#   middle1 = float2time(start + (1*24*3600))

#   middle2 = float2time(start + (2*24*3600))

#   end = float2time(start + (3*24*3600))

#   start = float2time(start)

    

    # dH/dt

    H = []

    for index, value in enumerate(minute_bx):

        H.append(math.sqrt(minute_bx[index]**2 + minute_by[index]**2))

        

    #dH_time, dH = slope(minute_time, H, 60)

    dH_time, dH = slope_refined(minute_time, H)

    #dH_time, dH = slope(minute_time, H, 60)

    

    

    D, D_rad = [], []

    for index, value in enumerate(H):

        D_rad.append(math.asin(minute_by[index]/H[index]))

    

    for index, value in enumerate(D_rad):

        D.append(180*value/math.pi)


        #minute_H in position minute_bx, 
        #Minute D in position minute_by
    
    plt.clf()

    fig1 = plt.figure(1)

    plt.subplots_adjust(bottom = 0.1, top = 0.93, left = 0.1, right = 0.94, hspace=0.1)
    plt.style.use('classic')
    

    a1 = plt.subplot(311)

    plt.ticklabel_format(useOffset=False)

    plt.plot(float2time(minute_time), D, 'r')

    

    plt.locator_params(axis='y', nbins=4)

    plt.ylabel("D (degrees)", fontsize = 14)

    plt.title(str(sitefull)+" 3-Day D, H, dH/dt", fontsize = 16)

#   plt.ylim([2.35, 2.65])

    plt.xlim([start, end])

    

    plt.setp(a1.get_xticklabels(), visible=False)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,6)))

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tick_params('x', length=5, width=1, which='both')

    plt.axvline(middle1, color = 'k', lw = 1)

    plt.axvline(middle2, color = 'k', lw = 1)

    

    plt.grid()

    

    

    a2 = plt.subplot(312)

    plt.ticklabel_format(useOffset=False)

    plt.plot(float2time(minute_time), H, 'b')

    

    plt.locator_params(axis='y', nbins=5)

    plt.ylabel("H (nT)", fontsize = 14)

    

    plt.setp(a2.get_xticklabels(), visible=False)

    plt.xlim([start, end])

    H_up_lim = (int(max(H))/10)*10+10

    H_low_lim = (int(min(H))/10)*10-10

    plt.ylim([H_low_lim, H_up_lim])

    

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,6)))

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tick_params('x', length=5, width=1, which='both')

    plt.axvline(middle1, color = 'k', lw = 1)

    plt.axvline(middle2, color = 'k', lw = 1)

    

    plt.grid()

    

    a3 = plt.subplot(313)

    plt.ticklabel_format(useOffset=False)

    plt.plot(float2time(dH_time), dH, 'g')

    

    plt.locator_params(axis='y', nbins=4)

    plt.xlabel('{}-{}-{}                         {}-{}-{}                         {}-{}-{}'.format(start.day, start.strftime("%B")[0:3], start.year, middle1.day, middle1.strftime("%B")[0:3], middle1.year, middle2.day, middle2.strftime("%B")[0:3], middle2.year), fontsize = 14)

    plt.ylabel("dH/dt (nT/min)", fontsize = 14)

    

    plt.xlim([start, end])

    plt.ylim([-5, 5])

    

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,6)))

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tick_params('x', length=5, width=1, which='both')

    plt.axvline(middle1, color = 'k', lw = 1)

    plt.axvline(middle2, color = 'k', lw = 1)

    

    plt.grid()

    
    plt.figtext(0.65, 0.02, timestamp_string, fontsize = 11, style='italic')



    if sitefull=='Armagh':
        plt.figtext(0.02, 0.02, "TCD/DIAS/ARM", fontsize = 11, style = 'italic')   
        
    elif sitefull=='Valentia':
        plt.figtext(0.02, 0.02, "TCD/DIAS/Met Eireann", fontsize = 11, style = 'italic')
        
    else:
        plt.figtext(0.02, 0.02, "TCD/DIAS", fontsize = 11, style = 'italic')
    

    fig1.set_size_inches(9, 8.5)

    #fig1.savefig("C:\\Users\\john_\\Documents\\MagIE\\MagIE_all_files\\HDdH\\"+str(site_l)+str(day_str)+"_DHdH.png")
    fig1.savefig(str(save_address)+"_DHdH.png")
    fig1.savefig(str(save_address2)+"_DHdH.png")
    #fig1.savefig('/mnt/data.magie.ie/magnetometer_archive/'+year_str+'/'+month_str+'/'+day_str+'/png/'+str(site_l)+'_DHdH'+year_str+month_str+day_str+'.png')  

    

    ##########################################################################

    #bx, by, bz



    plt.clf()

    fig1 = plt.figure(1)

    plt.subplots_adjust(bottom = 0.1, top = 0.93, left = 0.1, right = 0.94, hspace=0.1)

    

    a1 = plt.subplot(311)

    plt.ticklabel_format(useOffset=False)

    plt.plot(float2time(minute_time), minute_bx, 'r')

    

    plt.locator_params(axis='y', nbins=6)

    plt.ylabel("Bx (nT)", fontsize = 14)

    plt.title(str(sitefull)+" 3-Day Bx, By, Bz", fontsize = 16)

    

    plt.setp(a1.get_xticklabels(), visible=False)

    plt.xlim([start, end])

    Bx_up_lim = (int(max(minute_bx))/10)*10+10

    Bx_low_lim = (int(min(minute_bx))/10)*10-10

    plt.ylim([Bx_low_lim, Bx_up_lim])

    

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,6)))

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tick_params('x', length=5, width=1, which='both')

    plt.axvline(middle1, color = 'k', lw = 1)

    plt.axvline(middle2, color = 'k', lw = 1)

    

    plt.grid()

    

    a2 = plt.subplot(312)

    plt.ticklabel_format(useOffset=False)

    plt.plot(float2time(minute_time), minute_by, 'b')

    

    plt.locator_params(axis='y', nbins=5)

    plt.ylabel("By (nT)", fontsize = 14, labelpad = 0)

    

    plt.setp(a2.get_xticklabels(), visible=False)

    plt.xlim([start, end])

    By_up_lim = (int(max(minute_by))/10)*10+10

    By_low_lim = (int(min(minute_by))/10)*10-10

    plt.ylim([By_low_lim, By_up_lim])

    

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,6)))

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tick_params('x', length=5, width=1, which='both')

    plt.axvline(middle1, color = 'k', lw = 1)

    plt.axvline(middle2, color = 'k', lw = 1)

    

    plt.grid()

    

    a3 = plt.subplot(313)

    plt.ticklabel_format(useOffset=False)

    plt.plot(float2time(minute_time), minute_bz, 'g')

    

    plt.locator_params(axis='y', nbins=5)

    plt.xlabel('{}-{}-{}                         {}-{}-{}                         {}-{}-{}'.format(start.day, start.strftime("%B")[0:3], start.year, middle1.day, middle1.strftime("%B")[0:3], middle1.year, middle2.day, middle2.strftime("%B")[0:3], middle2.year), fontsize = 14)

    plt.ylabel("Bz (nT)", fontsize = 14)

    

    plt.xlim([start, end])

    Bz_up_lim = (int(max(minute_bz))/10)*10+10

    Bz_low_lim = (int(min(minute_bz))/10)*10-10

    plt.ylim([Bz_low_lim, Bz_up_lim])

    

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,6)))

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tick_params('x', length=5, width=1, which='both')

    plt.axvline(middle1, color = 'k', lw = 1)

    plt.axvline(middle2, color = 'k', lw = 1)

    

    plt.grid()

    

    plt.figtext(0.65, 0.02, timestamp_string, fontsize = 11, style='italic')



    if sitefull=='Armagh':
        plt.figtext(0.02, 0.02, "TCD/DIAS/ARM", fontsize = 11, style = 'italic')   
        
    elif sitefull=='Valentia':
        plt.figtext(0.02, 0.02, "TCD/DIAS/Met Eireann", fontsize = 11, style = 'italic')
        
    else:
        plt.figtext(0.02, 0.02, "TCD/DIAS", fontsize = 11, style = 'italic')
        


    fig1.set_size_inches(9, 8.5)

    #fig1.savefig("C:\\Users\\john_\\Documents\\MagIE\\MagIE_all_files\\XYZ\\"+str(site_l)+str(day_str)+"_bxbybz.png")
    fig1.savefig(str(save_address)+"_bxbybz.png")
    fig1.savefig(str(save_address2)+"_bxbybz.png")
#fig1.savefig('/mnt/data.magie.ie/magnetometer_archive/'+year_str+'/'+month_str+'/'+day_str+'/png/'+str(site_l)+'_bxbybz'+year_str+month_str+day_str+'.png')    

    return



def createfolder(directory):
    """Create a directory (and parents) if it does not already exist.

    This is a thin wrapper around `os.makedirs` that mirrors the original
    legacy behaviour and is used by `archive_maker` when building the
    yearly archive structure.

    Parameters
    ----------
    directory : str
        Absolute path of the directory to ensure exists.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory) #Only creates folders if they don't already exist
    except OSError:
        ('Error: Creating directory. ' + directory)
        


##########################################################################

def archive_maker(time_now, createfolder):
    """Create next year's archive directory tree on 31 December.

    If `time_now` corresponds to 31 December, this function:
        * Determines the following year.
        * Creates year/month/day subdirectories.
        * Under each day, creates both 'txt' and 'png' subfolders.

    The layout is:

        /mnt/data.magie.ie/magnetometer_archive/YYYY/MM/DD/txt
        /mnt/data.magie.ie/magnetometer_archive/YYYY/MM/DD/png

    Parameters
    ----------
    time_now : datetime.datetime
        Current UTC timestamp used to decide when to create the archive.
    createfolder : callable
        Function used to safely create directories (typically `createfolder`).
    """
    #Script made to create new folders for each year for Houdini archive
    if time_now.day==31 and time_now.month==12: #Set to create folders for new year at end of each year
        
        direct="/mnt/data.magie.ie/magnetometer_archive/" #Address archive is stored at
        
        #adding loops to create all folders for next year i.e for each day of year        
        next_year=time_now.year+1   
        
        extra_year=time_now.year+2
        
        for year in range(next_year,extra_year,1):
#
            for month in range(1,13,1):
                
                if month==1 or month==3 or month==5 or month==7 or month==8 or month==10 or month==12:
                   daymax=31
                if month==2:
                    daymax=28
                    
                    if year % 4==0 and year % 100!=0  : #account for leap years
                        daymax=29
                        
                if month==4 or month==6 or month==9 or month==11:
                    
                    daymax=30
                                   
            
                for day in range(1,daymax+1,1):
        
                    
                    createfolder(direct+str(year)) #Creates year folder
                    if month<10:
                        createfolder(direct+str(year)+'/0'+str(month)) #Creates month folder            
                        if day<10:
        
                            createfolder(direct+str(year)+'/0'+str(month)+'/0'+str(day)) #Creates day folder
                            createfolder(direct+str(year)+'/0'+str(month)+'/0'+str(day)+'/txt')
                            createfolder(direct+str(year)+'/0'+str(month)+'/0'+str(day)+'/png')
                            
                        if day>9:
                            createfolder(direct+str(year)+'/0'+str(month)+'/'+str(day))
                            createfolder(direct+str(year)+'/0'+str(month)+'/'+str(day)+'/txt')
                            createfolder(direct+str(year)+'/0'+str(month)+'/'+str(day)+'/png')
        
        
                        
                    if month>9:
                        createfolder(direct+str(year)+'/'+str(month))
                        if day<10:
        
                            createfolder(direct+str(year)+'/'+str(month)+'/0'+str(day))
                            createfolder(direct+str(year)+'/'+str(month)+'/0'+str(day)+'/txt')
                            createfolder(direct+str(year)+'/'+str(month)+'/0'+str(day)+'/png')
                            
                        if day>9:
                            createfolder(direct+str(year)+'/'+str(month)+'/'+str(day))
                            createfolder(direct+str(year)+'/'+str(month)+'/'+str(day)+'/txt')
                            createfolder(direct+str(year)+'/'+str(month)+'/'+str(day)+'/png')
        print('Folders created')
    else:
        pass
SITE_K9 = {
    'arm': 630,
    'dun': 570,
    'val': 480,
}
def get_k9_for_site(site_code: str) -> float:
    """Resolve k9 from a short site code."""
    sc = site_code.lower()[:3]
    if sc not in SITE_K9:
        raise ValueError(f"Unknown site_code {site_code!r}, expected one of {list(SITE_K9)}")
    return SITE_K9[sc]
def compute_old_K(df_raw, site_code):
    """
    Reproduce the legacy K pipeline on a raw 1-sec DataFrame:

      - mag_filter (cars) for non-Valentia
      - minute_bin
      - three-stage K: k1 -> smooth/subtract -> k2 -> smooth/subtract -> k3
      - return final K as a pandas Series with UTC timestamps (3-hour bins)
    """

    df_raw = df_raw.sort_index()

    # --- 2.1 convert to arrays ---
    bx = df_raw["Bx"].to_numpy()
    by = df_raw["By"].to_numpy()
    bz = df_raw["Bz"].to_numpy()

    # legacy timedate_float
    timedate = df_raw.index.to_pydatetime().tolist()
    timedate_float = time2float(timedate)

    # --- 2.2 mag_filter for non-Val sites (exact same logic) ---
    if site_code.lower() != "val":
        bx_f, by_f, bz_f = mag_filter(bx, by, bz)
        bx = np.array(bx_f)
        by = np.array(by_f)
        bz = np.array(bz_f)

        # legacy code truncates time to same length as filtered arrays
        timedate_float = timedate_float[: len(bx)]

    # --- 2.3 convert seconds to minutes (legacy minute_bin) ---
    # n = 5 days (as in original script)
    minute_time, minute_bx, minute_by, minute_bz = minute_bin(
        timedate_float, bx, by, bz, n=5
    )

    # --- 2.4 legacy triple K sequence ---

    k9 = get_k9_for_site(site_code)

    # First K
    k1, k_ts1, k_time1 = k_index(minute_time, minute_bx, minute_by, k9)
    k_times = pd.to_datetime(
        [float2time(t) for t in k_ts1]
    )
    # return pd.Series(k1, index=k_times, name="K_index")
    # First smoothing
    sm_t1, sm_bx1, sm_by1 = fmi_smoothed2(
        minute_time, minute_bx, minute_by, k1, k_time1
    )
    # k_times = pd.to_datetime(
    #     [float2time(t) for t in sm_t1]
    # )
    # return pd.DataFrame({'Date_UTC': k_times, 'Bx':sm_bx1, 'By':sm_by1})
    sxt1, sbx1 = smoothed(sm_t1, sm_bx1, 3)
    sxt1b, sby1 = smoothed(sm_t1, sm_by1, 3)
    k_times = pd.to_datetime(
        [float2time(t) for t in sxt1]
    )
    # return pd.DataFrame({'Date_UTC': k_times, 'Bx':sbx1, 'By':sby1})
    sub_bx1, sub_by1 = subtracted(
        minute_time, sxt1, minute_bx, sbx1, minute_by, sby1
    )

    # Second K
    k2, k_ts2, k_time2 = k_index(minute_time, sub_bx1, sub_by1, k9)

    # Second smoothing
    sm_t2, sm_bx2, sm_by2 = fmi_smoothed2(
        minute_time, minute_bx, minute_by, k2, k_time2
    )
    sxt2, sbx2 = smoothed(sm_t2, sm_bx2, 3)
    sxt2b, sby2 = smoothed(sm_t2, sm_by2, 3)
    sub_bx2, sub_by2 = subtracted(
        minute_time, sxt2, minute_bx, sbx2, minute_by, sby2
    )

    # Final K (k_index3)
    k3, k_ts3, k_time3 = k_index(minute_time, sub_bx2, sub_by2, k9)

    # --- 2.5 make a pandas Series of final K with UTC timestamps ---
    k_times = pd.to_datetime(
        [float2time(t) for t in k_ts3]
    )
    K_old = pd.Series(k3, index=k_times, name="K_index")

    # If you want to mimic the 3-day plot trimming, uncomment this:
    #
    # a = minute_time[-1] - minute_time[-1] % (24*3600) - (2 * 24*60*60)
    # keep = [i for i, t in enumerate(k_ts3) if t >= a]
    # if keep:
    #     K_old = K_iloc[keep[0]:]

    return K_old._sort_index()


if __name__=='__main__':
    ##########################################################################
    ##########################################################################

    sites_obs=['Arm','Dun', 'Val'] 
    sitefull_list=['Armagh','Dunsink','Valentia']

    k_thres=[630 ,570, 480] 
    #the k_threshold is the minimum value to denote a kp 9 storm(in nT)
    #needs to match each site


    nowz = datetime.datetime.utcnow() #need utc time


    TEXTFULL='' #an empty string used for sending email alerts

    email_count=0
    site_count=0 #added to read different variables in lists above 
    year_str=str(nowz.year)
    month_str="%02d" %(nowz.month)
    day_str="%02d" %(nowz.day)
    date_str= day_str+'/'+month_str+'/'+year_str
    intensity_list=[]
    k_stamp_list=[]
    condition_list=[]
    #Creating a loop to read data for each site
    for k1, k2 in enumerate(sites_obs):
        
        k2_l = k2.lower() #
        k_max=k_thres[site_count] #ensures next k_thres value ran with each iteration of sites_obs
        sitefull_name=sitefull_list[site_count]
        site_count=site_count+1 
        nowz = datetime.datetime.utcnow() #need utc time
        year_str=str(nowz.year)
        month_str="%02d" %(nowz.month)
        day_str="%02d" %(nowz.day)
        save_address='/mnt/data.magie.ie/magnetometer_live/'+str(k2_l)    
        save_address2="/mnt/data.magie.ie/magnetometer_archive/"+year_str+"/"+month_str+'/'+day_str+'/png/'+str(k2_l)+year_str+month_str+day_str
        print(save_address2)
        folder="/mnt/data.magie.ie/magnetometer_archive/"
        file_full=[]
        folder_live="/mnt/data.magie.ie/magnetometer_live/"+str(k2_l)+"_mag_realtime.txt"   

        for j in range(0,4,1): #making strings to find files in archive


            
            time=nowz-datetime.timedelta(days=j)
            date_str= "%d%02d%02d" %(time.year,time.month,time.day)
            year_str=str(time.year)
            month_str="%02d" %(time.month)
            day_str="%02d" %(time.day)
            file_str=year_str+'/'+month_str+'/'+day_str+'/txt/'+str(k2_l)+date_str+'.txt'
            file_str2=folder+file_str
            file_full.append(file_str2)

        

            
        try: #try loop added in case site fails due to missing data

            file_list = []
            for k in file_full:
                print (k)
                if ".txt" not in k:
                    continue
                l = os.path.getmtime(k)
                file_list.append((l, k))
                #file_list = sorted(file_list, reverse = True)
            print(len(file_list))
        
        
        ########################################
        #Normal K index section
        
                
                    
            current_file = file_list[0]
            current_day_seconds = current_file[0] - current_file[0]%(24*60*60)
            
            datez, timez, bx, by, bz, tempfg = [], [], [], [], [], []
            
            
            for i in range(3, -1, -1): #reads last 4 files in folder, from end to start   
        
                filename = file_list[i][1]
        
                date1, time1, bx1, by1, bz1 = data_read(filename)
                if k2_l == 'val' or 'dun':
                    temp1 = np.loadtxt(filename, usecols = (5,), skiprows = 2)
                    
                else:
                    temp1 = np.loadtxt(filename, usecols = (10,), skiprows = 2)
                tempfg = np.concatenate((tempfg, temp1), axis = 0)
                #Adding a quick check to ensure that same file isn't repeated two days in a row
                if i<3:
                    #print(bx[0])
                    #print(bx1[0])
                    if bx_record[0] == bx1[0] and by_record[0]==by1[0] and bz_record[0]==bz1[0]:
                        print(filename)
                        print('copy')
                        bx2=np.ones(len(bx1))
                        bx1=np.array([99999.99*i for i in bx2])
                        by1=bx1
                        bz1=bx1
                bx_record=bx1
                by_record=by1
                bz_record=bz1
                bx = np.concatenate((bx,bx1), axis = 0)
                by = np.concatenate((by,by1), axis = 0)
                bz = np.concatenate((bz,bz1), axis = 0)
                datez = np.concatenate((datez,date1), axis = 0)
                timez = np.concatenate((timez,time1), axis = 0)
        
            
                timedate = timedatez(datez, timez)
                timedate_float = time2float(timedate)
                print(bx)
            if k2_l!='val':
                print('filtering')
                print(len(bx))
                bx,by,bz=mag_filter(bx,by,bz)
                bx=np.array(bx)
                by=np.array(by)
                bz=np.array(bz)
                timedate_float=timedate_float[0:len(bx)]
                print(len(timedate_float),'Timedate len')
                print(len(bx))
                ##########################################################
                #Added to remove errorsome data
                #Note: Don't need for current Valentia setup
                #...Only looking at Variations in Valenti
            print(len(bx))
            bx[bx >= 80000.0] = 'nan'
            bx[bx == 'infs'] = 'nan'
            bx[bx <= 0.0] = 'nan'
            nans, x = nan_helper(bx)
            bx[nans]= np.interp(x(nans), x(~nans), bx[~nans])
        
            
            by[by >= 50000.0] = 'nan'
            by[by == 'infs'] = 'nan'
            by[by <= -10000.0] = 'nan'
            nans, x = nan_helper(by)
            by[nans]= np.interp(x(nans), x(~nans), by[~nans])
            
            bz[bz >= 80000.0] = 'nan'
            bz[bz == 'infs'] = 'nan'
            bz[bz <= 10000.0] = 'nan'
            nans, x = nan_helper(bz)
            bz[nans]= np.interp(x(nans), x(~nans), bz[~nans])
            ###############################################################################
            ###############################################################################
            #Creating plots
            
            print( "Starting analysis\n")
            print ("Getting data in minute bins\n")
            minute_time, minute_bx, minute_by, minute_bz= minute_bin(timedate_float, bx, by, bz, 5)
            
            print ("Initial plots")
            do_other_plots(minute_time, minute_bx, minute_by, minute_bz, sitefull_name, save_address,save_address2)
            
            print ("First k_index\n")
            k_index1, k_timestamp1, k_time1 = k_index(minute_time, minute_bx, minute_by, k_max)
            
            
            print ("FMI-Smoothing 1\n")
            smoothed_time, smoothed_bx, smoothed_by = fmi_smoothed2(minute_time, minute_bx,
            minute_by, k_index1, k_time1)
            
            print( "Interpolated Univariate Spline smoothing 1\n")
            smooth_time1, smooth_bx1 = smoothed(smoothed_time, smoothed_bx, 3)
            smooth_time1, smooth_by1 = smoothed(smoothed_time, smoothed_by, 3)
            
            print ("Subtracting data from smoothed\n")
            subtracted_bx1, subtracted_by1 = subtracted(minute_time, smooth_time1,
            minute_bx, smooth_bx1, minute_by, smooth_by1)
            
            print ("Second k-index\n")
            k_index2, k_timestamp2, k_time2 = k_index(minute_time, subtracted_bx1, subtracted_by1, k_max)
            
            print ("FMI-Smoothing 2\n")
            smoothed_time, smoothed_bx, smoothed_by = fmi_smoothed2(minute_time, minute_bx, 
                                                    minute_by, k_index2, k_time2)
            
            print ("Interpolated Univariate Spline smoothing 2\n")
            smooth_time2, smooth_bx2 = smoothed(smoothed_time, smoothed_bx, 3)
            smooth_time2, smooth_by2 = smoothed(smoothed_time, smoothed_by, 3)
            
            print ("Subtracting data from smoothed\n")
            subtracted_bx2, subtracted_by2 = subtracted(minute_time, smooth_time2, minute_bx, smooth_bx2, minute_by, smooth_by2)
            
            print ("Final k_index\n")
            k_index3, k_timestamp3, k_time3 = k_index(minute_time, subtracted_bx2, subtracted_by2, k_max)
            
            a = minute_time[-1] - minute_time[-1]%(24*60*60) - (2 * 24* 60 * 60)
            for i, v in enumerate(k_timestamp3):
                if v >= a:
                    break
            k_index3 = k_index3[i:]
            k_timestamp3 = k_timestamp3[i:]
            k_time3 = k_time3[i:]
            
            k_time3 = [x - 8 for x in k_time3]
            
            print ("Plotting...")
            #do_k_plots(k_index1, k_time1, k_timestamp1eCallistpo, minute_time)
            
            do_k_plots(k_index3, k_time3, k_timestamp3, minute_time,sitefull_name, save_address,save_address2)
            

                
            
        except:
            pass

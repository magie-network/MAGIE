import numpy as np
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


def minute_bin_intermag(b):
    """
    o Input 1-second data
    o Can apply to any conventional coords, i.e. XYZ, HDZ, NEZ, etc.

    o Applies INTERMAG digital filter for 1 min data, which also bins to 1 min
    o The coefficients used for the intermagnet filter are given by intermag_halfgaussian
      (intermag_gaussian reflects them)
    o Note that the first and last minute will be missing, ideally load in day
      before /after so all points loaded
    o Returns 1 minute filtered data
    """

    intermag_halfgaussian=[0.000459,0.000548,0.000651,0.00077,0.000907,0.0010645,
                       0.0012445,0.0014492,0.0016809,0.0019419,0.0022347,0.0025614,
                       0.0029243,0.0033254,0.0037667,0.0042496,0.0047755,
                       0.0053454,0.0059596,0.0066181,0.0073204,0.0080653,0.0088509,
                       0.0096747,0.01053338,0.01142303,0.01233892,0.01327563,
                       0.01422707,0.01518651,0.01614667,0.01709976,0.01803763,
                       0.01895183,0.01983377,0.0206748,0.02146643,0.02220039,
                       0.02286881,0.02346437,0.0239804,0.02441104,0.02475132,0.02499727,
                       0.02514602,0.0251958]
    intermag_gaussian=[]

    #Reflecting intermag halfgaussian for full set
    for i in intermag_halfgaussian:
        intermag_gaussian.append(i)
    for i in range(len(intermag_halfgaussian)-2,-1,-1):
        intermag_gaussian.append(intermag_halfgaussian[i])


    #print(len(intermag_halfgaussian))
    bgauss=[]
    bcountmax=0
    for i in range(15,len(b)-90-45,60):

        bgauss2=[]

        brange=b[i:i+90+1]

        badcount=0
        for k in brange:
            if k >90000.0 or np.isnan(k):
                badcount=badcount+1

        if badcount > bcountmax:
            bcountmax=badcount
        #commenting lines for "smart method", should work but don't, do not use
        #check=all(x > 90000.99 for x in brange)
        #check2=any(x > 90000.99 for x in brange)
        # removing nans if only a few in a minute of data, else leaving them
        if badcount >0 and badcount <60:

            try:
                brange[brange >= 80000.0] = 'nan'
                brange[brange == 'infs'] = 'nan'
                brange[brange <= -80000.0] = 'nan'
                nans, x = nan_helper(brange)
                brange[nans]= np.interp(x(nans), x(~nans), brange[~nans])
            except ValueError:
                pass
        l=0
        for j in brange:
            bval=j*intermag_gaussian[l]
            l=l+1

            bgauss2.append(bval)



        if badcount<60:
            bgauss.append(np.sum(bgauss2))
        else: #99999.99 for an entire minute
            bgauss.append(99999.99)

    return bgauss
#Use like so
#X=minute_bin_intermag(X)
#Y=minute_bin_intermag(Y)
#Z=minute_bin_intermag(Z)
#H=minute_bin_intermag(H)


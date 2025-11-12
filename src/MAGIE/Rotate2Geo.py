import pandas as pd
import numpy as np
import gc
import vaex as vx
import os
def Huber_Mean(data, sd_lim=1.5, lim= 0.1, iter_lim= 20, rm_nan=True):
    """
    Calculates the Huber Weighted Mean

    Parameters
    ----------
    data : numpy.ndarray
        Values to calculate the Huber Weighted Mean.
    sd_lim : float/int, optional
        How many standard deviations from the mean before weighting begins. The default is 1.5.
    lim : float, optional
        As a multiple of the orignal mean how close must two consecutive iterations of the mean be before returning the Huber Weighted Mean. The default is 0.1.
    iter_lim : int, optional
        How many iterations before the code breaks and assumes no better difference in the means can be found. The default is 20.
    rm_nan : Bool, optional
        Should nans be removed? The default is True.

    Raises
    ------
    ValueError
        Raised when the mean is nan and subsequently a Huber Weighted Mean cannot be found.

    Returns
    -------
    mu: Huber Weighted Mean.

    """
    if rm_nan:
        data=data[np.isfinite(data)]
    mu= np.mean(data, axis=data.ndim-1)
    sigma= np.std(data)
    delta=9.999e3*np.max(abs(data))
    i= 0
    lim*=mu
    while np.all(delta> lim):
        if data.ndim==2:
            e= abs(data-np.vstack(mu))
        else:
            e= abs(data-mu)
        w= sd_lim*sigma/e
        w[w>=1]=1
        # w= np.min(np.vstack([np.ones(e.shape), sd_lim*sigma/e]), axis=0)
        mu2=np.average(data, weights= w, axis=data.ndim-1)
        delta= abs(mu-mu2)
        mu= mu2
        if i>iter_lim:
            print('iteration limit reached')
            break
        i+=1
        if np.any(np.isnan(mu)):
            raise ValueError('Mean is nan')
    return mu

def Huber(data):
    try:
        if not isinstance(data, np.ndarray):
            return Huber_Mean(data.values)
        else:
            return Huber_Mean(data)
    except ValueError as e:
        if 'Mean is nan' in str(e):
            print(f'Mean in nan for Huber Mean, length of data used in Huber Mean is : {len(data)}')
        else:
            raise
            

def get_average_field(df, average_function=np.mean, **average_function_kwargs):
    df["Year-Month"] = df["Date_UTC"].dt.to_period("M")
    monthly_averages= df.groupby('Year-Month')[['X', 'Y', 'Z']].agg(average_function)
    monthly_averages['Year']= monthly_averages.index.year
    return monthly_averages.groupby('Year')[['X', 'Y', 'Z']].agg(average_function).reset_index()
def get_average_field(df, average_function=np.mean, **average_function_kwargs):
    df["Year-Month"] = df['Date_UTC'].dt.year.astype(str)+'-'+df['Date_UTC'].dt.month.apply(lambda x: f'{x:02d}')
    monthly_averages= df.groupby('Year-Month', agg={'X': vx.agg.list('X'), 'Y': vx.agg.list('Y'), 'Z': vx.agg.list('Z') }).to_pandas_df().set_index('Year-Month').map(average_function).reset_index()
    monthly_averages['Year']= monthly_averages['Year-Month'].apply(lambda x : x[:4])
    return monthly_averages.groupby('Year')[['X', 'Y', 'Z']].agg(average_function).reset_index()
def get_chaos_field(glon, glat, years, month=1, day=1, radius= 6371.2, chaos_model='./CHAOS-8.1.mat'):
    from chaosmagpy import load_CHAOS_matfile
    from chaosmagpy.data_utils import mjd2000
    from pysymmetry import geodesy
    # Load CHAOS from matfile
    model= load_CHAOS_matfile(chaos_model)
    # geodetic to geocentric to align with CHAOS
    colatitude, radius, _, _ = geodesy.geod2geoc(glat, 0, 0, 0)
    # get static field
    B_radius_crust, B_theta_crust, B_phi_crust = model.synth_values_static(radius, colatitude, glon)
    Bx_chaos, By_chaos, Bz_chaos= [], [], []
    for year in years:
        # conversion to julian day (needed for CHAOS)
        time = mjd2000(year, month, day)
        # get time dependent field
        B_radius_core, B_theta_core, B_phi_core = model.synth_values_tdep(time, radius, colatitude, glon)
        # Sum radial field
        B_radius = B_radius_core + B_radius_crust
        # Sum theta field
        B_theta = B_theta_core + B_theta_crust
        # Sum phi field
        B_phi = B_phi_core + B_phi_crust
        # Return to geodetic system
        glat, h, B_north, B_radius = geodesy.geoc2geod(colatitude, radius, B_theta, B_radius)

        Bx_chaos.append(B_north)
        By_chaos.append(B_phi)
        Bz_chaos.append(B_radius)

    return np.array(Bx_chaos), np.array(By_chaos), np.array(Bz_chaos)
def get_horizontal_angle(measured, model):
    B_measured_horiz = measured[:, :2]
    B_model_horiz = model[:, :2]
    
    dot = np.sum(B_measured_horiz * B_model_horiz, axis=1)
    det = B_measured_horiz[:, 0] * B_model_horiz[:, 1] - B_measured_horiz[:, 1] * B_model_horiz[:, 0]
    
    angle_rad = np.arctan2(det, dot)
    return np.rad2deg(angle_rad)
def get_vertical_angle(measured, model):
    # Vertical component (Bz only)
    cos_theta_vert = np.dot(measured[-1], model[-1]) / (np.linalg.norm(measured[2]) * np.linalg.norm(model[2]))
    angle_vert = np.arccos(np.clip(cos_theta_vert, -1.0, 1.0))
    return np.rad2deg(angle_vert)

def rotate(X, Y, angle):
    angle= np.deg2rad(angle)
    return X*np.cos(angle) - Y*np.sin(angle), X*np.sin(angle)+ Y*np.cos(angle)


def rotate_north(X, Y, angle):
    angle = np.deg2rad(angle)
    return X * np.cos(angle) - Y * np.sin(angle)

def rotate_east(X, Y, angle):
    angle = np.deg2rad(angle)
    return X * np.sin(angle) + Y * np.cos(angle)


def Unknown2Geo(csv_file, glon, glat, X='Bx', Y='By', Z='Bz', datetime='Date_UTC', year_average_kwargs={},
                get_model_field= get_chaos_field, model_kwargs={'chaos_model':'../Data/CHAOS-8.1.mat'}):
    kwargs= {'average_function':Huber}
    kwargs.update(year_average_kwargs)
    df= vx.open(csv_file)
    for old_col, new_col in zip([X, Y, Z, datetime], ['X', 'Y', 'Z', 'Date_UTC']):
        df.rename(old_col, new_col)
    df=df.dropmissing(column_names=['X', 'Y', 'Z'])
    df['Y']*=-1
    print('step 1')
    del X, Y, Z, datetime
    yearly_average= get_average_field(df, **kwargs)
    yearly_average= yearly_average.join(pd.DataFrame({col: val for col, val in zip(['Model_X', 'Model_Y', 'Model_Z'],
                                                                                   get_model_field(glon, glat, np.array(yearly_average.Year.values),
                                                                                                   **model_kwargs))}))
    print('step 2')
    yearly_average['theta']= get_horizontal_angle(yearly_average[['X', 'Y', 'Z']].values,
                                                  yearly_average[['Model_X', 'Model_Y', 'Model_Z']].values)
    yearly_average=yearly_average.join(pd.DataFrame({col: val for col, val in zip(['Average_N_Geo', 'Average_E_Geo'],
                                                                   rotate(*yearly_average[['X', 'Y', 'theta']].values.T))}))
    yearly_average.to_csv('tmp_year_av.csv', index=False)
    del yearly_average
    yearly_average= vx.open('tmp_year_av.csv')
    print('step 3')
    df['Year']= df['Date_UTC'].dt.year
    df= df.join(yearly_average[['Year', 'Model_X', 'Model_Y', 'Model_Z', 'theta', 'Average_E_Geo', 'Average_N_Geo']], on='Year')
    df= df.dropmissing(column_names=['Average_E_Geo', 'Average_N_Geo'])
    print('step 4')
    del yearly_average
    # return df
    gc.collect()
    df['North']= df.apply(rotate_north, arguments=(df.X, df.Y, df.theta))
    # return df
    df['East']= df.apply(rotate_east, arguments=(df.X, df.Y, df.theta))
    print('step 5')
    return df

if __name__ =='__main__':
    from progressbar import progressbar
    columns= ['Date_UTC', 'Site', 'X', 'Y', 'Z', 'Model_X', 'Model_Y', 'Model_Z', 'theta', 'Average_E_Geo', 'Average_N_Geo', 'East', 'North']
    for site in progressbar(['Valentia', 'Dunsink', 'Armagh']):
        # Load the CSV file
        file_path = f"../Data/All_{site}.csv"  # Change this to your actual file path
        if site=='Dunsink':
            glat = 53.38  # Example: Dunsink Observatory latitude
            glon = 353.66  # Example: Dunsink Observatory longitude
        elif site=='Valentia':
            glat = 51.94 # Example: Valentia Observatory latitude
            glon = 349.76  # Example: Valentia Observatory longitude
        elif site=='Armagh':
            glat = 54.34 # Example: Valentia Observatory latitude
            glon = 353.34  # Example: Valentia Observatory longitude
        else:
            raise ValueError(f'Incorrect Site choice: {site}')
        df= Unknown2Geo(file_path, glon, glat)
        df.export_csv(f'../Data/{site}_rotated.csv')
        df[columns].export_hdf5(f'../Data/{site}_rotated.hdf5')
        os.remove('tmp_year_av.csv')
        

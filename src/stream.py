import netCDF4
import numpy as np
import os
import xarray as xr

PATH_NC = '../data/cmems_20200916_3.nc'


def load_nc(path_nc: str=PATH_NC):
    """
    Load the .nc file as a xarray.Dataset
    """
    assert os.path.isfile(path_nc)
    nc = xr.open_dataset(path_nc)
    return nc


def get_stream_velocity(nc, lat: float,  lon: float, date: str, coord_range: float=10):
    """
    Get stream velocities and their corresponding latitudes and longitudes
    
    Params
    ------
    nc : xarray.Dataset
    lat : float
        Boat position (latitude)
    lon : float
        Boat position (longitude)
    date : str
        Date in format YYYY-MM-DD
    coord_range : float, default=10
        Range for longitude and latitude
    
    Returns
    -------
    stream_velocities : np.array
        An (M, N, K)-dimensional array containing the K-dim
        values of the magnitude. Each MxN matrix
        element represents the value for the k_th dimension of
        that magnitude in the m_th, n_th latitude, longitude point
        on the discrete space.
    stream_velocities_latitudes : np.array
        An (M)-dimensional vector with the latitude values corresponding
        to the discrete points of the grid_array.
    stream_velocities_longitudes : np.array
        An (N)-dimensional vector with the longitude values corresponding
        to the discrete points of the grid_array.
    """
    lon_float = (max(lon - coord_range, -181),
                min(lon + coord_range, 181))
    lat_float = (max(lat - coord_range, -61),
                 min(lat + coord_range, 61))
    # index the variables 
    nc_sel = nc.sel(time=date,
                longitude=slice(*lon_float),
                latitude=slice(*lat_float))
    # u (velocity parallel to longitude)
    u = nc_sel['uo'].data
    # v (velocity parallel to latitude)
    v = nc_sel['vo'].data
    stream_velocities = np.concatenate((v, u))
    stream_velocities_latitudes = nc_sel['latitude'].data
    stream_velocities_longitudes = nc_sel['longitude'].data
    return stream_velocities, stream_velocities_latitudes, stream_velocities_longitudes

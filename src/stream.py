import netCDF4
import numpy as np
import os
import xarray as xr

PATH_NC = '../data/cmems_20200916_3.nc'


def load_nc(path_nc: str=PATH_NC):
    assert os.path.isfile(path_nc)
    nc = xr.open_dataset(path_nc)
    return nc


def get_stream_velocity(nc, lat: tuple,  lon: tuple, date: str, coord_range: float=20):
    # index the variables u (velocity parallel to longitude)
    # and y (velocity parallel to latitude)
    u = (nc['uo'].sel(time=date,
                      longitude=slice(*lon),
                      latitude=slice(*lat)).data)
    v = (nc['vo'].sel(time=date,
                      longitude=slice(*lon),
                      latitude=slice(*lat)).data)
    stream_velocity = np.concatenate((u, v))
    return stream_velocity
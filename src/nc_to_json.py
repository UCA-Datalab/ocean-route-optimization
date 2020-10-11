import json
import netCDF4
import numpy as np
import xarray as xr

# This following code allows the script
# to import the module
import os
import sys

path_module = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path_module)

# Module's functions
from src.stream import PATH_NC

PATH_DATA = os.path.join(path_module, 'data/')


def convert_to_json(date: str,
                    path_nc: str=PATH_NC, path_data: str=PATH_DATA,
                   max_points: int=1e6):
    
    path_json = os.path.join(PATH_DATA, 'nc_{}.json'.format(date))
    
    # load the NetCDF file (NetCDF4 library needed)
    assert os.path.isfile(path_nc)
    nc = xr.open_dataset(path_nc)
    
    number_points = nc.latitude.shape[0] * nc.longitude.shape[0]
    dx = float(np.mean(np.diff(nc.longitude.data)))
    dy = float(np.mean(np.diff(nc.latitude.data)))
    
    nc_sub = nc.sel(time=date)
    
    u = nc_sub['uo'].data[0,:,:]
    v = nc_sub['vo'].data[0,:,:]
    
    while number_points > max_points:
        u = u[::2,::2]
        v = v[::2,::2]
        dx = dx * 2
        dy = dy * 2
        number_points = u.size
    
    header = {
        "discipline":10,
        "disciplineName":"Oceanographic_products",
        "center":-3,
        "centerName":"Earth & Space Research",
        "refTime":"2020-02-25T00:00:00.000Z",
        "significanceOfRT":0,
        "significanceOfRTName":"Analysis",
        "parameterCategory":1,
        "parameterCategoryName":"Currents",
        "parameterNumber":2,
        "parameterNumberName":"U_component_of_current",
        "parameterUnit":"m.s-1",
        "forecastTime":0,
        "surface1Type":160,
        "surface1TypeName":"Depth below sea level",
        "surface1Value":15.0,
        "numberPoints":number_points,
        "shape":0,
        "shapeName":"Earth spherical with radius = 6,367,470 m",
        "scanMode":0,
        "nx":u.shape[0],
        "ny":v.shape[1],
        "lo1":float(nc.longitude.valid_min),
        "la1":float(nc.latitude.valid_min),
        "lo2":float(nc.longitude.valid_max),
        "la2":float(nc.latitude.valid_max),
        "dx":dx,
        "dy":dy
        }
    
    u = np.round(u.flatten(), decimals=2).tolist()
    u = [round(x, 2) for x in u]
    v = np.round(v.flatten(), decimals=2).tolist()
    v = [round(x, 2) for x in v]
    
    json_dict = {'U_component_of_current': {'header': header,
                'data': u},
                'V_component_of_current': {'header': header,
                                          'data': v}}

    with open(path_json, 'w') as json_file:
        json.dump(json_dict, json_file)

        
if __name__ == '__main__':
    convert_to_json('2020-09-16')
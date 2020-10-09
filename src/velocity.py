import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp2d

def interpolated_value(lat, long, vel_array, xy_positions, step = 0.01):
    """Returns the interpolated value of the tensor for the latitude and longitude requested.
    
    """
    
    idx = (np.abs(xy_positions[0] - lat)).argmin()
    idy = (np.abs(xy_positions[1] - long)).argmin()
    
    
    if xy_positions[0][idx] < lat: idx += 1
    if xy_positions[1][idy] < long: idy += 1
        
    x = xy_positions[0][idx:idx+2]
    y = xy_positions[1][idy:idy+2]
    
    small_array = vel_array[idx:idx+2,idy:idy+2,:]

    velx = interp2d(x, y, small_array[:,:,0])
    
    vely = interp2d(x, y, small_array[:,:,1])
    
    new_velx = velx(lat, long)
    new_vely = vely(lat, long)
    
    #Interpolated square
    lat_s = np.arange(xy_positions[0][idx], xy_positions[0][idx+1], step)
    long_s = np.arange(xy_positions[1][idx], xy_positions[1][idx+1], step)
    velx_s = velx(lat_s, long_s)
    vely_s = vely(lat_s, long_s)
        
    return new_velx, new_vely, lat_s, long_s, velx_s, vely_s
    
    
    
    
def velocity_composition(boat_velocity, boat_position, stream_velocitites, xy_stream_velocities):
    
    """Given the boat velocity and the stream velocity, computes the total velocity of the boat.
    """
    lat_boat, long_boat = boat_position
    velx, vely, *_ = interpolated_value(lat_boat, long_boat, stream_velocitites, xy_stream_velocities)
    
    return boat_velocity[0] + velx, boat_velocity[1] + vely
    


def boat_movement(boat_velocity, boat_position, stream_velocitites, xy_stream_velocities, ts):
    vx, vy, *_ = velocity_composition(boat_velocity, boat_position, stream_velocitites, xy_stream_velocities)
    
    return boat_position[0]+vx*ts, boat_position[1]+vy*ts


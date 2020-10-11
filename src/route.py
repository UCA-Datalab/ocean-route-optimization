import numpy as np

# This following code allows the script
# to import the module
import os
import sys

path_module = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path_module)

from src.stream import get_stream_velocity, load_nc, PATH_NC
from src.velocity import boat_movement
from src.consumption import fuel_consumption


def get_route_end_point(initial_position, boat_velocity_decisions, time_stamp,
                        date, path_nc:str = PATH_NC, verbose = False, coord_range = 10):
    """Given the coordinates and a set of M boat velocity decisions, returns
    the final vessel position.

    Params
    ------

    * initial_position: array.
        1D array with the initial coordinates as latitude, longitude
    * boat_velocity_decisions: array.
        (M, 2)-dim array with the velocity decisions. Each 2-value row is one
        velocity decision. The two values correspond to the latitude and
        longitude components of velocity.
    * time_stamp: float or array.
        Amount of time between boat velocity decisions. If float, then equal
        decision time intervals are assumed. Otherwise, it must be an (M) long
        vector.
    * date: string.
        Initial date with format format YYYY-MM-DD.
    * path_nc: string.
        Path for the nc file containing the stream velocities forecast.
    * verbose: bool. Default = False
        Whether to show the evolution.
    * coord_range: Range for magnitudes and longitudes

    Returns:
    -----

    * positions: array
        (M, 2)-dim array with the all the coordinates of the route as
        latitudes, longitudes
    * total_fuel_consumption: float
        Total fuel consumption
    *
    """
    initial_latitude = initial_position[0]
    initial_longitude = initial_position[1]

    # Total travel time
    total_decisions = boat_velocity_decisions.shape[0]
    if isinstance(time_stamp, float) or (len(time_stamp) == 1):
        time_stamp = np.ones([total_decisions]) * time_stamp

    total_time = np.sum(time_stamp)

    # Fuel consumption
    scalar_boat_velocity_decisions = np.sqrt(np.sum(
        boat_velocity_decisions**2, axis = 1))
    total_fuel_consumption = fuel_consumption(scalar_boat_velocity_decisions)

    # To seconds
    total_fuel_consumption = total_fuel_consumption * 24*3600

    # Total distance
    total_distance = np.sum(scalar_boat_velocity_decisions * total_decisions)

    # Get streams data
    nc = load_nc(path_nc)
    stream_velocities, stream_velocities_lats, stream_velocities_longs =\
        get_stream_velocity(nc, initial_latitude, initial_longitude, date,
                            coord_range)

    # Iterate over the hole route
    positions = np.zeros([total_decisions+1, 2])
    current_position = initial_position
    positions[0,:] = current_position
    for decision in np.arange(total_decisions):
        new_position = boat_movement(scalar_boat_velocity_decisions[decision],
                                     current_position[0],
                                     current_position[1],
                                     stream_velocities.T,
                                     stream_velocities_lats,
                                     stream_velocities_longs,
                                     time_stamp[decision])
        positions[decision + 1, 0] = new_position[0]
        positions[decision + 1, 1] = new_position[1]
        current_position = new_position

    return positions, total_fuel_consumption, total_distance, total_time

import numpy as np
import warnings
from scipy.interpolate import interp2d


def interpolate_discrete(lat, long, grid_array, array_latitudes,
                         array_longitudes, sector_interpolation=False,
                         step=0.01):
    """
    Given K 2D-arrays, representing some K-dim magnitude values, interpolates the K values for a given latitude and
    longitude. It can also return the interpolated values for the sector the lat, long pint yields on.

    Params:
    ------
    lat: float. Latitude.
    long: float. Longitude.
    grid_array: array. An (M, N, K)-dimensional array containing the
    K-dim values of the magnitude. Each MxN matrix element represents the value for the k_th dimension of that
    magnitude in the m_th, n_th latitude, longitude point on the discrete space.
    array_latitudes: array. An (M)-dimensional vector with the latitude values corresponding to the discrete points of the grid_array.
    array_longitudes: array. An (N)-dimensional vector with the longitude values corresponding to the discrete points
    of the grid_array.
    sector_interpolation: boolean. Default = False. Whether to interpolate the sector or not.
    step: float. Default = 0.01. If interpolating the sector, the step of the new discrete map.

    Returns:
    ------
    interpolated_values: array. (K)-dimensional vector with the interpolated values.
    lat_s, long_s: arrays. [ONLY IF sector_interpolation = True] (M')-dim and (N')-dim arrays containing the lats and
        longs of the interpolated sector
    interpolated_sector. [ONLY IF sector_interpolation = True] (M', N', K)-dim array containing the interpolated values
        for the whole sector

    """

    # Look for the closest grid points
    idx = (np.abs(array_latitudes - lat)).argmin()
    idy = (np.abs(array_longitudes - long)).argmin()

    # Check if the closest grid point is "above" or "bellow"
    if array_latitudes[idx] < lat:
        idx -= 1
    if array_longitudes[idy] < long:
        idy -= 1

    # Select the sector to interpolate
    small_lat = array_latitudes[idx:idx + 2]
    small_long = array_longitudes[idy:idy + 2]
    small_array = grid_array[idx:idx + 2, idy:idy + 2, :]

    k_dimensions = small_array.shape[2]

    # Interpolation for each matrix
    interpolated_values = np.zeros([k_dimensions])
    if sector_interpolation:
        lat_s = np.arange(array_latitudes[idx], array_latitudes[idx + 1], step)
        long_s = np.arange(array_longitudes[idx], array_longitudes[1][idx + 1],
                           step)
        interpolated_sector = np.zeros([len(lat_s), len(long_s), k_dimensions])
    for i, dimension in enumerate(np.arange(k_dimensions)):
        interpolation_function = interp2d(small_lat, small_long,
                                          small_array[:, :, dimension])

        # Interpolate value
        interpolated_values[i] = interpolation_function(lat, long)

        if sector_interpolation:
            interpolated_sector[:, :, i] = interpolation_function(lat_s, long_s)

    if sector_interpolation:
        return interpolated_values, lat_s, long_s, interpolated_sector
    else:
        return interpolated_values


def velocity_composition(boat_velocity, lat, long, stream_velocities,
                         stream_velocities_latitudes,
                         stream_velocities_longitudes):
    """
    Given streams velocities on a discrete grid of latitude and longitude points, and a determined boat velocity, it
    compounds the sum of both velocities, interpolating the stream velocity for the latitude and longitude
    position of the boat.

    Params:
    ------
    - boat_velocity: array. Vector containing the latitude and longitude
    - components of the boat velocity, in that order.
    * lat: float. Latitude.
    * long: float. Longitude.
    * grid_array: array. An (M, N, K)-dimensional array containing the K-dim
        values of the magnitude. Each MxN matrix
        element represents the value for the k_th dimension of that magnitude in
        the m_th, n_th latitude, longitude point on the discrete space.
    * array_latitudes: array. An (M)-dimensional vector with the latitude
        values corresponding to the discrete points of the grid_array.
    * array_longitudes: array. An (N)-dimensional vector with the longitude
        values corresponding to the discrete points of the grid_array.

    Returns:
    ------
    * velocity. array. A vector with the composed sum of velocities, with the
        latitude and longitude components on that order.
    """
    interpolated_stream_velocities = interpolate_discrete(lat, long,
                                                          stream_velocities,
                                                          stream_velocities_latitudes,
                                                          stream_velocities_longitudes)
    velocity = boat_velocity + interpolated_stream_velocities

    return velocity


def boat_movement(boat_velocity, lat, long, stream_velocities,
                  stream_velocities_latitudes, stream_velocities_longitudes,
                  ts):
    """
    Computes the new position of the boat given the boat velocity, the steams velocity and the timestamp. Timestamp
    is assumed to be so small that the stream velocity does not change from the initial point to the final one.

    Params:
    ------
    boat_velocity: array. Vector containing the latitude and longitude components of the boat velocity, in that order.
    lat: float. Latitude.
    long: float. Longitude.
    grid_array: array. An (M, N, K)-dimensional array containing the K-dim values of the magnitude. Each MxN matrix
        element represents the value for the k_th dimension of that magnitude in the m_th, n_th latitude, longitude point
        on the discrete space.
    array_latitudes: array. An (M)-dimensional vector with the latitude values corresponding to the discrete points of
        the grid_array.
    array_longitudes: array. An (N)-dimensional vector with the longitude values corresponding to the discrete points of
        the grid_array.
    ts: float. The time increment interval [THIS VALUE IS ASSUMED TO BE SMALL SO THE DISPLACEMENT IS SHORT].

    Returns:
    ------
    new_latitude, new_longitude. floats. The values for the latitude and longitude new positions.

    """

    velocity = velocity_composition(boat_velocity, lat, long, stream_velocities,
                                    stream_velocities_latitudes,
                                    stream_velocities_longitudes)
    new_latitude = lat + velocity[0] * ts
    new_longitude = long + velocity[1] * ts
    return new_latitude, new_longitude


def desired_velocity(initial_point, final_point,
                     stream_velocities, stream_velocities_latitudes,
                     stream_velocities_longitudes, ts, max_distance=1):
    """Given the stream speeds, a couple of points, A and B, assuming that the
    distance between those points is much smaller than the 'stream change
    distance', dL, returns the boat velocity and time spent by the boat to
    reach that point.

    Params:
    ------
    initial_point: array. 1D vector, containing the initial point
        coordinates, expressed as latitude and longitude values.
    final_point: array. 1D vector, containing the destination point
        coordinates, expressed as latitude and longitude values.
    grid_array: array. An (M, N, K)-dimensional array containing the K-dim
        values of the magnitude. Each MxN matrix element represents the value
        for the k_th dimension of that magnitude in the m_th, n_th latitude,
        longitude point on the discrete space.
    array_latitudes: array. An (M)-dimensional vector with the latitude values
        corresponding to the discrete points of the grid_array.
    array_longitudes: array. An (N)-dimensional vector with the longitude values
        corresponding to the discrete points of the grid_array.
    time_spent: float. Time spent on that journey.
    max_distance: float. Default = 1. Max distance that can be considered for
        the uniform approximation. In units of grid resolution distance.

    Returns:
    ------

    boat_velocity: array. 1D Vector containing the latitude and longitude
        components of the boat velocity, in that order.
    """

    # Ensure distance is small enough for uniform regime
    distance = np.sqrt(np.sum((final_point - initial_point) ** 2))

    # Assuming that the grid is squared
    grid_distance = np.abs(stream_velocities_latitudes[1]
                           - stream_velocities_latitudes[0])

    if distance >= max_distance * grid_distance:
        warnings.warn('Distance is {}, which may be too huge. Recommended maximum distance is {}'
            .format(distance, max_distance * grid_distance), Warning, stacklevel=2)

    lat = initial_point[0]
    long = initial_point[1]

    final_lat = final_point[0]
    final_long = final_point[1]

    stream_initial_velocity = interpolate_discrete(
        lat, long,
        stream_velocities,
        stream_velocities_latitudes,
        stream_velocities_longitudes)

    stream_final_velocity = interpolate_discrete(
        final_lat, final_long,
        stream_velocities,
        stream_velocities_latitudes,
        stream_velocities_longitudes)

    stream_mean_velocity = (stream_initial_velocity +
                            stream_final_velocity) / 2

    velocity = (final_point - initial_point) / ts
    boat_velocity = velocity - stream_mean_velocity

    return boat_velocity

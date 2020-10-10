import numpy as np

ALPHA = 0.1727
BETA =  0.219


def fuel_consumption(boat_velocities_scalars: np.array,
                     alpha: float=ALPHA, beta: float=BETA):
    """
    Params
    ------
    boat_velocities_scalars : np.array
        a vector containing several scalar velocities. [KNOTS]
    alpha : float, default=ALPHA
    beta : float, default=BETA
    
    Returns
    -------
    fuel : numpy.array
        [TONS PER DAY]
    """

    boat_velocities_scalars
    fuel = alpha * boat_velocities_scalars ** 2 - beta * boat_velocities_scalars
    return fuel
    

import numpy as np

ALPHA = 0.1727
BETA =  0.219


def fuel_consumption(velocity: np.array,
                     alpha: float=ALPHA, beta: float=BETA):
    """
    Params
    ------
    velocity : float
        En nudos
    alpha : float, default=ALPHA
    beta : float, default=BETA
    
    Returns
    -------
    fuel : numpy.array
        En toneladas por dia
    """
    fuel = alpha * velocity ** 2 - beta * velocity
    return fuel
    

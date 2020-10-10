from brkga_mp_ipr.types import BaseChromosome


class RouteProblem():
    def __init__(x_start, x_end, time, steps, v_min, v_max):
        # Start position (lat, lon)
        self.x_start = x_start
        # End position
        self.x_end = x_end
        # Route duration (in seconds)
        self.time = time
        # Steps
        self.steps = steps
        self.time_delta = time / steps
        # Speed limit (meters per second)
        self.v_min = v_min
        self.v_max = v_max
        


class RouteDecoder():
    """
    Simple Traveling Salesman Problem decoder. It creates a permutation of
    nodes induced by the chromosome and computes the cost of the tour.
    """

    def __init__(self, instance: RouteInstance):
        """
        instance
            Class containing the data
        """
        self.instance = instance

    ###########################################################################

    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:
        """
        Given a chromossome, builds a tour.
        Note that in this example, ``rewrite`` has not been used.
        """

        permutation = sorted(
            (key, index) for index, key in enumerate(chromosome)
        )

        cost = self.instance.distance(permutation[0][1], permutation[-1][1])
        for i in range(len(permutation) - 1):
            cost += self.instance.distance(permutation[i][1],
                                           permutation[i + 1][1])
        return cost

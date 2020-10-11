from copy import deepcopy
from datetime import datetime
from os.path import basename
from math import atan
import numpy as np
import random
import time

from brkga_mp_ipr.algorithm import BrkgaMpIpr
from brkga_mp_ipr.enums import ParsingEnum, Sense
from brkga_mp_ipr.types import BiasFunctionType

from brkga_mp_ipr.types import BaseChromosome
from math import pi

# This following code allows the script
# to import the module
import os
import sys

path_module = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path_module)

from src.consumption import fuel_consumption, ALPHA, BETA
from src.route import get_route_end_point
from src.stream import load_nc


class RouteInstance():
    def __init__(self, x_start, x_target, time, steps, v_min, v_max,
                 date,
                 w_distance=1e6,
                alpha=ALPHA, beta=BETA):
        # Start position (lat, lon)
        self.x_start = x_start
        # End position
        self.x_target = x_target
        # Route duration (in seconds)
        self.time = time
        self.date = date
        # Steps
        self.steps = steps
        self.time_delta = time / steps
        # Speed limit (meters per second)
        self.v_min = v_min
        self.v_max = v_max
        self.v_diff = v_max - v_min
        self.w_distance = w_distance
        # Angle of geodesic on Ecuator (radians)
        self.angle_geo = atan(abs(x_start[0] - x_target[0])
                             / abs(x_start[1] - x_target[1]))
        # Current dataset
        self.nc = load_nc()
        self.alpha = alpha
        self.beta = beta


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
        # Assuming angle is in radians
        chro = np.array(chromosome)
        angle = chro[:self.instance.steps] * 2 * pi + self.instance.angle_geo
        v = chro[self.instance.steps:] * self.instance.v_diff + self.instance.v_min
        fuel = fuel_consumption(v, alpha=self.instance.alpha, beta=self.instance.beta)
        # Transform velocity to (lat, lon)
        v_lon = v * np.cos(angle)
        v_lat = v * np.sin(angle)
        v_geo = np.stack((v_lat, v_lon), axis=1)
        x_end, fuel, _, _  = get_route_end_point(self.instance.x_start, v_geo,
                                   self.instance.time_delta, self.instance.date)
        #distance = FUNCION_DISTANCIA_TIERRA(x_end, self.instance.x_target)
        distance = abs(np.sum(np.array(x_end) - np.array(self.instance.x_target)))
        cost = sum(fuel) + distance * self.instance.w_distance
        return cost


###############################################################################
# Enumerations and constants
###############################################################################

class StopRule(ParsingEnum):
    """
    Controls stop criteria. Stops either when:
    - a given number of `GENERATIONS` is given;
    - or a `TARGET` value is found;
    - or no `IMPROVEMENT` is found in a given number of iterations.
    """
    GENERATIONS = 0
    TARGET = 1
    IMPROVEMENT = 2
    
    
class BRKGAParams():
    # Population size
    population_size = 100
    # Elite percentage
    elite_percentage = 0.30
    # Mutants percentage
    mutants_percentage = 0.15
    # Number of elite parents to mate
    num_elite_parents = 2
    # Total number of parents to mate
    total_parents = 3
    # Bias function to be used to rank the parents
    bias_type = BiasFunctionType.LOGINVERSE
    # Number of independent populations
    num_independent_populations = 3
    # Number of pairs of chromosomes to be tested to path relinking.
    pr_number_pairs = 0
    # Mininum distance between chromosomes to path-relink
    pr_minimum_distance = 0.15
    # Path relink type
    pr_type = 'PERMUTATION'
    # Individual selection to path-relink
    pr_selection = 'BESTSOLUTION'
    # Defines the block size based on the size of the population
    alpha_block_size = 1.0
    # Percentage/path size
    pr_percentage = 1.0
    # Interval at which elite chromosomes are exchanged (0 means no exchange)
    exchange_interval = 200
    # Number of elite chromosomes exchanged from each population
    num_exchange_indivuduals = 2


###############################################################################

def main() -> None:
    """
    Proceeds with the optimization. Create to avoid spread `global` keywords
    around the code.
    """
    
    seed = 2020
    stop_arg = 0
    stop_rule = StopRule(stop_arg)

    if stop_rule == StopRule.TARGET:
        stop_argument = float(stop_arg)
    else:
        stop_argument = int(stop_arg)

    maximum_time = float(1e3)

    if maximum_time <= 0.0:
        raise RuntimeError(f"Maximum time must be larger than 0.0. "
                           f"Given {maximum_time}.")

    perform_evolution = True

    ########################################
    # Load config file and show basic info.
    ########################################

    brkga_params = BRKGAParams()
    control_params = {}

    print(f"""------------------------------------------------------
        > Experiment started at {datetime.now()}
        > Algorithm Parameters:""", end="")

    if not perform_evolution:
        print(">    - Simple multi-start: on (no evolutionary operators)")
    else:
        output_string = ""
        for name, value in vars(brkga_params).items():
            output_string += f"\n>  -{name} {value}"
        for name, value in control_params.items():
            output_string += f"\n>  -{name} {value}"

        print(output_string)
        print(f"""> Seed: {seed}
        > Stop rule: {stop_rule}
        > Stop argument: {stop_argument}
        > Maximum time (s): {maximum_time}
        ------------------------------------------------------""")

    ########################################
    # Load instance and adjust BRKGA parameters
    ########################################

    print(f"\n[{datetime.now()}] Reading Route data...")

    x_start = (40, -9)
    x_target = (41.08, -70)
    travel_time = 60 * 60 * 24 * 9
    steps = 100
    v_min = 10
    v_max = 5
    date = '2020-09-16'
    instance = RouteInstance(x_start, x_target, travel_time,
                           steps, v_min, v_max, date)

    print(f"\n[{datetime.now()}] Generating initial tour...")

    ########################################
    # Build the BRKGA data structures and initialize
    ########################################

    print(f"\n[{datetime.now()}] Building BRKGA data...")
    
    # Build a decoder object.
    decoder = RouteDecoder(instance)

    # Chromosome size is the number of nodes.
    # Each chromosome represents a permutation of nodes.
    brkga = BrkgaMpIpr(
        decoder=decoder,
        sense=Sense.MINIMIZE,
        seed=seed,
        chromosome_size=instance.steps * 2,
        params=brkga_params,
        evolutionary_mechanism_on=perform_evolution
    )

    # To inject the initial tour, we need to create chromosome representing
    # that solution. First, we create a set of keys to be used in the
    # chromosome.
    random.seed(seed)
    keys = sorted([random.random() for _ in range(instance.steps * 2)])

    # Then, we visit each node in the tour and assign to it a key.
    initial_chromosome = [0] * instance.steps * 2

    # Inject the warm start solution in the initial population.
    brkga.set_initial_population([initial_chromosome])

    # NOTE: don't forget to initialize the algorithm.
    print(f"\n[{datetime.now()}] Initializing BRKGA data...")
    brkga.initialize()

    ########################################
    # Warm up the script/code
    ########################################

    # To make sure we are timing the runs correctly, we run some warmup
    # iterations with bogus data. Warmup is always recommended for script
    # languages. Here, we call the most used methods.
    print(f"\n[{datetime.now()}] Warming up...")

    bogus_alg = deepcopy(brkga)
    bogus_alg.evolve(2)
    # TODO (ceandrade): warm up path relink functions.
    # bogus_alg.path_relink(brkga_params.pr_type, brkga_params.pr_selection,
    #              (x, y) -> 1.0, (x, y) -> true, 0, 0.5, 1, 10.0, 1.0)
    bogus_alg.get_best_fitness()
    bogus_alg.get_best_chromosome()
    bogus_alg = None

    ########################################
    # Evolving
    ########################################

    print(f"\n[{datetime.now()}] Evolving...")
    print("* Iteration | Cost | CurrentTime")

    best_cost = 1e99
    best_chromosome = initial_chromosome

    iteration = 0
    last_update_time = 0.0
    last_update_iteration = 0
    large_offset = 0
    # TODO (ceandrade): enable the following when path relink is ready.
    # path_relink_time = 0.0
    # num_path_relink_calls = 0
    # num_homogenities = 0
    # num_best_improvements = 0
    # num_elite_improvements = 0
    run = True

    # Main optimization loop. We evolve one generation at time,
    # keeping track of all changes during such process.
    start_time = time.time()
    while run:
        iteration += 1

        # Evolves one iteration.
        brkga.evolve()

        # Checks the current results and holds the best.
        fitness = brkga.get_best_fitness()
        if fitness < best_cost:
            last_update_time = time.time() - start_time
            update_offset = iteration - last_update_iteration

            if large_offset < update_offset:
                large_offset = update_offset

            last_update_iteration = iteration
            best_cost = fitness
            best_chromosome = brkga.get_best_chromosome()

            print(f"* {iteration} | {best_cost:.0f} | {last_update_time:.2f}")
        # end if

        # TODO (ceandrade): implement path relink calls here.
        # Please, see Julia version for that.

        iter_without_improvement = iteration - last_update_iteration

        # Check stop criteria.
        run = not (
            (time.time() - start_time > maximum_time)
            or
            (stop_rule == StopRule.GENERATIONS and iteration == stop_argument)
            or
            (stop_rule == StopRule.IMPROVEMENT and
             iter_without_improvement >= stop_argument)
            or
            (stop_rule == StopRule.TARGET and best_cost <= stop_argument)
        )
    # end while
    total_elapsed_time = time.time() - start_time
    total_num_iterations = iteration

    print(f"[{datetime.now()}] End of optimization\n")

    print(f"Total number of iterations: {total_num_iterations}")
    print(f"Last update iteration: {last_update_iteration}")
    print(f"Total optimization time: {total_elapsed_time:.2f}")
    print(f"Last update time: {last_update_time:.2f}")
    print(f"Large number of iterations between improvements: {large_offset}")

    # TODO (ceandrade): enable when path relink is ready.
    # print(f"\nTotal path relink time: {path_relink_time:.2f}")
    # print(f"\nTotal path relink calls: {num_path_relink_calls}")
    # print(f"\nNumber of homogenities: {num_homogenities}")
    # print(f"\nImprovements in the elite set: {num_elite_improvements}")
    # print(f"\nBest individual improvements: {num_best_improvements}")

    ########################################
    # Extracting the best tour
    ########################################

    tour = []
    for (index, key) in enumerate(best_chromosome):
        tour.append((key, index))
    tour.sort()

    print(f"\n% Best tour cost: {best_cost:.2f}")
    print("% Best tour: ", end="")
    for _, node in tour:
        print(node, end=" ")

    print("\n\nInstance,Seed,NumNodes,TotalIterations,TotalTime,"
          #"TotalPRTime,PRCalls,NumHomogenities,NumPRImprovElite,"
          #"NumPrImprovBest,"
          "LargeOffset,LastUpdateIteration,LastUpdateTime,"
          "Cost")

    print(f"{seed},{instance.steps},{total_num_iterations},"
          f"{total_elapsed_time:.2f},"
          # f"{path_relink_time:.2f},{num_path_relink_calls},"
          # f"{num_homogenities},{num_elite_improvements},{num_best_improvements},"
          f"{large_offset},{last_update_iteration},"
          f"{last_update_time:.2f},{best_cost:.0f}")

###############################################################################

if __name__ == "__main__":
    main()
from copy import deepcopy
from datetime import datetime
from os.path import basename
import random
import time

import docopt

from brkga_mp_ipr.algorithm import BrkgaMpIpr
from brkga_mp_ipr.enums import ParsingEnum, Sense
from brkga_mp_ipr.types_io import load_configuration


from brkga_mp_ipr.types import BaseChromosome
from math import pi

from src.stream import load_nc
from src.consumption import fuel_consumption, ALPHA, BETA


class RouteProblem():
    def __init__(self, x_start, x_target, time, steps, v_min, v_max,
                 w_distance=1e6,
                alpha=ALPHA, beta=BETA):
        # Start position (lat, lon)
        self.x_start = x_start
        # End position
        self.x_target = x_target
        # Route duration (in seconds)
        self.time = time
        # Steps
        self.steps = steps
        self.time_delta = time / steps
        # Speed limit (meters per second)
        self.v_min = v_min
        self.v_max = v_max
        self.v_diff = v_max - v_min
        self.w_distance = w_distance
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
        angle = chromosome[:instance.steps] * 2 * pi
        v = chromosome[instance.steps:] * instance.v_diff + instance.v_min
        fuel = fuel_consumption(v, alpha=instance.alpha, beta=instance.beta)
        x_end = FUNCION_BORJA(v, angle, instance.x0, instance.nc, instance.time_delta)
        distance = FUNCION_DISTANCIA_TIERRA(x_end, instance.x_target)
        cost = fuel + distancia * instance.w_distance
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


###############################################################################

def main() -> None:
    """
    Proceeds with the optimization. Create to avoid spread `global` keywords
    around the code.
    """

    args = docopt.docopt(__doc__)
    # print(args)

    configuration_file = args["--config_file"]
    instance_file = args["--instance_file"]
    seed = int(args["--seed"])
    stop_rule = StopRule(args["--stop_rule"])

    if stop_rule == StopRule.TARGET:
        stop_argument = float(args["--stop_arg"])
    else:
        stop_argument = int(args["--stop_arg"])

    maximum_time = float(args["--max_time"])

    if maximum_time <= 0.0:
        raise RuntimeError(f"Maximum time must be larger than 0.0. "
                           f"Given {maximum_time}.")

    perform_evolution = not args["--no_evolution"]

    ########################################
    # Load config file and show basic info.
    ########################################

    brkga_params, control_params = load_configuration(configuration_file)

    print(f"""------------------------------------------------------
> Experiment started at {datetime.now()}
> Instance: {instance_file}
> Configuration: {configuration_file}
> Algorithm Parameters:""", end="")

    if not perform_evolution:
        print(">    - Simple multi-start: on (no evolutionary operators)")
    else:
        output_string = ""
        for name, value in vars(brkga_params).items():
            output_string += f"\n>  -{name} {value}"
        for name, value in vars(control_params).items():
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

    instance = RouteInstance(instance_file)
    print(f"Number of nodes: {instance.num_nodes}")

    print(f"\n[{datetime.now()}] Generating initial tour...")

    ########################################
    # Build the BRKGA data structures and initialize
    ########################################

    print(f"\n[{datetime.now()}] Building BRKGA data...")

    # Usually, it is a good idea to set the population size
    # proportional to the instance size.
    brkga_params.population_size = min(brkga_params.population_size,
                                       10 * instance.num_nodes)
    print(f"New population size: {brkga_params.population_size}")

    # Build a decoder object.
    decoder = RouteDecoder(instance)

    # Chromosome size is the number of nodes.
    # Each chromosome represents a permutation of nodes.
    brkga = BrkgaMpIpr(
        decoder=decoder,
        sense=Sense.MINIMIZE,
        seed=seed,
        chromosome_size=instance.num_nodes,
        params=brkga_params,
        evolutionary_mechanism_on=perform_evolution
    )

    # To inject the initial tour, we need to create chromosome representing
    # that solution. First, we create a set of keys to be used in the
    # chromosome.
    random.seed(seed)
    keys = sorted([random.random() for _ in range(instance.num_nodes)])

    # Then, we visit each node in the tour and assign to it a key.
    initial_chromosome = [0] * instance.num_nodes

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

    print(f"{basename(instance_file)},"
          f"{seed},{instance.num_nodes},{total_num_iterations},"
          f"{total_elapsed_time:.2f},"
          # f"{path_relink_time:.2f},{num_path_relink_calls},"
          # f"{num_homogenities},{num_elite_improvements},{num_best_improvements},"
          f"{large_offset},{last_update_iteration},"
          f"{last_update_time:.2f},{best_cost:.0f}")

###############################################################################

if __name__ == "__main__":
    main()
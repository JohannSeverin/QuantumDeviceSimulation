import sys

sys.path.append("../")

from simulation.experiment import SimulationResults
from .analysis import (
    sweep_analysis,
    plot_time_evolution,
    plot_time_evolution_with_single_sweep,
)


def automatic_analysis(results: SimulationResults):
    if results.number_of_expvals > 0:
        if results.number_of_sweeps == 1 and not results.only_store_final:
            print("plotting time evolution with single sweep")
            return plot_time_evolution_with_single_sweep(results)
        if results.number_of_sweeps == 0:
            if not results.only_store_final:
                print("plotting time evolution")
                return plot_time_evolution(results)
        if results.number_of_sweeps in [1, 2] and results.only_store_final:
            print("plotting sweep analysis with final expectation value")
            return sweep_analysis(results)
    print("No automatic analysis found.")
    return None

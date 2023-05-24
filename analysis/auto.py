import sys

sys.path.append("../")

from simulation.experiment import SimulationResults
from .analysis import sweep_analysis


def automatic_analysis(results: SimulationResults):
    if results.number_of_expvals > 0:
        if results.number_of_sweeps in [1, 2] and results.only_store_final:
            return sweep_analysis(results)
    else:
        return None

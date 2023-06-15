## Setup
import numpy as np
import matplotlib.pyplot as plt

import os, sys, pickle

sys.path.append("..")

# Paths and imports
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/punchout"


# Load devices/system
from devices.device import Transmon
from devices.system import QubitSystem
from devices.pulses import SquareCosinePulse
from simulation.experiment import MonteCarloExperiment, LindbladExperiment

from analysis.auto import automatic_analysis

## Define devices
qubit = Transmon(
    EC=15 / 100 * 2 * np.pi,
    EJ=15 * 2 * np.pi,
    n_cutoff=15,
    levels=4,
    ng=0.0,
    T1=np.power(10, np.linspace(1, 3, 10)),
)

qubit_pulse = SquareCosinePulse(
    amplitude=0.5,
    frequency=4.11,  # np.linspace(4.075, 4.125, 5),
    phase=0.0,
    duration=16,
    start_time=0.0,
)

# System
system = QubitSystem(qubit, qubit_pulse=qubit_pulse)

##### SIMULATIONS #####
name = "stochastic_tests"
experiment_name = os.path.join(experiment_path, name)
times = np.linspace(0, 16, 1600)

# Experiment
dispersive_schroedinger_experiment = MonteCarloExperiment(
    system,
    [system.get_states(0)],
    times,
    store_states=True,
    only_store_final=True,
    expectation_operators=[system.qubit_state_occupation_operator(1)],
    save_path=experiment_name,
    ntraj=1,
    # exp_val_method="average",
)

results = dispersive_schroedinger_experiment.run()

automatic_analysis(results)

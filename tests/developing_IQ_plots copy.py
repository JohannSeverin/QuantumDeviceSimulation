## Setup
import numpy as np
import matplotlib.pyplot as plt

import os, sys, pickle

sys.path.append("..")

# Paths and imports
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data"

# Load devices
from devices.device import Transmon, Resonator
from devices.system import QubitResonatorSystem
from devices.pulses import GaussianPulse


## Define devices
qubit = Transmon(
    EC=15 / 100 * 2 * np.pi,
    EJ=15 * 2 * np.pi,
    n_cutoff=15,
    levels=4,
    ng=0.0,
    T1=1000,
)

resonator_pulse = GaussianPulse(
    amplitude=0.1,
    frequency=6.035,
    phase=0,
    sigma=1,
    duration=16,
    start_time=0,
)

resonator = Resonator(frequency=6.00, levels=10, kappa=1 / 10)

## Define System
system = QubitResonatorSystem(
    qubit=qubit,
    resonator=resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=0.25 * 2 * np.pi * np.linspace(0, 1, 10),
)

## Simulate
from simulation.experiment import LindbladExperiment, MonteCarloExperiment

times = np.linspace(0, 100, 1000)

experiment = MonteCarloExperiment(
    system=system,
    states=[system.get_states(0, 0), system.get_states(1, 0)],
    times=times,
    store_states=True,
    only_store_final=False,
    # expectation_operators=[system.resonator_I(), system.resonator_Q()],
    ntraj=10,
)

results = experiment.run()


from analysis.auto import automatic_analysis
from analysis.Q_func import qfunc_plotter

qfunc_plotter(results, interval=4)

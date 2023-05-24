## Setup
import numpy as np
import matplotlib.pyplot as plt


import os

plt.style.use("../analysis/standard_plot_style.mplstyle")

# Paths and imports
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/punchout"

import sys, pickle, os

sys.path.append("..")

# Load devices/system
from devices.device import Resonator, Transmon
from devices.pulses import SquareCosinePulse
from devices.system import QubitResonatorSystem, DispersiveQubitResonatorSystem

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment, LindbladExperiment

# load Analysis tool
from analysis.auto import automatic_analysis

## Define devices
qubit = Transmon(EC=15 / 25, EJ=15, n_cutoff=15, levels=4, ng=0.0)  # h GHz


# Coupling strength
coupling_strength = 0.1 * 2 * np.pi
times = np.linspace(0, 100, 1000)

##### DISPERSIVE SIMULATIONS #####
name = "resonator_freq_drive"
experiment_name = os.path.join(experiment_path, name)

# SCANNING PARAMETERS
drive_frequency_scan = np.linspace(5.95, 6.05, 2)
resonator_freq_scan = np.linspace(5.95, 6.05, 2)

# Define Resonator
resonator = Resonator(resonator_freq_scan, levels=20, kappa=0.020)

# System
dispersive_system = DispersiveQubitResonatorSystem(
    qubit,
    resonator,
    drive_frequency=drive_frequency_scan,
    drive_amplitude=0.02,
    coupling_strength=coupling_strength,
)

# Experiment
dispersive_schroedinger_experiment = LindbladExperiment(
    dispersive_system,
    [dispersive_system.get_states(0, 0), dispersive_system.get_states(1, 0)],
    times,
    store_states=True,
    only_store_final=True,
    expectation_operators=[dispersive_system.photon_number_operator()],
    save_path=experiment_name,
)

results = dispersive_schroedinger_experiment.run()

analysis = automatic_analysis(results)

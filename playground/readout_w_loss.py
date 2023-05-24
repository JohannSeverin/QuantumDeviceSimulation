## Setup
import numpy as np
import matplotlib.pyplot as plt

import sys, pickle, os

sys.path.append("..")

experiment_name = "lindblad_test"
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/" + experiment_name


# Load devices/system
from devices.device import Resonator, Transmon
from devices.pulses import ReadoutCosinePulse
from devices.system import DispersiveQubitResonatorSystem

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment, LindbladExperiment

# load Analysis tool
from analysis.Q_func import qfunc_plotter


## Define devices
qubit = Transmon(EC=15 / 25, EJ=15, n_cutoff=15, levels=4, ng=0.0, T1=1000)  # h GHz

resonator = Resonator(6.02, levels=15, kappa=0.050)

system = DispersiveQubitResonatorSystem(
    qubit,
    resonator,
    drive_frequency=5.97,  # np.linspace(5.92, 6.02, 15),
    drive_amplitude=0.25,  # np.linspace(0.00, 0.25, 15),
    coupling_strength=2 * np.pi * 0.250,
)

# Create experiment
ground_state = system.get_states(0, 0)
excited_state = system.get_states(1, 0)

times = np.linspace(0, 1000, 1000)

experiment = LindbladExperiment(
    system,
    [ground_state, excited_state],
    times,
    store_states=True,
    only_store_final=True,
    expectation_operators=[system.photon_number_operator()],
    save_path=experiment_path,
)


# Run experiment
results = experiment.run()

# Plot QFunc
qfunc_plotter(results)


## Setup
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../analysis/standard_plot_style.mplstyle")

experiment_name = "test_amplitude"
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/dump"

import sys, pickle, os 
sys.path.append("..")

# Load devices/system
from devices.device import Resonator, Transmon
from devices.pulses import ReadoutCosinePulse
from devices.system import QubitResonatorSystem

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment


## Define devices 
qubit = Transmon(
    EC = 15 / 25,  # h GHz
    EJ = 15 , 
    n_cutoff = 15,  
    levels = 4,
    ng = 0.0
)

resonator = Resonator(
    6.02,
    levels = 10
)

## Define Pulse
readout_pulse = ReadoutCosinePulse(
    frequency = np.linspace(5.90, 6.05, 16),
    amplitude = np.linspace(0.01, 0.16, 16),
    phase     = 0.0
)

## Combine to system
coupling_strength       = 0.250
system = QubitResonatorSystem(qubit, resonator, resonator_pulse = readout_pulse, coupling_strength = coupling_strength)

# Create experiment
ground_state    = system.get_states(0, 0)
excited_state   = system.get_states(1, 0)

times        = np.linspace(0, 100, 1000)

experiment = SchroedingerExperiment(
    system, 
    [ground_state, excited_state], 
    times,
    store_states = False,
    only_store_final = True,
    expectation_opreators = [system.photon_number_operator()]
)

results = experiment.run()



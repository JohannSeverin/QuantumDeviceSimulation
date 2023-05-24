
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
from devices.system import DispersiveQubitResonatorSystem, QubitResonatorSystem

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment, LindbladExperiment

# load Analysis tool
from analysis.auto import AutomaticAnalysis


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
    levels = 100,
    kappa  = 0.050
)

resonator_drive = ReadoutCosinePulse(
    frequency = np.linspace(5.94, 6.00, 5),
    amplitude = np.linspace(0.00, 0.4,  2),
    phase     = 0.0,
)

system = QubitResonatorSystem(
    qubit, 
    resonator,
    resonator_drive,
    coupling_strength   = 2 * np.pi * 0.250
)

# Create experiment
ground_state    = system.get_states(0, 0)
excited_state   = system.get_states(1, 0)

times        = np.linspace(0, 50, 2000)

experiment = SchroedingerExperiment(
    system, 
    [ground_state, excited_state], 
    times,
    store_states = False,
    only_store_final = True,
    expectation_operators = [system.photon_number_operator()],
    save_path = experiment_path,
)


results = experiment.run()

analysis = AutomaticAnalysis(results)
analysis.plot()

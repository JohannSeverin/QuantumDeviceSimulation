## Setup
import numpy as np

import sys

sys.path.append("..")

experiment_name = "control_test"
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/" + experiment_name


# Load devices/system
from devices.device import Resonator, Transmon
from devices.pulses import SquareCosinePulse
from devices.system import QubitResonatorSystem

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment

# load Analysis tool
from analysis.auto import automatic_analysis

times = np.linspace(0, 50, 2000)

## Define devices
qubit = Transmon(EC=15 / 25, EJ=15, n_cutoff=15, levels=4, ng=0.0)

resonator = Resonator(6.02, levels=20, kappa=0.050)

resonator_drive = SquareCosinePulse(
    frequency=5.985,
    amplitude=np.linspace(0.0, 0.2, 2),
    phase=0.0,
    start_time=0,
    duration=32,
)

qubit_drive = SquareCosinePulse(
    frequency=7.86 + np.linspace(-0.4, 0.1, 2),
    amplitude=0.2,
    phase=0.0,
    start_time=32,
    duration=16,
)

# Define the system
system = QubitResonatorSystem(
    qubit,
    resonator,
    coupling_strength=2 * np.pi * 0.250,
    qubit_pulse=qubit_drive,
    resonator_pulse=resonator_drive,
)

# Create experiment
ground_state = system.get_states(0, 0)
excited_state = system.get_states(1, 0)


experiment = SchroedingerExperiment(
    system,
    [ground_state],
    times,
    store_states=False,
    only_store_final=True,
    expectation_operators=[
        system.qubit_state_operator(),
        system.photon_number_operator(),
    ],
    save_path=experiment_path,
)


results = experiment.run()

analysis = automatic_analysis(results)

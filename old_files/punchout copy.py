## Setup
import numpy as np
import matplotlib.pyplot as plt


import os

plt.style.use("../analysis/standard_plot_style.mplstyle")

# Plotting funcs
# Plot results
# def plot_results(results):
#     max_photon_number = np.max(results["exp_vals"])
#     extent = [results["sweep_list"][0].min(), results["sweep_list"][0].max(), results["sweep_list"][1].min(), results["sweep_list"][1].max()]

#     fig, axes = plt.subplots(ncols = 2, figsize = (14, 6), sharey = True)

#     from matplotlib.colors import LinearSegmentedColormap

#     cmap_ground = LinearSegmentedColormap.from_list("ground", ["white", "C0"])

#     img_ground  = axes[0].imshow(results["exp_vals"][..., 0].T, extent = extent,
#         aspect = "auto", origin = "lower", cmap = cmap_ground, vmin = 0, vmax = max_photon_number )
#     img_excited = axes[1].imshow(results["exp_vals"][..., 1].T, extent = extent,
#         aspect = "auto", origin = "lower", cmap = cmap_ground, vmin = 0, vmax = max_photon_number )


#     axes[0].set(
#         xlabel = "Frequency (GHz)",
#         ylabel = "Amplitude (a.u.)",
#         title = "Ground state"
#     )

#     axes[1].set(
#         xlabel = "Frequency (GHz)",
#         title = "Excited state"
#     )

#     fig.colorbar(img_ground, ax = axes[1], label = "Photon number")

#     fig.tight_layout()

#     return fig, axes

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

# Analysis tools
from analysis.auto import AutomaticAnalysis

## Define devices
qubit = Transmon(EC=15 / 25, EJ=15, n_cutoff=15, levels=4, ng=0.0)  # h GHz
resonator = Resonator(6.02, levels=20, kappa=0.020)

# Coupling strength
coupling_strength = 0.25 * 2 * np.pi
times = np.linspace(0, 100, 1000)


##### DISPERSIVE SIMULATIONS #####
name = "dispersive_schroedinger"
experiment_name = os.path.join(experiment_path, name)

# SCANNING PARAMETERS
drive_frequency_scan = np.linspace(5.94, 6.00, 10)
drive_amplitude_scan = np.linspace(0.00, 0.05, 5)

# System
dispersive_system = DispersiveQubitResonatorSystem(
    qubit,
    resonator,
    drive_frequency=drive_frequency_scan,
    drive_amplitude=drive_amplitude_scan,
    coupling_strength=coupling_strength,
)

print(f"dispersive shift: {dispersive_system.dispersive_shift() / 2 / np.pi}")

# Experiment
dispersive_schroedinger_experiment = SchroedingerExperiment(
    dispersive_system,
    [dispersive_system.get_states(0, 0), dispersive_system.get_states(1, 0)],
    times,
    store_states=False,
    only_store_final=True,
    expectation_operators=[dispersive_system.photon_number_operator()],
    save_path=experiment_name,
)

results = dispersive_schroedinger_experiment.run()

# Plot results
analysis_1 = AutomaticAnalysis(results)
fig, axes = analysis_1.plot()


##### DISPERSIVE SIMULATIONS #####
name = "dispersive_lindblad"
experiment_name = os.path.join(experiment_path, name)

# System
dispersive_system = DispersiveQubitResonatorSystem(
    qubit,
    resonator,
    drive_frequency=drive_frequency_scan,
    drive_amplitude=drive_amplitude_scan,
    coupling_strength=coupling_strength,
)

# Experiment
dispersive_lindblad_experiment = LindbladExperiment(
    dispersive_system,
    [dispersive_system.get_states(0, 0), dispersive_system.get_states(1, 0)],
    times,
    store_states=False,
    only_store_final=True,
    expectation_operators=[dispersive_system.photon_number_operator()],
    save_path=experiment_name,
)

results = dispersive_lindblad_experiment.run()

# Plot results
plot_results(results)


###### Full Schroedinger #####
name = "full_schroedinger"
experiment_name = os.path.join(experiment_path, name)

times = np.linspace(0, 100, 2500)

# Pulse
readout_pulse = ReadoutCosinePulse(
    frequency=drive_frequency_scan[::],  # SCANNING THESE TWO PARAMETERS
    amplitude=drive_amplitude_scan[::],  # SCANNING THESE TWO PARAMETERS
    phase=0.0,
)

# System
system = QubitResonatorSystem(
    qubit, resonator, resonator_pulse=readout_pulse, coupling_strength=coupling_strength
)

# Experiment
full_schroedinger_experiment = SchroedingerExperiment(
    system,
    [system.get_states(0, 0), system.get_states(1, 0)],
    times,
    store_states=False,
    only_store_final=True,
    expectation_operators=[system.photon_number_operator()],
    save_path=experiment_name,
)


results = full_schroedinger_experiment.run()

# plt.plot(times, results["exp_vals"][:, 1].T, label = "Ground")


# Plot results
plot_results(results)

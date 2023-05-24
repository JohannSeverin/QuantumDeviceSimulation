
## Setup
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../analysis/standard_plot_style.mplstyle")

experiment_name = "lindblad_test"
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/" + experiment_name

import sys, pickle, os 
sys.path.append("..")

# Load devices/system
from devices.device import Resonator, Transmon
from devices.pulses import ReadoutCosinePulse
from devices.system import DispersiveQubitResonatorSystem

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment, LindbladExperiment


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
    levels = 15,
    kappa  = 0.050
)


system = DispersiveQubitResonatorSystem(
    qubit, 
    resonator,
    drive_frequency     = np.linspace(5.92, 6.02, 15),
    drive_amplitude     = np.linspace(0.00, 0.25, 15),
    coupling_strength   = 2 * np.pi * 0.250
)

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
    expectation_operators = [system.photon_number_operator()],
    save_path = experiment_path,
)

# experiment = LindbladExperiment(
#     system,
#     [ground_state, excited_state],
#     times,
#     store_states = False,
#     only_store_final = True,
#     expectation_operators = [system.photon_number_operator()],
#     save_path = experiment_path,
# )

results = experiment.run()

# plt.plot(results["exp_vals"].reshape(-1, 100).T)

# Numbers for plotting
max_photon_number = np.max(results["exp_vals"])

# Plot results
fig, axes = plt.subplots(ncols = 2, figsize = (14, 6), sharey = True)

from matplotlib.colors import LinearSegmentedColormap

cmap_ground = LinearSegmentedColormap.from_list("ground", ["white", "C0"])

img_ground  = axes[0].imshow(results["exp_vals"][..., 0].T, extent = [5.92, 6.00, 0.01, 0.20], 
    aspect = "auto", origin = "lower", cmap = cmap_ground, vmin = 0, vmax = max_photon_number )
img_excited = axes[1].imshow(results["exp_vals"][..., 1].T, extent = [5.92, 6.00, 0.01, 0.20], 
    aspect = "auto", origin = "lower", cmap = cmap_ground, vmin = 0, vmax = max_photon_number )


axes[0].set(
    xlabel = "Frequency (GHz)",
    ylabel = "Amplitude (a.u.)",
    title = "Ground state"
)

axes[1].set(
    xlabel = "Frequency (GHz)",
    title = "Excited state"
)

fig.colorbar(img_ground, ax = axes[1], label = "Photon number")

fig.tight_layout()


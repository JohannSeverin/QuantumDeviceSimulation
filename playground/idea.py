import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../analysis/standard_plot_style.mplstyle")

experiment_name = "test_amplitude"
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/dump"

import sys, pickle, os 
sys.path.append("..")

from devices.basic_system   import QubitResonatorSystem
from devices.resonator      import Resonator
from devices.transmon       import Transmon
from devices.pulses         import ReadoutCosinePulse

from simulation.lindblad_evolution import LindbladExperiment


system = QubitResonatorSystem(
    qubit = Transmon(
        EC = 15 * 2 * np.pi / 25,
        EJ = 15 * 2 * np.pi,
        n_cutoff = 15,
        ng = 0.0,
        levels = 4
    ),
    resonator = Resonator(
        6.02,
        levels = 10
    ),
    resonator_drive = ReadoutCosinePulse(
        amplitude = 0.1,
        frequency = 6.02,
        phase = 0.0
    ),   
    coupling_strength = 0.250 * 2 * np.pi
)

states = system.get_states(qubit_states = 0, resonator_states = 0)

times = np.linspace(0, 50, 1000)

experiment = LindbladExperiment(
    system = system,
    times = times,
    states = states,
    running_states = False,
    running_exp_vals = [system.photon_number_operator()],
)

results = experiment.run_experiments()

with open(os.path.join(experiment_path, experiment_name + ".pkl"), "wb") as file:
    pickle.dump(results, file)


# ### ANALYSIS #### 
# plt.figure()
# plt.title("Mean Photon Number vs. Frequency")
# plt.plot(results["sweep_list"], results["exp_vals"][:, :, :].mean(axis = 2), label = ["$|0, 0\\rangle$", "$|1, 0\\rangle$", "$|2, 0\\rangle$"])
# plt.xlabel("Frequency (GHz)")
# plt.ylabel("Mean Photon Number")
# plt.legend()

# plt.figure()
# plt.title("Dynamics at Most Resonant Frequency")
# for i in range(3):
#     most_resonant = np.argmax(results["exp_vals"][:, i, :].mean(axis = 1))
#     plt.plot(times, results["exp_vals"][most_resonant, i, :], label = f"$|{i}, 0\\rangle$ at {results['sweep_list'][most_resonant]:.3f} GHz")
# plt.xlabel("Time (ns)")
# plt.ylabel("Mean Photon Number")
# plt.legend()



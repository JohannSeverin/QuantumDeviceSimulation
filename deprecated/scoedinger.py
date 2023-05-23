import qutip
import numpy as np

from experiment import SimulationExperiment




class SchrodingerExperiment(SimulationExperiment):
    def __init__(self, system, states, times, expectation_opreators = None, store_states = False, only_store_final = False):
        self.system = system
        
        super().__init__(system, times, states, expectation_operators = expectation_opreators, store_states = store_states, only_store_final = only_store_final)
    
    def simulate(self, state):
        """
        Simulate the system.
        """
        H_0 = self.system.hamiltonian

        if self.system.hamiltonian_t:
            H_1 = self.system.hamiltonian_t
            H   = [H_0, H_1]
        else:
            H = H_0

        return qutip.sesolve(
            H,
            psi0    = state,
            tlist   = self.times,
            options = self.options
        )



if __name__ == "__main__":
    import sys
    sys.path.append("../")
    
    from devices.resonator import Resonator
    from devices.transmon import Transmon
    from devices.pulses import ReadoutCosinePulse
    from devices.basic_system import QubitResonatorSystem

    n_cutoff    = 15
    EJ          = 15        # h GHz
    EC          = EJ / 25   # h GHz

    qubit = Transmon(EC, EJ, n_cutoff, 0.0, levels = 4)

    resonator_states        = 20
    resonator_frequency     = 6.02  # GHz

    resonator = Resonator(resonator_frequency, levels = resonator_states)

    amplitude   = 0.1
    frequency   = 6.02 * np.linspace(0.98, 1.02, 50)
    phase       = 0.0

    readout_pulse = ReadoutCosinePulse(frequency, amplitude, phase)

    coupling_strength       = 0.250
    system = QubitResonatorSystem(qubit, resonator, resonator_pulse = readout_pulse, coupling_strength = coupling_strength)

    # Create experiment
    ground_state    = system.get_states(0, 0)
    excited_state   = system.get_states(1, 0)

    times        = np.linspace(0, 100, 1000)
    
    experiment = SchrodingerExperiment(
        system, 
        [ground_state, excited_state], 
        times,
        store_states = True,
        only_store_final = True,
        expectation_opreators = [system.photon_number_operator()]
    )

    result = experiment.run()
    
    import matplotlib.pyplot as plt 
    
    plt.plot(result["sweep_list"], result["exp_vals"][:, 0])
    plt.plot(result["sweep_list"], result["exp_vals"][:, 1])
    
    

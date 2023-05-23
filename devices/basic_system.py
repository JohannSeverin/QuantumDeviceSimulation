import qutip
import numpy as np

from qutip import tensor, basis, ket2dm

from devices.system import System



class QubitResonatorSystem(System):
    """
    A class to create a simple qubit-resonator system.

    Parameters:
    ----
    qubit: A qubit class, only Transmon exists at the moment. 
    resonator: a resonator class
    coupling: float, the strength of the coupling between qubit and resonator given in h * GHz
    """
    def __init__(self, qubit, resonator, resonator_pulse, coupling_strength):
        """
        A class to create a simple qubit-resonator system.

        Parameters
        ----
        qubit: A qubit class, only Transmon exists at the moment
        resonator: a resonator class
        coupling: float, the strength of the coupling between qubit and resonator
        """
        self.devices = {
            "qubit": qubit,
            "resonator": resonator,
            "resonator_pulse": resonator_pulse
        }

        # Set parameters
        self.system_parameters = {"coupling": coupling_strength * 2 * np.pi}

        # Set methods to be updated
        self.update_methods = [self.set_opreators]

        super().__init__()

    def set_opreators(self):
        # devices
        qubit           = self.devices["qubit"]
        resonator       = self.devices["resonator"]
        resonator_pulse = self.devices["resonator_pulse"]
        
        # system parameters
        coupling = self.parameters["system"]["coupling"]

        # Time independent hamiltonian
        H_0_qubit       = tensor(qubit.hamiltonian, qutip.qeye(resonator.levels))
        H_0_resonator   = tensor(qutip.qeye(qubit.levels), resonator.hamiltonian)
        H_interaction   = coupling * tensor(qubit.charge_matrix, resonator.a_dag + resonator.a)

        self.hamiltonian = H_0_qubit + H_0_resonator + H_interaction

        # Time dependent hamiltonian
        resonator_coupling_operator = tensor(qutip.qeye(qubit.levels), resonator.coupling_operator)

        self.hamiltonian_t = [resonator_coupling_operator, resonator_pulse.pulse] 

    def get_states(self, qubit_states = 0, resonator_states = 0):
            # Only integers
            if isinstance(qubit_states, int) and isinstance(resonator_states, int):        
                qubit       = basis(self.devices["qubit"].levels, qubit_states)
                resonator   = basis(self.devices["resonator"].levels, resonator_states)
                return tensor(qubit, resonator)
            
            # Integer and list of integers
            elif isinstance(qubit_states, int) and isinstance(resonator_states, list):
                qubit       = basis(self.devices["qubit"].levels, qubit_states)
                resonator_states = [basis(self.devices["resonator"].levels, state) for state in resonator_states]
                return [tensor(qubit, resonator) for resonator in resonator_states]
            
            # List of integers and integer
            elif isinstance(qubit_states, list) and isinstance(resonator_states, int):
                qubit_states = [basis(self.devices["qubit"].levels, state) for state in qubit_states]
                resonator   = basis(self.devices["resonator"].levels, resonator_states)
                return [tensor(qubit, resonator) for qubit in qubit_states]

            # List of integers
            elif len(qubit_states) == len(resonator_states):
                qubits_states = [basis(self.devices["qubit"].levels, state) for state in qubit_states]
                resonator_states = [basis(self.devices["resonator"].levels, state) for state in resonator_states]
                return [tensor(qubit, resonator) for qubit, resonator in zip(qubits_states, resonator_states)]

    def photon_number_operator(self):
        return tensor(qutip.qeye(self.devices["qubit"].levels), self.devices["resonator"].a_dag * self.devices["resonator"].a)
    
    def qubit_state_operator(self):
        return tensor(qutip.num(self.devices["qubit"].levels), qutip.qeye(self.devices["resonator"].levels))




if __name__ == "__main__":
    
    from resonator import Resonator
    from transmon import Transmon
    from pulses import ReadoutCosinePulse

    n_cutoff    = 15
    EJ          = 15     * 2 * np.pi # h GHz
    EC          = EJ / 25

    qubit = Transmon(EC, EJ, n_cutoff, 0.0, levels = 4)

    resonator_states        = 10
    resonator_frequency     = 6.02

    resonator = Resonator(resonator_frequency, levels = resonator_states)

    amplitude   = 0.1  * np.linspace(0, 1, 10)
    frequency   = 6.02 * np.linspace(0, 1, 10)
    phase       = 0.0

    readout_pulse = ReadoutCosinePulse(frequency, amplitude, phase)

    coupling_strength       = 0.250
    system = QubitResonatorSystem(qubit, resonator, resonator_pulse = readout_pulse, coupling_strength = coupling_strength)

    # print(system.hamiltonian.full())
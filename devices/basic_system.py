import qutip
import numpy as np

from qutip import tensor, basis, ket2dm



class QubitResonatorSystem:


    def __init__(self, qubit, resonator, coupling_strength):
        """
        A class to create a simple qubit-resonator system.

        Parameters
        ----
        qubit: A qubit class, only Transmon exists at the moment
        resonator: a resonator class
        coupling: float, the strength of the coupling between qubit and resonator
        """

        # Load qubit and resonator
        self.qubit = qubit
        self.resonator = resonator

        self.coupling = coupling_strength
    
    def get_states(self, qubit_states = 0, resonator_states = 0):
        # Only integers
        if isinstance(qubit_states, int) and isinstance(resonator_states, int):        
            qubit       = basis(self.qubit.levels, qubit_states)
            resonator   = basis(self.resonator.levels, resonator_states)
            return ket2dm(tensor(qubit, resonator))
        
        # Integer and list of integers
        elif isinstance(qubit_states, int) and isinstance(resonator_states, list):
            qubit       = basis(self.qubit.levels, qubit_states)
            resonator_states = [basis(self.resonator.levels, state) for state in resonator_states]
            return [ket2dm(tensor(qubit, resonator)) for resonator in resonator_states]
        
        # List of integers and integer
        elif isinstance(qubit_states, list) and isinstance(resonator_states, int):
            qubit_states = [basis(self.qubit.levels, state) for state in qubit_states]
            resonator   = basis(self.resonator.levels, resonator_states)
            return [ket2dm(tensor(qubit, resonator)) for qubit in qubit_states]

        # List of integers
        elif len(qubit_states) == len(resonator_states):
            qubits_states = [basis(self.qubit.levels, state) for state in qubit_states]
            resonator_states = [basis(self.resonator.levels, state) for state in resonator_states]
            return [ket2dm(tensor(qubit, resonator)) for qubit, resonator in zip(qubits_states, resonator_states)]
        
    def get_hamiltonian(self):
        # Create the full Hamiltonian
        non_interacting_hamiltonian = tensor(self.qubit.hamiltonian, qutip.qeye(self.resonator.levels)) + tensor(qutip.qeye(self.qubit.levels), self.resonator.hamiltonian)

        # Interaction Hamiltonian
        interacting_hamiltonian = self.coupling * tensor(self.qubit.charge_matrix, self.resonator.a_dag + self.resonator.a)

        return non_interacting_hamiltonian + interacting_hamiltonian

    def dispersive_shift(self, frequency):
        # Calculate dispersive shifts
        # Multi qubit shifts
        g_squared_matrix    = self.coupling ** 2 * self.qubit.charge_matrix.full() ** 2
    
        omega_ij_matrix     = np.expand_dims(self.qubit.hamiltonian.diag(), 1) - np.expand_dims(self.qubit.hamiltonian.diag(), 0)
        
        chi_matrix = g_squared_matrix * (1 / (omega_ij_matrix - frequency) + 1 / (omega_ij_matrix + frequency)) 

        # The dispersive shifts
        dispersive_shifts = chi_matrix.sum(axis = 1)

        return dispersive_shifts

    def dispersive_hamiltonian(self, frequency = None):
        """
        Do the dispersive appriximation. Include the frequency of the reference frame to enter. 

        Parameters:
        ----
        frequency: float / None, the frequency of the reference frame. If None use resonator frequencvy
        """

        if frequency is None:
            frequency = self.resonator.frequency

        # The dispersive shifts
        dispersive_shifts = self.dispersive_shift(frequency)

        # Create the dispersive Hamiltonian
        resonator_hamiltonian = (self.resonator.frequency - frequency) * tensor(qutip.qeye(self.qubit.levels), self.resonator.a_dag * self.resonator.a)
        qubit_hamiltonian     = tensor(qutip.Qobj(np.diag(dispersive_shifts, 0)), self.resonator.a_dag * self.resonator.a)

        return resonator_hamiltonian + qubit_hamiltonian

    def dispersive_drive(self, amplitude, frequency = None):
        """
        The contribution to the Hamiltonian from driving the resonator.

        Parameters:
        ----
        amplitude: float, the amplitude of the drive
        frequency: float / None, the frequency of the drive. If None use resonator frequencvy. This assumes that frequency of drive is equal to the frequency of the reference frame.
        """

        if frequency is None:
            frequency = self.resonator.frequency

        drive_hamiltonian = amplitude * tensor(qutip.qeye(self.qubit.levels), self.resonator.a_dag + self.resonator.a)

        return drive_hamiltonian

    def get_resonator_interaction(self):
        I_drive = tensor(qutip.qeye(self.qubit.levels),       self.resonator.a_dag + self.resonator.a )
        # Q_drive = tensor(qutip.qeye(self.qubit.levels), 1j * ( self.resonator.a_dag - self.resonator.a))

        return I_drive # , Q_drive

    def get_qubit_decay_operators(self, T1 = None):
        # Assuming that rate from n-1 to n is sqrt(n) / T1
        qubit_decay = qutip.destroy(self.qubit.levels)

        gamma1 = 1 / T1 if T1 is not None else 0

        return np.sqrt(gamma1) * qubit_decay

    def get_resonator_decay_operators(self, kappa = None):
        # kappa is the linewidth of the resonator
        resonator_decay = qutip.destroy(self.resonator.levels)

        return kappa * resonator_decay
    
    def get_collapse_operators(self):
        c_ops = []

        if self.qubit.T1:
            c_ops.append(self.set_qubit_decay_operators(self.qubit.T1))

        if self.resonator.kappa:
            c_ops.append(self.set_resonator_decay_operators(self.resonator.kappa))

        return c_ops

    def photon_number_operator(self):
        # Define function of type f(t, state) -> expectation value
        return (self.resonator.a_dag * self.resonator.a, 1)



if __name__ == "__main__":
    
    from resonator import Resonator
    from transmon import Transmon


    n_cutoff    = 15
    EJ          = 15     * 2 * np.pi # h GHz
    EC          = EJ / 25

    qubit = Transmon()
    qubit.from_device_parameters(EC, EJ, n_cutoff, 0.0, levels = 4)


    resonator_states        = 10
    resonator_frequency     = 6.02 * 2 * np.pi    

    resonator = Resonator(resonator_frequency, levels = resonator_states)

    coupling_strength       = 0.250 * 2 * np.pi

    system = QubitResonatorSystem(qubit, resonator, coupling_strength)

    full_hamiltoian = system.full_hamiltoian()
    dispersive_hamiltonian = system.dispersive_hamiltonian()
    dispersive_hamiltonian += system.dispersive_drive(0.1)

    initial_state = system.get_state(0, 0)

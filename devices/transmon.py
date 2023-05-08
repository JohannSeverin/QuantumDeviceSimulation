import qutip
import numpy as np

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh



class Transmon:

    def __init__(self):
        """
        A Container for variables for a transmon. 

        The transmon can be intilized in two ways. Either from device params where it is simulated for EC, EJ in a discrete basis n. 
        The other option is to take it from the calibrated parameters, where a given frequency and anharmonicity can be used.
        """

    def from_device_parameters(self, EC, EJ, n_cutoff, ng, levels = 3, T1 = None):
        """
        Simulate transmon from device parameters.

        Parameters
        ----
        EC: float : The capacitor energy of the transmon
        EJ: float : The Josephson energy of the transmon
        n_cutoff: int : The number of levels to simulate (the size of the hilbert space is 2 * n_cutoof + 1)
        ng: float : The offset charge of the transmon

        levels : int : The number of levels to carry over in the reduced hamiltonian        
        """
        self.levels = levels
        self.T1     = T1

        # Define basis
        n_matrix = diags(np.arange(-n_cutoff, n_cutoff + 1) - ng, 0, shape = (2 * n_cutoff + 1, 2 * n_cutoff + 1))

        # Define the flux operators
        exp_i_flux = diags(np.ones(2 * n_cutoff), offsets = -1)
        cos_flux   = (exp_i_flux.getH() + exp_i_flux) / 2 
        
        # Calculate energy
        kinetic     = 4 * EC * n_matrix ** 2
        potential   = - EJ * cos_flux

        H = kinetic + potential

        # Diagonalize
        eigenvalues, eigenvectors = eigsh(H, k = levels, which = 'SA')

        # Calculate charge matrix
        charge_matrix = eigenvectors.conj().T @ n_matrix @ eigenvectors
        charge_matrix[np.isclose(charge_matrix, 0.0, atol = 1e-10)] = 0.0

        # Store
        self.hamiltonian    = qutip.Qobj(diags(eigenvalues))
        self.charge_matrix  = qutip.Qobj(csr_matrix(charge_matrix)) 

    

        return self


    def from_calibrated_parameters(self, frequencies, charge_matrix = None):
        """
        Simulate transmno from calibrated parameters.

        Parameters
        ----
        frequencies: array float : The frequency of the transmon. This will be the diagonal of the hamiltonian.
        charge_matrix: array (d x d) array: The matrix elements of the charge matrix: <i|n|j>. This is used for interaction with the resonator 
        """
        raise NotImplementedError




if __name__ == '__main__':

    n_cutoff    = 15
    EJ          = 15     * 2 * np.pi # h GHz
    EC          = EJ / 25

    system = Transmon()
    system.from_device_parameters(EC, EJ, n_cutoff, 0.0, levels = 4)

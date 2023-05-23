import qutip
import numpy as np

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh


from devices.device import Device

class Transmon(Device):

    def __init__(self, EC, EJ, n_cutoff, ng, levels = 3, T1 = None):
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
        self.param_type = "device_params"

        self.parameters = {
            "EC"        : EC * 2 * np.pi,
            "EJ"        : EJ * 2 * np.pi,
            "ng"        : ng,
            "T1"        : T1,
        }

        self.levels     = levels
        self.n_cutoff   = n_cutoff

        self.update_methods = [self.set_operators, self.set_dissipators]

        super().__init__()

    def set_operators(self):
        """
        Set the operators used for dynamics. Everything changeable will be in the self.parameters dictionary.
        """
        # Unpack parameters
        EC          = self.parameters["EC"]
        EJ          = self.parameters["EJ"]
        ng          = self.parameters["ng"]

        # Define basis
        n_matrix = diags(np.arange(-self.n_cutoff, self.n_cutoff + 1) - ng, 0, shape = (2 * self.n_cutoff + 1, 2 * self.n_cutoff + 1))

        # Define the flux operators
        exp_i_flux = diags(np.ones(2 * self.n_cutoff), offsets = -1)
        cos_flux   = (exp_i_flux.getH() + exp_i_flux) / 2 
        
        # Calculate energy
        kinetic     = 4 * EC * n_matrix ** 2
        potential   = - EJ * cos_flux

        H = kinetic + potential

        # Diagonalize
        eigenvalues, eigenvectors = eigsh(H, k = self.levels, which = 'SA')

        # Calculate charge matrix
        charge_matrix = eigenvectors.conj().T @ n_matrix @ eigenvectors
        charge_matrix[np.isclose(charge_matrix, 0.0, atol = 1e-10)] = 0.0

        # Store
        self.hamiltonian    = qutip.Qobj(diags(eigenvalues))
        self.charge_matrix  = qutip.Qobj(csr_matrix(charge_matrix)) 

    def set_dissipators(self):
        """
        Set the dissipators used for dynamics. Everything changeable will be in the self.parameters dictionary.
        """
        # Unpack parameters
        T1 = self.parameters["T1"]

        # Set the dissipator
        if T1 is not None:
            self.dissipators = [np.sqrt(1 / T1) * qutip.destroy(self.levels)]
        else:
            self.dissipators = []


if __name__ == '__main__':

    n_cutoff    = 15
    EJ          = 15     * 2 * np.pi # h GHz
    EC          = EJ / 25
    T1          = np.linspace(0.1, 10, 10)

    system = Transmon(EC, EJ, n_cutoff, 0.0, levels = 4, T1 = T1)

import qutip
import numpy as np

from qutip import tensor

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh


class Device:
    """
    Parent class for all devices. This is intended to take of common operations like sweeping parameters for an individual class and propagating them to the system.
    """

    def __init__(self):
        """
        Initialize the device.
        """
        if not hasattr(self, "parameters"):
            raise ValueError("The device must have a parameters attribute.")

        if not hasattr(self, "update_methods"):
            raise ValueError("The device must have an update_methods attribute.")

        self.parameters_to_be_swept()

        if self.should_be_swept:
            for key in self.sweep_parameters:
                self.parameters[key] = self.sweep_list[key][0]

        self.update(self.parameters)

    def parameters_to_be_swept(self):
        """
        Returns a list of the parameters that should be swept.
        """
        self.should_be_swept = np.any(
            [
                isinstance(self.parameters[key], np.ndarray)
                for key in self.parameters.keys()
            ]
        )

        if self.should_be_swept:
            self.sweep_parameters = [
                key
                for key in self.parameters.keys()
                if isinstance(self.parameters[key], np.ndarray)
            ]
            self.sweep_list = {
                key: self.parameters[key] for key in self.sweep_parameters
            }
        else:
            self.sweep_parameters = None
            self.sweep_list = None

    def update(self, new_parameters):
        """
        Update the parameters of the device.
        """
        self.parameters.update(new_parameters)

        for update_func in self.update_methods:
            update_func()


class Transmon(Device):
    def __init__(self, EC, EJ, n_cutoff, ng, levels=3, T1=None):
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
            "EC": EC * 2 * np.pi,
            "EJ": EJ * 2 * np.pi,
            "ng": ng,
            "T1": T1,
        }

        self.levels = levels
        self.n_cutoff = n_cutoff

        self.update_methods = [self.set_operators, self.set_dissipators]

        super().__init__()

    def set_operators(self):
        """
        Set the operators used for dynamics. Everything changeable will be in the self.parameters dictionary.
        """
        # Unpack parameters
        EC = self.parameters["EC"]
        EJ = self.parameters["EJ"]
        ng = self.parameters["ng"]

        # Define basis
        n_matrix = diags(
            np.arange(-self.n_cutoff, self.n_cutoff + 1) - ng,
            0,
            shape=(2 * self.n_cutoff + 1, 2 * self.n_cutoff + 1),
        )

        # Define the flux operators
        exp_i_flux = diags(np.ones(2 * self.n_cutoff), offsets=-1)
        cos_flux = (exp_i_flux.getH() + exp_i_flux) / 2

        # Calculate energy
        kinetic = 4 * EC * n_matrix**2
        potential = -EJ * cos_flux

        H = kinetic + potential

        # Diagonalize
        eigenvalues, eigenvectors = eigsh(H, k=self.levels, which="SA")

        # Calculate charge matrix
        charge_matrix = eigenvectors.conj().T @ n_matrix @ eigenvectors
        charge_matrix[np.isclose(charge_matrix, 0.0, atol=1e-10)] = 0.0

        # Store
        self.hamiltonian = qutip.Qobj(diags(eigenvalues))
        self.charge_matrix = qutip.Qobj(csr_matrix(charge_matrix))

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


class Resonator(Device):
    def __init__(self, frequency, levels=10, kappa=None):
        """
        A container for resonator variables. This is defined by the paramters:

        Parameters
        ----
        frequency: list: The frequency of the resonator given in GHz
        levels: int: The size of the hilbert space for the resonator
        """
        self.parameters = {"frequency": frequency, "kappa": kappa}

        self.update_methods = [self.set_operators, self.set_dissipators]

        self.levels = levels

        self.a = qutip.destroy(levels)
        self.a_dag = self.a.dag()

        super().__init__()

    def set_operators(self):
        # Set the hamiltonian
        frequency = self.parameters["frequency"]
        self.hamiltonian = 2 * np.pi * frequency * (self.a_dag * self.a + 1 / 2)

        # Coupling to the drive
        self.coupling_operator = self.a + self.a_dag

    def set_dissipators(self):
        kappa = self.parameters["kappa"]

        if kappa is not None:
            self.dissipators = [np.sqrt(kappa) * self.a]
        else:
            self.dissipators = []

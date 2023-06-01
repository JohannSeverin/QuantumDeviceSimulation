from abc import ABC, abstractmethod
from typing import Union

import qutip
import numpy as np

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh


################## Abstract Device Class ##################
class Device(ABC):
    """
    Parent class for all devices. This is intended to take of common operations like sweeping parameters for an individual class and propagating them to the system.
    """

    sweepable_parameters: list[str]
    update_methods: list[callable]

    def __init__(self) -> None:
        """
        Initialize the device.
        """
        self.parameters_to_be_swept()

        for param in self.sweep_parameters.keys():
            setattr(self, param, self.sweep_parameters[param][0])

        self.update()

    def set_operators(self) -> None:
        """
        The class should be able to set the operators used for dynamics from the parameters saved.
        """
        pass

    def set_parameter(self, key: str, value: any) -> None:
        """
        Set a parameter of the device.

        Parameters
        ----
        key: str: The key of the parameter
        value: float: The value of the parameter
        """
        if key in self.sweepable_parameters:
            setattr(self, key, value)
        else:
            raise ValueError(f"{key} is not a sweepable parameter.")

    def get_parameter(self, key: str) -> any:
        """
        Get a parameter of the device.

        Parameters
        ----
        key: str: The key of the parameter
        """
        return getattr(self, key)

    def parameters_to_be_swept(self) -> None:
        """
        Returns a list of the parameters that should be swept.
        """
        # Find the parameters which have a list
        to_sweep = [
            key
            for key in self.sweepable_parameters
            if isinstance(self.get_parameter(key), np.ndarray)
        ]

        self.should_be_swept = len(to_sweep) > 0

        self.sweep_parameters = {key: self.get_parameter(key) for key in to_sweep}

    def update(self, new_parameters: dict = {}) -> None:
        """
        Update the parameters of the device.
        """
        for key in new_parameters.keys():
            self.set_parameter(key, new_parameters[key])

        for update_func in self.update_methods:
            update_func()


################## Qubits ##################
class Transmon(Device):
    def __init__(
        self,
        EC: float,
        EJ: float,
        n_cutoff: int = 20,
        ng: float = 0.0,
        levels: int = 3,
        T1: float = 0.0,
    ) -> None:
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

        # Load Parameters
        self.EC = EC
        self.EJ = EJ
        self.n_cutoff = n_cutoff
        self.ng = ng
        self.levels = levels
        self.T1 = T1

        # Define sweepable parameters
        self.sweepable_parameters = ["EC", "EJ", "ng", "T1"]

        # Define the update methods
        self.update_methods = [self.set_operators, self.set_dissipators]

        super().__init__()

    def set_operators(self) -> None:
        """
        Set the operators used for dynamics. Everything changeable will be in the self.parameters dictionary.
        """
        # Unpack parameters
        EC = self.EC
        EJ = self.EJ
        ng = self.ng

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

    def set_dissipators(self) -> None:
        """
        Set the dissipators used for dynamics. Everything changeable will be in the self.parameters dictionary.
        """
        # Unpack parameters
        T1 = self.T1

        # Set the dissipator
        if T1 > 0:
            self.dissipators = [np.sqrt(1 / T1) * qutip.destroy(self.levels)]
        else:
            self.dissipators = []


################## Resonator ##################
class Resonator(Device):
    def __init__(self, frequency: float, levels=10, kappa: float = 0) -> None:
        """
        A container for resonator variables. This is defined by the paramters:

        Parameters
        ----
        frequency: list: The frequency of the resonator given in GHz
        levels: int: The size of the hilbert space for the resonator
        """
        # Load params
        self.frequency = frequency
        self.kappa = kappa
        self.levels = levels

        # Lists to parent class
        self.sweepable_parameters = ["frequency", "kappa"]
        self.update_methods = [self.set_operators, self.set_dissipators]

        # Useful operators
        self.a = qutip.destroy(levels)
        self.a_dag = self.a.dag()

        # Call parent class
        super().__init__()

    def set_operators(self) -> None:
        # Set the hamiltonian
        frequency = self.frequency
        self.hamiltonian = 2 * np.pi * frequency * (self.a_dag * self.a + 1 / 2)

        # Coupling to the drive
        self.coupling_operator = self.a + self.a_dag

    def set_dissipators(self) -> None:
        kappa = self.kappa

        if kappa > 0:
            self.dissipators = [np.sqrt(kappa) * self.a]
        else:
            self.dissipators = []


################## Test ##################
if __name__ == "__main__":
    qubit = Transmon(EC=7.5 / 25, EJ=7.5, T1=0, levels=3)
    resonator = Resonator(frequency=np.linspace(5.0, 7.0, 21), levels=10, kappa=0.1)

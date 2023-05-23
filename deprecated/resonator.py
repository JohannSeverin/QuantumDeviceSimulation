import qutip
import numpy as np

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh

from devices.device import Device


class Resonator(Device):

    def __init__(self, frequency, levels = 10, kappa = None):
        """
        A container for resonator variables. This is defined by the paramters:

        Parameters 
        ---- 
        frequency: list: The frequency of the resonator given in GHz
        levels: int: The size of the hilbert space for the resonator
        """
        self.parameters = {
            "frequency" : frequency,
            "kappa"     : kappa
        }

        self.update_methods = [self.set_operators, self.set_dissipators]

        self.levels     = levels
   
        self.a      = qutip.destroy(levels)
        self.a_dag  = self.a.dag()

        super().__init__()

   
    def set_operators(self):
        # Set the hamiltonian
        frequency = self.parameters["frequency"]
        self.hamiltonian = 2 * np.pi * frequency * (self.a_dag * self.a + 1/2)

        # Coupling to the drive
        self.coupling_operator = self.a + self.a_dag
        

    def set_dissipators(self):
        kappa = self.parameters["kappa"]
        
        if kappa is not None:
            self.dissipators = [np.sqrt(kappa) * self.a]
        else:
            self.dissipators = []





if __name__ == '__main__':

    frequency = 5.0 + np.linspace(0, 1, 11)
    levels    = 10

    res = Resonator(frequency, levels = levels)

import qutip
import numpy as np

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh



class Resonator:

    def __init__(self, frequency, levels = 10, kappa = None):
        """
        A container for resonator variables. This is defined by the paramters:

        Parameters 
        ---- 
        frequency: list: The frequency of the resonator given in GHz
        levels: int: The size of the hilbert space for the resonator
        """

        self.frequency  = frequency
        self.levels     = levels
        self.kappa      = kappa

        self.a      = qutip.destroy(levels)
        self.a_dag  = self.a.dag()

        self.hamiltonian = 2 * np.pi * frequency * (self.a_dag * self.a + 1/2)

    
    def create_pulse(self, pulse):
        """
        Creates a pulse for the resonator. This is defined by the paramters:

        Parameters 
        ---- 
        pulse: list: The pulse to be created
        """

        self.pulse = pulse




if __name__ == '__main__':

    frequency = 5.0
    levels    = 10

    res = Resonator(frequency, levels = levels)

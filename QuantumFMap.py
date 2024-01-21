import numpy as np

class QuantumFMap:
    """
    Attributes:
        cutoff_dim (int): The cutoff dimension for sf simulation.
        modes (int): The number of modes in the photonic circuit.
        vals (list): A list of values used to QFM.
        r (float): Squeezing parameter for the squeezed states.
        Squeezed_QFM (numpy.ndarray): QFM using squeezed states.

    Methods:
        __init__(self, cutoff_dim, modes, vals, r):
            Initializes a QuantumFMap instance with the specified parameters.

        Squeezed_QFM(self):
            Calculates the QFM with squeezed states based on the given parameters.
    """
    def __init__(self, cutoff_dim, modes, vals,r):
        self.cutoff_dim = cutoff_dim
        self.modes = modes
        self.vals = vals
        self.r=r
        self.Squeezed_QFM = self.Squeezed_QFM()

    def Squeezed_QFM(self):
        """
        This function converts a list with data into QFM.
         The cutoff dimension depends on the cutoff_dim of the sf simulation
          and the number of modes used to make the VQFM
          
          Returns:
            numpy.ndarray: Quantum Frequency Map with squeezed states.
        """
        QFM1 = []
        for val in self.vals:
            target_state = []
            for n in range(self.cutoff_dim ** self.modes):
                coeff = (
                    (1 / np.sqrt(np.cosh(self.r)))
                    * (np.sqrt(np.math.factorial(2 * n)))
                    / ((2 ** n) * np.math.factorial(n))
                    * (np.tanh(self.r) * np.exp((-1) ** (1 / 2) * (val*np.pi + np.pi))) ** n
                )
                target_state.append(coeff)
            QFM1.append(target_state)

        return np.array(QFM1)


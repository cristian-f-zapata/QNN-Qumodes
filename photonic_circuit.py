
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
import tensorflow as tf
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")
physical_devices = tf.config.experimental.list_physical_devices('CPU')
tf.config.experimental.set_visible_devices(physical_devices[0])



from init_weights import WeightInitializer


class photonic_circuit():
    def __init__(self, cutoff_dim, modes, r, layers, weights_shape):
        self.cutoff_dim = cutoff_dim
        self.modes = modes
        self.r = r
        self.layers = layers
        self.weights_shape = weights_shape

    def input_qnn_layer(self, hid, q):
        """
        Applies an input layer to a quantum neural network.

        This function introduces data to the QNN by applying a sequence
        of quantum gates to the input qumode.

        Args:
            hid (list): List of parameters for the quantum gates in the input layer.

        Returns:
            None
        """
        with tf.name_scope('inputlayer'):
            ops.Rgate(hid[0]) | q[0]
            ops.Rgate(hid[0]) | q[0]
            ops.Rgate(hid[0]) | q[0]

            ops.Rgate(hid[0]) | q[1]
            ops.Rgate(hid[0]) | q[1]
            ops.Rgate(hid[0]) | q[1]

    def qnn_layer(self, params, layer_number, q):
        with tf.name_scope('layer_{}'.format(layer_number)):
            N = len(q)
            M = int(self.modes * (self.modes - 1))

            int1 = params[:M]
            r1 = params[M:M + N]
            sm = params[M + N:M + 2 * N]
            sp = params[M + 2 * N:M + 3 * N]
            int2 = params[M + 3 * N:2 * M + 3 * N]
            r2 = params[2 * M + 3 * N:2 * M + 4 * N]
            r3 = params[2 * M + 4 * N:2 * M + 5 * N]
            r4 = params[2 * M + 5 * N:2 * M + 6 * N]

            theta1 = int1[:len(int1) // 2]
            phi1 = int1[len(int1) // 2:]

            theta2 = int2[:len(int2) // 2]
            phi2 = int2[len(int2) // 2:]

            for k, (q1, q2) in enumerate(combinations(q, 2)):
                ops.BSgate(theta1[k], phi1[k]) | (q1, q2)
            for i in range(N):
                ops.Rgate(r1[i]) | q[i]
            for i in range(N):
                ops.Sgate(sm[i], sp[i]) | q[i]
            for k, (q1, q2) in enumerate(combinations(q, 2)):
                ops.BSgate(theta2[k], phi2[k]) | (q1, q2)
            for i in range(N):
                ops.Rgate(r2[i]) | q[i]
            for i in range(N):
                ops.Rgate(r3[i]) | q[i]
            for i in range(N):
                ops.Rgate(r4[i]) | q[i]


    def build_circuit(self):
        global eng, qnn, sf_params, hid_params, layers

        num_params = np.prod(weights_shape)
        hidden_units = 3 * self.modes

        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        qnn = sf.Program(self.modes)

        sf_params = np.arange(num_params).reshape(weights_shape).astype(str)
        sf_params = np.array([qnn.params(*i) for i in sf_params])

        hid_params = np.arange(num_params, num_params + hidden_units).reshape(hidden_units, 1).astype(str)
        hid_params = np.array([qnn.params(*i) for i in hid_params])

        with qnn.context as q:
            self.input_qnn_layer(hid_params, q)
            for k in range(self.layers):
                self.qnn_layer(sf_params[k], k, q)

    def circuit(self):
        # Call build_circuit to initialize the circuit
        self.build_circuit()
        return qnn  # Return the quantum circuit
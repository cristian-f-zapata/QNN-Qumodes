import tensorflow as tf
class QuantumEncoder:
    def __init__(self, sf_params, hid_params, weights, eng, qnn):
        self.sf_params = sf_params
        self.hid_params = hid_params
        self.weights = weights
        self.eng = eng
        self.qnn = qnn

    def state_VQFM(self, x):
        """make vqfm of a data, putting it as a parameter in the encoding layer
        
        Generates a Quantum State using a Variational QFM (VQFM) encoding layers.

        This function takes input data 'x' and evaluate the quantum state by mapping it through
        a Variational circuit.

        Args:
            x (float): Input data to be encoded in the quantum state.

        Returns:
            tf.Tensor: Quantum state represented as a complex-valued tensor.
        """
        mapping_wt = {p.name: w for p, w in zip(self.sf_params.flatten(), tf.reshape(self.weights, [-1]))}
        mapping_hid = {p.name: w for p, w in zip(self.hid_params, tf.fill((6,), x))}
        mapping_wt.update(mapping_hid)

        results = self.eng.run(self.qnn, args=mapping_wt)
        ket = results.state.ket()
        ket = tf.reshape(ket, [-1])
        ket = tf.cast(ket, tf.complex128)
        if self.eng.run_progs:
            self.eng.reset()
        return ket

    def VQFM(self, vector_x):
        """Vectorized Quantum Feature Map (VQFM).
        
        This function applies a Variational Quantum Feature Map (VQFM) to a vector of input data. 
        The VQFM is implemented using a quantum circuit, and each element of the input vector is 
        individually mapped to a quantum state.

        Args:
            vector_x (tf.Tensor): Input vector of data to be encoded in quantum states.

        Returns:
            tf.Tensor: Quantum states represented as complex-valued tensors.

        See Also:
            state_VQFM: The non-vectorized version of this function that applies the VQFM to a single data point.

        Example:
            >>> result_states = VQFM(tf.constant([0.1, 0.2, 0.3]))
            >>> print(result_states)
        """
        def state_VQFM_(x):
            return self.state_VQFM(x)

        return tf.map_fn(lambda x: state_VQFM_(x), vector_x, dtype=tf.complex128)

import tensorflow as tf

class WeightInitializer:
    def __init__(self, modes, layers, active_sd=0.1, passive_sd=1):
        self.modes = modes
        self.layers = layers
        self.active_sd = active_sd
        self.passive_sd = passive_sd
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes weights for QNN.

        This function generates random weights for the parameters of a quantum neural network
        with specified modes and layers.

        Returns:
            tf.Variable: TensorFlow Variable containing the initialized weights.
        """
        # Number of interferometer parameters:
        M = int(self.modes * (self.modes - 1)) 
        # TensorFlow variables
        int1_weights = tf.random.normal(shape=[self.layers, M], stddev=self.passive_sd)
        r1_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=self.passive_sd)
        int2_weights = tf.random.normal(shape=[self.layers, M], stddev=self.passive_sd)
        s_mag_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=self.active_sd)
        s_phase_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=self.passive_sd)
        r2_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=self.passive_sd)
        r3_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=self.passive_sd)
        r4_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=self.passive_sd)
        k_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=self.active_sd)

        weights = tf.concat([int1_weights, r1_weights, s_mag_weights, s_phase_weights, int2_weights, 
                            r2_weights, r3_weights, r4_weights, k_weights], axis=1)
        weights = tf.Variable(weights)
        
        return weights



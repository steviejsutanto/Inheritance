import time

import numpy as np
import tensorflow as tf

import hparams as HP
from utils import py_gen_decay_indices, tf_gen_decay_indices


class MemoryLayer(tf.keras.Model):
    def __init__(self, hidden_units):
        super(MemoryLayer, self).__init__()
        self.hidden_units = hidden_units

        # Initialize Layers
        self.gru = tf.keras.layers.GRU(
            self.hidden_units, return_sequences=False, return_state=False
        )

        self.fc_o = tf.keras.layers.Dense(self.hidden_units // 2, activation="relu")

    def call(self, x):
        x = self.gru(x)
        x = self.fc_o(x)
        return x


class MotoricLayer(tf.keras.Model):
    def __init__(self, bits, hidden_units, unroll=False):
        super(MotoricLayer, self).__init__()
        quantize_dim = 2 ** bits
        self.hidden_units = hidden_units
        self.unroll = unroll

        # Initialize layers:
        self.gru1 = tf.keras.layers.GRU(
            self.hidden_units,
            return_sequences=True,
            return_state=True,
            unroll=self.unroll,
        )

        self.gru2 = tf.keras.layers.GRU(
            self.hidden_units,
            return_sequences=True,
            return_state=True,
            unroll=self.unroll,
        )

        self.fc_o1 = tf.keras.layers.Dense(self.hidden_units, activation="relu")
        self.fc_o2 = tf.keras.layers.Dense(quantize_dim)

    def call(self, c, x, h1, h2):
        seq_len = tf.shape(x)[1]

        x, h1 = self.gru1(x, h1)

        c = tf.tile(tf.expand_dims(c, 1), [1, seq_len, 1])
        xc = tf.concat([c, x], axis=-1)

        xc, h2 = self.gru2(xc, h2)

        x_out = self.fc_o1(x + xc)
        x_out = self.fc_o2(x_out)

        return x_out, h1, h2


class InheritanceModel(tf.keras.Model):
    def __init__(self, bits, mem_len, unroll=False):
        super(InheritanceModel, self).__init__()
        self.hidden_units = HP.hidden_units
        self.bits = bits
        self.mem_len = mem_len
        self.unroll = unroll

        self.memory_layer = MemoryLayer(self.hidden_units)
        self.motoric_layer = MotoricLayer(self.bits, self.hidden_units, self.unroll)

    def call(self, mem, mot, h1, h2):
        c = self.memory_layer(mem)
        x, h1, h2 = self.motoric_layer(c, mot, h1, h2)

        return x, h1, h2

    def generate(self, n_gen_samples, temperature):
        # Initialize inputs
        h1 = tf.zeros([1, self.hidden_units])
        h2 = tf.zeros([1, self.hidden_units])

        previous = tf.random.uniform(
            minval=125, maxval=129, dtype=tf.int32, shape=[1, 1, 1]
        )
        memory = tf.random.uniform(
            minval=125, maxval=129, dtype=tf.int32, shape=[1, self.mem_len, 1]
        )

        init_output = tf.concat([memory, previous], axis=1)
        init_output = init_output.numpy()

        output = np.empty([1, n_gen_samples, 1], dtype=np.int32)
        output = np.concatenate([init_output, output], axis=1)

        time_start = time.time()
        # Start generating
        for step in range(n_gen_samples):
            if step % self.mem_len == 0:
                c_indices = py_gen_decay_indices(step + self.mem_len, self.mem_len)
                memory_buffer = output[:, c_indices, :]
                memory_buffer = tf.cast(memory_buffer, tf.float32)
                memory_buffer = memory_buffer / ((2 ** self.bits - 1) * 0.5) - 1
                c = self.memory_layer(memory_buffer)
                print(
                    "Generated {} / {} samples at {:.4f} samples/sec".format(
                        step, n_gen_samples, self.mem_len / (time.time() - time_start)
                    )
                )
                time_start = time.time()

            pred, h1, h2 = self.generate_step(
                c, previous, h1, h2, self.bits, temperature
            )
            previous = tf.expand_dims(pred, 0)
            output[:, step + self.mem_len + 1, :] = np.expand_dims(pred, axis=1)

        output = np.squeeze(output, 0)

        return output

    @tf.function
    def generate_step(self, c, previous, h1, h2, bits, temperature):
        previous = tf.cast(previous, tf.float32)
        previous = previous / ((2 ** bits - 1) * 0.5) - 1

        logits, h1, h2 = self.motoric_layer(c, previous, h1, h2)

        pred = tf.random.categorical(
            tf.squeeze(logits, 0) / temperature, num_samples=1, dtype=tf.int32
        )

        return pred, h1, h2

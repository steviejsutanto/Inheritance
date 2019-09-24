import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import hparams as HP


def load_audio(file_path, tgt_sr, mono=True):
    x, src_sr = sf.read(file_path, always_2d=True)  # shape: [len, n_channels]
    x = resampy.resample(x, src_sr, tgt_sr, axis=0)
    if mono:
        x = np.mean(x, axis=-1, dtype=np.float32, keepdims=True)
    return x


def encode_mulaw(x, bits):
    # Non-linear quantization with mu-law
    mu = float(2 ** bits - 1)
    x = x * ((1 - 1 / 2 ** bits) / np.max(np.abs(x)))  # normalizes the audio
    x = np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))
    x = (x + 1) * mu // 2
    return x


def decode_mulaw(x, bits):
    # Inverse mu-law
    mu = 2 ** bits - 1
    x = x / (mu * 0.5) - 1
    x = np.sign(x) * (((mu + 1) ** np.abs(x) - 1) / mu)
    return x


def encode_16bit(x):
    # Input range is -1 to 1, output range is -32768 to 32767
    x = np.round(x * ((1 - 1 / 2 ** 16) * (2 ** 15))).astype(np.int16)
    return x


def gen_decay_indices(limit, n):
    # generates indices for STM input that mimic a decay-ing behaiviour of memory
    limit = limit.numpy()
    n = n.numpy()
    result = [1]
    if n > 1:
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            result.append(next_value)
        else:
            result.append(result[-1] + 1)
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))

    indices = list(map(lambda x: -round(x) + limit, result[::-1]))
    indices = np.asarray(np.stack(indices), dtype=np.int32)
    return indices


def py_gen_decay_indices(limit, n):
    # generates indices for STM input that mimic a decay-ing behaiviour of memory
    result = [1]
    if n > 1:
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            result.append(next_value)
        else:
            result.append(result[-1] + 1)
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))

    indices = list(map(lambda x: -round(x) + limit, result[::-1]))
    return indices


def tf_gen_decay_indices(limit, n):
    return tf.py_function(gen_decay_indices, [limit, n], tf.int32)


if __name__ == "__main__":
    # Check preprocessing
    x = load_audio(HP.train_path, HP.sr)
    x = encode_mulaw(x, HP.bits)
    x = decode_mulaw(x, HP.bits)
    x = encode_16bit(x)

    sf.write("outputs/final/preprocess_check.wav", x, HP.sr)

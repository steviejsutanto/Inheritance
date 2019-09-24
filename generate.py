import datetime
import os
import time

import soundfile as sf
import tensorflow as tf

import hparams as HP
from model import InheritanceModel
from utils import decode_mulaw, encode_16bit

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == "__main__":
    # Build model
    model = InheritanceModel(HP.bits, HP.mem_len, unroll=True)
    global_step = tf.Variable(0)

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(model=model, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(ckpt, HP.ckpt_path, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Latest checkpoint restored!!")

    # Write output to WAV
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = "outputs/final/{}Hz_{}bit_{}_{}s_{}.wav".format(
        HP.sr, HP.bits, global_step.numpy(), (HP.n_gen_samples // HP.sr), current_time
    )

    print("\nGenerating audio...\n")
    time_start = time.time()
    output = model.generate(HP.n_gen_samples, HP.temperature)
    output = decode_mulaw(output, HP.bits)
    output = encode_16bit(output)
    sf.write(output_path, output, HP.sr)
    print(
        "\nFinished generating audio at {:.2f}samples/sec\n".format(
            HP.n_gen_samples / (time.time() - time_start)
        )
    )

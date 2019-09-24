import os
import random
import time

import soundfile as sf
import tensorflow as tf

import hparams as HP
from model import InheritanceModel
from utils import (
    decode_mulaw,
    encode_16bit,
    encode_mulaw,
    load_audio,
    tf_gen_decay_indices,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# Define methods
def load_and_preprocess(file_path, sr, bits):
    x = load_audio(file_path.numpy(), sr)
    x = encode_mulaw(x, bits)
    return x


def tf_load_and_preprocess(file_path, sr, bits):
    return tf.py_function(load_and_preprocess, [file_path, sr, bits], tf.int32)


def get_random_example(x, tbptt, mem_len):
    rand_idx = tf.random.uniform(
        [], minval=mem_len + 1, maxval=tf.shape(x)[0] - (tbptt + 1) - 1, dtype=tf.int32
    )
    c_indices = tf_gen_decay_indices(rand_idx, mem_len)
    condition = tf.gather(x, c_indices, axis=0)
    motoric_input = x[rand_idx : rand_idx + tbptt + 1]

    return condition, motoric_input


def loss_fn(target, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        target, logits, from_logits=True
    )

    return tf.reduce_mean(loss)


@tf.function
def train_step(mem, mot, h1, h2, model, optimizer, bits, log_loss, log_acc):

    mem_in = tf.cast(mem, tf.float32)
    mem_in = mem_in / ((2 ** bits - 1) * 0.5) - 1

    mot_in = tf.cast(mot[:, :-1, :], tf.float32)
    mot_in = mot_in / ((2 ** bits - 1) * 0.5) - 1

    target = mot[:, 1:, :]

    with tf.GradientTape() as tape:
        logits, h1, h2 = model(mem_in, mot_in, h1, h2)
        loss = loss_fn(target, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    log_loss(loss)
    log_acc(target, logits)

    return loss, gradients, h1, h2


def train(
    model, optimizer, train_dataset, batch_size, ckpt_manager, global_step, sr, bits
):
    # Initialize tensorboard variables
    train_log_dir = "logs/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    log_loss = tf.keras.metrics.Mean()
    log_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    # Initialize hidden states
    h1 = tf.zeros([batch_size, model.hidden_units])
    h2 = tf.zeros([batch_size, model.hidden_units])

    time_start = time.time()

    # Start training
    for mem, mot in train_dataset:
        passHidden = random.choice([True, False])
        if passHidden:
            h1 = tf.zeros([batch_size, model.hidden_units])
            h2 = tf.zeros([batch_size, model.hidden_units])

        running_loss, gradients, h1, h2 = train_step(
            mem, mot, h1, h2, model, optimizer, bits, log_loss, log_acc
        )

        running_loss = running_loss.numpy()
        step = global_step.numpy()

        if step % 5000 == 4999:
            ckpt_manager.save()
            print("\nCheckpoint saved!\n")

        if step % 20000 == 19998:
            print("\nGenerating audio...\n")
            output = model.generate(10 * sr, 1.0)
            output = decode_mulaw(output, bits)
            log_output = tf.cast(tf.expand_dims(output, 0), tf.float32)

            with train_summary_writer.as_default():
                tf.summary.audio(
                    "Audio/{}Hz_{}bit".format(sr, bits), log_output, sr, step, 1, "wav"
                )

            output = encode_16bit(output)
            sf.write(
                "outputs/{}Hz_{}bit_{}step_{}.wav".format(
                    sr, bits, step, int(time.time())
                ),
                output,
                sr,
            )

        if step % 100 == 0:
            grads_norm = tf.linalg.global_norm(gradients)
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", log_loss.result(), step=step)
                tf.summary.scalar("acc", log_acc.result(), step=step)
                tf.summary.scalar("grads_norm", grads_norm, step=step)

                for grad, var in zip(gradients, model.trainable_variables):
                    tf.summary.histogram(
                        "gradients/{}".format(var.name), grad, step=step
                    )
                    tf.summary.histogram("weights/{}".format(var.name), var, step=step)

            log_loss.reset_states()
            log_acc.reset_states()

        global_step.assign_add(1)

        time_end = time.time() - time_start

        print(
            "Step: {} \t || \t Loss: {:.4f} \t || \t Speed: {:.2f}sec/step".format(
                step, running_loss, time_end
            )
        )

        time_start = time.time()


if __name__ == "__main__":
    # Create input pipeline
    train_dataset = (
        tf.data.Dataset.list_files(HP.train_path)
        .map(lambda x: tf_load_and_preprocess(x, HP.sr, HP.bits))
        .cache()
        .map(lambda x: get_random_example(x, HP.tbptt, HP.mem_len))
        .repeat()
        .batch(HP.batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # for mem, mot in train_dataset.take(4):
    #     print(mem[0])
    #     print(mot[0])
    #     plt.subplot(211)
    #     plt.plot(mem[0])
    #     plt.subplot(212)
    #     plt.plot(mot[0])
    #     plt.show()

    # Initialize model
    model = InheritanceModel(HP.bits, HP.mem_len)
    optimizer = tf.keras.optimizers.Adam(HP.lr, clipnorm=HP.clipnorm)
    global_step = tf.Variable(0)

    # Initialize checkpoint
    ckpt = tf.train.Checkpoint(
        model=model, optimizer=optimizer, global_step=global_step
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, HP.ckpt_path, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    train(
        model,
        optimizer,
        train_dataset,
        HP.batch_size,
        ckpt_manager,
        global_step,
        HP.sr,
        HP.bits,
    )

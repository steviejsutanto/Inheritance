# Model parameters
sr = 16000
bits = 8
hidden_units = 512
mem_len = 1024


# Training parameters
tbptt = 960
lr = 1e-4
batch_size = 32
clipnorm = 5.0


# Sampling parameters
n_gen_samples = 240 * sr
temperature = 1.0


# Paths
train_path = "training_data/*"
ckpt_path = "ckpt"

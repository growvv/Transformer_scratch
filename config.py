import torch

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False
save_model = False

# Training hyperparameters
num_epochs = 100
learning_rate = 3e-4
batch_size = 4


# Model hyperparameters
src_vocab_size = 1000
trg_vocab_size = 1000
embedding_size = 256
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_length = 100
forward_expansion = 4
src_pad_idx = 0
trg_pad_idx = 0

# dataset
file_root = "./numbers.csv"
# save model
model_root = "my_checkpoint.pth.tar"

# generate_data
entry_num = 100


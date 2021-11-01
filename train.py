"""
Seq2Seq using Transformers on the Multi30k
dataset. In this video I utilize Pytorch
inbuilt Transformer modules, and have a
separate implementation for Transformers
from scratch. Training this model for a
while (not too long) gives a BLEU score
of ~35, and I think training for longer
would give even better results.
"""

from sys import float_repr_style
from numpy import True_
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformer import Transformer
from train_utils import load_checkpoint, save_checkpoint, my_predict
from generate_data import SeqDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import ipdb
import config


# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False
save_model = True

# Training hyperparameters
num_epochs = 10
learning_rate = 3e-4
batch_size = 4

# Model hyperparameters
src_vocab_size = 100
trg_vocab_size = 100
embedding_size = 256
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = 0
trg_pad_idx = 0

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot2")
step = 0

model = Transformer(
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    embedding_size,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    num_heads,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


dataset = SeqDataset("./numbers.csv", max_length=config.max_length)
train_iterator = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,  collate_fn=None)

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    # 评估
    model.eval()
    translated_sentence = my_predict(
        model, device, max_len
    )
    print(f"Translated example sentence: \n {translated_sentence}")

    # 训练
    # ipdb.set_trace()
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        src, trg = batch
        src = src.to(device)
        trg = trg.to(device)
        # ipdb.set_trace()

        # Forward prop
        output = model(src, trg)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it

        # output: [batch_size, len, trg_vocab_size]
        # trg: [batch_size, len]  one-hot编码
        _batch_size, _len, _trg_vocab_size = output.shape
        # ipdb.set_trace()
        assert _batch_size == batch_size
        assert _trg_vocab_size == trg_vocab_size

        optimizer.zero_grad()

        # ipdb.set_trace()
        output = output.reshape(-1, trg_vocab_size)
        trg = trg.reshape(-1)
        loss = criterion(output, trg)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        if batch_idx % 100 == 0:
            print(epoch, batch_idx, loss.item())
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    # ipdb.set_trace()
    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# running on entire test data takes a while
# print(f"Bleu score {score * 100:.2f}")
print("finish!!")

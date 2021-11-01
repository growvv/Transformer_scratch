import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformer import Transformer
from train_utils import load_checkpoint, save_checkpoint, my_predict, draw_atten
from seq_dataset import SeqDataset
from torch.utils.data import DataLoader
import config
import ipdb
import time


model = Transformer(
    config.src_vocab_size,
    config.trg_vocab_size,
    config.src_pad_idx,
    config.trg_pad_idx,
    config.embedding_size,
    config.num_encoder_layers,
    config.num_decoder_layers,
    config.forward_expansion,
    config.num_heads,
    config.dropout,
    config.max_length,
    config.device,
).to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if config.load_model:
    load_checkpoint(torch.load(config.model_root), model, optimizer)


dataset = SeqDataset(config.file_root, max_length=config.max_length)
train_iterator = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=0,  collate_fn=None)

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
global_step = 0
log_step = 1
tic_train = time.time()

for epoch in range(config.num_epochs):
    print(f"[Epoch {epoch} / {config.num_epochs}]")

    # 训练
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        if batch_idx == 3:
            break
        src, trg = batch
        src = src.to(config.device)
        trg = trg.to(config.device)
        # target = trg

        # Forward prop
        output = model(src, trg)
        # ipdb.set_trace()
        # best_guess = output.argmax(2)
        # print(best_guess.shape)

        optimizer.zero_grad()

        output = output.reshape(-1, config.trg_vocab_size)
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

        global_step += 1
        if global_step % log_step == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, batch_idx, loss, global_step / (time.time() - tic_train), 
                ))
        writer.add_scalar("Training loss", loss, global_step=global_step)
        # writer.add_graph(model, [src, target])
        # writer.add_histogram("weight", model.decoder.layers[2].attn.atten ,step)

    # ipdb.set_trace()
    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

    if config.save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=config.model_root)

    # 评估
    model.eval()
    translated_sentence = my_predict(
        model, config.device, config.max_length
    )
    # print(f"Translated example sentence: \n {translated_sentence}")

    # 可用于attention可视化
    # print(model.encoder.layers[2].attn.atten.shape)
    # draw_atten(model.encoder.layers[2].attn.atten)

print("finish!!")
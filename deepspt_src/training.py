from torch import Tensor
import torch
from torch import nn
from torch.utils.data import Dataset


def train_epoch(model, optimizer, train_loader, device):
    train_loss = 0
    train_acc = 0
    train_count = 0
    for batch_idx, xb in enumerate(train_loader):
        x, y = xb
        batch_size = x.size(0)
        loss, acc = train_batch(model, optimizer, xb)

        train_loss += loss * batch_size
        train_acc += acc * batch_size
        train_count += batch_size

    average_loss = train_loss / train_count
    average_acc = train_acc / train_count

    return average_loss, average_acc.item()


def train_batch(model, optimizer, xb):
    model.train()
    optimizer.zero_grad()

    loss, acc, _ = model(xb)
    loss.backward()
    optimizer.step()

    return loss.item(), acc


def validate(model, optimizer, validation_loader, device):
    model.eval()

    val_loss = 0
    val_acc = 0
    val_count = 0
    with torch.no_grad():
        for batch_idx, xb in enumerate(validation_loader):
            x, y = xb
            batch_size = x.size(0)
            loss, acc, _ = model(xb)

            val_loss += loss.item() * batch_size
            val_acc += acc * batch_size
            val_count += batch_size

        average_loss = val_loss / val_count
        average_acc = val_acc / val_count

    return average_loss, average_acc.item()

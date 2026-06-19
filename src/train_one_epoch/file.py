import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def train_one_epoch(
    model,
    dataloader,
    device,
    optimizer,
    scheduler: LambdaLR,
):
    model.train()
    
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device) 

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
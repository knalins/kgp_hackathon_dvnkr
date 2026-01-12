# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import train_classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

book_file_path = os.path.join(os.path.dirname(__file__), "Books")

def get_batch(input_file_path):
    # treat the file as bytes
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    
    # ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    max_start = len(data) - BLOCK_SIZE - 1
    num_possible_chunks = max_start // 256
    
    # Select random starting points aligned to stride
    chunk_indices = torch.randint(0, max(1, num_possible_chunks), (BATCH_SIZE,))
    ix = chunk_indices * 256
    ix = torch.clamp(ix, 0, max_start)
    
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def train_novel():
    books_list = os.listdir(book_file_path)
    for book in books_list:
        print(f"Training on book: {book}")
        novel_file_path = os.path.join(book_file_path, book)
        
        model = bdh.BDH(BDH_CONFIG).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        x, y = get_batch(novel_file_path)
        loss_acc = 0
        loss_steps = 0
        for step in range(MAX_ITERS):
            with ctx:
                logits, loss = model(x, y)
            loss_acc += loss
            loss_steps += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if step % LOG_FREQ == 0:
                print(f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
                loss_acc = 0
                loss_steps = 0

        torch.save(model.state_dict(), f'{book[:-4].lower()}.pt')
        print("Training done... ")
        model.eval()
        
if __name__ == "__main__":
    train_novel()
    train_classifier.train_classifier_kfold()     # training
    # train_classifier.evaluate_classifier()     # evaluation
    train_classifier.predict()      # prediction
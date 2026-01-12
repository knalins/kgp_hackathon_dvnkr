import os
from contextlib import nullcontext
from pydoc import text

import bdh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
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
print(f"Using device: {device} with dtype {dtype}")


BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512  # Max sequence length per chunk
CHUNK_STRIDE = 256  # Overlap between chunks (sliding window)
BATCH_SIZE = 8
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LOG_FREQ = 10

class BackstoryClassifier(nn.Module):
    def __init__(self, config: bdh.BDHConfig, num_classes: int = 2):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        self.backbone = bdh.BDH(config)
        
        # Classification head
        # Pool embeddings -> project to classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, num_classes)
        )
        
    def get_embeddings(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from backbone (before LM head).
        Returns: (batch, seq_len, n_embd) tensor
        """
        C = self.config
        B, T = idx.size()
        
        x = self.backbone.embed(idx).unsqueeze(1)
        x = self.backbone.ln(x)
        
        # Pass through transformer layers
        for _ in range(C.n_layer):
            D = C.n_embd
            nh = C.n_head
            N = D * C.mlp_internal_dim_multiplier // nh
            
            x_latent = x @ self.backbone.encoder
            x_sparse = F.relu(x_latent)
            
            yKV = self.backbone.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.backbone.ln(yKV)
            
            y_latent = yKV @ self.backbone.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            
            xy_sparse = self.backbone.drop(xy_sparse)
            
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.backbone.decoder
            )
            y = self.backbone.ln(yMLP)
            x = self.backbone.ln(x + y)
        
        # x shape: (B, 1, T, D) -> (B, T, D)
        return x.squeeze(1)
    
    def forward(self, idx: torch.Tensor, chunk_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with chunking support.
        
        Args:
            idx: (batch, seq_len) token indices
            chunk_mask: (batch, seq_len) mask for valid tokens (1=valid, 0=padding)
        
        Returns:
            logits: (batch, num_classes) classification logits
        """
        # Get embeddings from backbone
        embeddings = self.get_embeddings(idx)  # (B, T, D)
        
        # Mean pooling over sequence (with mask if provided)
        if chunk_mask is not None:
            # Mask out padding tokens
            mask_expanded = chunk_mask.unsqueeze(-1).float()  # (B, T, 1)
            embeddings = embeddings * mask_expanded
            pooled = embeddings.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling
            pooled = embeddings.mean(dim=1)  # (B, D)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


class BackstoryDataset(Dataset):
    LABEL_MAP = {"consistent": 0, "contradict": 1}
    
    def __init__(self, csv_name: str, block_size: int = BLOCK_SIZE, stride: int = CHUNK_STRIDE):
        self.block_size = block_size
        self.stride = stride
        self.samples = {}
        
        df = pd.read_csv(csv_name)
        df["caption"] = df["caption"].fillna("[NO_CAPTION]")
        self.length = len(df)
        for row in df.itertuples(index=False):
            if self.samples.get(row.book_name) is None:
                self.samples[row.book_name] = []
                
            self.samples[row.book_name].append({
                    'label': self.LABEL_MAP[row.label],
                    'content': row.content,
                    'char': row.char,
                    'caption': row.caption
                })
    
    def text_to_tokens(self, text: str) -> torch.Tensor:
        tokens = list(text.encode('utf-8'))
        return torch.tensor(tokens, dtype=torch.long)
    
    def chunk_tokens(self, tokens: torch.Tensor) -> tuple:
        seq_len = len(tokens)
        if seq_len <= self.block_size:
            # Pad short sequences
            padded = torch.zeros(self.block_size, dtype=torch.long)
            padded[:seq_len] = tokens
            mask = torch.zeros(self.block_size, dtype=torch.long)
            mask[:seq_len] = 1
            return padded.unsqueeze(0), mask.unsqueeze(0)
        
        # Create overlapping chunks using sliding window
        chunks = []
        masks = []
        
        for start in range(0, seq_len - self.block_size + 1, self.stride):
            chunk = tokens[start:start + self.block_size]
            chunks.append(chunk)
            masks.append(torch.ones(self.block_size, dtype=torch.long))
        
        # Handle remaining tokens if any
        if (seq_len - self.block_size) % self.stride != 0:
            last_chunk = tokens[-self.block_size:]
            chunks.append(last_chunk)
            masks.append(torch.ones(self.block_size, dtype=torch.long))
        
        return torch.stack(chunks), torch.stack(masks)
    
    def __len__(self):
        return self.length

def train_classifier_kfold():
    """
    K-Fold Cross-Validation Training.
    Trains on K-1 folds, validates on 1 fold, and reports average accuracy.
    """
    train_dataset = BackstoryDataset("train.csv")
    
    for book_name in train_dataset.samples.keys():
        samples = train_dataset.samples[book_name]
        n_samples = len(samples)
        indices = np.arange(n_samples)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=33)
        fold_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            model = BackstoryClassifier(BDH_CONFIG, num_classes=2).to(device)
            model_path = os.path.join(os.path.dirname(__file__), f"{book_name.lower()}.pt")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
            
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=LEARNING_RATE, 
                weight_decay=WEIGHT_DECAY
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=MAX_EPOCHS * len(train_dataset)
            )
            
            # Loss function
            class_counts = [0, 0]
            for item in samples:
                class_counts[item['label']] += 1
            total_samples = sum(class_counts)
            
            class_weights = torch.tensor(
                [total_samples / (2 * max(c, 1)) for c in class_counts], 
                device=device
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # Training loop
            model.train()
            for epoch in range(MAX_EPOCHS):
                epoch_loss = 0
                correct = 0
                total = 0
                
                for idx in train_idx:
                    item = samples[idx]
                    text = "Caption: " + item['caption'] + " Character: " + item['char'] + " Content: " + item['content']
                    tokens = train_dataset.text_to_tokens(text)
                    chunks, mask = train_dataset.chunk_tokens(tokens)
                    chunks = chunks.to(device)
                    mask = mask.to(device)
                    label = torch.tensor([item['label']], device=device)
                    
                    optimizer.zero_grad()
                    
                    with ctx:
                        all_logits = []
                        for chunk_idx in range(chunks.size(0)):
                            chunk = chunks[chunk_idx:chunk_idx+1]
                            chunk_mask = mask[chunk_idx:chunk_idx+1]
                            logits = model(chunk, chunk_mask)
                            all_logits.append(logits)
                        
                        final_logits = torch.stack(all_logits).mean(dim=0)
                        loss = criterion(final_logits, label)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    pred = final_logits.argmax(dim=1)
                    correct += (pred == label).sum().item()
                    total += 1
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{MAX_EPOCHS} | Loss: {epoch_loss/total:.4f} | Train Acc: {correct/total*100:.1f}%")
            
            # Validation on held-out fold
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for idx in val_idx:
                    item = samples[idx]
                    text = "Caption: " + item['caption'] + " Character: " + item['char'] + " Content: " + item['content']
                    tokens = train_dataset.text_to_tokens(text)
                    chunks, mask = train_dataset.chunk_tokens(tokens)
                    chunks = chunks.to(device)
                    mask = mask.to(device)
                    label = item['label']
                    
                    all_logits = []
                    for chunk_idx in range(chunks.size(0)):
                        with ctx:
                            logits = model(chunks[chunk_idx:chunk_idx+1], mask[chunk_idx:chunk_idx+1])
                        all_logits.append(logits)
                    
                    final_logits = torch.stack(all_logits).mean(dim=0)
                    pred = final_logits.argmax(dim=1).item()
                    
                    val_correct += (pred == label)
                    val_total += 1
            
            fold_acc = val_correct / val_total * 100
            fold_accuracies.append(fold_acc)
            print(f"  Fold {fold+1} Validation Accuracy: {fold_acc:.1f}%")
        torch.save(model.state_dict(), f'{book_name.lower()}.pt')    
        print(book_name + " " + str(fold_accuracies))

def predict():
    df = pd.read_csv('test.csv')
    df["caption"] = df["caption"].fillna("[NO_CAPTION]")
    df["label"] = ''  # to store predictions
    
    samples = {}
    row_id = 0
    for row in df.itertuples(index=False):
        if samples.get(row.book_name.lower()) is None:
            samples[row.book_name.lower()] = []
                
        samples[row.book_name.lower()].append({
            'content': row.content,
            'char': row.char,
            'caption': row.caption,
            'row_id': row_id,
        })
        row_id += 1
        
    for book_name in samples.keys():
        model = BackstoryClassifier(BDH_CONFIG, num_classes=2).to(device)
        model_path = os.path.join(os.path.dirname(__file__), f"{book_name.lower()}.pt")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        model.eval()
    
        for item in samples[book_name]:
            text = "Caption: " + item['caption'] + " Character: " + item['char'] + " Content: " + item['content']
            tokens = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
            
            if len(tokens) <= BLOCK_SIZE:
                padded = torch.zeros(BLOCK_SIZE, dtype=torch.long)
                padded[:len(tokens)] = tokens
                mask = torch.zeros(BLOCK_SIZE, dtype=torch.long)
                mask[:len(tokens)] = 1
                chunks = padded.unsqueeze(0).to(device)
                masks = mask.unsqueeze(0).to(device)
            else:
                chunk_list = []
                mask_list = []
                for start in range(0, len(tokens) - BLOCK_SIZE + 1, CHUNK_STRIDE):
                    chunk_list.append(tokens[start:start + BLOCK_SIZE])
                    mask_list.append(torch.ones(BLOCK_SIZE, dtype=torch.long))
                chunks = torch.stack(chunk_list).to(device)
                masks = torch.stack(mask_list).to(device)
            
            with torch.no_grad():
                all_logits = []
                for i in range(chunks.size(0)):
                    with ctx:
                        logits = model(chunks[i:i+1], masks[i:i+1])
                    all_logits.append(logits)
                
                final_logits = torch.stack(all_logits).mean(dim=0)
                probs = F.softmax(final_logits, dim=1)
                pred = final_logits.argmax(dim=1).item()
            
            df.loc[item['row_id'], 'label'] = 'consistent' if pred == 0 else 'contradict'
            confidence = probs[0, pred].item()
            print(f"confidence for row id {item['row_id']}: {confidence:.4f}")
            
    df = df[['id', 'label']]
    df.to_csv("submission.csv", index=False)
"""
Train the MUTAG-style GIN on the AMES mutagenicity benchmark (~6,500 molecules).
Upgraded with Learning Rate Scheduler and Best-Model Checkpointing.
"""

import numpy as np
for _name, _val in [("int", int), ("float", float), ("object", object), ("complex", complex)]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

import pandas as pd
import urllib.request
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from molecule import smiles_to_graph, UnsupportedAtomError
from atom_map import NUM_TAGS
from gin.model import GIN

HIDDEN = 64
LAYERS = 5
EPOCHS = 150   # Increased training time
BATCH = 32
LR = 0.005     # Slightly lower initial LR for better convergence

print("Downloading AMES dataset from GitHub mirror...")
url = "https://raw.githubusercontent.com/mathworks/Chemistry-Deep-Learning-GCN-Mutagenicity-Classification/main/AMES-csv-Data/AMES_All_Data.csv"
df = pd.read_csv(url, header=None, names=["SMILES", "label"])
print(f"Downloaded {len(df)} molecules.")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = df.iloc[:int(0.8 * len(df))]
valid_df = df.iloc[int(0.8 * len(df)):int(0.9 * len(df))]
test_df  = df.iloc[int(0.9 * len(df)):]

def build_dataset(data_df):
    graphs, skipped = [], 0
    for smi, label in zip(data_df['SMILES'], data_df['label']):
        try:
            edge_index, tags, _mol = smiles_to_graph(str(smi))
            x = torch.zeros(len(tags), NUM_TAGS)
            x[torch.arange(len(tags)), tags] = 1.0
            graphs.append(Data(x=x, edge_index=edge_index,
                               y=torch.tensor(int(label), dtype=torch.long)))
        except UnsupportedAtomError:
            skipped += 1
        except Exception:
            skipped += 1
    print(f"  kept {len(graphs)} graphs, skipped {skipped}")
    return graphs

print("Building train/valid/test graphs...")
train_dataset = build_dataset(train_df)
valid_dataset = build_dataset(valid_df)
test_dataset  = build_dataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=256)
test_loader  = DataLoader(test_dataset,  batch_size=256)

model = GIN(in_dim=NUM_TAGS, hidden_dim=HIDDEN, num_classes=2, num_layers=LAYERS)
# Added weight_decay to prevent overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
# Automatically drops LR by 50% if validation accuracy stalls for 10 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
criterion = torch.nn.CrossEntropyLoss()

@torch.no_grad()
def accuracy(loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        correct += (out.argmax(dim=1) == batch.y).sum().item()
        total += batch.num_graphs
    return correct / max(total, 1)

print("Starting training...")
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        
    val_acc = accuracy(valid_loader)
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch:03d}  loss {total_loss/len(train_dataset):.4f}  "
          f"val_acc {val_acc:.3f}  lr {current_lr:.5f}")
          
    # Save the model ONLY when it hits a new high score
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"in_dim": NUM_TAGS, "hidden_dim": HIDDEN,
                       "num_classes": 2, "num_layers": LAYERS},
            "tag_list": sorted(range(NUM_TAGS)),
        }, "checkpoint_ames.pt")

print(f"\nBest Validation accuracy: {best_val_acc:.3f}")
print(f"Final Test accuracy (using best model): {accuracy(test_loader):.3f}")
print("Saved best checkpoint to checkpoint_ames.pt")
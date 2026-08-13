"""
Train an ENSEMBLE of GIN models on the AMES benchmark (~6,500 molecules).
Prints progress every 10 epochs so the terminal never looks stuck.

Run:  python train_ames.py
Produces: checkpoint_ames_ensemble.pt  (+ refreshes train_smiles.txt)
"""

import numpy as np
# Safe numpy-1.x alias shim for old libraries (never overwrites existing attrs).
for _name, _val in [("int", int), ("float", float), ("object", object), ("complex", complex)]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

import time
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from molecule import smiles_to_graph, UnsupportedAtomError
from atom_map import NUM_TAGS
from gin.model import GIN

HIDDEN = 64
LAYERS = 5
EPOCHS = 150
BATCH = 32
LR = 0.005
ENSEMBLE_SIZE = 3   # number of models in the ensemble

# ------------------------------------------------------------- dataset
print("Downloading AMES dataset from GitHub mirror...")
url = "https://raw.githubusercontent.com/mathworks/Chemistry-Deep-Learning-GCN-Mutagenicity-Classification/main/AMES-csv-Data/AMES_All_Data.csv"
df = pd.read_csv(url, header=None, names=["SMILES", "label"])
print(f"Downloaded {len(df)} molecules.")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = df.iloc[:int(0.8 * len(df))]
valid_df = df.iloc[int(0.8 * len(df)):int(0.9 * len(df))]
test_df  = df.iloc[int(0.9 * len(df)):]

# Refresh the applicability-domain file so it always matches this training set
train_df["SMILES"].to_csv("train_smiles.txt", index=False, header=False)
print("Saved train_smiles.txt for the applicability-domain checker.")

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

# ------------------------------------------------------------- ensemble
all_state_dicts = []

@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        correct += (out.argmax(dim=1) == batch.y).sum().item()
        total += batch.num_graphs
    return correct / max(total, 1)

for seed in range(ENSEMBLE_SIZE):
    print(f"\n=== Ensemble model {seed+1}/{ENSEMBLE_SIZE} (seed {seed}) ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GIN(in_dim=NUM_TAGS, hidden_dim=HIDDEN, num_classes=2, num_layers=LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    t0 = time.time()

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

        val_acc = accuracy(model, valid_loader)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}/{EPOCHS}  loss {total_loss/len(train_dataset):.4f}  "
                  f"val_acc {val_acc:.3f}  lr {optimizer.param_groups[0]['lr']:.5f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"  Model {seed+1} best val acc: {best_val_acc:.3f} "
          f"(trained in {time.time()-t0:.0f}s)")
    all_state_dicts.append(best_state)

# ------------------------------------------------------------- save + test
torch.save({
    "state_dicts": all_state_dicts,
    "config": {"in_dim": NUM_TAGS, "hidden_dim": HIDDEN,
               "num_classes": 2, "num_layers": LAYERS},
    "tag_list": sorted(range(NUM_TAGS)),
}, "checkpoint_ames_ensemble.pt")

print("\nSaved ensemble to checkpoint_ames_ensemble.pt")
print("Done. Now run: python app.py")
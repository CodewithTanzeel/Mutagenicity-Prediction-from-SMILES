"""
Trains one GIN model on the full MUTAG dataset for DEPLOYMENT (not
evaluation) -- your existing train.py presumably follows the paper's
10-fold protocol to report a comparable accuracy number, which is right
for the paper but means each fold's model only saw 90% of the data. For a
demo that scores arbitrary user molecules, training on all 188 graphs
gives the model the most information; there's no held-out test set here
because this script isn't trying to reproduce a benchmark number.

Not run/verified in the authoring session -- no working torch install
available there. Run and sanity-check locally:
    python train_and_export.py --epochs 200 --hidden_dim 64 --num_layers 5

Saves checkpoint.pt containing:
    state_dict, config (constructor args for gin.model.GIN), tag_list,
    label_list -- everything app.py needs to reload the exact model.
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from gin.data import load_mutag, build_node_features
from gin.batch import collate
from gin.model import GIN
from atom_map import ATOM_MAP, NUM_TAGS


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="dataset/MUTAG/MUTAG.txt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--learn_eps", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoint.pt")
    args = p.parse_args()

    # same seeding + device convention as train.py, for consistency with
    # the CV numbers you already trust
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs, tag_list, label_list = load_mutag(args.data_path)
    in_dim = build_node_features(graphs, tag_list)

    # sanity check against atom_map.py's assumptions -- fail loudly rather
    # than silently training a model whose feature indices don't match
    # what molecule.py will produce for user-submitted SMILES.
    assert tag_list == sorted(ATOM_MAP.keys()), (
        f"tag_list {tag_list} from {args.data_path} doesn't match "
        f"atom_map.ATOM_MAP keys {sorted(ATOM_MAP.keys())} -- re-derive "
        f"the atom mapping before trusting this checkpoint."
    )
    assert in_dim == NUM_TAGS

    num_classes = len(label_list)
    batch = collate(graphs)
    x, ei, bv = batch.x.to(device), batch.edge_index.to(device), batch.batch_vec.to(device)
    labels = batch.labels.to(device)

    model = GIN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learn_eps=args.learn_eps,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits = model(x, ei, bv, batch.num_graphs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                logits = model(x, ei, bv, batch.num_graphs)
                acc = (logits.argmax(dim=1) == labels).float().mean().item()
            model.train()
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  train_acc {acc:.3f}")

    # final full-data accuracy (NOT a generalization estimate -- this model
    # saw every graph during training; for that number, use your existing
    # 10-fold train.py instead)
    model.eval()
    with torch.no_grad():
        logits = model(x, ei, bv, batch.num_graphs)
        final_acc = (logits.argmax(dim=1) == labels).float().mean().item()
    print(f"final train-set accuracy (fit, not generalization): {final_acc:.3f}")
    model.to("cpu")

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "in_dim": in_dim,
            "hidden_dim": args.hidden_dim,
            "num_classes": num_classes,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learn_eps": args.learn_eps,
        },
        "tag_list": tag_list,
        "label_list": label_list,
    }, args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()

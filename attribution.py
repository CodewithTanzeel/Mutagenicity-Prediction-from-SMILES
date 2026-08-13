"""
Atom-level attribution + ensemble prediction.
Runs the molecule through ALL ensemble models, averages logits, and reports
mean probability ± spread (high spread = model is unsure).
"""

import torch
import torch.nn.functional as F
from rdkit import Chem

from atom_map import ATOM_MAP, CLASS_NAMES

_GROUP_PATTERNS = [
    ("nitro group (-NO2)", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("carbonyl (C=O)", "[CX3]=[OX1]"),
    ("aromatic amine (-NH2 on ring)", "[NX3;H2][c]"),
    ("aromatic ring carbon", "[c]"),
    ("halogen substituent", "[F,Cl,Br,I]"),
    ("Michael acceptor (alpha,beta-unsaturated carbonyl)", "[CX3]=[CX3]-[CX3]=[OX1]"),
]


def build_one_hot(tags, num_tags):
    x = torch.zeros(tags.shape[0], num_tags)
    x[torch.arange(tags.shape[0]), tags] = 1.0
    x.requires_grad_(True)
    return x


def predict_and_attribute(models, edge_index, tags, num_tags=7):
    """Ensemble forward pass. `models` is a list of GIN models."""
    if not isinstance(models, list):
        models = [models]
    for m in models:
        m.eval()

    x = build_one_hot(tags, num_tags)
    batch_vec = torch.zeros(tags.shape[0], dtype=torch.long)

    all_logits, all_probs = [], []
    for m in models:
        logit = m(x, edge_index, batch_vec, num_graphs=1)
        all_logits.append(logit)
        all_probs.append(F.softmax(logit, dim=-1).squeeze(0).detach())

    mean_logit = torch.stack(all_logits, dim=0).mean(dim=0)
    probs = F.softmax(mean_logit, dim=-1).squeeze(0)
    probs_std = torch.stack(all_probs, dim=0).std(dim=0, unbiased=False)
    pred_class = int(torch.argmax(probs).item())

    if x.grad is not None:
        x.grad.zero_()
    for m in models:
        m.zero_grad(set_to_none=True)

    # Backprop through the ENSEMBLE MEAN logit (gradients accumulate from all models)
    mean_logit[0, pred_class].backward()

    grad = x.grad
    importance = (grad * x.detach()).sum(dim=1)
    max_abs = importance.abs().max().clamp(min=1e-8)
    importance_norm = 0.5 + 0.5 * (importance / max_abs)

    return {
        "probs": probs.detach(),
        "probs_std": probs_std,
        "pred_class": pred_class,
        "importance": importance.detach(),
        "importance_norm": importance_norm.detach(),
    }


def explain(mol, tags, importance, top_k=3):
    atom_to_groups = {i: [] for i in range(mol.GetNumAtoms())}
    for name, smarts in _GROUP_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            for atom_idx in match:
                atom_to_groups[atom_idx].append(name)

    order = torch.argsort(importance.abs(), descending=True)
    top_idx = [int(i) for i in order[:top_k]]

    parts = []
    for idx in top_idx:
        elem = ATOM_MAP.get(int(tags[idx]), "?")
        groups = atom_to_groups.get(idx, [])
        direction = "pushed toward" if importance[idx] > 0 else "pushed away from"
        group_text = f", part of a {groups[0]}" if groups else ""
        parts.append(f"atom {idx} ({elem}{group_text}) {direction} the prediction")

    return "Most influential: " + "; ".join(parts) + "." if parts else \
        "No single atom stood out strongly for this prediction."


def format_prediction(probs, probs_std, pred_class):
    name = CLASS_NAMES.get(pred_class, str(pred_class))
    conf = float(probs[pred_class])
    std = float(probs_std[pred_class])
    return f"{name} ({conf:.1%} ± {std:.1%} confidence)"
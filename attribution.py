"""
Atom-level attribution and a short plain-English explanation.

Attribution method: Gradient x Input ("Grad" saliency), the standard
baseline used in the MUTAG-explainability literature (GNNExplainer,
SubgraphX and others report it alongside their own methods). Because each
node's input x_v is one-hot, grad(logit) . x_v collapses to the gradient
component at that node's active atom-type index -- i.e. "how much would
the predicted-class logit change if this atom's presence were scaled up".

This is a simple, fast, honest baseline -- NOT a claim of being the most
faithful attribution method available (Integrated Gradients, GNNExplainer,
and SubgraphX all report different, sometimes-better fidelity in the
literature). Good enough for a 24h demo; worth flagging as a limitation
in the writeup.
"""

import torch
import torch.nn.functional as F
from rdkit import Chem

from atom_map import ATOM_MAP, CLASS_NAMES

# SMARTS patterns for a handful of MUTAG-relevant functional groups, used
# only to generate human-readable explanation text -- NOT fed to the model.
_GROUP_PATTERNS = [
    ("nitro group (-NO2)", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("carbonyl (C=O)", "[CX3]=[OX1]"),
    ("aromatic amine (-NH2 on ring)", "[NX3;H2][c]"),
    ("aromatic ring carbon", "[c]"),
    ("halogen substituent", "[F,Cl,Br,I]"),
]


def build_one_hot(tags, num_tags):
    """[N] long tags -> [N, num_tags] float one-hot, gradient-tracked."""
    x = torch.zeros(tags.shape[0], num_tags)
    x[torch.arange(tags.shape[0]), tags] = 1.0
    x.requires_grad_(True)
    return x


def predict_and_attribute(model, edge_index, tags, num_tags=7):
    """Runs one forward+backward pass on a single molecule.

    Returns dict with:
      probs        : [num_classes] float tensor (softmax probabilities)
      pred_class   : int
      importance   : [N] float tensor, raw Grad-x-Input score per atom
      importance_norm : [N] float tensor, importance rescaled to [0, 1]
                       for coloring (sign-preserving: 0.5 = neutral)
    """
    model.eval()
    x = build_one_hot(tags, num_tags)
    batch_vec = torch.zeros(tags.shape[0], dtype=torch.long)

    logits = model(x, edge_index, batch_vec, num_graphs=1)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_class = int(torch.argmax(probs).item())

    model.zero_grad(set_to_none=True)
    logits[0, pred_class].backward()

    grad = x.grad  # [N, num_tags]
    importance = (grad * x.detach()).sum(dim=1)  # [N], Grad x Input

    # sign-preserving normalization for a diverging color scale (0.5 = neutral)
    max_abs = importance.abs().max().clamp(min=1e-8)
    importance_norm = 0.5 + 0.5 * (importance / max_abs)

    return {
        "probs": probs.detach(),
        "pred_class": pred_class,
        "importance": importance.detach(),
        "importance_norm": importance_norm.detach(),
    }


def explain(mol, tags, importance, top_k=3):
    """Rule-based, human-readable explanation string built from the top-k
    most important atoms and whatever functional group (if any) they
    belong to, via RDKit SMARTS matching."""
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
        elem = ATOM_MAP[int(tags[idx])]
        groups = atom_to_groups.get(idx, [])
        direction = "pushed toward" if importance[idx] > 0 else "pushed away from"
        group_text = f", part of a {groups[0]}" if groups else ""
        parts.append(f"atom {idx} ({elem}{group_text}) {direction} the prediction")

    if not parts:
        return "No single atom stood out strongly for this prediction."

    return "Most influential: " + "; ".join(parts) + "."


def format_prediction(probs, pred_class):
    name = CLASS_NAMES.get(pred_class, str(pred_class))
    confidence = float(probs[pred_class])
    return f"{name} ({confidence:.1%} confidence)"

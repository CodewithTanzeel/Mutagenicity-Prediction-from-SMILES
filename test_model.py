import torch
from gin.model import GIN
from molecule import smiles_to_graph
from atom_map import NUM_TAGS, CLASS_NAMES
from attribution import predict_and_attribute
from alerts import find_alerts

ckpt = torch.load("checkpoint_ames.pt", map_location="cpu")
model = GIN(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])

for smi in ["C=CC(=O)N", "Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]", "CCO"]:
    edge_index, tags, mol = smiles_to_graph(smi)
    r = predict_and_attribute(model, edge_index, tags, num_tags=NUM_TAGS)
    gin_verdict = CLASS_NAMES[r["pred_class"]]
    alerts = find_alerts(mol)
    final = "Mutagenic" if alerts else gin_verdict
    note = f" [expert rule: {alerts[0]}]" if alerts else ""
    print(f"{smi:45s} -> {final} (GIN alone: {gin_verdict} "
          f"{float(r['probs'][r['pred_class']]):.0%}){note}")
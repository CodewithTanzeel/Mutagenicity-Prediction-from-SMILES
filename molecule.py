"""
SMILES -> (edge_index, tags) graph.
Upgraded to use atomic numbers as tags, supporting the whole periodic table.
"""
import torch
from rdkit import Chem

class UnsupportedAtomError(ValueError):
    def __init__(self, symbol):
        super().__init__(f"Atom '{symbol}' is not supported.")
        self.symbol = symbol

def smiles_to_graph(smiles):
    """Returns (edge_index [2,E] LongTensor, tags [N] LongTensor, mol)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    # Use atomic number directly as the tag! (1=H, 6=C, 7=N, 8=O, 15=P, 16=S, etc.)
    # This means no atoms are ever "unsupported" anymore.
    tags = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.append(i); dst.append(j)
        src.append(j); dst.append(i)

    if len(src) == 0 and mol.GetNumAtoms() > 1:
        raise ValueError("Parsed molecule has no bonds between its atoms.")

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else \
        torch.zeros((2, 0), dtype=torch.long)
    tags_t = torch.tensor(tags, dtype=torch.long)

    return edge_index, tags_t, mol
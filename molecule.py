"""
SMILES -> (edge_index, tags) graph, built to match gin/data.py's
conventions exactly:
  - edge_index stores BOTH directions per bond (undirected message passing)
  - tags are raw integer atom-type tags per atom_map.ELEMENT_TO_TAG
  - no bond-type / edge features are used anywhere in this project, so we
    don't need to distinguish single/double/aromatic bonds here either --
    only which atoms are bonded.

Requires rdkit (pip install rdkit). Not verified to run in this authoring
session (no working package-install network here) -- verify locally.
"""

import torch
from rdkit import Chem

from atom_map import ELEMENT_TO_TAG


class UnsupportedAtomError(ValueError):
    """Raised when a SMILES string contains an element the model was never
    trained on (MUTAG only has C, N, O, F, I, Cl, Br)."""
    def __init__(self, symbol):
        super().__init__(
            f"Atom '{symbol}' is not one of the 7 atom types the model was "
            f"trained on ({', '.join(ELEMENT_TO_TAG)}). This molecule can't "
            f"be scored by a model trained only on MUTAG's atom vocabulary."
        )
        self.symbol = symbol


def smiles_to_graph(smiles):
    """Returns (edge_index [2,E] LongTensor, tags [N] LongTensor, mol).
    Raises ValueError for unparseable SMILES, UnsupportedAtomError for
    out-of-vocabulary atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    tags = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ELEMENT_TO_TAG:
            raise UnsupportedAtomError(symbol)
        tags.append(ELEMENT_TO_TAG[symbol])

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.append(i); dst.append(j)
        src.append(j); dst.append(i)

    if len(src) == 0 and mol.GetNumAtoms() > 1:
        # disconnected/no-bond edge case -- shouldn't happen for valid
        # organic SMILES, but don't silently produce a bond-free graph
        raise ValueError("Parsed molecule has no bonds between its atoms.")

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else \
        torch.zeros((2, 0), dtype=torch.long)
    tags_t = torch.tensor(tags, dtype=torch.long)

    return edge_index, tags_t, mol

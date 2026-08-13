"""
Atom and class mappings.
Upgraded to support the entire periodic table (atomic numbers 1-118)
so the model can learn from Sulfur, Phosphorus, etc., instead of skipping them.
"""
from rdkit import Chem

# Use RDKit's periodic table to get symbols for atomic numbers 1 to 118.
# We reserve tag 0 for padding (unused). Tags 1 to 118 are atomic numbers.
ATOM_MAP = {i: Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 119)}

NUM_TAGS = 119  # 0 to 118

# Keep this for backward compatibility, though molecule.py won't strictly need it anymore
ELEMENT_TO_TAG = {v: k for k, v in ATOM_MAP.items()}

CLASS_NAMES = {0: "Non-mutagenic", 1: "Mutagenic"}
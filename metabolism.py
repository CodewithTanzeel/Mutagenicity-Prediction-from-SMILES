"""
One-step Phase I metabolite screening (pro-mutagen detection).

Applies a small set of conservative SMARTS biotransformations, then checks
each predicted metabolite for DNA-reactive warning structures (epoxides,
N-hydroxy arylamines, Michael acceptors) or a high ensemble mutagenicity
probability. Used to flag pro-mutagens whose PARENT looks clean.
"""
from rdkit import Chem
from rdkit.Chem import rdChemReactions

_BIOTRANSFORMATIONS = [
    ("alkene epoxidation",            "[CX3:1]=[CX3:2] >> [CX4:1]1[O][CX4:2]1"),
    ("aromatic amine N-hydroxylation","[NX3;H2:1][c:2] >> [NX3;H1:1]([OX2H1])[c:2]"),
    ("nitro reduction to amine",      "[N+;X3:1](=O)[O-] >> [NX3;H2:1]"),
]

METABOLITE_ALERTS = [
    ("epoxide", "C1OC1"),
    ("N-hydroxy aromatic amine", "[NX3;H1]([OX2H1])[c]"),
    ("Michael acceptor", "[CX3]=[CX3]-[CX3]=[OX1]"),
]


def generate_metabolites(mol, max_total=6):
    """Returns [(rule_name, metabolite_mol, canonical_smiles), ...] one-step only."""
    out, seen = [], set()
    parent = Chem.MolToSmiles(mol)
    for name, smarts in _BIOTRANSFORMATIONS:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        if rxn is None:
            continue
        for products in rxn.RunReactants((mol,))[:8]:
            for p in products:
                try:
                    Chem.SanitizeMol(p)
                except Exception:
                    continue
                frags = Chem.GetMolFrags(p, asMols=True, sanitizeFrags=False)
                if not frags:
                    continue
                big = max(frags, key=lambda m: m.GetNumAtoms())
                if big.GetNumAtoms() < 3:
                    continue
                smi = Chem.MolToSmiles(big)
                if not smi or smi == parent or smi in seen:
                    continue
                seen.add(smi)
                out.append((name, big, smi))
                if len(out) >= max_total:
                    return out
    return out


def metabolite_alerts(mol):
    hits = []
    for name, smarts in METABOLITE_ALERTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            hits.append(name)
    return hits
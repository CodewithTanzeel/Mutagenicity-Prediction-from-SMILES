"""
Applicability-domain (AD) checker.

A prediction is only as trustworthy as its distance to the chemistry the
model was trained on. We measure that with Morgan-fingerprint Tanimoto
similarity to the nearest training-set molecule (same idea the OECD QSAR
Toolbox uses for its domain check).
"""
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs


class DomainChecker:
    def __init__(self, train_smiles):
        # Use the modern RDKit generator API (silences the deprecation warning)
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        self.fps = []
        for smi in train_smiles:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                self.fps.append(self.fpgen.GetFingerprint(m))

    def score(self, mol):
        """Nearest-neighbour Tanimoto similarity to the training set (0..1)."""
        if not self.fps:
            return 0.0
        fp = self.fpgen.GetFingerprint(mol)
        return max(DataStructs.TanimotoSimilarity(fp, t) for t in self.fps)


def domain_label(sim):
    if sim >= 0.6:
        return "HIGH reliability", "In-domain: very similar compounds were tested"
    if sim >= 0.35:
        return "MEDIUM reliability", "Borderline domain: only partly similar compounds seen"
    return "LOW reliability", "Out-of-domain: treat the verdict as advisory only"
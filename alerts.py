"""Expert rule-based structural alerts (ICH M7 second methodology)."""
from rdkit import Chem

MUTAGENIC_ALERTS = [
    ("Alkyl sulfonate ester (alkylating agent)", "[SX4](=[OX1])(=[OX1])-[OX2]-[CX4]"),
    ("Michael acceptor (alpha,beta-unsaturated carbonyl)", "[CX3]=[CX3]-[CX3]=[OX1]"),
    ("Michael acceptor (alpha,beta-unsaturated nitrile)", "[CX3]=[CX3]-[C]#[N]"),
    ("Epoxide", "C1OC1"),
    ("Aziridine", "C1NC1"),
    ("N-nitrosamine", "[N;X3]-[N;X2]=O"),
]

def find_alerts(mol):
    hits = []
    for name, smarts in MUTAGENIC_ALERTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            hits.append(name)
    return hits
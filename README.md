# MUTAG GIN Explainer (Dash demo)

## IMPORTANT — what's verified vs. not

Everything here was **written but not executed** — this authoring session
had no working package-install network (`pip install torch/rdkit/dash`
all failed here). Before your demo:

1. Place `MUTAG.txt` at `dataset/MUTAG/MUTAG.txt` (matches your own
   `train.py`'s default path), or pass `--data_path`.
2. Run `python train_and_export.py` and check the printed train accuracy
   is reasonable (>80%, sanity-checking against your existing 10-fold
   `train.py` results — this one trains on all 188 graphs so it should be
   at least as good, likely close to or above your CV accuracy since it's
   not held-out).
3. Run `python app.py`, try the example molecules, and check:
   - images render with sensible highlight colors
   - predictions look directionally right (nitrobenzene should trend
     mutagenic; benzene alone should not)
   - no exceptions in the RDKit `highlightAtomRadii`/`highlightAtomColors`
     calls (API has shifted across rdkit versions — pin a version if so)

## Files

- `gin/` — your original `data.py`, `batch.py`, `model.py`, unmodified.
- `atom_map.py` — the atom-tag mapping, derived empirically from your
  `MUTAG.txt` (frequency + valence analysis) — see docstring for the full
  derivation and the one flagged uncertainty (F vs I, 2 vs 1 atoms total).
- `molecule.py` — SMILES → graph, matching `gin/data.py`'s conventions.
- `attribution.py` — Gradient×Input saliency (a standard, simple baseline
  — not the most faithful method in the literature, worth naming as a
  limitation) + rule-based SMARTS explanation text.
- `render.py` — RDKit rendering with attribution-colored atoms → base64 PNG.
- `train_and_export.py` — trains on the FULL dataset (not cross-val) for
  deployment, saves `checkpoint.pt`.
- `app.py` — the Dash app itself.

## Known gaps / things to add if time allows

- `EXAMPLES` in `app.py` includes a 2,4-dinitrotoluene SMILES I'm not
  100% certain is canonically correct — verify it against RDKit's own
  canonicalization or a trusted source (e.g. PubChem) before using it live.
- No handling yet for disconnected molecules (salts, multiple components
  in one SMILES via `.`) — `Chem.MolFromSmiles` will parse them but the
  graph conversion assumes one connected component.
- `CLASS_NAMES` (0=Non-mutagenic, 1=Mutagenic) is based on the documented
  125/63 MUTAG class split, not a value your own code confirmed — flag if
  your training pipeline defines this differently.

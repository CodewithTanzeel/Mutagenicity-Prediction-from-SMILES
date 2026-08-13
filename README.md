\<div align="center"\>

# 🧬 Molecular Mutagenicity Explainer

**Predict whether a chemical is mutagenic — straight from its SMILES
string — and see** ***why***\*\*.\*\*

**🔗 Live app:**
[**mutagenai.up.railway.app**](https://mutagenai.up.railway.app/)

\</div\>

------------------------------------------------------------------------

## 📸 Screenshots


### 🧠 Model Documentation

![OECD QSAR model documentation and methodology](https://raw.githubusercontent.com/CodewithTanzeel/Mutagenicity-Prediction-from-SMILES/main/docs/model-documentation.png)

### 🔬 Interactive Prediction & Explanation

![Molecular mutagenicity prediction with atom-level attribution](https://raw.githubusercontent.com/CodewithTanzeel/Mutagenicity-Prediction-from-SMILES/main/docs/prediction.png)

### 📋 Generalization Test Sheet

![Generalization test sheet with known compounds](https://raw.githubusercontent.com/CodewithTanzeel/Mutagenicity-Prediction-from-SMILES/main/docs/generalization-test.png)

------------------------------------------------------------------------

## What it does

Type a chemical in as a **SMILES string** or a **common name** (e.g.
`Aspirin`), and get back:

| Layer What it tells you            |                                                                                                                                                             |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🧠 **3-model GIN ensemble**        | Mean predicted probability ± ensemble disagreement, from a Graph Isomorphism Network trained on ~6,500 Ames mutagenicity results                            |
| 🚨 **Expert structural alerts**    | Deterministic SMARTS rules for known DNA-reactive groups (nitro aromatics, Michael acceptors, epoxides, aziridines, N-nitrosamines, alkyl sulfonate esters) |
| 🧪 **One-step metabolite screen**  | Catches "pro-mutagens" — compounds that are only dangerous *after* the body metabolizes them                                                                |
| 🎯 **Applicability domain check**  | A High / Medium / Low reliability score based on similarity to the training data, so you know when to trust the verdict less                                |
| 🔬 **Atom-level explanation**      | Grad×Input attribution rendered directly on the molecule (red = pushes toward the verdict, blue = pushes away)                                              |
| 📋 **Generalization test sheet**   | An editable, spreadsheet-style tab to grade the whole pipeline against your own known compounds                                                             |
| 📄 **Exportable report**           | One click generates a standalone HTML safety report for any prediction                                                                                      |
| 📚 **OECD QMRF documentation tab** | The model's scope, algorithm, domain, and named limitations, laid out against the five OECD QSAR principles                                                 |

The verdict follows **ICH M7**'s two-methodology principle: a positive
from *either* the trained model *or* an expert alert is treated as a
mutagenicity concern — mirroring how this is actually done in regulatory
toxicology.

> ⚠️ **This is an educational/hackathon demo, not a validated regulatory
> tool.** Predictions must not be used for safety-critical decisions
> without expert review and experimental (in-vitro/in-vivo)
> verification.

------------------------------------------------------------------------

## Why this matters

The Ames test (bacterial reverse mutation assay) is one of the first
genotoxicity screens run on any new drug candidate or industrial
chemical, but it's slow and resource-intensive to run experimentally. A
model that gives a fast, *explainable* first-pass estimate — flagging
both the compound **and** its likely metabolites, with a built-in sense
of when it's out of its depth — is directly useful earlier in a
discovery pipeline, as a triage tool rather than a replacement for the
wet-lab assay.

------------------------------------------------------------------------

## Model

The core predictor is a **Graph Isomorphism Network (GIN)** ensemble (3
seeds), which represents each molecule as a graph — atoms as nodes,
bonds as edges — and learns to classify the whole graph as mutagenic or
not.

GIN was chosen because it's provably the most expressive message-passing
GNN architecture under the Weisfeiler-Lehman graph isomorphism test —
meaning it can, in principle, distinguish molecular structures that
architectures like GCN or GraphSAGE (mean/max-pooling aggregators)
provably cannot tell apart. This is exactly the citation used below.

> Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). **How Powerful are
> Graph Neural Networks?** *International Conference on Learning
> Representations (ICLR 2019).*
> [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)

    @inproceedings{xu2019how,
      title     = {How Powerful are Graph Neural Networks?},
      author    = {Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
      booktitle = {International Conference on Learning Representations (ICLR)},
      year      = {2019},
      url       = {https://arxiv.org/abs/1810.00826}
    }

------------------------------------------------------------------------

## Tech stack

- [**Dash**](https://dash.plotly.com/) (Plotly) — the web UI
- [**PyTorch**](https://pytorch.org/) (CPU build) — the GIN ensemble
- [**RDKit**](https://www.rdkit.org/) — SMILES parsing, SMARTS alerts,
  molecule rendering
- [**Gunicorn**](https://gunicorn.org/) — production WSGI server
- [**PubChem PUG REST
  API**](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) — name → SMILES
  resolution
- [**Railway**](https://railway.app/) — deployment

------------------------------------------------------------------------

## Project structure

    .
    ├── app.py                       # the Dash app (UI + callbacks)
    ├── gin/                         # GIN model definition
    │   └── model.py
    ├── atom_map.py                  # atom-tag <-> feature mapping
    ├── molecule.py                  # SMILES -> graph conversion
    ├── attribution.py                # Grad×Input attribution + explanation text
    ├── alerts.py                    # SMARTS-based expert structural alerts
    ├── ad.py                         # applicability-domain (reliability) scoring
    ├── metabolism.py                  # one-step metabolite generation + screening
    ├── render.py                     # RDKit molecule rendering -> base64 PNG
    ├── pubchem.py                    # PubChem name resolution helper
    ├── train_ames.py                  # training script (Ames dataset)
    ├── train_and_export.py            # trains + exports a deployable checkpoint
    ├── make_ad_file.py                # builds the training-SMILES file for the AD check
    ├── test_model.py                  # basic model sanity checks
    ├── train_smiles.txt               # training-set SMILES used for the AD check
    ├── checkpoint_ames_ensemble.pt    # trained 3-model ensemble checkpoint (Git LFS)
    ├── requirements.txt
    ├── Dockerfile
    └── apt.txt                        # extra system packages

------------------------------------------------------------------------

## Running locally

**Requirements:** Python 3.11

    git clone https://github.com/CodewithTanzeel/Mutagenicity-Prediction-from-SMILES.git
    cd Mutagenicity-Prediction-from-SMILES

    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate

    pip install -r requirements.txt
    python app.py

Open **http://127.0.0.1:8050**.

### Running with Docker

    docker build -t mutagenicity-app .
    docker run -p 8080:8080 -e CHECKPOINT_URL=<your-checkpoint-url> mutagenicity-app

------------------------------------------------------------------------

## Deployment notes

Deployed on Railway using Gunicorn, not Dash's dev server. `app.py`
exposes the Flask server directly for this:

    app = Dash(__name__)
    server = app.server

**Start command:**

    gunicorn app:server --workers 1 --threads 4 --timeout 120

`--workers 1` is important: each Gunicorn worker loads its own copy of
the 3-model ensemble, so more than one worker can push a small container
into an out-of-memory kill.

### The checkpoint and Git LFS

`checkpoint_ames_ensemble.pt` is tracked with **Git LFS**. Most git-push
PaaS deploys (Railway included) check out the repo *without* resolving
LFS objects, so the file that lands in the container is a ~130-byte LFS
*pointer*, not the real weights — loading it fails with
`invalid load key, 'v'` or a `weights_only` unpickling error.

The fix used here: `app.py` downloads the real checkpoint at startup
from a `CHECKPOINT_URL` env var, overwriting anything already on disk
that's suspiciously small (i.e. a stale pointer):

    CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL")

**Setup:**

1.  Set `CHECKPOINT_URL` on your host to a URL serving the *raw binary*,
    not an HTML page. GitHub's own raw-resolving endpoint works
    (`raw.githubusercontent.com` does **not** — it serves the LFS
    pointer text):
        https://github.com/CodewithTanzeel/Mutagenicity-Prediction-from-SMILES/raw/refs/heads/main/checkpoint_ames_ensemble.pt
2.  Redeploy. Logs should show:
        Checkpoint missing or too small (likely an LFS pointer) - downloading from CHECKPOINT_URL...Checkpoint downloaded: <size> bytesMODEL OK: 3 modelsAD OK: <n> fingerprints

### Environment variables

| Variable Required Description |                  |                                                                    |
|-------------------------------|------------------|--------------------------------------------------------------------|
| `CHECKPOINT_URL`              | Yes (for deploy) | Direct download URL for `checkpoint_ames_ensemble.pt`              |
| `PYTHON_VERSION`              | No               | Pins the Python version on some buildpacks (e.g. Railway Nixpacks) |

------------------------------------------------------------------------

## Retraining the model

    python train_and_export.py

Trains the GIN ensemble on the Ames dataset and exports a new
`checkpoint_ames_ensemble.pt` (`config`, `state_dicts`, `tag_list`). Run
`make_ad_file.py` to regenerate `train_smiles.txt` for the
applicability-domain check.

------------------------------------------------------------------------

## Limitations

- Ames endpoint only — not a substitute for in-vivo genotoxicity assays
  (micronucleus, comet).
- Metabolic activation is approximated by one-step rules; multi-step
  bioactivation isn't modeled.
- No stereochemistry, salt, or pH-dependent speciation handling.
- Educational/hackathon demo — not validated for regulatory submissions
  (no external benchmark audit, no calibration guarantees).

------------------------------------------------------------------------

## Citation

If you use this project or build on it, please cite the underlying GIN
architecture:

    @inproceedings{xu2019how,
      title     = {How Powerful are Graph Neural Networks?},
      author    = {Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
      booktitle = {International Conference on Learning Representations (ICLR)},
      year      = {2019},
      url       = {https://arxiv.org/abs/1810.00826}
    }

## License

MIT — see [LICENSE](https://claude.ai/chat/LICENSE).

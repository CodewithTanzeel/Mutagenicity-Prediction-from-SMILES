"""
Dash demo app: SMILES -> mutagenicity prediction + atom-level explanation.
3-model GIN ensemble (AMES) + expert alerts (ICH M7) + AD reliability score
+ one-step metabolite screen + OECD QMRF tab + Report Export + PubChem name search.
Run:  python app.py
"""

import base64
import datetime

import requests
import torch
from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL, no_update, dash_table
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')  # Silence noisy parse errors when trying names as 

from gin.model import GIN
from atom_map import NUM_TAGS, CLASS_NAMES
from molecule import smiles_to_graph, UnsupportedAtomError
from attribution import predict_and_attribute, explain, format_prediction
from alerts import find_alerts
from ad import DomainChecker, domain_label
from metabolism import generate_metabolites, metabolite_alerts
from render import render_molecule_png

import os
import urllib.request

CHECKPOINT_PATH = "checkpoint_ames_ensemble.pt"
CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL")  # set this in your host's env vars
MIN_CHECKPOINT_BYTES = 10_000  # a real .pt is way bigger than a ~130-byte LFS pointer file

_needs_download = (
    not os.path.exists(CHECKPOINT_PATH)
    or os.path.getsize(CHECKPOINT_PATH) < MIN_CHECKPOINT_BYTES
)

if _needs_download and CHECKPOINT_URL:
    print("Checkpoint missing or too small (likely an LFS pointer) - downloading from CHECKPOINT_URL...")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
        size = os.path.getsize(CHECKPOINT_PATH)
        print("Checkpoint downloaded:", size, "bytes")
        if size < MIN_CHECKPOINT_BYTES:
            raise RuntimeError(
                f"Downloaded file is only {size} bytes - CHECKPOINT_URL is probably still "
                "pointing at an LFS pointer or an HTML page, not the real .pt file."
            )
    except Exception as e:
        print("CHECKPOINT DOWNLOAD FAILED:", e)
        raise
elif _needs_download:
    print("WARNING: checkpoint missing/invalid locally and no CHECKPOINT_URL set")

# ---- notebook-page background --------------------------------------
NOTEBOOK_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M20 0H0V20" fill="none" stroke="rgba(70,100,170,0.11)" stroke-width="1"/>
    </pattern>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="rgba(32,40,58,0.20)"/>
    </marker>
  </defs>
  <rect width="640" height="640" fill="url(#grid)"/>
  <line x1="66" y1="0" x2="66" y2="640" stroke="rgba(170,60,50,0.16)" stroke-width="1.5"/>

  <g fill="none" stroke="rgba(32,40,58,0.22)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="146,96 176,113 176,147 146,164 116,147 116,113" transform="rotate(-4 146 130)"/>
    <circle cx="146" cy="130" r="15" transform="rotate(-4 146 130)"/>
    <line x1="176" y1="113" x2="205" y2="97" transform="rotate(-4 146 130)"/>
    <polygon points="486,232 516,249 516,283 486,300 456,283 456,249" transform="rotate(9 486 266)"/>
    <circle cx="486" cy="266" r="15" transform="rotate(9 486 266)"/>
    <line x1="456" y1="249" x2="428" y2="234" transform="rotate(9 486 266)"/>
    <polygon points="286,452 314,468 314,500 286,516 258,500 258,468" transform="rotate(-7 286 484)"/>
    <circle cx="286" cy="484" r="14" transform="rotate(-7 286 484)"/>
    <polygon points="566,536 590,550 590,578 566,592 542,578 542,550" transform="rotate(5 566 564)"/>
    <path d="M28,318 L50,302 L72,318 L94,302"/>
    <path d="M356,66 L414,66" stroke-width="1.8" marker-end="url(#arrow)"/>
  </g>

  <g font-family="Kalam, cursive" fill="rgba(32,40,58,0.24)">
    <text x="90" y="70" font-size="24" transform="rotate(-4 90 70)">NH&#8322;</text>
    <text x="180" y="178" font-size="22" transform="rotate(3 180 178)">CH&#8323;</text>
    <text x="420" y="230" font-size="23" transform="rotate(-3 420 230)">COOH</text>
    <text x="500" y="330" font-size="22" transform="rotate(4 500 330)">OH</text>
    <text x="330" y="470" font-size="22" transform="rotate(-5 330 470)">C=O</text>
    <text x="600" y="520" font-size="22" transform="rotate(3 600 520)">OR</text>
    <text x="200" y="30" font-size="22" transform="rotate(-2 200 30)">R</text>
    <text x="330" y="82" font-size="22" transform="rotate(-2 330 82)">+</text>
  </g>
</svg>
"""
NOTEBOOK_BG_URI = "data:image/svg+xml;base64," + base64.b64encode(NOTEBOOK_SVG.encode("utf-8")).decode("ascii")

CSS = """
:root{
 --paper:#f4eed4; --paper-raised:rgba(250,246,230,.82); --paper-deep:#e7ddb9;
 --paper-tint:rgba(250,246,230,.62);
 --ink:#20283a; --ink-soft:#525d78; --ink-faint:#8890a6;
 --line:rgba(32,40,58,.18); --line-soft:rgba(32,40,58,.10);
 --moss:#2f6b4c; --moss-bg:rgba(47,107,76,.11);
 --brick:#a13f2c; --brick-bg:rgba(161,63,44,.11);
 --amber:#a97a1f; --amber-bg:rgba(169,122,31,.13);
}
*{box-sizing:border-box}
body{margin:0;background-color:var(--paper);color:var(--ink);
 font-family:"IBM Plex Sans",system-ui,sans-serif;
 background-image:url('""" + NOTEBOOK_BG_URI + """');
 background-size:640px 640px;background-repeat:repeat}
.wrap{max-width:1020px;margin:0 auto;padding:28px 20px 60px}
.hero{position:relative;padding:30px 28px 26px;border:1px solid var(--line);border-radius:2px;
 border-left:3px solid var(--ink);background:var(--paper-raised);overflow:hidden;
 box-shadow:0 1px 0 var(--paper-deep) inset}
.hero-badge{position:relative;display:inline-block;font-family:"IBM Plex Mono",monospace;
 font-size:10.5px;letter-spacing:.12em;font-weight:600;color:var(--ink-soft);
 border:1px dashed var(--line);border-radius:2px;padding:5px 11px;
 background:var(--paper-tint);text-transform:uppercase}
.hero h1{position:relative;margin:16px 0 8px;font-family:"IBM Plex Sans",system-ui,sans-serif;
 font-size:30px;font-weight:600;letter-spacing:-.01em;color:var(--ink)}
.hero h1 span{color:var(--amber)}
.hero p{position:relative;margin:0;color:var(--ink-soft);max-width:760px;line-height:1.6;font-size:14px}
.hero .meta{position:relative;margin-top:16px;font-family:"IBM Plex Mono",monospace;
 font-size:12px;color:var(--ink-soft);border-top:1px dashed var(--line);padding-top:12px}
.hero .meta b{color:var(--ink);font-weight:600}
.card{background:var(--paper-raised);border:1px solid var(--line);border-radius:2px;padding:22px;margin-top:16px}
.card h3{margin:0 0 4px;font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;
 font-weight:600;color:var(--ink)}
.hint{color:var(--ink-soft);font-size:12.5px;margin:0 0 16px;line-height:1.55}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input.smiles{flex:1;min-width:240px;background:var(--paper-tint);border:1px solid var(--line);color:var(--ink);
 border-radius:2px;padding:11px 14px;font-family:"IBM Plex Mono",monospace;font-size:13.5px;outline:none}
input.smiles::placeholder{color:var(--ink-faint)}
input.smiles:focus{border-color:var(--amber)}
button.btn{background:var(--ink);border:1px solid var(--ink);color:var(--paper-raised);
 font-family:"IBM Plex Mono",monospace;font-weight:600;letter-spacing:.02em;
 border-radius:2px;padding:11px 18px;cursor:pointer;font-size:13px}
button.btn:hover{background:var(--amber);border-color:var(--amber);color:#1d1503}
button.btn.ghost{background:transparent;color:var(--ink-soft);border:1px solid var(--line)}
button.btn.ghost:hover{border-color:var(--ink-soft);color:var(--ink)}
.chips{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}
.chips button{background:var(--paper-tint);border:1px solid var(--line);color:var(--ink-soft);
 border-radius:2px;font-family:"IBM Plex Mono",monospace;
 padding:6px 13px;font-size:11.5px;cursor:pointer}
.chips button:hover{border-color:var(--amber);color:var(--amber);background:var(--amber-bg)}
.mol-card{background:rgba(253,253,247,.90);border:1px solid var(--line);border-radius:3px;padding:12px;margin-top:18px}
.expl{color:var(--ink-soft);font-size:13.5px;line-height:1.65;margin-top:10px}
.tabs .tab{background:transparent;border:1px solid var(--line);border-bottom:2px solid var(--paper-deep);
 color:var(--ink-soft);font-family:"IBM Plex Mono",monospace;border-radius:3px 3px 0 0;
 padding:10px 18px;font-weight:600;font-size:12.5px;letter-spacing:.02em;cursor:pointer}
.tabs .tab--selected{background:var(--paper-raised);color:var(--ink);border-color:var(--line);
 border-bottom:2px solid var(--amber)}
.summary{margin-top:16px;font-family:"IBM Plex Mono",monospace;font-size:14px;font-weight:600;color:var(--ink)}
.qmrf h4{margin:20px 0 6px;font-family:"IBM Plex Mono",monospace;font-size:13px;
 text-transform:uppercase;letter-spacing:.06em;color:var(--amber)}
.qmrf p{margin:0 0 8px;color:var(--ink-soft);font-size:13px;line-height:1.65}
.qmrf code{font-family:"IBM Plex Mono",monospace;background:var(--paper-tint);
 border:1px solid var(--line);border-radius:2px;padding:1px 5px;font-size:12px}
footer{color:var(--ink-faint);font-family:"IBM Plex Mono",monospace;font-size:11px;
 margin-top:30px;padding-top:16px;border-top:1px dashed var(--line);text-align:center;
 letter-spacing:.03em}
"""

EXAMPLES = [
    ("Acrylamide (Michael acceptor)", "C=CC(=O)N"),
    ("Styrene (pro-mutagen)", "C=Cc1ccccc1"),
    ("Benzene (simple aromatic)", "c1ccccc1"),
    ("Aniline", "c1ccc(cc1)N"),
    ("Nitrobenzene (nitro group)", "c1ccc(cc1)[N+](=O)[O-]"),
    ("2,4-Dinitrotoluene (known mutagen)", "Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]"),
    ("Ethanol (non-mutagen)", "CCO"),
]

DEFAULT_SHEET = [
    {"name": "Acrylamide",            "smiles": "C=CC(=O)N",                          "expected": "Mutagenic",     "predicted": "", "verdict": ""},
    {"name": "Acrylonitrile",         "smiles": "C=CC#N",                             "expected": "Mutagenic",     "predicted": "", "verdict": ""},
    {"name": "Ethylene oxide",        "smiles": "C1CO",                               "expected": "Mutagenic",     "predicted": "", "verdict": ""},
    {"name": "2,4-Dinitrotoluene",    "smiles": "Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]", "expected": "Mutagenic",    "predicted": "", "verdict": ""},
    {"name": "Methyl methanesulfonate","smiles": "CS(=O)(=O)OC",                      "expected": "Mutagenic",     "predicted": "", "verdict": ""},
    {"name": "Styrene",               "smiles": "C=Cc1ccccc1",                        "expected": "Mutagenic",     "predicted": "", "verdict": ""},
    {"name": "Caffeine",              "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",         "expected": "Non-mutagenic", "predicted": "", "verdict": ""},
    {"name": "Aspirin",               "smiles": "CC(=O)Oc1ccccc1C(=O)O",              "expected": "Non-mutagenic", "predicted": "", "verdict": ""},
    {"name": "Ethanol",               "smiles": "CCO",                                "expected": "Non-mutagenic", "predicted": "", "verdict": ""},
    {"name": "Glucose",               "smiles": "OCC(O)C(O)C(O)C(O)C=O",              "expected": "Non-mutagenic", "predicted": "", "verdict": ""},
]

import traceback

print("="*50)
print("STARTING APP.PY - DO NOT CRASH")
print("="*50)

# ---- load ensemble once at startup -----------------------------------
try:
    _ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    _models = []
    for sd in _ckpt["state_dicts"]:
        m = GIN(**_ckpt["config"])
        m.load_state_dict(sd)
        m.eval()
        _models.append(m)
    print("MODEL OK:", len(_models), "models")
    assert _ckpt["tag_list"] == sorted(range(NUM_TAGS))
except Exception as e:
    print("MODEL FAILED:", e)
    raise

# ---- AD setup --------------------------------------------------------
def _load_train_smiles():
    try:
        with open("train_smiles.txt") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        print("WARNING: train_smiles.txt missing")
        return []

try:
    _domain = DomainChecker(_load_train_smiles())
    print("AD OK:", len(_domain.fps), "fingerprints")
except Exception as e:
    print("AD FAILED:", e)
    raise

_AD_COLORS = {
    "HIGH reliability": "#2f6b4c",
    "MEDIUM reliability": "#a97a1f",
    "LOW reliability": "#a13f2c",
}

def final_verdict(mol, result):
    """ICH M7: positive from EITHER the ensemble GIN or the expert alerts = Mutagenic."""
    return "Mutagenic" if find_alerts(mol) else CLASS_NAMES[result["pred_class"]]

@torch.no_grad()
def ensemble_p_mutagenic(smi):
    """Mean ensemble probability of class 1 for a SMILES string (or None)."""
    try:
        edge_index, tags, _ = smiles_to_graph(smi)
    except Exception:
        return None
    if len(tags) == 0:
        return None
    x = torch.zeros(len(tags), NUM_TAGS)
    x[torch.arange(len(tags)), tags] = 1.0
    batch_vec = torch.zeros(len(tags), dtype=torch.long)
    ps = [float(torch.softmax(m(x, edge_index, batch_vec, num_graphs=1), dim=-1)[0, 1])
          for m in _models]
    return sum(ps) / len(ps)

def resolve_to_smiles(user_input):
    """Tries to parse as SMILES first; if that fails, queries PubChem by name."""
    if Chem.MolFromSmiles(user_input) is not None:
        return user_input, None  # Already a valid SMILES -- instant, no network

    try:
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{requests.utils.quote(user_input)}/property/IsomericSMILES/JSON")
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            smi = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
            return smi, f"Resolved '{user_input}' via PubChem"
    except Exception:
        pass
    return None, f"Could not resolve '{user_input}'. Enter a valid SMILES or chemical name."

def build_report_html(data):
    verdict_color = "#a13f2c" if "MUTAGENIC" in data["verdict"].upper() else "#2f6b4c"
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Mutagenicity Report: {data['smiles']}</title>
    <style>
      body {{ font-family: "Segoe UI", sans-serif; color: #20283a; background: #f4eed4; padding: 40px; max-width: 800px; margin: auto; }}
      h1 {{ border-bottom: 3px solid #20283a; padding-bottom: 10px; margin-top: 0; }}
      .meta {{ font-family: monospace; color: #525d78; font-size: 13px; margin-bottom: 20px; }}
      .card {{ background: white; padding: 25px; border: 1px solid #8890a6; border-radius: 4px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
      .verdict {{ font-size: 22px; font-weight: bold; color: {verdict_color}; margin-bottom: 10px; }}
      .img-box {{ text-align: center; background: white; padding: 20px; border: 1px solid #8890a6; margin-bottom: 20px; }}
      .img-box img {{ max-width: 100%; max-height: 400px; }}
      footer {{ margin-top: 40px; font-size: 11px; color: #8890a6; border-top: 1px dashed #8890a6; padding-top: 15px; text-align: center; }}
      strong {{ color: #20283a; }}
    </style>
    </head>
    <body>
      <h1>⚠️ Mutagenicity Safety Report</h1>
      <div class="meta">
        <strong>Generated:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}<br>
        <strong>Query:</strong> {data['query']}<br>
        <strong>Resolved SMILES:</strong> {data['smiles']}
      </div>

      <div class="img-box">
        <img src="{data['img_src']}" alt="Molecule Structure">
      </div>

      <div class="card">
        <div class="verdict">{data['verdict']}</div>
        <p><strong>Confidence:</strong> {data['gin_text']}</p>
        <p><strong>Reliability:</strong> {data['ad_text']}</p>
      </div>

      <div class="card">
        <h3>Detailed Explanation & Alerts</h3>
        <p>{data['explanation']}</p>
      </div>

      <footer>
        <strong>DISCLAIMER:</strong> This report is generated by an educational AI demo tool
        (3-model GIN ensemble on AMES data + expert alerts). It is NOT a validated regulatory
        submission and must not be used for safety-critical decision-making without expert
        review and experimental verification.
      </footer>
    </body>
    </html>
    """

# ---- app ---------------------------------------------------------------
app = Dash(__name__)
app.title = "Molecular Mutagenicity Explainer"

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Sans:wght@400;500;600'
    '&family=IBM+Plex+Mono:wght@400;500;600'
    '&family=Kalam:wght@400;700&display=swap" rel="stylesheet">'
)

app.index_string = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}"
    + FONTS +
    "<style>" + CSS + "</style></head><body>{%app_entry%}"
    "<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
)

app.layout = html.Div(
    className="wrap",
    children=[
        html.Div(
            className="hero",
            children=[
                html.Div("GIN ensemble · AMES · ICH M7 · metabolite screen", className="hero-badge"),
                html.H1(["Molecular ", html.Span("Mutagenicity"), " Explainer"]),
                html.P(
                    "Type any chemical — as a SMILES string or a common name like 'Aspirin' — "
                    "or use the test sheet below. The verdict combines a 3-model GNN ensemble "
                    "trained on ~6,500 Ames experiments with expert structural-alert rules "
                    "(ICH M7 two-methodology), a one-step metabolite screen for pro-mutagens, "
                    "and a reliability score based on training-domain similarity."),
                html.Div(className="meta", children=[
                    html.B("~6,500"), " compounds (AMES) · ",
                    html.B("3"), "-model ensemble · ",
                    html.B("4"), " layers: ensemble + alerts + metabolism + domain check",
                ]),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            className="tabs",
            value="explorer",
            children=[
                dcc.Tab(label="Molecule Explorer", value="explorer"),
                dcc.Tab(label="Generalization Test Sheet", value="sheet"),
                dcc.Tab(label="Model Documentation", value="docs"),
            ],
        ),
        # ---------------- Tab 1: explorer ----------------
        html.Div(
            id="tab-explorer",
            className="card",
            children=[
                html.H3("Explore a single molecule"),
                html.P("Paste a SMILES string or a common chemical name (e.g. Aspirin), "
                       "then press Predict. Red atoms pushed toward the verdict, blue away "
                       "from it. A wide ± spread means the ensemble disagrees; a 'pro-mutagen' "
                       "verdict means a predicted metabolite is the toxic species.", className="hint"),
                html.Div(className="row", children=[
                    dcc.Input(id="smiles-input", type="text", value="", className="smiles",
                              placeholder="e.g. Aspirin, Ibuprofen, or C=CC(=O)N"),
                    html.Button("Predict", id="predict-btn", n_clicks=0, className="btn"),
                ]),
                html.Div(className="chips", children=[
                    html.Button(name, id={"type": "example-btn", "index": i}, n_clicks=0)
                    for i, (name, _) in enumerate(EXAMPLES)
                ]),
                html.Div(id="error-output", style={"color": "#a13f2c", "marginTop": "10px",
                                                    "fontFamily": '"IBM Plex Mono",monospace',
                                                    "fontSize": "12.5px"}),
                html.Div(id="mol-wrap", className="mol-card", style={"display": "none"},
                         children=[html.Img(id="mol-image", style={"maxWidth": "100%"})]),
                html.H4(id="prediction-output", style={"marginTop": "14px"}),
                html.Div(id="ad-output", className="expl", style={"marginTop": "4px",
                         "fontFamily": '"IBM Plex Mono",monospace', "fontWeight": "600"}),
                html.P(id="explanation-output", className="expl"),
                html.Button("⬇ Download Safety Report", id="download-btn", className="btn ghost",
                            style={"marginTop": "20px", "display": "block", "width": "100%"}),
            ],
        ),
        # ---------------- Tab 2: sheet ----------------
        html.Div(
            id="tab-sheet",
            className="card",
            style={"display": "none"},
            children=[
                html.H3("Generalization test sheet"),
                html.P("This works like a mini spreadsheet. Edit any row or add your own "
                       "compounds, set what is KNOWN from the literature ('Expected'), then press "
                       "'Evaluate sheet'. The app grades the full pipeline against your sheet.",
                       className="hint"),
                html.Div(className="row", children=[
                    html.Button("▶ Evaluate sheet", id="eval-sheet-btn", n_clicks=0, className="btn"),
                    html.Button("+ Add row", id="add-row-btn", n_clicks=0, className="btn ghost"),
                    html.Button("Reset sheet", id="reset-sheet-btn", n_clicks=0, className="btn ghost"),
                ]),
                html.Div(
                    dash_table.DataTable(
                        id="sheet-table",
                        columns=[
                            {"name": "Compound", "id": "name"},
                            {"name": "SMILES", "id": "smiles"},
                            {"name": "Expected (known)", "id": "expected", "presentation": "dropdown"},
                            {"name": "Model says", "id": "predicted", "editable": False},
                            {"name": "✓ / ✗", "id": "verdict", "editable": False},
                        ],
                        data=DEFAULT_SHEET,
                        editable=True,
                        dropdown={"expected": {"options": [
                            {"label": v, "value": v} for v in ["Mutagenic", "Non-mutagenic", ""]]}},
                        style_table={"overflowX": "auto", "marginTop": "14px"},
                        style_header={"backgroundColor": "rgba(231,221,185,.88)", "color": "#20283a",
                                      "border": "1px solid rgba(32,40,58,.18)", "fontWeight": "600",
                                      "fontFamily": '"IBM Plex Mono",monospace',
                                      "fontSize": "12px", "letterSpacing": ".02em",
                                      "textTransform": "uppercase", "padding": "10px"},
                        style_cell={"backgroundColor": "rgba(250,246,230,.85)", "color": "#20283a",
                                    "border": "1px solid rgba(32,40,58,.18)", "padding": "9px 12px",
                                    "fontFamily": '"IBM Plex Mono",monospace',
                                    "fontSize": "12.5px", "textAlign": "left", "minWidth": "90px"},
                        style_data_conditional=[
                            {"if": {"column_id": "verdict", "filter_query": '{verdict} = "✓"'},
                             "color": "#2f6b4c", "fontWeight": "700"},
                            {"if": {"column_id": "verdict", "filter_query": '{verdict} = "✗"'},
                             "color": "#a13f2c", "fontWeight": "700"},
                            {"if": {"column_id": "predicted"}, "color": "#a97a1f", "fontWeight": "600"},
                        ],
                    ),
                ),
                html.Div(id="sheet-summary", className="summary"),
            ],
        ),
        # ---------------- Tab 3: OECD documentation ----------------
        html.Div(
            id="tab-docs",
            className="card qmrf",
            style={"display": "none"},
            children=[
                html.H3("Model documentation (OECD QSAR principles)"),
                html.P("Summary of the model following the five OECD QSAR principles (QMRF "
                       "style), so the tool's scope and limits are explicit and auditable.",
                       className="hint"),
                html.H4("1 · Defined endpoint"),
                html.P(["Binary outcome of the bacterial reverse mutation assay (Ames test): ",
                        html.Code("Mutagenic"), " / ", html.Code("Non-mutagenic"),
                        ". Training data: the curated Hansen benchmark (~6,500 compounds with "
                        "experimental Ames results)."]),

                html.H4("2 · Unambiguous algorithm"),
                html.P("Statistical method: Graph Isomorphism Network (GIN) — 5 message-passing "
                       "layers, hidden dimension 64, one-hot atomic-number node features, "
                       "sum-pooled layer readouts — trained as an ensemble of 3 seeds. "
                       "Predictions are the softmax of the averaged logits; the ensemble spread "
                       "is reported as the ± confidence."),
                html.P("Expert method: deterministic SMARTS structural alerts (nitro aromatics, "
                       "Michael acceptors, epoxides, aziridines, N-nitrosamines, alkyl sulfonate "
                       "esters) plus a one-step Phase-I metabolite screen. The two methodologies "
                       "are combined per ICH M7: a positive from EITHER method = mutagenic "
                       "concern."),

                html.H4("3 · Defined applicability domain"),
                html.P(["Each query is scored by Tanimoto similarity (Morgan fingerprints, "
                       "radius 2) to its nearest training-set neighbour: ",
                       html.Code("≥ 0.60"), " in-domain (high reliability), ",
                       html.Code("0.35–0.60"), " borderline (medium), ",
                       html.Code("< 0.35"), " out-of-domain (advisory only). Out-of-domain "
                       "verdicts are explicitly marked in the UI."]),

                html.H4("4 · Goodness of fit & robustness"),
                html.P("Ensemble held-out AMES test accuracy ≈ 77%. Robustness is surfaced live "
                       "via the ensemble confidence spread (a large ± means model disagreement) "
                       "and via the interactive Generalization Test Sheet, which grades the "
                       "pipeline against a user-editable set of literature-known compounds."),

                html.H4("5 · Mechanistic interpretation"),
                html.P("Every prediction ships with atom-level Grad×Input attribution (red = "
                       "supports the verdict, blue = opposes it) and named toxicophore alerts, "
                       "making each decision auditable by a chemist."),

                html.H4("6 · Limitations"),
                html.P("• Ames endpoint only — not a substitute for in-vivo genotoxicity assays "
                       "(micronucleus, comet)."),
                html.P("• Metabolic activation is approximated by one-step rules; multi-step "
                       "bioactivation is not modelled."),
                html.P("• No stereochemistry, salt, or pH-dependent speciation handling."),
                html.P("• Educational/demo tool — NOT validated for regulatory submissions "
                       "(no external benchmark audit, no calibration guarantees)."),
            ],
        ),
        html.Footer("Educational demo — not for regulatory decision-making. "
                    "Pipeline: 3-model GIN ensemble (AMES) + expert alerts + one-step metabolism."),

        # Hidden components for state and download
        dcc.Store(id='prediction-store'),
        dcc.Download(id="download-report"),
    ],
)

# ---- callbacks ---------------------------------------------------------
@app.callback(Output("tab-explorer", "style"), Output("tab-sheet", "style"),
              Output("tab-docs", "style"),
              Input("tabs", "value"))
def show_tab(tab):
    return ({"display": "block" if tab == "explorer" else "none"},
            {"display": "block" if tab == "sheet" else "none"},
            {"display": "block" if tab == "docs" else "none"})

@app.callback(Output("smiles-input", "value"),
              Input({"type": "example-btn", "index": ALL}, "n_clicks"),
              prevent_initial_call=True)
def fill_example(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return no_update
    return EXAMPLES[ctx.triggered_id["index"]][1]

@app.callback(
    Output("mol-wrap", "style"),
    Output("mol-image", "src"),
    Output("prediction-output", "children"),
    Output("prediction-output", "style"),
    Output("ad-output", "children"),
    Output("ad-output", "style"),
    Output("explanation-output", "children"),
    Output("error-output", "children"),
    Output("prediction-store", "data"),
    Input("predict-btn", "n_clicks"),
    State("smiles-input", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, smiles):
    if not smiles or not smiles.strip():
        return (no_update,) * 7 + ("Enter a SMILES string or chemical name first.", None)

    user_query = smiles.strip()
    resolved_smi, resolve_msg = resolve_to_smiles(user_query)

    if resolved_smi is None:
        return (no_update,) * 7 + (resolve_msg, None)

    try:
        edge_index, tags, mol = smiles_to_graph(resolved_smi)
    except UnsupportedAtomError as e:
        return (no_update,) * 7 + (str(e), None)
    except ValueError as e:
        return (no_update,) * 7 + (f"Couldn't parse that structure: {e}", None)

    result = predict_and_attribute(_models, edge_index, tags, num_tags=NUM_TAGS)
    alerts = find_alerts(mol)
    verdict = final_verdict(mol, result)
    gin_text = format_prediction(result["probs"], result["probs_std"], result["pred_class"])

    sim = _domain.score(mol)
    band, why = domain_label(sim)
    ad_text = f"Reliability: {band} (similarity to nearest trained compound {sim:.2f}). {why}."
    ad_style = {"marginTop": "6px", "color": _AD_COLORS[band]}

    met_lines, pro_flag = [], None
    for rule_name, met_mol, met_smi in generate_metabolites(mol):
        met_alert = metabolite_alerts(met_mol)
        p_mut = ensemble_p_mutagenic(met_smi)
        if met_alert or (p_mut is not None and p_mut >= 0.5):
            reason = met_alert[0] if met_alert else f"ensemble p(mut) {p_mut:.0%}"
            met_lines.append(f"• {rule_name} → {met_smi} (flag: {reason})")
            if pro_flag is None:
                pro_flag = (rule_name, met_smi, reason)

    if alerts:
        pred_text = f"Verdict: MUTAGENIC  ·  expert rule: {alerts[0]}  (ensemble: {gin_text})"
        expl = ("Flagged by an expert structural alert — a DNA-reactive warning structure. "
                "Under ICH M7, a positive from either method means treat as a mutagenic concern.")
    elif pro_flag:
        verdict = "Mutagenic"
        pred_text = (f"Verdict: MUTAGENIC (pro-mutagen)  ·  parent {gin_text}, "
                     f"but a predicted metabolite is DNA-reactive")
        expl = (f"Predicted pro-mutagen: one-step metabolism via {pro_flag[0]} yields "
                f"{pro_flag[1]}, a flagged reactive species ({pro_flag[2]}).")
    else:
        pred_text = f"Verdict: {verdict.upper()}  ·  {gin_text}"
        expl = "No structural alerts and no flagged one-step metabolites."

    expl += " Ensemble atom attribution: " + explain(mol, tags, result["importance"])
    if met_lines:
        expl += " Metabolite screen: " + "  ".join(met_lines)
    if resolve_msg:
        expl += f" ({resolve_msg}: {resolved_smi})"

    color = "#a13f2c" if verdict == "Mutagenic" else "#2f6b4c"
    img_src = render_molecule_png(mol, result["importance_norm"])

    store_data = {
        "query": user_query,
        "smiles": resolved_smi,
        "verdict": pred_text,
        "gin_text": gin_text,
        "ad_text": ad_text,
        "explanation": expl,
        "img_src": img_src,
    }

    return ({"display": "block", "marginTop": "16px"},
            img_src,
            pred_text, {"color": color, "marginTop": "14px",
                        "fontFamily": '"IBM Plex Sans",system-ui,sans-serif'},
            ad_text, ad_style,
            expl, "", store_data)

@app.callback(
    Output("download-report", "data"),
    Input("download-btn", "n_clicks"),
    State("prediction-store", "data"),
    prevent_initial_call=True,
)
def generate_report(n_clicks, store_data):
    if not store_data:
        return no_update

    html_content = build_report_html(store_data)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_string(html_content, filename=f"mutagenicity_report_{timestamp}.html")

@app.callback(
    Output("sheet-table", "data", allow_duplicate=True),
    Output("sheet-summary", "children"),
    Input("eval-sheet-btn", "n_clicks"),
    State("sheet-table", "data"),
    prevent_initial_call=True,
)
def evaluate_sheet(n, rows):
    new_rows, correct, total = [], 0, 0
    for row in rows:
        r = dict(row)
        smi = (row.get("smiles") or "").strip()
        try:
            edge_index, tags, mol = smiles_to_graph(smi)
            res = predict_and_attribute(_models, edge_index, tags, num_tags=NUM_TAGS)
            pred = final_verdict(mol, res)
            if pred != "Mutagenic":
                for rule_name, met_mol, met_smi in generate_metabolites(mol):
                    if metabolite_alerts(met_mol) or \
                       ((ensemble_p_mutagenic(met_smi) or 0) >= 0.5):
                        pred = "Mutagenic"
                        break
        except Exception:
            pred = "⚠ unreadable"
        r["predicted"] = pred
        exp = (row.get("expected") or "").strip()
        if exp in ("Mutagenic", "Non-mutagenic") and pred in ("Mutagenic", "Non-mutagenic"):
            total += 1
            ok = exp == pred
            correct += ok
            r["verdict"] = "✓" if ok else "✗"
        else:
            r["verdict"] = "—"
        new_rows.append(r)

    if total:
        acc = correct / total
        comment = ("Excellent generalization to unseen chemistry." if acc >= 0.8
                   else "Reasonable generalization — misses are usually Ames-vs-regulatory "
                          "disagreements (e.g. compounds mutagenic only in vivo)." if acc >= 0.6
                   else "Many of these compounds lie far outside the training domain.")
        summary = f"Sheet accuracy: {correct}/{total} correct ({acc:.0%}) — {comment}"
    else:
        summary = "Fill in the 'Expected' column to compute an accuracy score."
    return new_rows, summary

@app.callback(Output("sheet-table", "data", allow_duplicate=True),
              Input("add-row-btn", "n_clicks"),
              State("sheet-table", "data"),
              prevent_initial_call=True)
def add_row(n, rows):
    return rows + [{"name": "", "smiles": "", "expected": "", "predicted": "", "verdict": ""}]

@app.callback(Output("sheet-table", "data", allow_duplicate=True),
              Input("reset-sheet-btn", "n_clicks"),
              prevent_initial_call=True)
def reset_sheet(n):
    return [dict(r) for r in DEFAULT_SHEET]

if __name__ == "__main__":
    app.run(debug=True)
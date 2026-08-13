"""
Dash demo app: SMILES -> mutagenicity prediction + atom-level explanation.
Statistical GIN (AMES benchmark) + expert structural alerts (ICH M7).
Includes a Generalization Test Sheet for non-technical users.

Run:  python app.py
"""

import torch
from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL, no_update, dash_table

from gin.model import GIN
from atom_map import NUM_TAGS, CLASS_NAMES
from molecule import smiles_to_graph, UnsupportedAtomError
from attribution import predict_and_attribute, explain, format_prediction
from alerts import find_alerts
from render import render_molecule_png

CHECKPOINT_PATH = "checkpoint_ames.pt"

CSS = """
:root{
 --paper:#eef1e5; --paper-raised:#f8f9f1; --paper-deep:#e4e8d9;
 --ink:#1d2a20; --ink-soft:#5c6b5e; --ink-faint:#8b9a8d;
 --line:rgba(29,42,32,.18); --line-soft:rgba(29,42,32,.10);
 --moss:#2f6b4c; --moss-bg:rgba(47,107,76,.11);
 --brick:#a13f2c; --brick-bg:rgba(161,63,44,.11);
 --amber:#a97a1f; --amber-bg:rgba(169,122,31,.13);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font-family:"IBM Plex Sans",system-ui,sans-serif;
 background-image:radial-gradient(circle at 1px 1px, rgba(29,42,32,.13) 1px, transparent 0);
 background-size:24px 24px}
.wrap{max-width:1020px;margin:0 auto;padding:28px 20px 60px}
.hero{position:relative;padding:30px 28px 26px;border:1px solid var(--line);border-radius:4px;
 background:var(--paper-raised);overflow:hidden;
 box-shadow:0 1px 0 var(--paper-deep) inset}
.hero::before{content:"";position:absolute;inset:0;pointer-events:none;
 background-image:radial-gradient(circle at 1px 1px, rgba(29,42,32,.10) 1.2px, transparent 0);
 background-size:16px 16px;opacity:.6}
.hero::after{content:"";display:block;position:absolute;left:0;right:0;bottom:0;height:5px;
 background:linear-gradient(90deg,var(--moss) 0%,var(--moss) 32%,var(--amber) 32%,var(--amber) 66%,var(--brick) 66%,var(--brick) 100%)}
.hero-badge{position:relative;display:inline-block;font-family:"IBM Plex Mono",monospace;
 font-size:10.5px;letter-spacing:.12em;font-weight:600;color:var(--ink-soft);
 border:1px dashed var(--line);border-radius:2px;padding:5px 11px;
 background:var(--paper);text-transform:uppercase}
.hero h1{position:relative;margin:16px 0 8px;font-family:"Space Grotesk",system-ui,sans-serif;
 font-size:32px;font-weight:600;letter-spacing:-.01em;color:var(--ink)}
.hero h1 span{color:var(--amber)}
.hero p{position:relative;margin:0;color:var(--ink-soft);max-width:760px;line-height:1.6;font-size:14px}
.stats{position:relative;display:flex;gap:10px;margin-top:20px;flex-wrap:wrap}
.stat{flex:1;min-width:160px;background:var(--paper);border:1px solid var(--line);border-radius:3px;
 padding:11px 14px;border-top:2px solid var(--ink-faint)}
.stat b{display:block;font-family:"IBM Plex Mono",monospace;font-size:17px;font-weight:600;color:var(--ink)}
.stat span{font-size:11.5px;color:var(--ink-soft)}
.card{background:var(--paper-raised);border:1px solid var(--line);border-radius:4px;padding:22px;margin-top:16px}
.card h3{margin:0 0 4px;font-family:"Space Grotesk",system-ui,sans-serif;font-size:16px;
 font-weight:600;color:var(--ink)}
.hint{color:var(--ink-soft);font-size:12.5px;margin:0 0 16px;line-height:1.55}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input.smiles{flex:1;min-width:240px;background:var(--paper);border:1px solid var(--line);color:var(--ink);
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
.chips button{background:var(--paper);border:1px solid var(--line);color:var(--ink-soft);
 border-radius:999px;font-family:"IBM Plex Mono",monospace;
 padding:6px 13px;font-size:11.5px;cursor:pointer}
.chips button:hover{border-color:var(--amber);color:var(--amber);background:var(--amber-bg)}
.mol-card{background:#fdfdfa;border:1px solid var(--line);border-radius:3px;padding:12px;margin-top:18px}
.expl{color:var(--ink-soft);font-size:13.5px;line-height:1.65;margin-top:10px}
.tabs .tab{background:transparent;border:1px solid var(--line);border-bottom:2px solid var(--paper);
 color:var(--ink-soft);font-family:"IBM Plex Mono",monospace;border-radius:3px 3px 0 0;
 padding:10px 18px;font-weight:600;font-size:12.5px;letter-spacing:.02em;cursor:pointer}
.tabs .tab--selected{background:var(--paper-raised);color:var(--ink);border-color:var(--line);
 border-bottom:2px solid var(--amber)}
.summary{margin-top:16px;font-family:"IBM Plex Mono",monospace;font-size:14px;font-weight:600;color:var(--ink)}
footer{color:var(--ink-faint);font-family:"IBM Plex Mono",monospace;font-size:11px;
 margin-top:30px;padding-top:16px;border-top:1px dashed var(--line);text-align:center;
 letter-spacing:.03em}
"""

EXAMPLES = [
    ("Acrylamide (Michael acceptor)", "C=CC(=O)N"),
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

# ---- load model once at startup --------------------------------------
_ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
_model = GIN(**_ckpt["config"])
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()
assert _ckpt["tag_list"] == sorted(range(NUM_TAGS))

def final_verdict(mol, result):
    """ICH M7: positive from EITHER the GIN or the expert alerts = Mutagenic."""
    return "Mutagenic" if find_alerts(mol) else CLASS_NAMES[result["pred_class"]]

# ---- app ---------------------------------------------------------------
app = Dash(__name__)
app.title = "Molecular Mutagenicity Explainer"

# Inject the stylesheet (html.Style doesn't exist in this Dash version)
FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Space+Grotesk:wght@500;600;700'
    '&family=IBM+Plex+Sans:wght@400;500'
    '&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
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
                html.Div("GIN · AMES benchmark · ICH M7 two-methodology", className="hero-badge"),
                html.H1(["Molecular ", html.Span("Mutagenicity"), " Explainer"]),
                html.P(
                    "Type any chemical (as SMILES) or use the test sheet below. The verdict "
                    "combines a graph neural network trained on ~6,500 Ames mutagenicity "
                    "experiments with expert structural-alert rules — the same two-methodology "
                    "approach used in regulatory safety assessment (ICH M7)."),
                html.Div(className="stats", children=[
                    html.Div(className="stat", children=[html.B("~6,500"), html.Span("molecules in training set (AMES)")]),
                    html.Div(className="stat", children=[html.B("~77%"), html.Span("held-out test accuracy")]),
                    html.Div(className="stat", children=[html.B("2"), html.Span("methods: GIN + expert alerts")]),
                ]),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            className="tabs",
            value="explorer",
            children=[
                dcc.Tab(label="🔬 Molecule Explorer", value="explorer"),
                dcc.Tab(label="📋 Generalization Test Sheet", value="sheet"),
            ],
        ),
        # ---------------- Tab 1: explorer ----------------
        html.Div(
            id="tab-explorer",
            className="card",
            children=[
                html.H3("Explore a single molecule"),
                html.P("Paste a SMILES string (or click a chip), then press Predict. "
                       "Red atoms pushed toward the verdict, blue away from it.", className="hint"),
                html.Div(className="row", children=[
                    dcc.Input(id="smiles-input", type="text", value="", className="smiles",
                              placeholder="e.g. C=CC(=O)N"),
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
                html.P(id="explanation-output", className="expl"),
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
                       "'Evaluate sheet'. The app grades the model against your sheet and reports "
                       "an accuracy score — a quick check of how well it generalizes to chemistry "
                       "it was never trained on.", className="hint"),
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
                        style_header={"backgroundColor": "#e4e8d9", "color": "#1d2a20",
                                      "border": "1px solid rgba(29,42,32,.18)", "fontWeight": "600",
                                      "fontFamily": '"IBM Plex Mono",monospace',
                                      "fontSize": "12px", "letterSpacing": ".02em",
                                      "textTransform": "uppercase", "padding": "10px"},
                        style_cell={"backgroundColor": "#f8f9f1", "color": "#1d2a20",
                                    "border": "1px solid rgba(29,42,32,.18)", "padding": "9px 12px",
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
        html.Footer("Educational demo — not for regulatory decision-making. "
                    "Statistical model: GIN on AMES; expert layer: structural alerts."),
    ],
)

# ---- callbacks ---------------------------------------------------------
@app.callback(Output("tab-explorer", "style"), Output("tab-sheet", "style"),
              Input("tabs", "value"))
def show_tab(tab):
    return ({"display": "block" if tab == "explorer" else "none"},
            {"display": "block" if tab == "sheet" else "none"})

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
    Output("explanation-output", "children"),
    Output("error-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("smiles-input", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, smiles):
    if not smiles or not smiles.strip():
        return no_update, no_update, no_update, no_update, no_update, "Enter a SMILES string first."
    try:
        edge_index, tags, mol = smiles_to_graph(smiles.strip())
    except UnsupportedAtomError as e:
        return no_update, no_update, no_update, no_update, no_update, str(e)
    except ValueError as e:
        return no_update, no_update, no_update, no_update, no_update, f"Couldn't parse that SMILES: {e}"

    result = predict_and_attribute(_model, edge_index, tags, num_tags=NUM_TAGS)
    alerts = find_alerts(mol)
    verdict = final_verdict(mol, result)
    gin_text = format_prediction(result["probs"], result["pred_class"])

    if alerts:
        pred_text = f"Verdict: MUTAGENIC  ·  expert rule: {alerts[0]}  (GIN alone: {gin_text})"
        expl = ("Flagged by an expert structural alert — a DNA-reactive warning structure. "
                "Under ICH M7, a positive from either method means treat as a mutagenic concern. "
                "GIN atom attribution: " + explain(mol, tags, result["importance"]))
    else:
        pred_text = f"Verdict: {verdict.upper()}  ·  {gin_text}"
        expl = explain(mol, tags, result["importance"])

    color = "#a13f2c" if verdict == "Mutagenic" else "#2f6b4c"
    return ({"display": "block", "marginTop": "16px"},
            render_molecule_png(mol, result["importance_norm"]),
            pred_text, {"color": color, "marginTop": "14px",
                        "fontFamily": '"Space Grotesk",system-ui,sans-serif'}, expl, "")

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
            res = predict_and_attribute(_model, edge_index, tags, num_tags=NUM_TAGS)
            pred = final_verdict(mol, res)
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
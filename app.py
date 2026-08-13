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
*{box-sizing:border-box}
body{margin:0;background:#161311;color:#f4ede4;font-family:"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:28px 20px 60px}
.hero{padding:26px;border:1px solid #3a322c;border-radius:16px;background:
 radial-gradient(1100px 300px at 0% 0%,rgba(255,122,26,.16),transparent 60%),
 radial-gradient(900px 260px at 100% 0%,rgba(255,93,93,.12),transparent 60%),#201b18}
.hero-badge{display:inline-block;font-size:11px;letter-spacing:.14em;font-weight:700;
 color:#ffb347;border:1px solid rgba(255,179,71,.45);border-radius:999px;padding:4px 12px;
 background:rgba(255,122,26,.08);text-transform:uppercase}
.hero h1{margin:12px 0 6px;font-size:30px}
.hero h1 span{color:#ff7a1a}
.hero p{margin:0;color:#b3a89b;max-width:780px;line-height:1.55;font-size:14px}
.stats{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap}
.stat{flex:1;min-width:160px;background:#28221e;border:1px solid #3a322c;border-radius:12px;padding:10px 14px}
.stat b{display:block;font-size:16px;color:#ffb347}
.stat span{font-size:12px;color:#b3a89b}
.card{background:#201b18;border:1px solid #3a322c;border-radius:16px;padding:20px;margin-top:18px}
.card h3{margin:0 0 4px;font-size:16px;color:#ff7a1a}
.hint{color:#b3a89b;font-size:12.5px;margin:0 0 14px;line-height:1.5}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input.smiles{flex:1;min-width:240px;background:#161311;border:1px solid #3a322c;color:#f4ede4;
 border-radius:10px;padding:11px 14px;font-size:14px;outline:none}
input.smiles:focus{border-color:#ff7a1a}
button.btn{background:linear-gradient(135deg,#ff7a1a,#ff5d5d);border:none;color:#1b120c;
 font-weight:800;border-radius:10px;padding:11px 18px;cursor:pointer;font-size:14px}
button.btn:hover{filter:brightness(1.12)}
button.btn.ghost{background:transparent;color:#ffb347;border:1px solid rgba(255,179,71,.4)}
.chips{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
.chips button{background:#28221e;border:1px solid #3a322c;color:#b3a89b;border-radius:999px;
 padding:6px 12px;font-size:12px;cursor:pointer}
.chips button:hover{border-color:#ff7a1a;color:#ffb347}
.mol-card{background:#fff;border-radius:14px;padding:10px;margin-top:16px}
.expl{color:#b3a89b;font-size:13.5px;line-height:1.6;margin-top:10px}
.tabs .tab{background:#201b18;border:1px solid #3a322c;color:#b3a89b;border-radius:10px 10px 0 0;
 padding:10px 18px;font-weight:700;cursor:pointer;border-bottom:none}
.tabs .tab--selected{background:#28221e;color:#ffb347;border-color:#ff7a1a}
.summary{margin-top:14px;font-size:15px;font-weight:700;color:#ffb347}
footer{color:#7d7367;font-size:12px;margin-top:26px;text-align:center}
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
app.index_string = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}"
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
                html.Div(id="error-output", style={"color": "#ff5d5d", "marginTop": "10px"}),
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
                        style_header={"backgroundColor": "#28221e", "color": "#ffb347",
                                      "border": "1px solid #3a322c", "fontWeight": "700",
                                      "fontSize": "13px", "padding": "10px"},
                        style_cell={"backgroundColor": "#201b18", "color": "#f4ede4",
                                    "border": "1px solid #3a322c", "padding": "9px 12px",
                                    "fontSize": "13px", "textAlign": "left", "minWidth": "90px"},
                        style_data_conditional=[
                            {"if": {"column_id": "verdict", "filter_query": '{verdict} = "✓"'},
                             "color": "#4cd478", "fontWeight": "800"},
                            {"if": {"column_id": "verdict", "filter_query": '{verdict} = "✗"'},
                             "color": "#ff5d5d", "fontWeight": "800"},
                            {"if": {"column_id": "predicted"}, "color": "#ffb347"},
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

    color = "#ff5d5d" if verdict == "Mutagenic" else "#4cd478"
    return ({"display": "block", "marginTop": "16px"},
            render_molecule_png(mol, result["importance_norm"]),
            pred_text, {"color": color, "marginTop": "14px"}, expl, "")

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
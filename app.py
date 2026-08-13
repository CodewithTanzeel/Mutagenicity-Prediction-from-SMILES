"""
Dash demo app: SMILES -> mutagenicity prediction + atom-level explanation.

Statistical model: GIN trained on the AMES mutagenicity benchmark (~6,500
molecules). Expert layer: rule-based structural alerts (ICH M7 second
methodology) flag DNA-reactive warning structures (Michael acceptors,
epoxides, ...). A positive from EITHER method = mutagenic concern.

Run:
    python train_ames.py     # once, produces checkpoint_ames.pt
    python app.py
"""

import torch
from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL, no_update

from gin.model import GIN
from atom_map import NUM_TAGS, CLASS_NAMES
from molecule import smiles_to_graph, UnsupportedAtomError
from attribution import predict_and_attribute, explain, format_prediction
from alerts import find_alerts
from render import render_molecule_png

CHECKPOINT_PATH = "checkpoint_ames.pt"

EXAMPLES = [
    ("Acrylamide (Michael acceptor)", "C=CC(=O)N"),
    ("Benzene (simple aromatic)", "c1ccccc1"),
    ("Chlorobenzene", "c1ccc(Cl)cc1"),
    ("Aniline", "c1ccc(cc1)N"),
    ("Nitrobenzene (nitro group)", "c1ccc(cc1)[N+](=O)[O-]"),
    ("2,4-Dinitrotoluene (known mutagen)",
     "Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]"),
    ("Ethanol (non-mutagen)", "CCO"),
]

# ---- load model once at startup --------------------------------------
_ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
_model = GIN(**_ckpt["config"])
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()
assert _ckpt["tag_list"] == sorted(range(NUM_TAGS)), (
    "checkpoint's tag_list doesn't match atom_map.py's expected 0..118 -- "
    "don't trust predictions until this is resolved."
)

# ---- app ---------------------------------------------------------------
app = Dash(__name__)
app.title = "Molecular Mutagenicity Explainer"

app.layout = html.Div(
    style={"maxWidth": "780px", "margin": "40px auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("Molecular Mutagenicity Explainer"),
        html.P(
            "Enter a SMILES string, or click an example. Mutagenicity is judged "
            "by a GIN trained on the AMES benchmark, cross-checked with expert "
            "structural alerts (ICH M7 two-methodology): a flag from EITHER "
            "method means treat as a mutagenic concern. Atoms are highlighted by "
            "gradient attribution (red = pushed toward the prediction, "
            "blue = pushed away)."
        ),
        dcc.Input(
            id="smiles-input", type="text", value="",
            placeholder="e.g. C=CC(=O)N",
            style={"width": "70%", "padding": "8px"},
        ),
        html.Button("Predict", id="predict-btn", n_clicks=0,
                    style={"marginLeft": "8px", "padding": "8px 16px"}),
        html.Div(
            style={"marginTop": "12px"},
            children=[
                html.Button(name, id={"type": "example-btn", "index": i},
                            n_clicks=0,
                            style={"marginRight": "6px", "marginBottom": "6px",
                                   "fontSize": "12px"})
                for i, (name, _) in enumerate(EXAMPLES)
            ],
        ),
        html.Hr(),
        html.Div(id="error-output", style={"color": "#b00020"}),
        html.Img(id="mol-image", style={"maxWidth": "100%"}),
        html.H4(id="prediction-output"),
        html.P(id="explanation-output"),
    ],
)


@app.callback(
    Output("smiles-input", "value"),
    Input({"type": "example-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def fill_example(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return no_update
    triggered_id = ctx.triggered_id
    idx = triggered_id["index"]
    return EXAMPLES[idx][1]


@app.callback(
    Output("mol-image", "src"),
    Output("prediction-output", "children"),
    Output("explanation-output", "children"),
    Output("error-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("smiles-input", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, smiles):
    if not smiles or not smiles.strip():
        return None, "", "", "Enter a SMILES string first."

    try:
        edge_index, tags, mol = smiles_to_graph(smiles.strip())
    except UnsupportedAtomError as e:
        return None, "", "", str(e)
    except ValueError as e:
        return None, "", "", f"Couldn't parse that SMILES: {e}"

    result = predict_and_attribute(_model, edge_index, tags, num_tags=NUM_TAGS)
    img_src = render_molecule_png(mol, result["importance_norm"])

    # ICH M7 two-methodology: statistical GIN + expert structural alerts
    alerts = find_alerts(mol)
    gin_text = format_prediction(result["probs"], result["pred_class"])

    if alerts:
        prediction_text = (
            f"Mutagenic (expert rule: {alerts[0]}; GIN alone: {gin_text})"
        )
        explanation_text = (
            f"Flagged by expert structural alert: {alerts[0]} -- a DNA-reactive "
            f"warning structure. Under ICH M7, a positive from either the "
            f"statistical model or the expert rules means treat as a mutagenic "
            f"concern. GIN atom attribution: "
            + explain(mol, tags, result["importance"])
        )
    else:
        prediction_text = gin_text
        explanation_text = explain(mol, tags, result["importance"])

    return img_src, prediction_text, explanation_text, ""


if __name__ == "__main__":
    app.run(debug=True)
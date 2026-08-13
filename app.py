"""
Dash demo app: SMILES -> mutagenicity prediction + atom-level explanation.

Not run/verified in the authoring session (no working Dash/torch install
available there -- see conversation). Run locally:
    python train_and_export.py     # once, produces checkpoint.pt
    python app.py

Things worth checking yourself once it's running, since I couldn't:
  - that highlight colors render sensibly (RDKit highlightAtomRadii/Colors
    API has changed across versions -- pin rdkit version if it errors)
  - that CLASS_NAMES in atom_map.py (0=Non-mutagenic, 1=Mutagenic) matches
    your own understanding of the label convention
"""

import torch
from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL, no_update

from gin.model import GIN
from atom_map import NUM_TAGS, CLASS_NAMES
from molecule import smiles_to_graph, UnsupportedAtomError
from attribution import predict_and_attribute, explain, format_prediction
from render import render_molecule_png

CHECKPOINT_PATH = "checkpoint.pt"

EXAMPLES = [
    ("Benzene (simple aromatic)", "c1ccccc1"),
    ("Chlorobenzene", "c1ccc(Cl)cc1"),
    ("Aniline", "c1ccc(cc1)N"),
    ("Nitrobenzene (nitro group)", "c1ccc(cc1)[N+](=O)[O-]"),
    ("2,4-Dinitrotoluene (known mutagen -- verify SMILES locally)",
     "Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]"),
]

# ---- load model once at startup --------------------------------------
_ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
_model = GIN(**_ckpt["config"])
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()
assert _ckpt["tag_list"] == sorted(range(NUM_TAGS)), (
    "checkpoint's tag_list doesn't match atom_map.py's expected 0..6 -- "
    "don't trust predictions until this is resolved."
)

# ---- app ---------------------------------------------------------------
app = Dash(__name__)
app.title = "MUTAG GIN Explainer"

app.layout = html.Div(
    style={"maxWidth": "780px", "margin": "40px auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("Molecular Mutagenicity Explainer"),
        html.P(
            "Enter a SMILES string, or click an example. Predicts mutagenicity "
            "using a GIN trained on MUTAG, and highlights which atoms drove the "
            "prediction (red = pushed toward it, blue = pushed away)."
        ),
        dcc.Input(
            id="smiles-input", type="text", value="",
            placeholder="e.g. c1ccc(cc1)[N+](=O)[O-]",
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


# Example buttons use Dash's pattern-matching callback API (ALL) since
# there's a variable number of them, generated from the EXAMPLES list.
@app.callback(
    Output("smiles-input", "value"),
    Input({"type": "example-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def fill_example(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return no_update
    triggered_id = ctx.triggered_id  # dict like {"type": "example-btn", "index": i}
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
    prediction_text = format_prediction(result["probs"], result["pred_class"])
    explanation_text = explain(mol, tags, result["importance"])

    return img_src, prediction_text, explanation_text, ""


if __name__ == "__main__":
    app.run(debug=True)

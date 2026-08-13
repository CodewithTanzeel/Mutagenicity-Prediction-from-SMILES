"""
Render a molecule with atoms colored by attribution score.

Deliberately avoids matplotlib as a dependency (one less thing to install
under a 24h deadline) -- the diverging blue/white/red colormap below is a
small hand-rolled linear interpolation instead.
"""

import base64
from io import BytesIO

from rdkit.Chem.Draw import rdMolDraw2D


# diverging colormap: negative (pushes away from prediction) -> blue,
# neutral -> white, positive (supports prediction) -> red
_NEG = (0.20, 0.40, 0.85)
_NEU = (1.00, 1.00, 1.00)
_POS = (0.85, 0.20, 0.20)


def _color_for(value_0_to_1):
    """value: 0.0 = most negative, 0.5 = neutral, 1.0 = most positive."""
    if value_0_to_1 < 0.5:
        t = value_0_to_1 / 0.5
        a, b = _NEG, _NEU
    else:
        t = (value_0_to_1 - 0.5) / 0.5
        a, b = _NEU, _POS
    return tuple(a[i] + t * (b[i] - a[i]) for i in range(3))


def render_molecule_png(mol, importance_norm, width=450, height=400):
    """importance_norm: [N] float tensor/list in [0, 1] (0.5 = neutral).
    Returns a base64-encoded PNG string, ready for html.Img(src=...)."""
    highlight_atoms = list(range(mol.GetNumAtoms()))
    highlight_colors = {
        i: _color_for(float(importance_norm[i])) for i in highlight_atoms
    }
    # scale marker radius slightly with |importance| so strong atoms pop visually
    highlight_radii = {
        i: 0.3 + 0.25 * abs(float(importance_norm[i]) - 0.5) * 2
        for i in highlight_atoms
    }

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"

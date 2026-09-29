"""Shared visual identity: supermarket shelf-edge labels.

Slate ink on a cool paper background, with shelf-tag yellow reserved for the
one thing that needs attention. Risk colours are used only for risk.
"""
import matplotlib as mpl

PAPER = "#F4F6F3"
INK = "#1B2A2F"
MUTED = "#66777C"
RULE = "#D5DBD8"
TAG = "#FFD23F"          # shelf-edge label yellow
TEAL = "#1F5F6B"
SAGE = "#8DB3A8"
HIGH = "#C8332F"
MEDIUM = "#D9822B"
LOW = "#2E7D5B"
SERIES = [TEAL, "#C99A1E", SAGE, "#4B6F8A", "#A45A52", "#6E8B3D", "#8A6FA0"]
RISK_COLORS = {"HIGH": HIGH, "MEDIUM": MEDIUM, "LOW": LOW}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": RULE, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.titlelocation": "left", "font.size": 10, "figure.dpi": 110,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })

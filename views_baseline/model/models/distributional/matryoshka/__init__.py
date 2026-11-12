"""MatryoshkaConflictology (ADR-024): one idea per module.

geometry         the grid and each cell's neighbours and country
size_classes     the seven size classes the knobs are fitted on
doll_counts      per-month class counts of each cell's four dolls (sparse)
forecast_probability  exact class probabilities, the fitting score and its slope
knob_fit         the backtest search for the knobs per target (8 with all four dolls)
draw             the three-step draw used by predict
model            the model class
"""

from views_baseline.model.models.distributional.matryoshka.model import MatryoshkaConflictology

__all__ = ["MatryoshkaConflictology"]

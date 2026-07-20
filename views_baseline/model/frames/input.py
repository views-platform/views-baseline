"""The one input boundary: normalize model input to a ``views_frames.FeatureFrame``.

This is the **single pandas reader** for the model layer (the DIP boundary of ADR-019).
Models call :func:`to_feature_frame` at entry so their internals depend only on the
``FeatureFrame`` abstraction, never on the concrete pandas ``MultiIndex`` API:

* a ``pandas.DataFrame`` is lifted to a ``FeatureFrame`` here (pandas is confined to this
  branch, and imported lazily so the FeatureFrame path is genuinely pandas-free);
* a ``FeatureFrame`` passes through after a declared-``loa`` ↔ frame-``level`` agreement
  check (ADR-003 — declarations are authoritative).

:func:`panel` is the numpy view the ported model internals read: ``(time, unit,
{target -> (N,) values})``, extracted from the frame's arrays with no pandas.

The ``views_frames`` leaf is imported **lazily inside the functions** (mirroring the
``frames/output.py`` seam) so ``model/`` stays importable without the platform leaf
present (ADR-002/ADR-013).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.spatial import resolve_level, spatial_level

if TYPE_CHECKING:
    from views_frames import FeatureFrame


def to_feature_frame(x, *, loa: str, targets: list[str]) -> "FeatureFrame":
    """Normalize model input to a ``FeatureFrame`` (the ADR-019 input boundary).

    Args:
        x: Either a ``pandas.DataFrame`` indexed by ``(time, entity)`` with the
            ``targets`` as columns, or an already-built ``FeatureFrame``.
        loa: Declared level of analysis (``"cm"`` / ``"pgm"``) — authoritative (ADR-003).
        targets: Target/feature column names the model consumes.

    Returns:
        A ``FeatureFrame`` whose features are ``targets`` (in order) and whose
        ``float64`` values carry the observed data as a ``(N, F, 1)`` block.

    Raises:
        ValueError: if a passed ``FeatureFrame``'s ``level`` disagrees with ``loa``, if
            required ``targets`` are absent, or (DataFrame) if the index names do not
            match the declared level.
        TypeError: if ``x`` is neither a ``FeatureFrame`` nor a ``pandas.DataFrame``.
    """
    from views_frames import FeatureFrame

    if isinstance(x, FeatureFrame):
        return _validate_feature_frame(x, loa=loa, targets=targets)

    # Lazy pandas import — the FeatureFrame path above never needs pandas (DIP: the
    # concrete data library is a detail confined to this one branch).
    import pandas as pd

    if isinstance(x, pd.DataFrame):
        return _from_dataframe(x, loa=loa, targets=targets)

    raise TypeError(
        f"to_feature_frame expects a pandas.DataFrame or a views_frames.FeatureFrame, "
        f"got {type(x).__name__}."
    )


def _validate_feature_frame(ff: "FeatureFrame", *, loa: str, targets: list[str]) -> "FeatureFrame":
    """Passthrough a ``FeatureFrame`` after checking loa↔level and target presence."""
    expected = spatial_level(loa)
    if ff.index.level != expected:
        raise ValueError(
            f"FeatureFrame level {ff.index.level.name} disagrees with declared loa={loa!r} "
            f"({expected.name}). Declarations are authoritative (ADR-003); the input frame "
            f"does not match the declared spatial level."
        )
    missing = [t for t in targets if t not in ff.feature_names]
    if missing:
        raise ValueError(
            f"FeatureFrame is missing required target feature(s) {missing}; "
            f"it carries {list(ff.feature_names)}."
        )
    return ff


def _from_dataframe(df, *, loa: str, targets: list[str]) -> "FeatureFrame":
    """Lift a ``(time, entity)``-indexed DataFrame to a ``FeatureFrame`` (pandas boundary)."""
    from views_frames import FeatureFrame, SpatioTemporalIndex

    level = resolve_level(loa, df.index.names)

    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required target column(s) {missing}; "
            f"it has columns {list(df.columns)}."
        )

    time = df.index.get_level_values(0).to_numpy(dtype=np.int64)
    unit = df.index.get_level_values(1).to_numpy(dtype=np.int64)
    # float64 block matching the downstream numpy pools (byte-identity with the pandas path).
    block = df[list(targets)].to_numpy(dtype=np.float64)
    index = SpatioTemporalIndex(time=time, unit=unit, level=level)
    return FeatureFrame.from_2d(block, index, list(targets))


def panel(ff: "FeatureFrame", targets: list[str]) -> tuple:
    """Return the numpy view the ported model internals read from a ``FeatureFrame``.

    Args:
        ff: The input ``FeatureFrame`` (observed data — a single sample per cell).
        targets: The target features to extract, in the caller's order.

    Returns:
        ``(time, unit, values)`` where ``time`` and ``unit`` are the ``(N,)`` int
        identifier arrays and ``values[target]`` is the ``(N,)`` ``float64`` observed
        series for that target (the ``S == 1`` observed slice of ``ff.values``).

    Raises:
        ValueError: if a requested target is not among the frame's features.
    """
    feature_names = list(ff.feature_names)
    values = {}
    for t in targets:
        if t not in feature_names:
            raise ValueError(
                f"target {t!r} is not a feature of the FeatureFrame {feature_names}."
            )
        col = ff.values[:, feature_names.index(t), :]  # (N, S) — FeatureFrame is float32
        # Upcast to float64 so downstream windowing/fitting matches the float64 pandas
        # path's arithmetic. NOTE: FeatureFrame storage is float32 by design, so the
        # observed magnitudes are already float32-rounded — upcasting does not restore
        # precision. This is the crux of the byte-identity limit (see C-32 status).
        values[t] = col[:, 0].astype(np.float64)
    return ff.index.time, ff.index.unit, values

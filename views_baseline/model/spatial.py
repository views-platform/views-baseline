from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from views_frames import SpatialLevel


def spatial_level(loa: str) -> "SpatialLevel":
    """Parse a declared level-of-analysis string into a views-frames ``SpatialLevel``.

    The single-sourced ``loa`` → ``SpatialLevel`` parse (ADR-003/ADR-020). Both
    :func:`resolve_level` (DataFrame input, which also cross-checks the index) and the
    ``to_feature_frame`` boundary adapter (FeatureFrame input, which checks the frame's
    own ``level``) depend on this one parser rather than re-implementing it.

    Args:
        loa: Declared level of analysis — the ``SpatialLevel`` value (``"cm"`` or
            ``"pgm"``), as carried in ``config["level"]``.

    Returns:
        The resolved ``SpatialLevel``.

    Raises:
        ValueError: if ``loa`` is not a known level.
    """
    from views_frames import SpatialLevel

    try:
        return SpatialLevel(loa)
    except ValueError:
        valid = [lvl.value for lvl in SpatialLevel]
        raise ValueError(
            f"Unknown level of analysis loa={loa!r}; expected one of {valid} "
            f"(ADR-020 spatial-level contract)."
        ) from None


def resolve_level(loa: str, index_names) -> "SpatialLevel":
    """Resolve the declared level-of-analysis to a views-frames ``SpatialLevel``.

    The declared ``loa`` is authoritative (ADR-003); the DataFrame index is a
    cross-check that must agree. ``views_frames.SpatialLevel`` is the single
    source of truth for the ``(time, entity)`` index vocabulary per level, so we
    validate against ``level.index_names`` rather than hardcoding column names.
    This makes the required spatial ``level`` (carried into every output
    ``PredictionFrame``) a declared-and-validated quantity, not an inference from
    the positional entity index (ADR-020; closes risk C-18, mitigates C-05).

    Args:
        loa: Declared level of analysis — the ``SpatialLevel`` value (``"cm"`` or
            ``"pgm"``), as carried in ``config["level"]``.
        index_names: The DataFrame's ``index.names`` — expected ``(time, entity)``.

    Returns:
        The resolved ``SpatialLevel``.

    Raises:
        ValueError: if ``loa`` is not a known level, or the index names do not
            match the declared level's ``(time, entity)`` vocabulary.
    """
    level = spatial_level(loa)
    expected = level.index_names
    actual = tuple(index_names)
    if actual != expected:
        raise ValueError(
            f"Index does not match declared level: loa={loa!r} -> {level.name} "
            f"expects index names {expected}, but the DataFrame index is {actual}. "
            f"Declarations are authoritative (ADR-003); the input does not match "
            f"the declared spatial level."
        )
    return level

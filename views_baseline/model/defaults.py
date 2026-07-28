"""Package-level default sentinels for baseline models.

Zero-Magic / Explicit-Defaults (ADR-021): the single source of truth for the
default RNG seed. It is a **sentinel for direct/test construction only** —
production configs MUST declare ``seed`` (a required, audited genome key). The
catalog reads ``config["seed"]`` strictly and never falls back to this constant.
"""

DEFAULT_SEED = 42

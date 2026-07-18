"""Per-run observability scaffolding (Phase 2.5 — the RL debugger)."""

from anvil.observe.jlens import (
    JLENS_SCHEMA_VERSION,
    build_jlens_record,
    compute_signals,
    digitseq_hit_layers,
    intermediate_order_score,
    jlens_order_collapsed,
    solve_order_score,
)
from anvil.observe.metrics import (
    JLENS_FILENAME,
    METRICS_FILENAME,
    PROBES_FILENAME,
    SCHEMA_VERSION,
    RunMetricsWriter,
    advantage_collapsed,
    read_jsonl,
)

__all__ = [
    "METRICS_FILENAME",
    "PROBES_FILENAME",
    "JLENS_FILENAME",
    "JLENS_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "RunMetricsWriter",
    "advantage_collapsed",
    "jlens_order_collapsed",
    "build_jlens_record",
    "compute_signals",
    "digitseq_hit_layers",
    "intermediate_order_score",
    "solve_order_score",
    "read_jsonl",
]

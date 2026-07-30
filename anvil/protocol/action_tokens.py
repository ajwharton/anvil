"""Text-tokenized robot actions (Phase 4.A).

Continuous control vectors become **space-separated tokens** that a small LM
can predict with ordinary next-token CE — OpenVLA-style discrete bins, or
decimal continuous text (Bridge converter default).

This is a **recipe/data concern** (how the assistant response is spelled), not
a new training verb. Decode on the robot only after a supervisor gate.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from anvil.data.convert import format_action_text

ActionScheme = Literal["bins", "continuous"]

# OpenVLA-style defaults: 256 bins over a clipped continuous range.
DEFAULT_N_BINS = 256
DEFAULT_MIN_ACTION = -1.0
DEFAULT_MAX_ACTION = 1.0

_WS_SPLIT = re.compile(r"\s+")
_BIN_TOKEN = re.compile(r"^-?\d+$")
_FLOAT_TOKEN = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True, slots=True)
class ActionTokenizer:
    """Encode / decode robot actions as text for SFT / offline policy learning.

    Parameters
    ----------
    scheme:
        ``bins`` — discretize each dim into ``[0, n_bins)`` integer tokens
        (OpenVLA-like). ``continuous`` — decimal text via
        :func:`~anvil.data.convert.format_action_text`.
    n_bins:
        Number of bins per dimension when ``scheme="bins"``.
    min_action / max_action:
        Clip + scale range for binning. Per-dim lists override the scalar.
    decimals:
        Rounding for continuous text.
    """

    scheme: ActionScheme = "bins"
    n_bins: int = DEFAULT_N_BINS
    min_action: float | Sequence[float] = DEFAULT_MIN_ACTION
    max_action: float | Sequence[float] = DEFAULT_MAX_ACTION
    decimals: int = 4

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if self.scheme not in ("bins", "continuous"):
            raise ValueError(f"unknown scheme {self.scheme!r}")

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(self.min_action, Sequence) and not isinstance(
            self.min_action, (str, bytes)
        ):
            d["min_action"] = [float(x) for x in self.min_action]
        if isinstance(self.max_action, Sequence) and not isinstance(
            self.max_action, (str, bytes)
        ):
            d["max_action"] = [float(x) for x in self.max_action]
        return d

    @classmethod
    def from_public(cls, d: Mapping[str, Any] | None) -> ActionTokenizer:
        if not d:
            return cls()
        return cls(
            scheme=str(d.get("scheme") or "bins"),  # type: ignore[arg-type]
            n_bins=int(d.get("n_bins") or DEFAULT_N_BINS),
            min_action=_bound_field(d.get("min_action"), DEFAULT_MIN_ACTION),
            max_action=_bound_field(d.get("max_action"), DEFAULT_MAX_ACTION),
            decimals=int(d.get("decimals") or 4),
        )

    def encode(self, action: Any) -> str:
        """Action → assistant target string."""
        if action is None:
            raise ValueError("action is None")
        if isinstance(action, str):
            s = action.strip()
            if not s:
                raise ValueError("empty action string")
            # Already tokenized (bin indices or continuous text) — pass through.
            return s
        if self.scheme == "continuous":
            return format_action_text(action, decimals=self.decimals)
        vec = _as_float_vector(action)
        if not vec:
            raise ValueError(f"cannot bin empty action: {action!r}")
        bins = [self._value_to_bin(v, i) for i, v in enumerate(vec)]
        return " ".join(str(b) for b in bins)

    def decode(self, text: str) -> list[float]:
        """Assistant text → continuous vector (best-effort).

        For ``bins``, tokens are integer bin indices (bin centers). For
        ``continuous``, space-separated floats. Dict-style ``k=v`` is not
        decoded here — use continuous encode only for dict actions.
        """
        tokens = [t for t in _WS_SPLIT.split(text.strip()) if t]
        if not tokens:
            raise ValueError("empty action text")
        if self.scheme == "continuous":
            return [float(t) for t in tokens]
        out: list[float] = []
        for i, t in enumerate(tokens):
            if _BIN_TOKEN.match(t):
                out.append(self._bin_to_value(int(t), i))
            elif _FLOAT_TOKEN.match(t):
                out.append(float(t))
            else:
                raise ValueError(f"unparseable action token: {t!r}")
        return out

    def encode_many(self, actions: Sequence[Any]) -> list[str]:
        return [self.encode(a) for a in actions]

    def _bounds_for_dim(self, dim: int) -> tuple[float, float]:
        lo = self.min_action
        hi = self.max_action
        if isinstance(lo, Sequence) and not isinstance(lo, (str, bytes)):
            lo_f = float(lo[min(dim, len(lo) - 1)])
        else:
            lo_f = float(lo)
        if isinstance(hi, Sequence) and not isinstance(hi, (str, bytes)):
            hi_f = float(hi[min(dim, len(hi) - 1)])
        else:
            hi_f = float(hi)
        if hi_f <= lo_f:
            raise ValueError(f"max_action must be > min_action for dim {dim}: {lo_f} >= {hi_f}")
        return lo_f, hi_f

    def _value_to_bin(self, value: float, dim: int) -> int:
        lo, hi = self._bounds_for_dim(dim)
        v = min(hi, max(lo, float(value)))
        # Map [lo, hi] → [0, n_bins-1]
        t = (v - lo) / (hi - lo)
        idx = int(t * self.n_bins)
        if idx >= self.n_bins:
            idx = self.n_bins - 1
        if idx < 0:
            idx = 0
        return idx

    def _bin_to_value(self, bin_idx: int, dim: int) -> float:
        lo, hi = self._bounds_for_dim(dim)
        # Clamp index; use bin center
        b = min(self.n_bins - 1, max(0, int(bin_idx)))
        # Uniform bins: center of bin b
        return lo + (b + 0.5) * (hi - lo) / self.n_bins


def _bound_field(raw: Any, default: float) -> float | list[float]:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return float(raw)


def _as_float_vector(action: Any) -> list[float]:
    if isinstance(action, Mapping):
        # Stable key order for dict actions (same as format_action_text).
        return [float(action[k]) for k in sorted(action.keys(), key=str)]
    if isinstance(action, (list, tuple)):
        out: list[float] = []
        for x in action:
            if isinstance(x, (int, float)):
                out.append(float(x))
            else:
                raise ValueError(f"non-numeric action element: {x!r}")
        return out
    if isinstance(action, (int, float)):
        return [float(action)]
    raise ValueError(f"unsupported action type: {type(action).__name__}")


def default_edge_tokenizer() -> ActionTokenizer:
    """Smol / edge default: 256 bins over [-1, 1] (OpenVLA-style text bins)."""
    return ActionTokenizer(scheme="bins", n_bins=DEFAULT_N_BINS)

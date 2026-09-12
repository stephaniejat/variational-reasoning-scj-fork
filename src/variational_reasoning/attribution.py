"""Where the responsibility mass comes from: declaration vs derivation, prior vs answer.

Two diagnostics that leave the update untouched.

``answer_free_prefix`` locates the longest prefix of a completion that does not
contain the answer string. Any earlier cut leaves the answer in context, so the
statistic is near zero by copying; any later cut deletes derivation tokens that
do not carry the answer, so the statistic measures truncation damage instead.
The maximal answer-free prefix is the unique cut on that boundary.

``declaration_effect`` differences a foil answer scored on the same two contexts,
so context degradation from truncation enters both bracketed terms and cancels.

``factor_attribution`` reports which of the two responsibility log-factors the
weights are actually tracking, which the summed score cannot express.

These functions accept precomputed scores, like the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .answer_events import parse_gsm8k_answer_event
from .em import _group_softmax, _groups, _mask, _vector


@dataclass(frozen=True)
class TraceSplit:
    """Character-level cut separating derivation from the answer-bearing tail."""

    prefix_end: int | None
    answer_start: int | None
    length: int
    found: bool

    @property
    def prefix_fraction(self) -> float | None:
        """Share of the completion retained. Values near zero mean the answer
        appears immediately, so the difference statistic will be dominated by
        truncation rather than by declaration reliance."""

        if not self.found or self.length == 0:
            return None
        return self.prefix_end / self.length


def answer_free_prefix(
    text: str,
    answer: object,
    *,
    mode: str = "strict_terminal_marker",
    use_marker_fallback: bool = True,
) -> TraceSplit:
    """Longest prefix of ``text`` that does not contain the answer string.

    The answer is matched with and without thousands separators. When it never
    occurs but a canonical ``####`` marker does, the marker start is used as a
    fallback cut if ``use_marker_fallback``; otherwise the split is unfound and
    the trace belongs in its own bucket rather than in the statistic.
    """

    event = parse_gsm8k_answer_event(text, mode=mode)
    forms: list[str] = []
    if answer is not None:
        plain = str(answer).replace(",", "")
        forms = [plain]
        try:
            forms.append(f"{int(plain):,}")
        except ValueError:
            pass

    starts = [text.find(form) for form in forms]
    hits = [start for start in starts if start >= 0]
    if hits:
        start = min(hits)
        return TraceSplit(
            prefix_end=start,
            answer_start=start,
            length=len(text),
            found=True,
        )
    if use_marker_fallback and event.marker_start is not None:
        return TraceSplit(
            prefix_end=event.marker_start,
            answer_start=None,
            length=len(text),
            found=True,
        )
    return TraceSplit(
        prefix_end=None,
        answer_start=None,
        length=len(text),
        found=False,
    )


def boundary_set(text: str, *, separator: str = "\n") -> list[int]:
    """B(h): admissible cut offsets. Always contains 0 and len(text)."""

    offsets = [0]
    index = text.find(separator)
    while index >= 0:
        offsets.append(index + len(separator))
        index = text.find(separator, index + len(separator))
    offsets.append(len(text))
    return sorted(set(offsets))


def maximal_admissible_cut(
    text: str,
    tau: int,
    *,
    separator: str = "\n",
) -> int:
    """c* = max{b in B(h) : b < tau}.

    The leakage-free set C = {c : answer not in text[:c]} is down-closed because
    substring containment is monotone in c, so C = {0, ..., tau - 1} and tau is a
    sufficient statistic for admissibility. Intersecting with the boundary set B
    leaves a finite non-empty totally ordered set (0 is always in both when
    tau > 0), so the maximum exists and is unique. Maximising c minimises deleted
    derivation, which is the objective; c < tau is what identifies D at all,
    since any c >= tau leaves the answer in context and D collapses to zero by
    copying for every trace.
    """

    if tau <= 0:
        return 0
    candidates = [b for b in boundary_set(text, separator=separator) if b < tau]
    return max(candidates) if candidates else 0


def declaration_effect(
    answer_logp_full: Sequence[float],
    foil_logp_full: Sequence[float],
    answer_logp_prefix: Sequence[float],
    foil_logp_prefix: Sequence[float],
) -> np.ndarray:
    """Preference for the true answer over a foil that the answer-bearing tail supplies.

    ``D = [L(a|h) - L(a'|h)] - [L(a|prefix) - L(a'|prefix)]``

    Large values mean the tail carries the answer likelihood. The foil is scored
    on both contexts, so any degradation caused by truncating mid-derivation
    affects both terms of the second bracket and cancels. Draw the foil from the
    answers declared by other retained traces for the same question.
    """

    full = _vector(answer_logp_full, "answer_logp_full")
    full_foil = _vector(foil_logp_full, "foil_logp_full")
    prefix = _vector(answer_logp_prefix, "answer_logp_prefix")
    prefix_foil = _vector(foil_logp_prefix, "foil_logp_prefix")
    if not (full.shape == full_foil.shape == prefix.shape == prefix_foil.shape):
        raise ValueError("all four score vectors must align")
    return (full - full_foil) - (prefix - prefix_foil)


def adjusted_declaration_effect(
    declaration: Sequence[float],
    control: Sequence[float],
) -> np.ndarray:
    """D at c* minus the mean of matched interior-span removals.

    Snapping to a boundary deletes tau - 1 - c* extra derivation tokens, and
    deletion can only reduce answer information, so D_snap >= D_{tau-1}: the
    statistic is biased toward the hypothesis. The control removes a span of
    identical length from the interior, estimating that inflation. Under the null
    that the tail carries no more answer information than any other span of the
    same length, the difference has zero mean.
    """

    tail = _vector(declaration, "declaration")
    interior = _vector(control, "control")
    if tail.shape != interior.shape:
        raise ValueError("declaration and control statistics must align")
    return tail - interior


def bypass_effect(
    answer_logp_full: Sequence[float],
    answer_logp_empty: Sequence[float],
) -> np.ndarray:
    """``L(a | q, h) - L(a | q)``: what the trace adds over the question alone.

    Values near zero mean the answer is conditionally independent of the trace,
    so the latent is carrying nothing. ``answer_logp_empty`` is one score per
    question, broadcast to the aligned trace layout by the caller.
    """

    full = _vector(answer_logp_full, "answer_logp_full")
    empty = _vector(answer_logp_empty, "answer_logp_empty")
    if full.shape != empty.shape:
        raise ValueError("full and empty answer scores must align")
    return full - empty


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return sums[inverse] / counts[inverse]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    a, b = _rank(left), _rank(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a**2).sum() * (b**2).sum())
    if denominator == 0:
        return float("nan")
    return float((a * b).sum() / denominator)


def factor_attribution(
    trace_logp: Sequence[float],
    answer_logp: Sequence[float],
    weights: Sequence[float],
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> dict[object, tuple[float, float]]:
    """Rank correlation of the responsibility with each of its two log-factors.

    The summed score ``trace + answer`` is a projection of the pair onto a line,
    so the weight alone cannot say which factor produced it. Returns
    ``{question: (rho_trace, rho_answer)}``. ``rho_trace`` far exceeding
    ``rho_answer`` means the responsibility is ordering traces by how typical
    they already are under the model rather than by answer support.
    """

    trace = _vector(trace_logp, "trace_logp")
    answer = _vector(answer_logp, "answer_logp")
    weight = _vector(weights, "weights")
    if not (trace.shape == answer.shape == weight.shape):
        raise ValueError("log probabilities and weights must align")
    groups = _groups(question_ids, len(trace))
    keep = _mask(active, len(trace))

    report: dict[object, tuple[float, float]] = {}
    for group in dict.fromkeys(groups.tolist()):
        local = (groups == group) & keep & np.isfinite(trace) & np.isfinite(answer)
        if local.sum() < 2:
            continue
        report[group] = (
            _spearman(weight[local], trace[local]),
            _spearman(weight[local], answer[local]),
        )
    return report


def tempered_joint_weights(
    trace_logp: Sequence[float],
    answer_logp: Sequence[float],
    question_ids: Sequence[object],
    *,
    gamma: float = 1.0,
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    """Joint responsibilities with an exponent on the rationale prior factor.

    ``w propto p(h | q) ** gamma * p(a | q, h)``

    ``gamma`` is the proposal correction, not a tuning knob. For a support drawn
    from proposal ``r``, exact self-normalised weights carry ``1 / r``. A fixed
    enumerated support makes that constant and gives ``gamma = 1``
    (:func:`joint_weights`); a question-only prior proposal cancels the prior
    factor and gives ``gamma = 0`` (:func:`pis_weights`). Writing the proposal's
    deviation from the prior as ``r ~ p(h | q) ** (1 - gamma)`` recovers this
    family exactly, so ``gamma`` estimates how far answer-hinting flattens the
    question-only prior. Fit it by regressing the proposal score on the prior
    score across the support rather than sweeping it.
    """

    trace = _vector(trace_logp, "trace_logp")
    answer = _vector(answer_logp, "answer_logp")
    if trace.shape != answer.shape:
        raise ValueError("trace_logp and answer_logp must align")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    if gamma == 0.0:
        return _group_softmax(answer, question_ids, active)
    return _group_softmax(gamma * trace + answer, question_ids, active)


def fit_gamma(
    trace_logp: Sequence[float],
    proposal_logp: Sequence[float],
    question_ids: Sequence[object],
    active: Sequence[bool] | None = None,
) -> dict[object, tuple[float, float]]:
    """Estimate gamma per question from the proposal's deviation from the prior.

    Regresses ``log r(h | q, a)`` on ``log p(h | q)`` across the support. The
    slope is ``1 - gamma``. Returns ``{question: (gamma, r_squared)}``; a low
    r-squared means no scalar exponent is an adequate correction and the explicit
    proposal density is needed instead.
    """

    trace = _vector(trace_logp, "trace_logp")
    proposal = _vector(proposal_logp, "proposal_logp")
    if trace.shape != proposal.shape:
        raise ValueError("trace_logp and proposal_logp must align")
    groups = _groups(question_ids, len(trace))
    keep = _mask(active, len(trace))

    report: dict[object, tuple[float, float]] = {}
    for group in dict.fromkeys(groups.tolist()):
        local = (groups == group) & keep & np.isfinite(trace) & np.isfinite(proposal)
        if local.sum() < 2:
            continue
        x, y = trace[local], proposal[local]
        xc, yc = x - x.mean(), y - y.mean()
        variance = (xc**2).sum()
        if variance == 0:
            continue
        slope = float((xc * yc).sum() / variance)
        residual = yc - slope * xc
        total = (yc**2).sum()
        r_squared = 1.0 if total == 0 else float(1 - (residual**2).sum() / total)
        report[group] = (1.0 - slope, r_squared)
    return report

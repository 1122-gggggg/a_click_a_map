from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from statistics import NormalDist


@dataclass(frozen=True)
class ProportionInterval:
    """Closed confidence interval for a binomial proportion."""

    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate_count(events: int, trials: int) -> None:
    if isinstance(events, bool) or isinstance(trials, bool):
        raise ValueError("events and trials must be integers")
    if not isinstance(events, int) or not isinstance(trials, int):
        raise ValueError("events and trials must be integers")
    if trials < 0 or events < 0 or events > trials:
        raise ValueError("require 0 <= events <= trials")


def wilson_interval(
    events: int,
    trials: int,
    confidence: float = 0.95,
) -> ProportionInterval:
    """Return the Wilson score interval for a binomial event rate.

    Unlike a symmetric normal interval, the Wilson interval remains bounded
    in [0, 1] and behaves sensibly for small samples and rates near zero or one.
    """

    _validate_count(events, trials)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if trials == 0:
        return ProportionInterval(0.0, 1.0)

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = events / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return ProportionInterval(
        low=max(0.0, center - half_width),
        high=min(1.0, center + half_width),
    )


def empirical_bayes_prior(
    events: int,
    trials: int,
    *,
    strength: float = 8.0,
    pseudocount: float = 0.5,
) -> tuple[float, float]:
    """Construct a Beta prior exactly centered on a smoothed global rate.

    ``pseudocount`` is used only to keep the empirical rate away from the
    degenerate endpoints 0 and 1. ``strength`` is the total concentration of
    the returned Beta prior, so its meaning remains stable across datasets.
    """

    _validate_count(events, trials)
    if isinstance(strength, bool) or not isfinite(strength) or strength <= 0.0:
        raise ValueError("strength must be finite and > 0")
    if isinstance(pseudocount, bool) or not isfinite(pseudocount) or pseudocount <= 0.0:
        raise ValueError("pseudocount must be finite and > 0")

    rate = (events + pseudocount) / (trials + 2.0 * pseudocount)
    return strength * rate, strength * (1.0 - rate)


def beta_posterior_mean(
    events: int,
    trials: int,
    alpha: float,
    beta: float,
) -> float:
    """Return E[p | events, trials] for a Beta-Binomial model."""

    _validate_count(events, trials)
    if (
        isinstance(alpha, bool)
        or isinstance(beta, bool)
        or not isfinite(alpha)
        or not isfinite(beta)
        or alpha <= 0.0
        or beta <= 0.0
    ):
        raise ValueError("alpha and beta must be finite and > 0")
    return (events + alpha) / (trials + alpha + beta)

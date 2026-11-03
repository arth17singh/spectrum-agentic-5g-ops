"""Offline privacy accounting for the DP comparison run (needs `opacus`; imported lazily).

The FL service's framework picks the noise for a target epsilon with a Renyi-DP accountant (Opacus) and fixes
delta = (training-set size)^-2. Reading its code, the accountant is given `steps = local_epochs x rounds` with a
sampling probability q = 1 / (n // batch). If a local epoch is a full pass over the silo's data, an epoch is n // batch
optimizer steps, not one, so the same noise multiplier spends more epsilon than the service reports. This module
computes both so the declaration can state the service's number AND an independent one:

    service_epsilon(n, batch, noise_multiplier, epochs, rounds)       # what the service's own accounting gives
    standard_epsilon(n, batch, noise_multiplier, epochs, rounds)      # every optimizer step counted

Neither is a proof of the deployed behaviour: the numbers are a function of the code we read, not of a run.
"""

from __future__ import annotations


def _eps(n: int, batch: int, noise_multiplier: float, steps: int) -> float:
    from opacus.accountants import rdp as acc
    from opacus.accountants.analysis import rdp as an

    orders = acc.RDPAccountant().DEFAULT_ALPHAS
    q = 1.0 / (n // batch)
    rdp = an.compute_rdp(q=q, noise_multiplier=noise_multiplier, steps=steps, orders=orders)
    return float(an.get_privacy_spent(orders=orders, rdp=rdp, delta=float(n) ** -2)[0])


def delta_for(n: int) -> float:
    return float(n) ** -2


def service_epsilon(n: int, batch: int, noise_multiplier: float, epochs: int, rounds: int) -> float:
    return _eps(n, batch, noise_multiplier, epochs * rounds)


def standard_epsilon(n: int, batch: int, noise_multiplier: float, epochs: int, rounds: int) -> float:
    return _eps(n, batch, noise_multiplier, (n // batch) * epochs * rounds)

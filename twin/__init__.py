"""Team13 twin access layer.

ReplayTwin     deterministic in-process Maveric twin, rebuilt from (scenario, seed, config, action_log)
kpis           headline KPIs from a twin state dict
AdapterClient  client for the organisers' hosted Maveric / Viavi adapters (twin.http)
presets        "official" and "viavi-like" configurations
"""

from twin.kpis import kpis
from twin.presets import PRESETS, resolve_config
from twin.replay import ReplayMismatch, ReplayTwin

__all__ = ["PRESETS", "ReplayMismatch", "ReplayTwin", "kpis", "resolve_config"]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LedgerStore(Protocol):
    def put(self, record: dict) -> None: ...
    def get(self, seq: int) -> dict | None: ...


class AnchorStore(Protocol):
    def put_anchor(self, anchor: dict) -> dict: ...


class Signer(Protocol):
    def sign(self, key_id: str, message: str) -> str: ...
    def verify(self, key_id: str, message: str, signature: str) -> bool: ...


class PolicyEngine(Protocol):
    def is_authorized(self, principal: dict, action: str, resource: dict, context: dict) -> dict: ...


class Guardrails(Protocol):
    def apply(self, text: str, source: str) -> dict | None: ...


class ApprovalBroker(Protocol):
    def complete(self, task_token: str, decision: dict) -> None: ...


class MetricsSink(Protocol):
    def put(self, namespace: str, metrics: list[dict]) -> None: ...


class FlagSource(Protocol):
    def flags(self) -> dict: ...


@dataclass
class Adapters:
    mode: str
    services: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, str]:
        return {name: getattr(svc, "backing", type(svc).__name__) for name, svc in self.services.items()}

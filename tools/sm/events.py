"""/team13/runner: one JSON log event per real step of a run, naming the run id.

These events CORROBORATE; they never define a run's AWS window (PutLogEvents takes a caller-chosen
timestamp). check_unit compares each event's @timestamp with CloudWatch's own @ingestionTime and fails
the run if they differ by more than 30 s. Steps from the laptop: declared, job_submitted, job_finished,
summary_written, checked. Steps from inside the job (preferred: AWS clocks): job_started, module_finished,
result_written, job_entry_finished. Every event is also printed, so inside a job it lands in
/aws/sagemaker/TrainingJobs with SageMaker's timestamps too.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from tools.sm.spec import LOG_GROUP, LOG_RETENTION_DAYS, stream_name

MARK = "team13-runner"


def _code(e: Exception) -> str:
    r = getattr(e, "response", None) or {}
    return str((r.get("Error") or {}).get("Code", "")) or e.__class__.__name__


def ensure_group(logs, group: str = LOG_GROUP, retention_days: int = LOG_RETENTION_DAYS) -> bool:
    """Create the log group if it is missing (and set its retention). True if it was created."""
    try:
        logs.create_log_group(logGroupName=group, tags={"team": "team13"})
        created = True
    except Exception as e:  # noqa: BLE001
        if _code(e) != "ResourceAlreadyExistsException":
            raise
        created = False
    if created:
        logs.put_retention_policy(logGroupName=group, retentionInDays=retention_days)
    return created


class RunnerLog:
    """Writes the run's step events. strict=False (inside a job): a refused write is printed, not raised,
    because the same line still reaches /aws/sagemaker/TrainingJobs through stdout."""

    def __init__(self, logs, run_id: str, writer: str, group: str = LOG_GROUP, strict: bool = True,
                 echo: Callable[[str], None] | None = None):
        self.logs, self.run_id, self.writer, self.group, self.strict = logs, run_id, writer, group, strict
        self.stream = stream_name(run_id, writer)
        self.echo = echo or (lambda s: print(s, flush=True))
        self._ready = False
        self.failures: list[str] = []

    def _stream(self) -> None:
        if self._ready:
            return
        try:
            self.logs.create_log_stream(logGroupName=self.group, logStreamName=self.stream)
        except Exception as e:  # noqa: BLE001
            if _code(e) != "ResourceAlreadyExistsException":
                raise
        self._ready = True

    def emit(self, step: str, **fields: Any) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        body = {"team13_runner": 1, "run_id": self.run_id, "writer": self.writer, "step": step,
                "at": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"), **fields}
        line = json.dumps(body, sort_keys=True, default=str)
        self.echo(f"{MARK} {line}")
        if self.logs is None:
            return body
        try:
            self._stream()
            self.logs.put_log_events(logGroupName=self.group, logStreamName=self.stream,
                                     logEvents=[{"timestamp": int(time.time() * 1000), "message": line}])
        except Exception as e:  # noqa: BLE001
            msg = f"{MARK} warning: could not write {step} to {self.group}: {_code(e)}"
            self.failures.append(msg)
            if self.strict:
                raise
            self.echo(msg)
        return body

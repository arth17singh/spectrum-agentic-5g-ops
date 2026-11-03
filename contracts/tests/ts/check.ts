// Type-level check of the generated records.gen.ts (not part of the app build).
// Run from the repo root (after symlinking dashboard/frontend/node_modules):
//   dashboard/frontend/node_modules/.bin/tsc --noEmit --strict --target es2022 \
//     --module esnext --moduleResolution bundler contracts/tests/ts/check.ts
import type {
  AnyRecord,
  ChangeRecord,
  DecisionRecord,
  RecordByKind,
  RecordKind,
  ResultRecord,
} from "../../../dashboard/frontend/src/contracts/records.gen";
import { RECORD_KINDS, SCHEMA_VERSION } from "../../../dashboard/frontend/src/contracts/records.gen";

const kinds: readonly RecordKind[] = RECORD_KINDS;
const v: "1" = SCHEMA_VERSION;

const change: ChangeRecord = {
  schema_version: "1",
  kind: "change",
  run_id: "t3-run-001",
  team: "team13",
  created_at: "2026-11-02T21:30:00.000Z",
  synthetic: true,
  case_id: "case-0001",
  change_id: "chg-0123456789ab",
  item_id: "step-3",
  twin: { kind: "maveric-inprocess", version: "1.2.9", scenario: "nominal", seed: 7, step: 3 },
  change: [{ cell: 0, d_power_db: -1, d_tilt_deg: 0.5 }],
  change_text: "cell=0;d_power_db=-1.0;d_tilt_deg=+0.5",
  baseline_kpis: { coverage_ratio: 0.86, sla_ratio: 0.8, sinr_p10_db: 1.0, sinr_mean_db: 9 },
  measured_kpis: { coverage_ratio: 0.88, sla_ratio: 0.8, sinr_p10_db: 1.2, sinr_mean_db: 9.1 },
  measured_delta: { coverage_ratio: 0.02, sla_ratio: 0, sinr_p10_db: 0.2, sinr_mean_db: 0.1 },
  scored: false,
  unscored_reason: "probe",
  probe: true,
  approved_by: "person:op-blue",
  approved_at: "2026-11-02T21:29:58.000Z",
};

// decision is v0: extra keys allowed.
const decision: DecisionRecord = {
  schema_version: "1",
  kind: "decision",
  run_id: "t3-run-001",
  team: "team13",
  created_at: "2026-11-02T21:30:00.000Z",
  synthetic: true,
  case_id: "case-0001",
  time: "2026-11-02T21:30:00.000Z",
  agent: "planner",
  tool: "ran___apply_change",
  decision: "DENY",
  source: "/team13/checkpoint",
  span_attr_unknown_yet: 1,
};

function describe(r: AnyRecord): string {
  switch (r.kind) {
    case "change":
      return `${r.change_text} ${r.measured_delta.coverage_ratio} ${r.error?.summary ?? "probe"}`;
    case "result":
      return `${r.unit} ${r.passed ? "PASS" : "FAIL"} ${r.target.op}`;
    case "decision":
      return `${r.tool} ${r.decision}`;
    default:
      return r.kind;
  }
}

function byKind<K extends RecordKind>(k: K, r: RecordByKind[K]): string {
  return `${k}:${r.run_id}`;
}

// @ts-expect-error: numbers are numbers, not strings
const bad: ResultRecord["value"] = "0.9";

export const out = [kinds, v, describe(change), describe(decision), byKind("change", change), bad];

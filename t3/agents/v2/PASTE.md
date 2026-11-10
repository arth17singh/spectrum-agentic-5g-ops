# v2 prompts: a verified lesson may become a candidate change

For the person at the Agent Builder console. These are prompt changes only: no tool, Cedar rule, token,
gateway target or Lambda changes. Pasting them changes how live agents behave, so paste only with the
lead's go, and **never while a measured run is in progress** (a run whose agents change mid-run is not one
experiment). Write down the UTC time you pasted and which version is live; the next run's declaration
should name it (for example `prompts=v2` in `T3_SWARM_MODEL_ID`).

## What changes

| Agent | Paste this file | Instead of | What is new |
|---|---|---|---|
| ran_monitor | `t3/agents/v2/ran_monitor.md` | `t3/agents/ran_monitor.md` | When the task says "Include per_cell", the answer adds each cell's power_dbm and tilt_deg, copied from ran___observe. Otherwise the answer is exactly v1's. |
| ran_optimizer | `t3/agents/v2/ran_optimizer.md` | `t3/agents/ran_optimizer.md` | Steps 5 and 6: when the task carries a lesson line ("- L-...") and the monitor's per_cell, it also turns the first lesson's fix into a concrete change, checks it with its own ran___predict and ran___preflight, and returns the eligible candidate (preflight positive, not clipped) an operator would sign: the one changing fewer cells; then the smaller total \|d_power_db\|; within 0.5 dB, the larger preflight coverage+SLA gain; a full tie goes to the policy. It answers with `source`, `lesson_id` and `alternative`. Without a lesson, exactly v1. |
| noc_lead | `t3/agents/v2/noc_lead.md` | `t3/agents/noc_lead.md` | Rule 3a: only when the message has a "Verified past lessons" block, it asks the monitor to "Include per_cell" and copies the lesson lines to the optimizer. Rule 5a: a lesson's change is applied only as the optimizer's grounded, preflight-positive answer; the lead never writes a change itself. The report names the source. |
| ran_explainer | nothing | `t3/agents/ran_explainer.md` stays | unchanged (`t3/agents/v2/ran_explainer.md` is an identical copy, so the v2 set is complete) |

Answer schemas: `t3/agents/v2/schemas/` (additive fields only; every v1 answer still validates).

## How

Order does not matter for safety (an old monitor ignores "Include per_cell", and then the optimizer skips
the lesson step), but paste the students first.

For each of ran_monitor, ran_optimizer, noc_lead: Agents, open the agent, Configuration, Customized,
replace the whole System prompt with the whole v2 file, Save. Leave the model, MCP server, token and
every other setting as they are.

## Re-test after pasting

1. Agent Chat, `noc_lead`: "Cell 2 has poor coverage. Find the cause and propose a fix." As in the
   checklist (section 6): it asks the students, calls ran___apply_change and stops with an approval id.
   There is no lesson, so the optimizer's answer must have no `source` field. Deny that approval (or let it
   expire); do not approve a test change.
2. Agent Chat, `noc_lead`, a ticket with a lesson block (the shape `t3/evals/lessons.py:lesson_block`
   writes; the numbers here are only an example):

   ```
   Network KPIs dropped after the last configuration step: coverage 0.950 -> 0.820, SLA 0.970 -> 0.930,
   p10 SINR 2.5 -> -0.6 dB. Restore service with at most 3 changes. Observe first, use preflight, and send
   every change as the canonical string.
   Verified past lessons (from practice incidents; each fix was replayed on the twin and recovered service). Use them as hints, still observe and preflight first:
   - L-01 (similarity 0.91): when coverage fell 0.130, SLA 0.040, p10 SINR -3.1 dB -> raise power by 6 dB on the cell with the lowest power worked.
   ```

   Expect: the monitor's task says "Include per_cell" and its answer has 4 per_cell entries; the
   optimizer calls ran___predict and ran___preflight twice (policy, then lesson) and answers with
   `source`, `lesson_id` "L-01" and `alternative`. The lesson's one-cell fix wins over a policy change
   that touches more cells whenever both are eligible, even if the policy's preflight gain is larger;
   then the lead's reason starts "lesson L-01:" and its report names the source. Deny the approval.
3. Laptop, kit folder: `python checkpoint_kit.py check --full` (every line PASS; prompts do not touch the
   rules, this proves it), then from the repo `python -m t3.agents.brain_smoke` (dry run).

## Roll back

Paste the v1 files back into the same three agents: `t3/agents/ran_monitor.md`,
`t3/agents/ran_optimizer.md`, `t3/agents/noc_lead.md` (unchanged in the repo; also
`git show 082922a:t3/agents/<name>.md`). Nothing else needs undoing. Note the UTC time of the rollback.

## Known gaps

- The optimizer has no observe tool: the lesson step depends on the lead passing the monitor's JSON with
  per_cell. If per_cell is missing, the optimizer skips the lesson (safe, but the lesson is unused).
- The lesson's fix is turned into cells by a student model (gpt-oss-20b) following written steps. A wrong
  cell only produces a weaker candidate, which preflight and the choice rule then drop, and a person still
  approves; it does not bypass anything.
- Two-part lessons: the prompt ranks among all 4 cells and steps past a used cell, which round-trips on
  the network the lesson came from; `t3/evals/lessons.py:resolve` re-ranks the remaining cells instead,
  so on another network the two can pick different cells for the second part (single-part lessons agree;
  `t3/suite/tests/test_prompts_v2.py` checks both).
- Lesson lines arrive in the ticket. A forged lesson can at most add a candidate: it still needs a positive
  preflight, the choice rule, the checkpoint's Cedar rules and a person's approval.
- The choice rule is minimal change first, by design: the policy's broad power boost often shows the bigger
  preflight gain because it over-shoots the healthy network (Live Fix run 2 ended at coverage 0.715
  against 0.605 healthy), so a gain-first rule would bury targeted fixes. The cost: a smaller fix can win
  while restoring less than the boost would. Both must be preflight-positive (the worst cell improves and
  no other cell is hurt), and the twin's grade at the grade step still decides whether the incident
  recovered.

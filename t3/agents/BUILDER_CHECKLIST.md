# Agent Builder checklist (team13, Track 3)

For the person at the console. Every step is manual and needs the lead's go: it launches or changes things in the team account (us-east-2). Never paste a token, the instance IP or the account id into a file in this repo, a chat or a prompt. Keep `kit.json`, `issuer-key.pem` and `tokens/` in a private folder outside the repo.

## 1. Account prerequisites (once per account, CloudShell)
- [ ] Console, region **US East (Ohio) us-east-2**, open CloudShell.
- [ ] Actions, Upload file: `agentbuilder-prereqs.sh` (from the organisers' hub).
- [ ] `bash agentbuilder-prereqs.sh` (defaults: IAM path `/bb26/`, boundary `bb26-boundary/bb26-Boundary`). Running it twice is harmless.
- [ ] Copy the `AB_` lines it prints at the end. AccessDenied: send the full error to Tupl.

## 2. Launch the Agent Builder instance
EC2, AMIs, Owned by me / Private images, search `agentbuilder`, newest `agentbuilder-YYYYMMDD-HHMM`, Launch instance from AMI:
- [ ] Name `team13-agentbuilder`; instance type `m7i.4xlarge`; key pair: proceed without.
- [ ] Network: auto-assign public IP **Enable**; new security group, **HTTPS from the internet only** (no HTTP, no SSH).
- [ ] Storage 150 GiB (default).
- [ ] Advanced, IAM instance profile: type `AgentBuilderInstanceProfile`, choose "Specify a custom value".
- [ ] Advanced, User data: the `AB_` lines from step 1, then these two lines:
```
AB_BEDROCK_MODELS=openai.gpt-oss-120b-1:0,openai.gpt-oss-20b-1:0,us.anthropic.claude-opus-5-5,us.anthropic.claude-haiku-5-5,us.moonshotai.kimi-k3,us.zai.glm-5.3,qwen.qwen3-235b-a22b-2507-v1:0
AB_DEFAULT_MODEL=openai.gpt-oss-120b-1:0
```
- [ ] Launch. Wait about 10 minutes. **Never stop the instance**: a restart changes its IP and every URL.
- [ ] Log in at `https://<ip-with-dashes>.sslip.io` as `ta_admin`, password = the instance id; set a new password (store it in the team's password manager, not here).

## 3. Deploy the checkpoint (laptop, private kit folder)
- [ ] Copy `t3/agents/kit.json.template` to the kit folder as `kit.json`; set `builder_host` to the `.sslip.io` name. Leave the A2A values as they are for now.
- [ ] `python checkpoint_kit.py deploy` (about 4 minutes). Every rule line must say ACTIVE. Note the Gateway URL (it goes into `T3_GATEWAY_URL` at run time, not into the repo).
- [ ] Our tools and rules are W3's deploy step (`t3/deploy/`), run by the lead after the kit deploy.

## 4. Create the four agents (Agents, + New agent), one per name, exactly:
`noc_lead`, `ran_monitor`, `ran_optimizer`, `ran_explainer` (the builder adds a tenant prefix to the display name; Cedar uses the token's name).
For each:
- [ ] Configuration, Customized. Prompts: paste the whole file `t3/agents/<name>.md` as the System prompt.
- [ ] Model: AWS Bedrock. `noc_lead`: `openai.gpt-oss-120b-1:0`, **Response Timeout 300**. Students: `openai.gpt-oss-20b-1:0`.
- [ ] MCP, Add MCP server, By URL: name `cp`, URL = the Gateway URL. Change Auth type from OAuth 2.1 to **Bearer token** and paste that agent's own token (`cat tokens/<name>.txt`). "No tools" is expected; leave the tool list empty (= all tools).
- [ ] No other MCP server, Delegate to sub-agents empty, Local access off, Memories/documents/Skills off.
- [ ] Create agent, then switch it to **Enabled**.

## 5. Expose over A2A, without linking
- [ ] For each student and for `noc_lead`: Sub-agents tab, switch on **Expose as sub-agent**, write a one-line description, switch on **Also expose via A2A**, Save. Leave **Delegate to sub-agents empty** on every agent.
- [ ] Copy each AgentCard URL; the a2a id is the part in `/tupl-agent/a2a/<a2a id>/.well-known/agent-card.json`. Put the ids into `kit.json` `a2a_agents`.
- [ ] A2A clients, + New client, label `team-checkpoint`, Issue token. Copy it now (shown once) into `kit.json` `a2a_token`.
- [ ] In the client's list, switch on all four agents: each row reads "Granted, callable via A2A".
- [ ] `python checkpoint_kit.py deploy` again.

## 6. Check
- [ ] `python checkpoint_kit.py check --full`: every line PASS. Save the output to `evidence/logs/<run_id>/kit-check.txt` (the organisers' check is our independent grader, contract decision 6).
- [ ] Agent Chat (`https://chat.<host>`): pick `noc_lead`, ask "Cell 2 has poor coverage. Find the cause and propose a fix." It must ask the students, call apply_change and stop with an approval id.
- [ ] Smoke test from the laptop: `python -m t3.agents.brain_smoke` (dry run), then with `--live`.

## Never
- Never run `checkpoint_kit.py destroy`: it deletes the approvals table, every target and every rule.
- Never name a pause rule `team13_pause_nlm...` by hand: the kit's `resume nlm` deletes every rule with that prefix. Our pauses are `team13_ranpause_...` (pause_floor.py).
- Never approve your own suite run's change from the runner: a person runs `checkpoint_kit.py approve chg-... "why"`.

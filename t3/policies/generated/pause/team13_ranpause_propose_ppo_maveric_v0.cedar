// team13_ranpause_propose_ppo_maveric_v0  validationMode=IGNORE_ALL_FINDINGS  (template: created only to pause)
forbid (principal, action == AgentCore::Action::"ran___propose", resource == @GW@)
unless { context.input has model_version && context.input.model_version != "v0" };

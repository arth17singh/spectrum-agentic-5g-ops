// team13_ranpause_predict_t2_response_v0_degraded  validationMode=IGNORE_ALL_FINDINGS  (template: created only to pause)
forbid (principal, action == AgentCore::Action::"ran___predict", resource == @GW@)
unless { context.input has model_version && context.input.model_version != "v0-degraded" };

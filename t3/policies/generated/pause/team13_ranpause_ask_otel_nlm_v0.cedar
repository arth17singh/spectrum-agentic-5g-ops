// team13_ranpause_ask_otel_nlm_v0  validationMode=IGNORE_ALL_FINDINGS  (template: created only to pause)
forbid (principal, action in [AgentCore::Action::"nlm___ask", AgentCore::Action::"nlm___ask_nlm"], resource == @GW@)
when { !(context.input has model_version) || context.input.model_version == "v0" };

"""remediation-v1: a fault-aware drop-in replacement for ppo_sm_50k behind ran___propose.

ppo_sm_50k was trained for coverage-capacity optimisation, not for faults: it proposes nearly the same
change on every incident (cell 2 +7..+8 dB, whatever the faulted cell). remediation-v1 has the SAME
artifact format (SB3 PPO zip + .meta.json sidecar, loaded by t1.rl.policy.load), the same actor
architecture and the same observation and action conventions as t1.rl.policy.Policy, but it is trained
to put the network back to its commissioned configuration:

    data.py      twin-labelled incidents (no LLM): the t3 incident generator on fresh dev-split seeds;
                 every state is labelled with the exact way back to the commissioned configuration
                 (the injected fault's inverse; zero on healthy or already-fixed states), and an incident
                 is kept only if the twin verifies that the inverse recovers it
    bc.py        behaviour cloning (+ DAgger rounds on the policy's own states) into the SB3 PPO actor
    job_train.py SageMaker entry: data + training, one item per candidate, artifacts to {model_dir}
    evaluate.py  SageMaker entry: v1 vs ppo_sm_50k vs config_revert vs rule_baseline on the dev holdout,
                 the frozen test split and the confirmation suite, graded by t3/evals/graders.py
"""

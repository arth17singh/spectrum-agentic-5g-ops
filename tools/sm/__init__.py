"""Team13's SageMaker job runner: every scored run (evals, data generation, RL evaluate, Viavi runs,
offline baselines) is a SageMaker training job launched from here; the laptop only launches and reads.

  spec.py     fixed facts (bucket, role, images, prices) and the pure request builders
  package.py  the job's source: `git archive <sha>` of a pathspec plus a generated requirements.txt
  events.py   /team13/runner log events (corroboration only; never an AWS window)
  run.py      the launcher CLI: python -m tools.sm.run --help
  entry.py    the job side: runs the module, uploads and attests its evidence
"""

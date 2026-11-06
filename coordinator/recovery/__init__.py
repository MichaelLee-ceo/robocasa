"""Recovery control (proposal §3, role 3): what to do about a confirmed trigger.
base.py      RecoveryPolicy: decide(memory, obs, trigger) -> Intervention (executed by loop.Coordinator.execute)
policies.py  RetraceRetry (the method behind the current result), Ladder (the proposal's fixed order). Work.
vlm.py       VLMRecovery: the VLM picks the intervention from the memory (reference v0)."""

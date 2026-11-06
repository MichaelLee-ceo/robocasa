"""Progress tracking (proposal §3, role 1): which sub-goals are done, "where am I?".
base.py    ProgressTracker: make_plan(instruction, obs) -> [SubGoal]; update(memory, obs) -> StageStatus | None
oracle.py  OracleTracker (reads the simulator: the upper-bound ablation), task_plan (the task's stage sentences)
vlm.py     VLMTracker (a VLM judges the checklist from the memory: context.progress_request)"""

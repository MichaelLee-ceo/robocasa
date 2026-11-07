"""Progress tracking (proposal §3, role 1): which sub-goals are done, "where am I?".
base.py    ProgressTracker: make_plan(instruction, obs) -> [SubGoal]; update(memory, obs) -> StageStatus | None
oracle.py  OracleTracker (reads the simulator: the upper-bound ablation), task_plan (the task's stage sentences)
vlm.py     VLMTracker (a to-do list: the VLM writes it, checks the current item at every check, progress dots;
           context.item_request / plan_request)"""

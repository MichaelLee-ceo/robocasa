# runs/
- `20261012_groot_repro_h1x/`: reproduction of RoboCasa365 Table 11 (GR00T N1.5, pretrain + target post-train, Composite-Unseen), 16 tasks × 20 seeds,
  original horizon. `summary_by_task.csv`, `summary.json`, `results.json`, `paper_table11_composite_unseen.json`. Result 42.5% [37.2, 48.0] vs paper 42.1%.
- `20261012_groot_oracle/`: oracle planner / oracle reset diagnosis on 8 tasks (see its README).
- `smoke_20261012/`: first SmolVLA CloseFridge smoke test.
- `scripts/`: groot_eval.py (seeded eval client), oracle_eval.py, oracle.py (reset + subtask tracker), demo_utils.py, validate_tracker.py, analyze_*.py
- Envs: `conda activate ~/Desktop/robocasa/envs/groot` (GR00T + robocasa, py3.10, torch 2.7.1 cu128); `conda activate robocasa` (LeRobot/SmolVLA, py3.12).
  Server: `python third_party/Isaac-GR00T/scripts/run_eval.py --server --model_path checkpoints/gr00t_n1-5/... --task_set composite_unseen --split target --port 5555`

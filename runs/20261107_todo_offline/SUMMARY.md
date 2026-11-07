# Offline evaluation of the to-do list tracker

VLM vllm:Qwen/Qwen3.8-27B@8104, plan vlm, windows ['attempt', 'recent']

```
dir | runs | plan_ok | checks | errors | item_acc | false_done | missed | ticks | premature | runs_premature | lag_median | lag_p90 | never | latency_median | images_median
20261107_todo_offline/attempt | 90 | 90 | 11304 | 3 | 0.83 | 0.009 | 0.161 | 148 | 21 | 18 | 2 | 26 | 28/162 | 13.51 | 16
    plan: {'KettleBoiling': '30/30', 'PackIdenticalLunches': '30/30', 'SteamInMicrowave': '30/30'}  hand-off: {'pass_ok': 127, 'pass_wrong': 21, 'block_ok': 13, 'block_wrong': 69}  previous_ok false: 69/120 true
    stall: {3: {'caught': '27/48', 'lead_median': 27, 'false_per_run': 1.61}, 4: {'caught': '26/48', 'lead_median': 29.5, 'false_per_run': 1.19}, 6: {'caught': '23/48', 'lead_median': 31, 'false_per_run': 0.68}, 8: {'caught': '23/48', 'lead_median': 29, 'false_per_run': 0.47}}
20261107_todo_offline/recent | 90 | 89 | 11304 | 21 | 0.866 | 0.022 | 0.112 | 163 | 40 | 29 | 1 | 22 | 17/160 | 13.6 | 18.0
    plan: {'KettleBoiling': '30/30', 'PackIdenticalLunches': '29/30', 'SteamInMicrowave': '30/30'}  hand-off: {'pass_ok': 123, 'pass_wrong': 40, 'block_ok': 32, 'block_wrong': 21}  previous_ok false: 46/93 true
    stall: {3: {'caught': '26/47', 'lead_median': 21.0, 'false_per_run': 1.09}, 4: {'caught': '26/47', 'lead_median': 22.0, 'false_per_run': 0.81}, 6: {'caught': '25/47', 'lead_median': 21, 'false_per_run': 0.4}, 8: {'caught': '24/47', 'lead_median': 20.5, 'false_per_run': 0.29}}

```

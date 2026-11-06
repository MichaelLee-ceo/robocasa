# oracle_retrace with the first coordinator skeleton (2026-11-06)

86 runs: the 43 phase 0 failure points x policy seeds 0-1, `oracle_retrace`, with the first skeleton of the coordinator
(then in runs/coordinator; superseded by the top-level `coordinator/` package, which replaced it).

| | fixed | success |
|---|---|---|
| this skeleton | 24 / 85 | 11 / 85 |
| runs/20261105_recovery loop.py retrace v2, same points and seeds | 31 / 85 | 19 / 85 |
| control (same-scene reruns), same points and seeds | 10 / 85 | 2 / 85 |

These rates count 85 runs: the comparison ran before the last run (PackIdenticalLunches s14 p0) finished.

The mechanism matched loop.py exactly. The first retrace was identical in all 85 runs (path length, entries, chunks),
and each retry attempt got the same GR00T time (median 24 chunks in both). The runs differ only in GR00T's noise
streams (other stream names) and in MSAA render noise. Per point, the skeleton was worse at 9 and better at 2-3
(sign test p 0.07-0.15).

One scoring flaw, fixed in the package: at "undone" points, the skeleton recorded the failed stage's first completion
during the replay. A later fix therefore did not count as fixed.

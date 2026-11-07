# oracle_retrace: 172 runs (paired with loop.py's retrace arm: 172)

| | control (GR00T alone) | v2 retrace (loop.py) | this batch | runs only this / only v2 (sign p) |
|---|---|---|---|---|
| fixed | 20/172 | 64/172 (37%) | 66/172 (38%) | 17 / 15 (0.86) |
| success | 7/172 | 38/172 (22%) | 37/172 (22%) | 15 / 16 (1.00) |

Per failure point, fixed: this batch better at 8, worse at 7, same at 28 (sign p 1.00)

Per failure point, success: this batch better at 6, worse at 8, same at 29 (sign p 0.79)

| task | runs | fixed v2 / this | success v2 / this |
|---|---|---|---|
| KettleBoiling | 28 | 12 / 10 | 12 / 10 |
| PackIdenticalLunches | 76 | 22 / 21 | 3 / 1 |
| SteamInMicrowave | 68 | 30 / 35 | 23 / 26 |

Retraces: 402; arrived 391; blocked at first 8, arrived after unstick 3; not arrived without a block 6
Retrace chunks: median 38, p90 81, max 297; at the 400-chunk cap: 0
Unstick sentences: [('Close the fridge door.', 12), ('Pick the fish from the fridge and place it in the tupperware.', 1)]
Run time: median 756 s, max 2015 s

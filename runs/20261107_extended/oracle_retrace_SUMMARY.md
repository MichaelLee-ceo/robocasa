# oracle_retrace: 172 runs (paired with loop.py's retrace arm: 172)

| | control (GR00T alone) | v2 retrace (loop.py) | this batch | runs only this / only v2 (sign p) |
|---|---|---|---|---|
| fixed | 20/172 | 64/172 (37%) | 80/172 (47%) | 27 / 11 (0.01) |
| success | 7/172 | 38/172 (22%) | 48/172 (28%) | 21 / 11 (0.11) |

Per failure point, fixed: this batch better at 10, worse at 2, same at 31 (sign p 0.04)

Per failure point, success: this batch better at 8, worse at 4, same at 31 (sign p 0.39)

| task | runs | fixed v2 / this | success v2 / this |
|---|---|---|---|
| KettleBoiling | 28 | 12 / 18 | 12 / 17 |
| PackIdenticalLunches | 76 | 22 / 27 | 3 / 2 |
| SteamInMicrowave | 68 | 30 / 35 | 23 / 29 |

Retraces: 605; arrived 565; blocked at first 42, arrived after unstick 13; not arrived without a block 11
Retrace chunks: median 46, p90 83, max 400; at the 400-chunk cap: 5
Unstick sentences: [('Close the fridge door.', 39), ('Pick the steak from the counter and place it in the tupperware.', 6), ('Pick the squash from the counter and place it in the tupperware.', 6), ('Pick the steak from the fridge and place it in the tupperware.', 4), ('Pick the lemon wedge from the counter and place it in the tupperware.', 4), ('Pick the chicken drumstick from the counter and place it in the tupperware.', 3), ('Pick the sweet potato from the fridge and place it in the tupperware.', 2), ('Pick the garlic from the fridge and place it in the tupperware.', 2)]
Run time: median 921 s, max 7464 s

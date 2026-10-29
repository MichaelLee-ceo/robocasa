| # | case (dir, tag, t, cand, boundary b) | situation | mode / yaw | after 1 step: arm q / arm v / fingers q / base q | after 16 steps: arm q / fingers q / base q / torso q / free objects q | held object max dev (m) | flag |
|---|---|---|---|---|---|---|---|
| 1 | v2/s0 KettleBoiling_s0 t0 c0 b0 | t=0 point (true hidden state = right after load_scene) | - | 0 / 9e-06 / 0.006 / 0 | 3e-05 / 0.0004 / 8e-07 / 0 / 0 |  |  |
| 2 | v2/s0 KettleBoiling_s0 t0 c0 b16 | arm, gripper opening | arm / 0.1 | 1e-05 / 0.0007 / 0.008 / 7e-07 | 0.0002 / 0.0007 / 5e-06 / 0 / 0 |  |  |
| 3 | v2/s2 KettleBoiling_s27 t160 c6 b128 | KB s0 arm, holding kettle | arm / 0.7 | 0 / 2e-06 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | obj 0.0 |  |
| 4 | v2/s1 KettleBoiling_s9 t320 c4 b48 | KB gripper misread: true closed, settle opening | arm / 0.8 | 0.002 / 0.041 / 0.008 / 2e-05 | 0.003 / 0.0007 / 8e-05 / 1e-05 / 0 |  | **>1e-3** |
| 5 | v2/s1 KettleBoiling_s9 t320 c4 b128 | KB s1 knob turning (knob joint moving, fingers on knob) | arm / 2.2 | 1e-06 / 3e-05 / 0 / 0 | 8e-06 / 0 / 0 / 0 / 0 |  |  |
| 6 | v2/s0 KettleBoiling_s29 t160 c2 b144 | KB mode switch inside chunk | switch / 0.8 | 0.003 / 0.082 / 0 / 1e-05 | 0.013 / 8e-07 / 1e-06 / 2e-06 / 0.004 |  | **>1e-3** |
| 7 | v2/s2 KettleBoiling_s2 t400 c4 b208 | KB gripper misread: true open, settle closing, free | arm / -1.7 | 1e-05 / 0.0006 / 0.008 / 1e-06 | 0.0002 / 0.0008 / 9e-06 / 0 / 9e-07 |  |  |
| 8 | v2/s1 KettleBoiling_s5 t0 c0 b192 | KB mid-ramp boundary | arm / -0.0 | 0.0002 / 0.004 / 0.004 / 4e-06 | 0.0008 / 0.0001 / 1e-05 / 0 / 0.002 | obj 0.0028 |  |
| 9 | v2/s1 KettleBoiling_s5 t0 c0 b304 | KB gripper misread: true open, settle closing, touching kettle | arm / -0.3 | 0.001 / 0.031 / 0.007 / 2e-05 | 0.069 / 0.0009 / 0.0007 / 0.0002 / 0.567 | obj 0.7214 | **>1e-3** |
| 10 | v2/s1 KettleBoiling_s4 t240 c2 b288 | KB s0 arm, gripper open, free | arm / 2.6 | 1e-06 / 3e-05 / 0 / 0 | 2e-06 / 0 / 0 / 0 / 0 |  |  |
| 11 | v2/s1 KettleBoiling_s4 t240 c2 b384 | KB s0 arm, closed empty | arm / 2.9 | 2e-06 / 4e-05 / 0 / 0 | 3e-06 / 0 / 0 / 0 / 0 |  |  |
| 12 | SteamInMicrowave_v2 SteamInMicrowave_s0 t0 c0 b0 | t=0 point (true hidden state = right after load_scene) | - | 0 / 1e-05 / 0.006 / 0 | 5e-05 / 0.0003 / 1e-06 / 0 / 8e-07 |  |  |
| 13 | SteamInMicrowave_fix3 SteamInMicrowave_s26 t480 c2 b32 | SIM s1 holding bowl, base mode, yaw>160 | base / 180.5 | 0.002 / 0.048 / 5e-06 / 5e-05 | 0.012 / 6e-06 / 3e-05 / 0 / 0.006 | bowl 0.0044 | **>1e-3** |
| 14 | SteamInMicrowave_fix3 SteamInMicrowave_s13 t640 c2 b64 | SIM s1 holding bowl, arm mode after base mode (prev chunk ended in base) | arm / -142.8 | 0.0002 / 0.006 / 1e-06 / 9e-06 | 0.0003 / 1e-06 / 1e-05 / 0 / 0.002 | bowl 0.0003 |  |
| 15 | dense_SteamInMicrowave SteamInMicrowave_s4 t96 c4 b144 | SIM s0 holding vegetable, arm | arm / 0.2 | 8e-07 / 2e-05 / 0 / 0 | 2e-06 / 0 / 0 / 0 / 9e-07 | vegetable 0.0 |  |
| 16 | SteamInMicrowave_fix3 SteamInMicrowave_s27 t480 c4 b32 | SIM s1 holding bowl, base mode, yaw 70-110 | base / -72.1 | 0.004 / 0.122 / 1e-05 / 5e-05 | 0.042 / 2e-05 / 7e-05 / 1e-06 / 0.015 | bowl 0.0064 | **>1e-3** |
| 17 | SteamInMicrowave_fix3 SteamInMicrowave_s27 t480 c4 b96 | SIM s1 holding bowl, base mode, yaw<20 | base / -5.7 | 0.002 / 0.048 / 5e-05 / 9e-06 | 0.013 / 1e-05 / 6e-06 / 0 / 0.008 | bowl 0.0084 | **>1e-3** |
| 18 | SteamInMicrowave_fix3 SteamInMicrowave_s27 t480 c4 b144 | SIM s1 holding bowl, switch inside chunk | switch / 8.0 | 0.002 / 0.050 / 1e-05 / 4e-06 | 0.010 / 2e-06 / 1e-06 / 0 / 0.070 | bowl 0.0012 | **>1e-3** |
| 19 | SteamInMicrowave_fix3 SteamInMicrowave_s20 t1280 c4 b208 | SIM gripper misread: true open, settle closing, free | arm / -159.2 | 2e-05 / 0.0004 / 0.008 / 0 | 7e-05 / 0.0008 / 0 / 0 / 8e-05 |  |  |
| 20 | SteamInMicrowave_fix3 SteamInMicrowave_s2 t1120 c4 b240 | SIM s3 button: fingers on microwave in stage 3 | arm / -174.1 | 6e-07 / 2e-05 / 0 / 0 | 1e-06 / 0 / 0 / 0 / 0.001 |  |  |
| 21 | dense_SteamInMicrowave SteamInMicrowave_s14 t1216 c0 b256 | SIM s2 door: fingers on microwave, door joint moving | arm / -95.4 | 8e-07 / 2e-05 / 0 / 0 | 3e-06 / 0 / 0 / 0 / 0.0005 |  |  |
| 22 | SteamInMicrowave_fix3 SteamInMicrowave_s16 t800 c2 b352 | SIM gripper misread: true open, settle closing, near/at object | arm / -21.1 | 0.0003 / 0.006 / 8e-05 / 9e-07 | 0.023 / 0.005 / 0.0002 / 3e-05 / 0.013 | bowl 0.0027 | **>1e-3** |
| 23 | dense_SteamInMicrowave SteamInMicrowave_s1 t544 c6 b96 | SIM gripper misread: true closed, settle opening, holding | arm / 174.0 | 0.002 / 0.058 / 0.007 / 3e-05 | 0.013 / 0.013 / 0.0003 / 2e-05 / 0.087 | bowl 0.0229 | **>1e-3** |
| 24 | dense_SteamInMicrowave SteamInMicrowave_s1 t544 c6 b656 | SIM s1 base mode, gripper free, yaw>160 | base / 174.2 | 0.006 / 0.189 / 5e-06 / 1e-05 | 0.062 / 0 / 1e-05 / 0 / 7e-05 |  | **>1e-3** |
| 25 | PackIdenticalLunches PackIdenticalLunches_s0 t0 c0 b0 | t=0 point (true hidden state = right after load_scene) | - | 0 / 8e-06 / 0.006 / 0 | 5e-05 / 0.0003 / 1e-06 / 0 / 2e-05 |  |  |
| 26 | dense_PackIdenticalLunches PackIdenticalLunches_s14 t32 c2 b48 | PIL gripper misread: true open, settle closing, touching item | arm / 0.6 | 0.0006 / 0.025 / 0.007 / 4e-05 | 0.012 / 0.004 / 0.0004 / 3e-06 / 0.048 | meat0 0.0132 | **>1e-3** |
| 27 | dense_PackIdenticalLunches PackIdenticalLunches_s21 t1792 c6 b64 | PIL gripper misread: true closed, settle opening | arm / -0.4 | 0.0004 / 0.009 / 0.007 / 1e-05 | 0.004 / 0.002 / 4e-05 / 2e-05 / 0.0002 |  | **>1e-3** |
| 28 | dense_PackIdenticalLunches PackIdenticalLunches_s21 t1408 c4 b16 | PIL mode switch inside chunk, holding | switch / -1.3 | 0.002 / 0.060 / 0 / 8e-06 | 0.013 / 0 / 3e-06 / 0 / 0.004 | vegetable1 0.0004 | **>1e-3** |
| 29 | dense_PackIdenticalLunches PackIdenticalLunches_s21 t1408 c4 b80 | PIL holding item, arm mode | arm / 0.1 | 0 / 4e-06 / 0 / 0 | 2e-06 / 7e-05 / 0 / 0 / 0.002 | vegetable1 0.0228 |  |
| 30 | dense_PackIdenticalLunches PackIdenticalLunches_s15 t128 c2 b80 | PIL holding item, base mode, yaw<20 | base / 3.9 | 0.002 / 0.048 / 0 / 3e-05 | 0.024 / 4e-05 / 4e-05 / 2e-06 / 0.008 | meat0 0.0026 | **>1e-3** |
| 31 | dense_PackIdenticalLunches PackIdenticalLunches_s15 t128 c2 b112 | PIL holding item, base mode, yaw 60-120 | base / -68.1 | 0.009 / 0.274 / 4e-06 / 0.0001 | 0.060 / 7e-05 / 0.0001 / 0 / 0.026 | meat0 0.0263 | **>1e-3** |
| 32 | PackIdenticalLunches_fix3 PackIdenticalLunches_s22 t1280 c2 b544 | PIL fridge contact, fridge joint moving | arm / 15.7 | 0.001 / 0.021 / 0.006 / 3e-05 | 0.003 / 0.001 / 4e-05 / 2e-06 / 0 |  | **>1e-3** |

# PreSoakPan end-state audit (ASUS run 20261014_asus_seen5)

All numbers come from reloading the frozen scene and the episode's `_end.npz` (EGL on GPU0; OSMesa is unavailable in this env).
Scripts are in this folder: `audit_presoak.py` (Q1–Q3), `audit_init_states.py` (containment at the initial and end states, s0–19),
`audit_extra.py` (s23/s27 and the scene table), `audit_grip.py` (gripper distances). Raw output is in `audit_*.json` and `audit_grip.out`. Frames are in `frames/`.

## Two bugs that explain all of the mismatches
1. **The note labels are wrong, not the state.** `export_episodes.py:24` labels tracker step 2 as "faucet on/off", but step 2's check is
   `e._check_success()` (`runs/scripts/atomic_steps.py:74-76`), which requires water_on, pan in sink, sponge in sink and the gripper far from both objects
   (`pre_soak_pan.py:85-97`). So "faucet off" really means "task not complete". The labels "pan/sponge (not) in sink" are the step-0/1 checks,
   which also include `gripper_obj_far(th=0.25)` (`atomic_steps.py:70,73`; `object_utils.py:645-652`).
2. **The containment tolerance does not scale correctly.** `OU.obj_inside_of` (`robocasa/utils/object_utils.py:46-55`) compares dot products with the
   *unnormalised* region edge vectors u,v,w against `±th` (th=0.05). In metres the tolerance is therefore 0.05/|edge|. For example, s19's right basin
   (0.240 × 0.412 × 0.141 m) gets +0.209 m in x, +0.121 m in y and +0.354 m in z. The check also tests all 8 corners of the object's axis-aligned bbox, so the empty corners
   next to a pan handle count as "pan". It returns True if the object fits in **any** of the regions basin / basin_left / basin_right
   (`sink.py:35-36`, `fixture.py:576-615`).

## Q1 Faucet
- `get_handle_state` (`robocasa/models/fixtures/sink.py:142-176`) reads `qpos[get_joint_qpos_addr(handle_joint)]`. That is a qpos address, so there is no index bug,
  even though the variable is named `handle_joint_id`; in these scenes the joint id and the address are equal anyway (14/57/16). The value is taken mod 2π and
  `water_on = 0.40 < q < π`. The joint range is [0, 0.5] (model and XML agree). There is no Python-level cached flag. The water stream you see is
  `model.site_rgba[water][3]`, which `Sink.update_state` sets from the same qpos (`sink.py:73-92`). That runs after every step (`kitchen.py:1492-1503`) and after
  `oracle.set_state` (`runs/scripts/oracle.py:45-51`). So the MuJoCo state fully determines water_on.
- Reloaded end states:
  - s12: q=0.4936, water_on=True, alpha=0.5, spout 'right'. Fails only because sponge_in_sink=False.
  - s13: q=0.5000, water_on=True, alpha=0.5. Fails only because pan_in_sink=False.
  - s19 (success): q=0.4615, water_on=True, and every _check_success term is True.
  - Across s0–19, the end state reproduces the rollout outcome: _check_success is True for exactly the 12 successes.
- **Root cause:** a labelling bug in export_episodes (bug 1). The faucet really is on in s12 and s13, and the video is correct.

## Q2 s13 pan (pan_4, sink_left_group Sink012, double basin)
- The library check returns False. Pan body at (0.416, -2.441, 0.795). It is strongly tilted (bbox corner z ranges from 0.68 to 1.00, so the handle end is raised by about 0.3 m and leans on the rim).
- Gripper distance is fine (grip_far=True), so this is a genuine containment-check failure.
- basin_left (edge lengths u 0.300 / v 0.371 / w 0.211 m, so effective tolerances are 0.167 / 0.135 / 0.237 m): 7 of 8 corners pass. **Corner 1** (object-local (+0.127, -0.2785, -0.019))
  is outside along v by **0.0108 m beyond the tolerance**, which is 0.146 m beyond the actual basin edge.
- basin_right: 6 of 8 corners fail, which is expected because the pan is in the other basin.
- Corner 1 is on the **handle end** of the bbox. The mesh profile along the long axis (y) is only about 0.023 m wide for y < -0.158 (the handle), while the bowl is about 0.25 m wide.
  The failing point is the bbox corner at x=+0.127 beside the narrow handle, not pan material.
- `sink.get_obj_basin_loc` gives 'none' with all 8 corners and 'left' using the centre point.
- **Root cause:** the axis-aligned bbox includes the handle's empty corner, and the pan is tilted onto the rim. Not a basin-selection issue.

## Q3 s19 sponge (sink_main_group Sink020, double basin)
The sponge was placed with its near bbox face d outside the basin edge, at its initial counter height (z=0.932), using the end state, then `sim.forward()` was called.

| side of basin_right | 3 cm | 6 cm | 10 cm |
|---|---|---|---|
| +x (outer counter; sink outer edge is 3.5 cm past the basin edge) | IN | IN | IN |
| -y (front) | IN | out | out |
| +y (back) | IN | out | out |
| -x (between basins) | IN | IN | IN |

basin_left's outer side (-x) is out at 3, 6 and 10 cm; its front and back are IN at 3 cm only.
- At the **initial** frozen state, s19's sponge (on the counter, sink-local x=0.50, so its near face is about 5 cm past the basin edge) already passes the library check. Its centre is outside, and with a metric 5 cm tolerance it fails (`audit_init_states.json`).
- s16's sponge also passes the check at the start.
- **Root cause:** step 1 passed at env step 352 because of the tolerance bug (bug 2). The sponge never had to move. At 352 the pan step had just passed, and the sponge was still at its start position.

## Q4 runs/20261013_scenes_v2/end_state_analysis.json (`runs/scripts/analyze_end_states.py`)
- `last_step_failures` (PreSoakPan rows):
  - `water_on` is False in all 35 rows. Computed via `get_handle_state` (analyze_end_states.py:28). **Not affected:** the reloaded faucet state is reliable (Q1).
  - `pan_in_sink` / `sponge_in_sink` come from `OU.obj_inside_of` (lines 29-30). **Affected** by bug 2: false positives up to about 0.1–0.35 m outside the basin, and false negatives from bbox corners and tilt.
  - `far_pan` / `far_sponge` are not affected.
  - Row selection depends on `reached`, which comes from the tracker (same check), so it is **affected** too.
  - The conclusion "PreSoakPan last-step failures are faucet-off" still holds.
- `presoak_chain_end`:
  - `pan_in_sink` / `sponge_in_sink` (lines 99-100) are **affected**.
  - `reached` / `success` come from the tracker and env reward. `reached` is **affected** because steps 0/1 use the same check; `success` (`_check_success`) is affected the same way.
  - `pan_tilt_deg` and `grip_pan` / `grip_sponge` are not affected.
- `presoak_single_handoff`:
  - The states are the chunk where tracker step k passed (k from `_states.npz`), so **which states were selected is affected**.
  - Simultaneous step-0/1 hits are suspect, because the sponge may still be on the counter: hscc s0 (576/576), s8 (976/976), s13 (496/496), s16 (352/352), s19 (304/304).
  - Its `pan_in_sink` / `sponge_in_sink` (lines 117-118) are **affected**. Tilt and grip distances are not.
- No field depends on cached Python faucet state. Also affected: the `note` field in `export_episodes.py` (bug 1, plus the gripper term), and `water_on` in `atomic_eval.py:174` is fine.

## A. s23 / s27 (pan_7, sink_main_group Sink025, layout 1)
- At the end state the library check says the pan **is** in the sink: basin_right passes with all 8 corners, the centre is in basin_right, and the tilt is 16.5° (s23) and 8.0° (s27).
  basin_left fails, which is expected.
- The sponge is also in the sink, and water_on=False (q=0.0007 for s23, 0.0002 for s27).
- The "pan not in sink" note comes from the gripper term: the end-effector is **0.239 m** from the pan (threshold >0.25 m) in both episodes, because the arm is hovering at the faucet.
- The pan did not leave the sink. At s27's tracker hit (env step 384) the video shows it in the right basin, and at the end it is still in basin_right. The exact pose change is not
  recoverable because no intermediate states were saved.
- So the pan-in-sink check itself is correct for s23/s27. The only problem is the mislabelled note (bug 1).

## B. Per-scene table (CSV: `presoak_scene_table.csv`)

|seed|layout|style|sink_fixture|sink_model|double_basin|handle_range|pan|asus_ok|asus_reached|asus_end|asus_history|asus_end_handle_qpos|asus_end_water_on|hscc_ok|hscc_reached|hscc_end|hscc_history|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|9|9|sink_left_group|Sink012|True|0 0.5|pan_5|True|3|672|0@256 1@528 2@672|0.4968|True|True|3|752|0@576 1@576 2@752|
|1|5|5|sink_left_group|Sink007|False|0 0.5|pan_6|True|3|1040|0@368 1@816 2@1040|0.5|True|True|3|768|0@336 1@576 2@768|
|2|10|10|sink_right_group|Sink014|False|0 0.5|pan_7|True|3|688|0@288 1@496 2@688|0.5001|True|True|3|864|0@272 1@656 2@864|
|3|5|5|sink_left_group|Sink007|False|0 0.5|pan_6|True|3|880|0@304 1@640 2@880|0.4129|True|True|3|1040|0@320 1@768 2@1040|
|4|8|8|sink_right_group|Sink039|False|0 0.5|pan_8|False|0|1600||0.0|False|True|3|688|0@224 1@448 2@688|
|5|10|10|sink_right_group|Sink014|False|0 0.5|pan_6|True|3|928|0@352 1@720 2@928|0.5001|True|True|3|752|0@288 1@576 2@752|
|6|5|5|sink_left_group|Sink007|False|0 0.5|pan_6|True|3|1104|0@256 1@544 2@1104|0.5001|True|False|2|1600|0@304 1@640|
|7|10|10|sink_right_group|Sink014|False|0 0.5|pan_7|True|3|1184|0@288 1@624 2@1184|0.47|True|True|3|720|0@256 1@480 2@720|
|8|8|8|sink_right_group|Sink039|False|0 0.5|pan_5|False|2|1600|0@304 1@560|0.0|False|False|2|1600|0@976 1@976|
|9|5|5|sink_left_group|Sink007|False|0 0.5|pan_6|True|3|848|0@304 1@608 2@848|0.5|True|True|3|832|0@304 1@592 2@832|
|10|8|8|sink_right_group|Sink039|False|0 0.5|pan_8|False|2|1600|0@288 1@560|0.0866|False|False|2|1600|0@304 1@528|
|11|2|2|sink_island_group|Sink009|False|0 0.5|pan_4|False|2|1600|0@336 1@624|0.0001|False|False|2|1600|0@320 1@608|
|12|1|1|sink_main_group|Sink025|True|0 0.5|pan_6|False|1|1600|0@640|0.4936|True|True|3|992|0@512 1@880 2@992|
|13|9|9|sink_left_group|Sink012|True|0 0.5|pan_4|False|0|1600||0.5|True|True|3|704|0@496 1@496 2@704|
|14|2|2|sink_island_group|Sink009|False|0 0.5|pan_8|True|3|848|0@304 1@608 2@848|0.4435|True|False|2|1600|0@384 1@672|
|15|10|10|sink_right_group|Sink014|False|0 0.5|pan_7|True|3|656|0@256 1@480 2@656|0.4026|True|True|3|800|0@272 1@480 2@800|
|16|6|6|sink_main_group|Sink020|True|0 0.5|pan_6|False|0|1600||0.0003|False|True|3|976|0@352 1@352 2@976|
|17|8|8|sink_right_group|Sink039|False|0 0.5|pan_8|False|2|1600|0@512 1@512|0.0|False|True|3|1312|0@368 1@608 2@1312|
|18|9|9|sink_left_group|Sink012|True|0 0.5|pan_7|True|3|752|0@224 1@560 2@752|0.5001|True|True|3|720|0@208 1@608 2@720|
|19|6|6|sink_main_group|Sink020|True|0 0.5|pan_6|True|3|1024|0@352 1@352 2@1024|0.4615|True|True|3|992|0@304 1@304 2@992|
|20|9|9|sink_left_group|Sink012|True|0 0.5|pan_7|True|3|1200|0@560 1@1056 2@1200|0.4864|True|||||
|21|4|4|sink_left_group|Sink006|True|0 0.5|pan_7|True|3|912|0@672 1@672 2@912|0.5002|True|||||
|22|8|8|sink_right_group|Sink039|False|0 0.5|pan_5|False|2|1600|0@304 1@576|0.0|False|||||
|23|1|1|sink_main_group|Sink025|True|0 0.5|pan_7|False|2|1600|0@1136 1@1152|0.0007|False|||||
|24|2|2|sink_island_group|Sink009|False|0 0.5|pan_6|False|1|1600|0@400|0.0002|False|||||
|25|6|6|sink_main_group|Sink020|True|0 0.5|pan_4|True|3|928|0@368 1@640 2@928|0.4383|True|||||
|26|9|9|sink_left_group|Sink012|True|0 0.5|pan_6|True|3|736|0@256 1@608 2@736|0.5029|True|||||
|27|1|1|sink_main_group|Sink025|True|0 0.5|pan_7|False|2|1600|0@384 1@768|0.0002|False|||||
|28|8|8|sink_right_group|Sink039|False|0 0.5|pan_5|False|2|1600|0@400 1@656|0.0|False|||||
|29|7|7|sink_island_group|Sink004|False|0 0.5|pan_4|True|3|944|0@256 1@480 2@944|0.5001|True|||||

- Every sink is class `Sink` with `handle_joint` range "0 0.5", axis (0,1,0) and the same water_on threshold.
- The faucet model differs per sink asset (SinkNNN). Scene layout and style ids are always equal to each other.

**Clustering:**
- The faucet-only failures (pan and sponge passed, water off) are s8, s10, s17, s22, s28 (layout 8, Sink039, single basin, end q=0.0–0.087), s11 (layout 2, Sink009, island), and s23, s27 (layout 1, Sink025; their notes are really caused by the gripper term).
- Layout 8 / Sink039 on ASUS: **0/6** (s4, s8, s10, s17, s22, s28; s4 never got the pan in).
- The same layout on hscc (s4, s8, s10, s17): **2/4** (s4 and s17 succeeded; s17 needed 1312 steps).
- Layout 1 / Sink025: 0/3 on ASUS (s12, s23, s27); hscc s12 succeeded.
- Every other sink has at least one ASUS success: Sink009 1/3, Sink012 4/5, Sink007 4/4, Sink014 4/4, Sink020 2/3, Sink006 1/1, Sink004 1/1.

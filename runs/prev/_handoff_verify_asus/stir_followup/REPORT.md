# StirVegetables follow-up (ASUS seen11 run 20261014, read-only, CPU, no env steps)

Scripts: `stir_followup.py` (items 2-4, output `stir_followup.json`, log `stir_followup.log`), `demo_stir_len.py` (item 5, `demo_stir_len.json`),
`tables.py` (renders `tables.md`), `explore.py`/`dbg_pot.py` (geom inspection). `s0_last.png`, `s1_last.png` = last video frame (ffmpeg) used to
confirm the pot really is tipped in success end states.

## Item 1: success logic (robocasa/environments/kitchen/composite/sauteing_vegetables/stir_vegetables.py)
See the final hand-off message / source lines 20-40, 59-62, 64-112, 114-160.
* `contact_with_spatula` is a local initialised `False` on every call (l.139): it does NOT persist across calls.
  Within one call it latches across the loop (l.152-153 only re-check while False), loop order `["veg1","veg2"]` (l.129):
  contact with veg1 makes veg2's contact condition true too; contact with veg2 alone never helps veg1. Net per-step condition:
  spatula touches **veg1**; veg2 contact is never required.
* Movement (l.143-148): xy norm of `body_xpos` minus `prev_obj_pos_<name>`, in cm (x1e2), must be > 0.05 cm (0.5 mm) per call.
  `prev_obj_pos_<name>` is set at the end of every loop iteration (l.158), i.e. on every `_detect_stirring` call, which only
  happens while `check_obj_grasped(spatula)` is true (l.127-129). Called once per env.step via `Kitchen._post_action -> update_state`
  (kitchen.py:1492). Very first call: getattr default = current pos -> movement 0. After a period without grasp the comparison is
  against the last grasped-step position (stale). The attributes are never reset anywhere (grep: only this file), so they survive env resets.
  `success_time` is cumulative, never decremented/reset on failure (only `_setup_scene` l.61): 5 qualifying steps in total, not consecutive.
* `check_obj_grasped` (object_utils.py:665-685) = gripper-object contact AND both finger joints < 0.035; finger_joint2 is negative
  for the Panda, so effectively only joint1 < 0.035.

## Methods (items 2-4)
* Start state: `reset_to(scene npz)`; end state: `set_state(end npz)` + `mj_forward` (no stepping). NB `oracle.set_state` calls
  `env.update_state()`, which mutates this analysis env's `success_time`/`prev_obj_pos_*` only.
* Geometry from collision geoms (contype/conaffinity != 0). Mesh vertices -> world via geom_xpos/geom_xmat (checked against geom_aabb, max err 0).
* Distances: `mujoco.mj_geomDistance` over all collision-geom pairs, distmax 1 m (mesh = convex hull of each decomposition piece);
  negative = penetration.
* Pot: pots are often **tipped** at the end (s0 47°, s2 58°, s4 36°...; verified visually on s0's last frame), so pot quantities are
  in the pot body frame: inner bottom = max local z of pot collision vertices within 0.5*R_int of the pot axis, rim = max local z of
  vertices within 1.3*R_int (R_int = `pot_reg_int` cylinder radius). Spatula low point = min pot-local z over spatula collision vertices
  ("above bottom" = low - bottom; "below rim" = rim - low; negative below-rim = entirely above rim). World-z variant also given.
* Spatula long axis = body axis with the largest extent of its collision vertices (body frame): local **y** in all 30 scenes.
  Tilt = angle between that axis and world z. Yaw change: ZYX-euler yaw from body quaternions (ill-conditioned when the spatula
  rolls) plus heading change of the long axis projected to xy, plus total rotation angle.
* Gripper opening = finger_joint1 - finger_joint2 qpos; pad separation = distance between the two pad-geom centres.

### Item 2: start state, all 30 scenes (xy distances from start robot base, cm)
| seed | res | L | knob | veg1 | veg2 | pot | spatula | base→pot | →veg1 | →veg2 | →spat | spatula on | spat→pot xy | vegs same counter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | succ(4/4) | 9 | front_center | potato_9 | tomato_5 | Pot069 | spatula_7 | 44.7 | 79.4 | 74.3 | 103.0 | counter_1_group | 85.7 | Y |
| s1 | succ(4/4) | 5 | front_left | tomato_6 | potato_7 | Pot065 | spatula_9 | 45.8 | 70.2 | 67.4 | 61.1 | counter_right_group | 69.6 | Y |
| s2 | succ(4/4) | 9 | front_center | lime_3 | cucumber_3 | Pot075 | spatula_9 | 40.6 | 79.1 | 86.3 | 78.0 | counter_1_group | 62.8 | Y |
| s3 | succ(4/4) | 9 | left | lime_3 | lime_3 | Pot074 | spatula_6 | 64.3 | 55.7 | 69.8 | 81.7 | counter_1_group | 113.2 | Y |
| s4 | succ(4/4) | 8 | front_left | bell_pepper_3 | bell_pepper_3 | Pot065 | spatula_8 | 48.0 | 71.1 | 73.2 | 105.0 | counter_2_left_group | 64.2 | Y |
| s5 | succ(4/4) | 7 | center | potato_9 | LemonWedge010 | Pot075 | spatula_5 | 54.2 | 53.3 | 54.8 | 104.8 | counter_group | 77.1 | Y |
| s6 | succ(4/4) | 5 | rear_left | onion_7 | lemon_5 | Pot065 | spatula_5 | 67.9 | 78.7 | 74.0 | 66.6 | counter_right_group | 77.7 | Y |
| s7 | fail(3/4) | 10 | front_right | sweet_potato_3 | potato_7 | Pot075 | spatula_7 | 42.2 | 67.5 | 59.7 | 83.4 | counter_1_group | 52.7 | Y |
| s8 | fail-step3(2/4) | 8 | rear_left | eggplant_4 | garlic_4 | Pot075 | spatula_6 | 60.0 | 77.1 | 64.0 | 88.5 | counter_1_left_group | 104.2 | Y |
| s9 | fail(1/4) | 5 | rear_left | eggplant_4 | eggplant_4 | Pot074 | spatula_5 | 64.3 | 86.5 | 80.0 | 73.3 | counter_right_group | 78.0 | Y |
| s10 | fail-hold(3/4) | 8 | rear_right | cucumber_4 | broccoli_3 | Pot069 | spatula_7 | 60.5 | 70.5 | 88.0 | 80.6 | counter_2_left_group | 86.9 | Y |
| s11 | fail-hold(3/4) | 2 | rear_center | avocado_8 | garlic_4 | Pot065 | spatula_8 | 64.2 | 66.7 | 84.3 | 81.4 | counter_group | 69.9 | Y |
| s12 | succ(4/4) | 7 | rear_right | tomato_5 | squash_16 | Pot071 | spatula_7 | 66.4 | 72.3 | 47.6 | 82.0 | counter_group | 86.0 | Y |
| s13 | succ(4/4) | 9 | left | sweet_potato_3 | squash_7 | Pot071 | spatula_8 | 67.6 | 62.4 | 57.2 | 77.2 | counter_1_group | 105.4 | Y |
| s14 | fail-hold(3/4) | 2 | front_right | garlic_4 | avocado_5 | Pot065 | spatula_7 | 47.7 | 75.4 | 80.2 | 86.7 | counter_group | 96.5 | Y |
| s15 | fail-hold(3/4) | 10 | rear_left | sweet_potato_3 | mushroom_8 | Pot071 | spatula_6 | 69.0 | 66.7 | 85.8 | 82.9 | counter_1_group | 72.9 | Y |
| s16 | fail-hold(3/4) | 6 | rear_left | garlic_4 | corn_3 | Pot069 | spatula_5 | 68.0 | 74.0 | 68.0 | 95.8 | counter_2_right_group | 95.5 | Y |
| s17 | fail-step3(2/4) | 8 | front_left | lime_2 | carrot_4 | Pot075 | spatula_6 | 45.1 | 74.8 | 75.5 | 97.8 | counter_2_left_group | 62.1 | Y |
| s18 | fail(3/4) | 9 | front_center | broccoli_3 | lemon_8 | Pot075 | spatula_7 | 45.1 | 78.7 | 85.4 | 89.9 | counter_1_group | 66.8 | Y |
| s19 | fail(3/4) | 6 | front_left | cucumber_5 | lemon_6 | Pot071 | spatula_5 | 44.7 | 80.3 | 67.1 | 69.3 | counter_1_right_group | 34.5 | Y |
| s20 | fail(3/4) | 9 | front_right | corn_4 | potato_9 | Pot071 | spatula_8 | 58.4 | 79.8 | 84.0 | 99.5 | counter_1_group | 51.8 | Y |
| s21 | succ(4/4) | 4 | front_left | corn_4 | sweet_potato_3 | Pot074 | spatula_9 | 64.9 | 57.8 | 58.8 | 68.2 | counter_1_right_group | 104.1 | Y |
| s22 | fail-step3(2/4) | 8 | front_right | LemonWedge009 | eggplant_3 | Pot071 | spatula_8 | 43.3 | 65.0 | 60.6 | 79.5 | counter_1_left_group | 39.6 | Y |
| s23 | fail-hold(3/4) | 1 | front_right | lemon_9 | avocado_6 | Pot075 | spatula_6 | 37.9 | 62.0 | 73.4 | 91.8 | counter_right_group | 60.1 | Y |
| s24 | succ(4/4) | 4 | front_center | squash_5 | potato_8 | Pot074 | spatula_9 | 37.8 | 70.0 | 71.9 | 94.2 | counter_1_right_group | 85.3 | Y |
| s25 | fail(3/4) | 6 | front_right | cucumber_5 | tomato_5 | Pot065 | spatula_8 | 42.4 | 68.6 | 77.6 | 85.0 | counter_1_right_group | 93.4 | Y |
| s26 | fail(0/4) | 9 | rear_center | broccoli_3 | bell_pepper_3 | Pot071 | spatula_8 | 67.9 | 62.2 | 67.5 | 65.1 | counter_1_group | 61.6 | Y |
| s27 | fail(3/4) | 1 | rear_left | LemonWedge009 | eggplant_3 | Pot065 | spatula_8 | 66.9 | 71.7 | 87.8 | 105.8 | counter_right_group | 99.9 | Y |
| s28 | fail(3/4) | 7 | center | bell_pepper_3 | onion_6 | Pot065 | spatula_9 | 52.4 | 79.0 | 77.5 | 98.6 | counter_group | 71.3 | Y |
| s29 | succ(4/4) | 10 | rear_left | carrot_6 | lime_2 | Pot075 | spatula_8 | 65.0 | 68.5 | 66.9 | 59.7 | counter_group | 41.6 | Y |

### Item 3: layout-8 spatula end state
| seed | res | spat dxy cm | dz cm | dyaw (quat ZYX) deg | long-axis heading Δ deg | total rot deg | gripper qpos (j1,j2) | opening j1-j2 cm | pad-centre sep cm | min pad↔spat cm | min finger/pad↔spat cm | grasped (official) | gripper contacts spat | spatula on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4 | succ | 75.4 | 22.4 | -179.2 | -64.3 | 163.6 | [0.0109, -0.0121] | 2.3 | 3.0 | -0.07 | -0.07 | True | True | robot0_base |
| s8 | fail-step3 | 2.3 | 0.0 | -7.2 | -7.2 | 7.2 | [0.0108, -0.006] | 1.7 | 2.4 | -0.11 | -0.11 | True | True | counter_1_left_group,robot0_base |
| s10 | fail-hold | 91.6 | 25.7 | -165.7 | 81.4 | 163.1 | [0.0128, -0.014] | 2.7 | 3.4 | -0.1 | -0.1 | True | True | robot0_base |
| s17 | fail-step3 | 3.3 | 1.5 | 79.5 | -9.0 | 90.1 | [0.0392, -0.04] | 7.9 | 8.6 | 54.51 | 54.3 | False | False | counter_2_left_group |
| s22 | fail-step3 | 2.5 | -0.1 | 12.0 | 12.0 | 12.0 | [0.0008, -0.0002] | 0.1 | 0.8 | 10.54 | 10.01 | False | False | counter_1_left_group |

### Item 4: spatula vs pot / vegetables at end (pot frame; negative distance = penetration)
| seed | res | pot tilt deg | pot depth cm | spat low above inner bottom cm | below rim cm | low pt radial cm (R_int) | frac spat verts inside R_int | world: low above bottom / below rim cm | min spat↔veg1 cm | spat↔veg2 cm | spat↔pot cm | contacts pot | contacts veg | vegs in pot | handle tilt from vertical deg | grasped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s10 | fail-hold | 8.2 | 14.2 | 12.2 | 2.0 | 9.7 (8.8) | 0.0 | 10.7 / 4.9 | 10.13 | 4.86 | 0.0 | False | - | veg1,veg2 | 32.6 | True |
| s11 | fail-hold | 0.1 | 11.8 | 9.0 | 2.9 | 17.1 (9.2) | 0.024 | 8.9 / 2.9 | 14.43 | 10.63 | 3.07 | False | - | veg1,veg2 | 40.5 | True |
| s14 | fail-hold | 0.4 | 11.8 | 3.7 | 8.2 | 2.0 (9.2) | 1.0 | 3.7 / 8.2 | 2.39 | 2.68 | 3.56 | False | - | veg1,veg2 | 12.8 | True |
| s15 | fail-hold | 0.0 | 12.5 | 8.0 | 4.5 | 3.1 (9.2) | 1.0 | 8.0 / 4.5 | 4.78 | 7.05 | 5.3 | False | - | veg1,veg2 | 9.6 | True |
| s16 | fail-hold | 0.0 | 14.2 | 5.3 | 8.9 | 4.6 (8.8) | 0.622 | 5.3 / 8.9 | 2.7 | 7.28 | 4.27 | False | - | veg1,veg2 | 31.9 | True |
| s23 | fail-hold | 0.0 | 13.3 | 16.9 | -3.6 | 5.7 (8.0) | 0.774 | 16.9 / -3.6 | 12.08 | 13.71 | 4.9 | False | - | veg1,veg2 | 35.0 | True |
| s0 | succ | 47.3 | 14.2 | 4.1 | 10.1 | 5.8 (8.8) | 0.955 | -0.5 / 18.4 | 0.08 | 2.06 | 0.28 | False | - | veg1,veg2 | 20.2 | True |
| s1 | succ | 47.3 | 11.8 | 2.7 | 9.2 | 8.3 (9.2) | 0.769 | -2.6 / 19.2 | -0.01 | -0.0 | -0.0 | True | veg1,veg2 | veg1,veg2 | 14.4 | True |
| s2 | succ | 58.5 | 13.3 | 2.4 | 10.9 | 5.2 (8.0) | 0.462 | -1.9 / 17.4 | -0.01 | -0.0 | -0.0 | True | veg1,veg2 | veg1,veg2 | 19.6 | True |
| s3 | succ | 0.0 | 18.5 | 3.1 | 15.4 | 2.3 (10.8) | 1.0 | 3.1 / 15.4 | -0.01 | -0.0 | 3.12 | False | veg1,veg2 | veg1,veg2 | 18.0 | True |
| s4 | succ | 36.3 | 11.8 | 5.8 | 6.1 | 6.9 (9.2) | 0.594 | 0.9 / 15.3 | 0.0 | 1.56 | 0.0 | False | - | veg1,veg2 | 23.4 | True |
| s5 | succ | 0.2 | 13.3 | 2.5 | 10.8 | 7.6 (8.0) | 0.64 | 2.5 / 10.8 | -0.0 | -0.04 | -0.01 | True | veg1,veg2 | veg1,veg2 | 2.5 | True |
| s6 | succ | 23.9 | 11.8 | 3.5 | 8.3 | 0.4 (9.2) | 0.543 | 2.9 / 12.4 | -0.0 | 1.91 | -0.0 | True | veg1 | veg1,veg2 | 72.7 | False |
| s12 | succ | 0.1 | 12.5 | 0.9 | 11.6 | 6.0 (9.2) | 0.883 | 0.9 / 11.6 | -0.05 | 3.85 | -0.03 | True | veg1 | veg1,veg2 | 25.3 | True |
| s13 | succ | 12.8 | 12.5 | 4.4 | 8.1 | 5.1 (9.2) | 0.907 | 3.3 / 11.5 | 0.02 | 0.01 | -0.0 | True | - | veg1,veg2 | 21.8 | True |
| s21 | succ | 31.9 | 18.5 | 7.2 | 11.3 | 8.6 (10.8) | 0.808 | 1.9 / 20.2 | 0.0 | 3.12 | -0.02 | True | - | veg1,veg2 | 38.0 | True |
| s24 | succ | 6.5 | 18.5 | 5.1 | 13.3 | 9.1 (10.8) | 1.0 | 4.1 / 15.6 | -0.03 | 1.85 | -0.01 | True | veg1 | veg1,veg2 | 18.8 | True |
| s29 | succ | 2.5 | 13.3 | 1.4 | 11.9 | 7.7 (8.0) | 0.861 | 1.7 / 11.8 | 1.5 | 5.24 | -0.02 | True | - | veg1,veg2 | 16.0 | True |

Notes: Item 2 counter = root body of the fixture the spatula contacts at start; vegs always rest on the same counter as the spatula (30/30).
Item 3: s8 still officially "grasped" with the spatula on the counter (moved 2.3 cm, z +0.0); s17 gripper fully open, pads 54 cm away,
spatula rolled ~90° in place (heading change only -9°); s22 gripper fully closed on nothing 10 cm from the spatula.
Item 4: in all 6 still-holding failures the spatula touches neither veg nor the pot; in s10/s11 it is (mostly) outside the pot
footprint, in s23 entirely above the rim. Successes: s6 no longer grasped in the end npz (the success counter is cumulative).

Policy step-4 durations in the successes (env steps, history granularity 16): s0 656, s1 112, s2 240, s3 592, s4 160, s5 144, s6 544,
s12 208, s13 192, s21 256, s24 592, s29 528.

## Item 5: training demos (StirVegetables/20250814/lerobot, 501 demos, 20 fps = 1 frame per env step)
Length in frames from end of last PickUpSpatula segment to episode end (includes NavigateKitchen in 74 demos and the 1-11 frame
`done` segment); second column = StirVegetables:execute frames only.

| layout | n | after-PickUpSpatula min/med/max | StirVegetables:execute only min/med/max |
|---|---|---|---|
| 1 | 56 | 98/279.5/755 | 97/263.5/754 |
| 2 | 51 | 134/263/671 | 88/217/670 |
| 3 | 57 | 126/255/521 | 120/244/513 |
| 4 | 59 | 90/224/521 | 82/205/520 |
| 5 | 43 | 20/241/653 | 12/198/652 |
| 6 | 51 | 95/203/474 | 94/199/470 |
| 7 | 46 | 105/234/440 | 97/223.5/438 |
| 8 | 39 | 133/232/468 | 111/220/467 |
| 9 | 47 | 108/223/514 | 107/213/513 |
| 10 | 52 | 96/208/450 | 83/204.5/448 |
| all | 501 | 20/236/755 | 12/220/754 |

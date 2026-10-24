import json, sys
R = json.load(open(sys.argv[1])); out = []
P = out.append
short = lambda s: s.split("/")[-1]
def cls(r):
    s = r["seed"]
    if r["success"]: return "succ"
    if s in (8, 17, 22): return "fail-step3"
    if s in (10, 11, 14, 15, 16, 23): return "fail-hold"
    return "fail"
P("### Item 2: start state, all 30 scenes (xy distances from start robot base, cm)")
P("| seed | res | L | knob | veg1 | veg2 | pot | spatula | base→pot | →veg1 | →veg2 | →spat | spatula on | spat→pot xy | vegs same counter |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in R:
    b = r["base_xy_cm"]; done = sum(x is not None for x in r["done"])
    P(f"| s{r['seed']} | {cls(r)}({done}/4) | {r['layout']} | {r['knob']} | {short(r['models']['veg1'])} | {short(r['models']['veg2'])} | {short(r['models']['pot'])} | {short(r['models']['spatula'])} | {b['pot']} | {b['veg1']} | {b['veg2']} | {b['spatula']} | {','.join(x.replace('_main','') for x in r['spatula_on'])} | {r['spatula_to_pot_xy_cm']} | {'Y' if r['veg_same_counter_as_spatula'] else 'N: v1=' + ','.join(r['veg1_on']) + ' v2=' + ','.join(r['veg2_on'])} |")
P("\n### Item 3: layout-8 spatula end state")
P("| seed | res | spat dxy cm | dz cm | dyaw (quat ZYX) deg | long-axis heading Δ deg | total rot deg | gripper qpos (j1,j2) | opening j1-j2 cm | pad-centre sep cm | min pad↔spat cm | min finger/pad↔spat cm | grasped (official) | gripper contacts spat | spatula on |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in R:
    if r["seed"] not in (8, 17, 22, 4, 10): continue
    e = r["end"]
    P(f"| s{r['seed']} | {cls(r)} | {e['spatula_dxy_cm']} | {e['spatula_dz_cm']} | {e['spatula_dyaw_deg_quat']} | {e['spatula_dheading_longaxis_deg']} | {e['spatula_rot_angle_total_deg']} | {e['gripper_qpos']} | {e['gripper_opening_cm']} | {e['pad_center_sep_cm']} | {e['min_pad_to_spatula_cm']} | {e['min_finger_or_pad_to_spatula_cm']} | {e['spatula_grasped_official']} | {e['gripper_contacts_spatula']} | {','.join(x.replace('_main','') for x in e['spatula_on_end'])} |")
P("\n### Item 4: spatula vs pot / vegetables at end (pot frame; negative distance = penetration)")
P("| seed | res | pot tilt deg | pot depth cm | spat low above inner bottom cm | below rim cm | low pt radial cm (R_int) | frac spat verts inside R_int | world: low above bottom / below rim cm | min spat↔veg1 cm | spat↔veg2 cm | spat↔pot cm | contacts pot | contacts veg | vegs in pot | handle tilt from vertical deg | grasped |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(R, key=lambda r: (r["success"], r["seed"])):
    if "end" not in r or "pot_tilt_deg" not in r["end"] or r["seed"] in (8, 17, 22): continue
    e = r["end"]
    P(f"| s{r['seed']} | {cls(r)} | {e['pot_tilt_deg']} | {e['pot_depth_cm']} | {e['spatula_low_above_bottom_cm']} | {e['spatula_low_below_rim_cm']} | {e['spatula_low_radial_cm']} ({e['pot_R_int_cm']}) | {e['spatula_frac_verts_inside_Rint']} | {e['world_spatula_low_above_bottom_cm']} / {e['world_spatula_low_below_rim_cm']} | {e['min_spatula_veg1_cm']} | {e['min_spatula_veg2_cm']} | {e['min_spatula_pot_cm']} | {e['spatula_contacts_pot']} | {','.join(e['spatula_contacts_veg']) or '-'} | {','.join(e['veg_in_pot']) or '-'} | {e['spatula_tilt_from_vertical_deg']} | {e['spatula_grasped_official']} |")
open(sys.argv[2], "w").write("\n".join(out) + "\n"); print("\n".join(out))

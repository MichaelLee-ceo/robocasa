# StoreLeftoversInBowl running notes

## Official criteria (from source)
- Stage 0/1: OU.check_obj_in_receptacle(food, bowl) = food in CONTACT with bowl AND xy centre dist < 0.7*bowl horizontal radius; plus gripper not touching the food. No upright check, no height check -> food under an upside-down bowl on the floor counts as "in bowl" (s0).
- Stage 2 = env._check_success(): both foods in bowl + fridge.check_rack_contact(bowl) + gripper_obj_far(bowl, 0.25).
  check_rack_contact: bowl's 8 bbox corners inside (5 cm tol) a FRIDGE-compartment SHELF region (not drawer, not freezer),
  region base z in [0.5, 1.5] m, region height >= 0.20 m, and bowl centre z between region bottom and 90% of region height.
- Packet bowl dist_to_target = distance to ANY fridge interior site (get_int_sites, all regions) -> d_end 0.0 with empty at_target
  means "inside the fridge but not on a qualifying shelf" (s21, s25, s26, s29).

## Flow (success s5)
veg pick -> bowl, drumstick pick -> bowl, grasp bowl by rim (often pinching a food item too), base drives to fridge
(side-by-side or French door already open), place bowl on a shelf, retreat. Bowl tilts a lot during carry even in successes (s5 tilt max 73).

## Layout counts
L9: s3 s4 s5* s8 s10 s11 s17* s18 s19 s20 s22 s27 (2/12 succ)
L10: s0 s2 s6* s7* s13* s14* s15* s25 s26 s29 (5/10)
L7: s1 s9* s24* ; L8: s12 s16* s28 ; L4: s21 ; L2: s23*

## Packet-first triage of failures
- bowl released at the fridge and fell ~1 m to floor (hold height_drop ~1.0-1.4): s0 s1 s2 s3 s8 s11 s12 s18 s22 s27 s20(2nd hold) s28(0.8)
- bowl inside fridge (d_end 0), upright, both foods in, but at_target empty (not on a qualifying shelf): s21 s25 s26 s29
- s4: bowl regrasped repeatedly at fridge to the end (bowl z 1.58, tilt 79), foods fell out
- s10: veg bounced out of bowl at 143-158 (stage1 never) ; bowl on shelf 805-1057 with drumstick only
- s19: bowl released 756 near fridge, drop 0.08 -> where? end z 0.05 (floor)

## Per-episode findings
- s0 (L10): drumstick held over bowl 410-633 (slow stage1). Rim pinch grasp w/ drumstick 902; bowl rotated on its side during carry (1096); held tilted against the FRONT EDGE of a high shelf 1180-1260, gripper opened 1192-1248, bowl fell to floor 1268. Root stage 2, placed_then_fell_or_slid. Food under inverted bowl on floor counts as in-bowl (criterion quirk).
- s1 (L7): rim pinch with drumstick 604, bowl near-vertical; base STALLED 820-1136 1.2 m short of fridge; bowl slipped out of closed fingers 1136-1144 (width 0.019->0.003), fell to floor. Then drove to fridge empty-handed. dropped_in_transit.
- s9 (L7 success, slow stage2 1237 steps): at fridge by ~900, bowl tipped 1069-1525 held against the side-by-side fridge shelves ~500 steps before placing 1489-1519.
- s2 (L10): 2 missed broccoli grasps (352, 548), slow stage1 541. Bowl tipped on side during carry 1108; held at FRONT EDGE of middle shelf, gripper opened 1192, bowl tumbled off to floor. Same mechanism as s0.
- KEY: L10 bowl end x: successes 0.18-0.28 (s6 .18, s13 .20, s14 .22, s15 .23, s7 .28) vs s25 .316, s26 .331, s29 .305 at same z (~1.08-1.09) -> s25/26/29 bowl sits on the shelf but sticks out over the front edge by ~3-10 cm; check_rack_contact needs all 8 bbox corners inside (5 cm tol). Near-miss placements. VERIFY visually.
- s3 (L9): at fridge ~860, held bowl at the opening of the narrow right compartment, released 968 at 8 cm outside interior -> floor. placed_off_target.
- L9 geometry: side-by-side fridge, base parks in front of the CLOSED left (blue, dispenser) door, arm reaches right into narrow open compartment. Successes s5/s17/(s10) put the bowl deeper on a middle shelf; many failures hold the bowl high near the top/front of the opening (s3 s11 s18 s19 s27 s4).
- Labelling rule: bowl released > 5 cm outside the interior -> placed_off_target; released at/on the shelf edge (<= 5 cm) then fell -> placed_then_fell_or_slid. (s0 had 0.105 -> should be placed_off_target? re-check at the end.)
- s4 (L9): bowl tipped >45 from 621 (rim grasp incl. lime); at fridge raised high, grip loosened, drumstick out ~785, lime ~845 -> floor. Then empty bowl wedged at top of compartment 870-1700. undid_previous_stage (spill) + stall.
- s8 (L9): bowl tilted from 609; raised to TOP of compartment; drumstick spilled ~805; gripper opened 816-824 with bowl not on a shelf -> bowl rode on open hand as arm withdrew -> fell 917 0.59 m outside. undid_previous_stage + placed_off_target.
- s10 (L9): lime released on bowl rim 143, rolled off to counter 158-176; never retrieved. Bowl (drumstick only) placed correctly on middle shelf 805 -> shows L9 shelf placement IS possible. Root stage 0, placed_then_fell_or_slid.
- s11 (L9): bowl tipped from 627; pressed sideways at upper shelf FRONT EDGE 700-784; foods slid out 766/778 (floor); gripper opened 784; bowl fell. undid_previous_stage primary (spill before release) + placed_then_fell_or_slid.
- s12 (L8): rim grasp clamped the drumstick lying across the rim (582); bowl hung by friction; slipped out ~876-888 at fridge approach (0.59 m out) -> floor. Drumstick kept in hand to end. dropped_in_transit.
- s18 (L9): bowl tipped from 707; raised to TOP shelf front edge, ~125-step fiddle 840-964; opened 968 at 3 cm outside -> floor. placed_then_fell_or_slid. Stage1 'undone' during tilted carry = xy-offset artefact, not spill.
- s19 (L9): top shelf FRONT EDGE again; opened 744-768 at 4.8 cm outside; bowl perched on shelf lip+open fingers (rack contact 759-761 only); arm withdrew 816-828 -> bowl fell. placed_then_fell_or_slid + disturbed_placed_object.
- s20 (L9): bowl rotated nearly inverted from 688; middle-shelf FRONT EDGE; opened 856-864 at 5.5 cm; bowl balanced on hand/lip, fell 927. Drumstick under inverted bowl on floor -> stage 0 'complete' at end (criterion quirk).
- s21 (L4, French door 083): bowl upright on TOP shelf (z 1.485) w/ both foods, gripper away from 880. Rack check excludes top shelf (z_range 0.5-1.5 or 90%-height filter). Visually success. placed_off_target + official_criterion_issue.
- s25 (L10, fridge 066): bowl upright on the correct shelf (z same as successes) but x 0.316 vs successes 0.18-0.28 (deeper = smaller x); shelf regions in 066 are shallower (half-depth 0.247 @+0.066) than main (0.38 @+0.031) -> front ~17 cm of physical shelf not covered. Near-miss. Unchanged 862-1700.
- s26 (L10, bowl_9 large): same near-miss, x 0.344 -> 0.331 after late nudge 1463-1481. s29 (L10, bowl_5): x 0.305 (s13 success same bowl at 0.197). All three: bowl static after release ~840 steps, no push-in.
- s22 (L9): bowl sideways (743) at the 2nd-highest shelf front inside opening; pinch slipped 816-824 (fingers closed through) -> bowl tumbled off -> floor. placed_then_fell_or_slid (grasp slip). Stage0 'complete' at end via drumstick touching bowl on floor.
- s27 (L9): bowl tilted from 558; raised ABOVE top shelf ~900; held ~270 steps with fingers slowly opening 0.031->0.054; lemon/drumstick slid out 1133-1166; bowl fell 1171 (0.22 m outside). stall_holding_at_target.
- s28 (L8): base parked against open left door; arm took bowl LOW beside door, tilted; opened 728-760 at 11 cm outside; bowl landed on its side at z 0.49 outside fridge. placed_off_target. (medium)
- NOTE: in L8 (s12 s16 s28) and L10 (s0) the drumstick plate is ~1 m from the bowl -> stage 1 includes a base move; long drumstick 'holds' there are partly navigation, not stalls. s0 still had a real hover 410-633 over the bowl.
- Successes: bowl tilt > 45 deg during the carry is COMMON in successes too (s5 s6 s7 s13 s14 s15 s16 s9) -> tilt alone is not the failure; what differs is levelling the bowl at the shelf and releasing it fully over the shelf.
- s15 (success): bowl carried via a grasp on the drumstick+rim (like s12) at 504-528, width 0.047; close_events has NO entry for it (tooling gap?).
- s17 (success): missed bowl grasp 377 (closed to 0.001), re-grasp 475 ok.
- s9 (success L7): ~500 steps (1000-1500) holding the bowl at the front of an upper shelf of the side-by-side fridge before placing 1489-1519 — the same 'hold at shelf front' behaviour that fails elsewhere, recovered here.
- s16 (success L8): stage 1 366 = base move with drumstick, not a stall.
## KEY quantitative finding: insertion depth at bowl release (saved states)
L9 (deeper = +x): successes release x 5.558 (s5), 5.632 (s17); s10 5.481 (rack contact true). Failures: s3 5.376, s11 5.415, s18 5.426, s19 5.410, s20 5.404, s22 5.463, s27 5.241, s8 4.87 (pulled out). Height: successes z 1.15-1.19; top-shelf failures s18 1.46, s19 1.45, s27 1.59-1.67, s4 1.5-1.6.
L10 (deeper = -x): successes end x 0.18-0.28; s25 .316, s26 .344, s29 .305 (stayed on shelf, region fail), s2 .375 and s0 .457 (fell).
L8: s16 success x 0.552; s28 released 0.79 (low, fell).
=> dominant stage-2 mechanism: bowl released 5-25 cm short of the shelf depth reached in successes (at/over the shelf front edge), often tilted; it either falls (most) or stays on the front lip and fails the region check (s25/26/29).

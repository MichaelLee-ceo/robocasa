# PackIdenticalLunches running notes

## Task facts (from env source)
- success: each tupperware has exactly 1 veg + 1 meat (check_obj_in_receptacle = contact with tupperware AND horiz dist < 0.7*radius), gripper >25cm from all in-items.
  -> an item resting ON another item (no tupperware contact) does NOT count.
- items: veg0/meat0 at fridge pos x=-0.25, veg1/meat1 at +0.25, z 1.0-1.5 (two fridge shelves). Veg/meat categories vary per episode (not only onion/drumstick).
- tupperwares sampled on counter near fridge; can be 0.23 m apart (layout 5) up to ~1.25 m (s5, layout 7). Robot is mobile: drive fridge <-> counter each item.
- packet tupperwares are NOT tracked as objects (no tupperware holds/contacts).

## Successes: s3, s8, s9, s18 (L2, L5, L5, L5). Stage medians 286/401/419/393.
Composition in successes: s3 v0+m0 | v1+m1; s8 v0+m0 | v1+m1; s9 v0+m1 | v1+m0; s18 v1+m0 | v0+m1.

## Final composition of failures (from end_pos clustering; verify in video)
- s1: v0,v1,m0 | m1  (two vegs together)
- s2: v0,m0,m1 | v1  (two meats together)
- s6: v0,v1,m0 | m1
- s20: v0,v1,m1 | m0
- s19: v0,v1 | m0,m1 (sorted by kind)
- s29: v0,v1 | m0 ; m1 dropped (z 0.65)
- s5: v0 | m0,m1 ; v1 in fridge
- s10: v0,m0 | m1 ; v1 on floor
- s17: v0,v1,m0 all one tupperware ; m1 on floor (dropped at ~207)
- s0: v1+m1 ; v0 floor, m0 at z1.11 elsewhere
- s11: v0 | v1 (good split), meats never lifted
- s12: m0 | m1 (good split), lemon wedges never lifted
- s13: v1+m1 ; v0, m0 never lifted

## Per-episode findings
- s24 (L2): m0 placed 329 (stage0 353). Then gripper open 336-2600, base never drove back to fridge; arm hovered/rotated at the counter, wrist cam toward oven/microwave. Second tupperware did NOT move (checked zooms 400-2600) -> could not confirm "grabbed the empty tupperware"; hand hovers at its front edge (500-700, 2140-2300) with open fingers.
- s25 (L2): garlic slipped out ~205 during turn, landed ON THE ROBOT BASE (end z 0.40), empty place 300-380, back at fridge 620-2600 stalling with closed fingers (closed on air 705, 1822), one failed pinch of a flat steak slice 2275.
- s14 (L4): steak hung by edge, fell to floor 157 at fridge; then 3 cycles close-on-air/slip -> drive to counter with closed empty fingers -> EMPTY PLACE (~500, ~1180, ~2000).
- s16 (L7): closes on air ~10cm above mushroom (89) and steak (777); empty place ~600; arm folded down looking at own base 960-2600 (frozen).
- s23 (L10): took tomato to a tupperware-FREE corner counter (other L10 tupperware site, where s7/s15 have them); hovered holding it 200-420, slipped to floor 427, stalled 1450 steps, 2nd item (fish 2083) taken to same wrong place.
- s27 (L10): lemon wedge slipped out at close 51; fingers stayed closed 72-2600, arm hovering (2530-step stall).
- s26 (L3): thin flat steak slices (tilt 53 = lying flat) unpinchable 128/151, 644; empty place ~800; pepper close 1170 no lift; folded stall 1350-2250; 2nd empty trip ~2400.
- s28 (L7): sweet potato held by its tip, slipped over the counter edge ~245 -> floor 28cm short of tupperware; then fish push 612, closes on air, empty trips.

## Recurring mechanisms (draft)
A. Missed / weak grasp then carries on: empty place at counter with closed fingers (s14, s16, s25, s26, s28), or stalls hovering with fingers closed for 1000s of steps (s25, s27, s16-fold, s26-fold).
B. Drops in transit from tip/edge grasps (s14 steak edge, s28 potato tip, s25 garlic, s23 tomato).
C. Flat steak slices (steak_4/5 lying flat, tilt ~53) are hard to pinch: s25, s26.
D. Navigation/destination errors: s24 never returned to fridge; s23 went to wrong counter.
- s4 (L2): sweet potato placed 307; thick steak slipped at 608 (lift 3.4cm); arm folded toward own base 700-2600 (frozen).
- s7 (L10): wedge #1 to correct LEFT (microwave-corner) tupperware 264; wedge #2 driven RIGHT to toaster counter (no tupperware), held 720-848, slipped to floor; steak driven farther to stove, released over floor 1673; wandered.
- s15 (L10): steak #1 to correct LEFT tupperware 310; steak #2 driven RIGHT to stove, carried 1000 steps with folded arm, slipped onto robot base 1640.
- s21 (L7): close on drumstick BONE TIP at 89, 1030 steps hovering closed, empty place ~1150, tomato placed 1501 (stage0 1519!), then close on air 1762 and closed stall to end.
- s22 (L2): broccoli placed 260 (no close event logged: wide object); then close on air 626, empty place ~980; closes on air 1251/1271, empty place ~1490; failed close on steak 1789; folded freeze.
- LAYOUT 10 has 2 tupperware sites: left microwave corner (s7, s15) and right counter ~(1.5,-0.4) (s17, s29, s23). Policy goes to the wrong one: s23 (went left, tupps right), s7/s15 (item 2 went right though tupps left and item 1 went left correctly).
- Folded-arm freeze (wrist cam looking at own base): s4, s16, s22, s26 (partial), s28?
- Packet: 1-step "object left target" undo events = contact flicker (s7, s21, s22), not real.
- Packet: close_events misses grasps of wide objects (broccoli s22) since width never < open threshold.
- s0 (L7): onion fell to floor at fridge ~170 (task impossible from here); empty place 300; v1+m1 correct pair (836, 1315); last drumstick slipped & LODGED ON ROBOT ARM ~1640-1687 (end z1.11); empty place right tupperware ~1760. meat1 on top of onion: at_target flickers (128 intervals) -> criterion fragility.
- s5 (L7): pumpkin miss + empty place (stage0 759); stage1 ok 1148; 1148-1920 idle hovering/drifting; close air 2039 + empty place ~2220; meat1 into the meat0 tupperware at 2586 (two meats) -> timeout.
- s11 (L2): fish x2 ungrasped (509 both fish, 1715 trout), each followed by empty place; placed both broccoli instead (v|v split).
- s12 (L3): drumsticks placed correctly (one per tupperware); lemon wedge squeezed out 1025; closed-finger stall 1048-2600.
- s13 (L7): v1+m1 pair OK by 722; flat lime wedge squeezed out x3 (1034, 1688, 2472) each -> empty place over right tupperware.
- WEDGES (LemonWedge*) are a failure magnet: s27, s12, s13 (and s7 dropped one at wrong counter).
- s1 (L2): veg1 (3rd item) into A already holding trout+butternut (1146) while B empty next to it -> root stage 2 placed_off_target; meat1 to B 2267.
- s2 (L5): 2nd fish into R with 1st fish (503), L empty beside it -> root stage 1. All 4 placed by 1297.
- s6 (L5): 4th item (lime) into R which already had fish+lime (1257); L had only fish -> root stage 3.
- s10 (L2): composition correct; last pepper grasped 1735, arm TUCKED (carry pose) and robot froze ~700 steps holding it at the fridge, pepper slipped 2513.
- s17 (L10 right site): T-bone held by edge fell to floor at 207 (first item) -> impossible; 2nd eggplant into the full tupperware (1626).
- s19 (L3): 2nd pepper into tupperware with 1st pepper (1545) -> sorted by kind 2 meat | 2 veg.
- s20 (L4): 2nd garlic onto 1st garlic in L (~950), only counted at 1027 when it tipped into contact with the tupperware (stacked item not 'in').
- s29 (L10 right site): 2nd broccoli into tupperware with 1st (821); drumstick fell to fridge bottom 1055.
COMPOSITION ERRORS (wrong tupperware): s1(st2), s2(st1), s6(st3), s19(st2), s20(st1), s29(st1), s5 (2 meats, st2, late), s17(st2, secondary). The policy has no memory of what is inside; it seems to pick a tupperware regardless (often the one nearest/first used).
TUCKED POSE: The 'folded arm looking at own base' is the carry/navigation pose; freezes in it: s4, s16, s22, s10 (holding pepper), s26, s15 (holding steak while driving).

## Successes done (s3, s8, s9, s18): all content-correct choices; s8 rim placement nudged in (603->650).
## Layout table: L5 (0.64 m fridge->tupperware) 20/20 items placed, 3/5 success (2 fails = composition). Other layouts 42/100 items, 1/25 success (s3 L2).
## L10 two tupperware sites: (0.7,-2.9) left/microwave corner [s7,s15] and (1.5,-0.4) right/toaster [s17,s29,s23]. s7/s15 trip 2 went to the right site (empty); s23 trip 1 went to the left site (empty). Site confusion.
## Root-cause stage tally: 0:10, 1:8, 2:6, 3:2. Primary: placed_off_target 6, missed_grasp 5, missed_grasp_continued_empty 5, dropped_in_transit 3, other-wrong-destination 3, wander_away 2, stall_pre_grasp 1, other-tucked-freeze 1.
## Tooling notes: lift_m relative to start height -> small for shelf->counter carries; close_events miss wide-object grasps (s22 broccoli, s3 mushroom); no tupperware membership field; tupperwares untracked; items on robot body not flagged.

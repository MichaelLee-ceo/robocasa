"""Plain-language completion definitions for every atomic step of the 16 composite-seen tasks.

Written from the simulator checks that produce the ground truth (runs/scripts/atomic_steps.py,
atomic_steps_seen11.py, and each composite task's _check_success for the last step), so a VLM judge is told the same
criterion the tracker uses. DEFS[task][k] = (counts as completed, does not count as completed).
"""

AWAY = "the robot has let go of it and moved its gripper away from it (about 25 cm or more)"
OFF = "the gripper is no longer touching it"

DEFS = {
    "KettleBoiling": [
        (f"the kettle rests on the stove top, centred within about 15 cm of a burner, and {AWAY}.",
         "the kettle is still held or in the air, still on the counter, on the stove but not over a burner, or the gripper is still at the kettle."),
        (f"the kettle sits on a burner (within about 15 cm), THAT burner is turned on, and {AWAY}.",
         "the burner under the kettle is still off, a different burner was turned on, or the kettle is not on a burner."),
    ],
    "PreSoakPan": [
        (f"the pan lies completely inside the sink basin and {AWAY}.",
         "the pan is still held, still on the counter, resting on the sink rim or only partly inside the basin, or the gripper is still at the pan."),
        (f"the sponge lies completely inside the sink basin and {AWAY}.",
         "the sponge is still held, still on the counter, only partly inside the basin, or the gripper is still at the sponge."),
        ("water is running from the faucet while the pan and the sponge are both completely inside the sink basin and the gripper is away from both.",
         "no water is running, or the pan or sponge is not fully inside the basin, or the gripper is still at one of them."),
    ],
    "StackBowlsCabinet": [
        (f"one of the two bowls is inside the open cabinet and {OFF}.",
         "neither bowl is inside the cabinet yet (still held, on the counter, or only partly inside)."),
        ("both bowls are inside the cabinet, one bowl is nested inside the other, and the gripper has moved away from both (about 25 cm or more).",
         "a bowl is still outside the cabinet or still held, the bowls sit side by side instead of stacked, or the gripper is still at a bowl."),
    ],
    "SteamInMicrowave": [
        (f"the vegetable lies in the bowl on the counter and {OFF}.",
         "the vegetable is still in the sink, still held, or outside the bowl."),
        (f"the bowl is inside the microwave and {AWAY}.",
         "the bowl is still on the counter, still held, only partly inside the microwave, or the gripper is still at the bowl."),
        ("the microwave door is closed.",
         "the door is still open or only partly closed."),
        ("the vegetable is in the bowl, the bowl is inside the microwave, the door is closed, and the microwave has been switched on (start button pressed).",
         "the start button has not been pressed, or the door is open, or the bowl / vegetable is not in place."),
    ],
    "PrepareCoffee": [
        (f"the mug stands under the coffee machine dispenser, in the pouring position on the machine's tray, and {AWAY}.",
         "the mug is still in the cabinet, still held, on the counter beside the machine, not under the spout, or the gripper is still at the mug."),
        ("the mug is under the dispenser, the coffee machine has been switched on with its button, and the gripper is away from both the mug and the button.",
         "the button has not been pressed, the mug is not under the dispenser, or the gripper is still at the button or the mug."),
    ],
    "ScrubCuttingBoard": [
        ("the gripper is holding the sponge and the sponge is lifted off the counter.",
         "the sponge still lies on the counter, or the gripper is near it but not holding it."),
        ("while holding the sponge, the robot has rubbed it on the cutting board at several different spots (at least 5, spread over at least 10 cm), and has then released the sponge and moved the gripper at least 15 cm away from it.",
         "the sponge has not touched the board, or only touched it in one small area, or the robot is still holding the sponge or still has its gripper next to it."),
    ],
    "WashLettuce": [
        ("water is running from the faucet.",
         "the faucet is still off (no water stream), even if the gripper is touching the handle."),
        ("the lettuce has been under the running water stream for a while (in total at least about one second).",
         "the water is off, or the lettuce has not yet been under the water stream long enough (it is still on the counter, held away from the stream, or only just reached it)."),
    ],
    "RinseSinkBasin": [
        ("water is running from the faucet.",
         "the faucet is still off (no water stream), even if the gripper is touching the handle."),
        ("while the water was running, the spout has been turned to all three positions — left, centre and right (in any order, over the course of the video).",
         "the water is off, or the spout has not yet been pointed in all three directions with water running."),
    ],
    "LoadDishwasher": [
        (f"at least one of the two dishes rests on the dishwasher rack and {OFF}.",
         "no dish is on the rack yet (dishes still on the counter, held, or on the open door)."),
        ("both dishes rest on the dishwasher rack and the gripper is not touching either of them.",
         "only one dish is on the rack, or the gripper is still holding / touching the second dish."),
        ("both dishes are on the rack and the dishwasher door is fully closed.",
         "the door is open or only partly closed, or a dish is not on the rack."),
    ],
    "StoreLeftoversInBowl": [
        (f"at least one of the two food items lies in the bowl and {OFF}.",
         "no food item is in the bowl yet (still on its plate, held, or dropped elsewhere)."),
        ("both food items lie in the bowl and the gripper is not touching them.",
         "only one food item is in the bowl, or the gripper is still holding the second one."),
        ("both food items are in the bowl, the bowl stands on a shelf inside the fridge, and the gripper has moved away from the bowl (about 25 cm or more).",
         "the bowl is still on the counter, still held, not on a fridge shelf, a food item fell out, or the gripper is still at the bowl."),
    ],
    "SetUpCuttingStation": [
        (f"the knife lies on the cutting board and {OFF}.",
         "the knife is still in the drawer, still held, or lying off the board."),
        ("the knife and the meat both lie on the cutting board, and the gripper has moved away from the knife and the board (about 25 cm or more).",
         "the meat is not on the board yet, the knife is no longer on the board, or the gripper is still at the board."),
    ],
    "StirVegetables": [
        (f"at least one of the two vegetables lies in the pot and {OFF}.",
         "no vegetable is in the pot yet (still on the counter, held, or dropped outside)."),
        ("both vegetables lie in the pot and the gripper is not touching them.",
         "only one vegetable is in the pot, or the gripper is still holding the second one."),
        ("the gripper is holding the spatula and the spatula is lifted off the counter.",
         "the spatula still lies on the counter, or the gripper touches it without holding it up."),
        ("while holding the spatula, the robot has pushed both vegetables around inside the pot (the vegetables visibly move), with the pot on the burner of the task.",
         "the spatula has not touched the vegetables, the vegetables did not move, or the robot is not holding the spatula."),
    ],
    "SearingMeat": [
        (f"the pan rests on the stove within about 15 cm of the named burner and {AWAY}.",
         "the pan is still in the cabinet, still held, on a different burner, or the gripper is still at the pan."),
        (f"the meat lies inside the pan and {OFF}.",
         "the meat is still in its container, still held, or outside the pan."),
        ("the named burner has been turned on, with the pan on it and the meat in the pan, and the gripper is away from the meat (about 25 cm or more).",
         "the burner is still off, a different burner was turned on, or the pan / meat is not in place."),
    ],
    "PackIdenticalLunches": [
        (f"at least one of the four food items lies in a tupperware on the counter and {AWAY}.",
         "no food item is in a tupperware yet (still in the fridge, held, or dropped elsewhere)."),
        (f"at least two of the four food items lie in the tupperwares and {AWAY}.",
         "fewer than two items are in the tupperwares, or the gripper is still at the last one placed."),
        (f"at least three of the four food items lie in the tupperwares and {AWAY}.",
         "fewer than three items are in the tupperwares, or the gripper is still at the last one placed."),
        ("each of the two tupperwares contains exactly one vegetable and exactly one meat item, and the gripper has moved away from all items (about 25 cm or more).",
         "an item is missing, a tupperware has two of the same kind, or the gripper is still at an item."),
    ],
    "DeliverStraw": [
        ("the drawer is pulled fully open (out to the end of its travel).",
         "the drawer is still closed or only partly open."),
        ("the straw stands in the glass cup on the dining counter and the robot has let go of it and moved its gripper away (about 25 cm or more).",
         "the straw is still in the drawer, still held, lying next to the cup, or the gripper is still at the straw."),
    ],
    "GetToastedBread": [
        ("the bread was in a toaster slot when the lever was pushed down (toaster switched on), and the toasting has since finished: the lever has popped back up.",
         "the lever has not been pushed down, the bread was not in the slot, or the lever is still down (still toasting)."),
        ("the toasted bread lies on the plate and the robot has let go of it and moved its gripper away (about 25 cm or more).",
         "the bread is still in the toaster, still held, not on the plate, or the gripper is still at the bread."),
    ],
}

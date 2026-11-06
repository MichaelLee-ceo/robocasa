"""Back off (proposal §3, recovery step 2): open the gripper and retreat a little, then let GR00T try again.

TODO (owner: ___). Contract:
    backoff(ep, trace, distance=0.10, chunks=3) -> dict(chunks=..., steps=..., ...)
Drive the robot with plain actions through ep.step (as retrace.py does with its override): gripper open, the hand
moved back along the approach direction (the hand's own -z in the base frame, or the reverse of its last motion from
trace.path), base still. Report `steps` (env steps used) so the coordinator can decide whether they count as GR00T's
time. Check it on recorded failure points as retrace.py was checked (runs/20261105_recovery/check_retrace.py)."""


def backoff(ep, trace, distance=0.10, chunks=3):
    raise NotImplementedError("backoff: see the module doc")

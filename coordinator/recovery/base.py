"""Interventions the coordinator can execute (loop.Coordinator.execute):
  continue     nothing
  resample     fresh noise for GR00T (a new seed stream), robot where it is
  retrace      play the robot's trace back to where the attempt began (motion/retrace.py), then fresh noise
  reinstruct   another sentence for GR00T (Intervention.instruction) from now on, fresh noise (mechanism only: GR00T
               barely acts on step-level sentences, runs/20261105_recovery README)
  backoff      open the gripper and retreat a little, then fresh noise (TODO: motion/backoff.py)
  stop         stop intervening for the rest of the episode"""
EXECUTABLE = ("continue", "resample", "retrace", "reinstruct", "stop")   # + "backoff" once motion/backoff.py exists


class RecoveryPolicy:
    uses_sim = False

    def decide(self, memory, obs, trigger):
        raise NotImplementedError


def used(memory, stage=None):
    return [iv for iv in memory.interventions if stage is None or iv["stage"] == stage]

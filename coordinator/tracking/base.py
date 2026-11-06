class ProgressTracker:
    """make_plan(instruction, obs) -> [SubGoal]  once, at the start
    update(memory, obs) -> StageStatus           every `track_every` chunks; StageStatus.done = None: no answer
    uses_sim                                     True if it reads the simulator (an oracle)"""
    uses_sim = False

    def make_plan(self, instruction, obs):
        raise NotImplementedError

    def update(self, memory, obs):
        raise NotImplementedError

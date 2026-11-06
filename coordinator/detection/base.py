class Detector:
    """check(memory, obs, status) -> [Trigger]; called after every chunk. status: this chunk's StageStatus, or None
    when the tracker did not run (or had no answer). uses_sim: True if it reads the simulator."""
    uses_sim = False

    def check(self, memory, obs, status):
        raise NotImplementedError


class Confirmer:
    """confirm(memory, obs, trigger) -> (act on it?, reason)"""
    uses_sim = False

    def confirm(self, memory, obs, trigger):
        raise NotImplementedError

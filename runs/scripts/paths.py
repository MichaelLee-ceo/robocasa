"""Where things are, relative to this checkout (2026-11-04).

ROOT is the repository: $ROBOCASA_ROOT if set (the shell launchers export it), else found from this file. RUNS is
ROOT/runs, and PY is the GR00T python ($GROOT_PY, else ROOT/envs/groot/bin/python).
Data written before 2026-11-04 stores absolute paths of the checkouts it was made in: jobs files and results rows hold
scene paths, and scene XMLs hold asset paths. local() maps them into this checkout. It is the identity in the original
checkout."""
import os

ROOT = os.path.abspath(os.path.expanduser(os.environ.get("ROBOCASA_ROOT")
                                          or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RUNS = os.path.join(ROOT, "runs")
PY = os.environ.get("GROOT_PY", os.path.join(ROOT, "envs", "groot", "bin", "python"))
OLD_ROOTS = ("/home/michaellee/mclee/robocasa", "/home/michaellee/Desktop/robocasa")  # the checkouts that wrote the data


def local(p):
    """p (a path, or a string holding paths such as a scene XML) with an earlier checkout's root replaced by ROOT"""
    for old in OLD_ROOTS:
        if old != ROOT:
            p = p.replace(old + "/", ROOT + "/")
    return p

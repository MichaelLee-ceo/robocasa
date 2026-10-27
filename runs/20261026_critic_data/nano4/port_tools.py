"""Ports for run_inside.sh: nano4 nodes are shared between jobs, so a port derived from the job id can collide (it did
on 2026-10-27: two shards on one node shared a port, and one shard's workers silently used the other shard's server).
usage: python port_tools.py free N          print N currently free TCP ports
       python port_tools.py owns PID PORT   exit 0 if process PID holds the listening socket on PORT"""
import os, socket, sys


def free(n):
    socks = []
    for _ in range(n):
        s = socket.socket(); s.bind(("", 0)); socks.append(s)
    print(" ".join(str(s.getsockname()[1]) for s in socks))
    for s in socks:
        s.close()


def listening_inodes(port):
    out = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = open(f).read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            p = row.split()
            if int(p[1].split(":")[1], 16) == port and p[3] == "0A":  # 0A = LISTEN
                out.add(p[9])
    return out


def owns(pid, port):
    ino = listening_inodes(port)
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False
    for fd in fds:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in ino:
            return True
    return False


if __name__ == "__main__":
    if sys.argv[1] == "free":
        free(int(sys.argv[2]))
    else:
        sys.exit(0 if owns(int(sys.argv[2]), int(sys.argv[3])) else 1)

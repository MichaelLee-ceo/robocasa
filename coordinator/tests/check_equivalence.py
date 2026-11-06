"""Is coordinator `oracle_retrace` the same method as runs/20261105_recovery/loop.py arm `retrace`?

The coordinator is driven by the exact GR00T chunks that loop.py recorded for the same failure point and seed
(loop/retrace/<id>_p<seed>.pkl.gz), instead of sampling GR00T. The physics and the retrace are deterministic, so if
the two loops make the same decisions, the coordinator asks for GR00T chunks at exactly the env chunks where loop.py
ran GR00T, retraces where loop.py retraced (same length), and ends with the same outcome. The first difference is
reported: where the coordinator asked for a GR00T chunk at a chunk where loop.py was retracing (or had stopped), a GR00T
request with another noise stream or instruction than loop.py's, or a retrace or outcome that differs.

usage: python -I coordinator/tests/check_equivalence.py <point id>:<seed> ...   (GR00T server port not used)
       prints one JSON line per run: equal, first_difference, and both sides' retraces and outcomes"""
import gzip, json, os, pickle, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from coordinator import configs, paths  # noqa: E402
from coordinator.points import load_point, run_from_point  # noqa: E402

paths.use_envkit()
import envkit  # noqa: E402


class Divergence(Exception):
    pass


class ReplayExecutor:
    """GR00T replaced by loop.py's recorded chunks. At each request it also checks GR00T's inputs other than the
    observation: the noise stream (the seed is chunk_seed(f"{tag}/{stream}", 0, chunk index) in both, so equal tags and
    streams mean equal seeds) and the instruction (None in loop.py's retrace arm)."""

    def __init__(self, chunks, gr00t_idx, log, memory_ref):
        self.chunks, self.ok, self.log, self.mem = chunks, set(gr00t_idx), log, memory_ref

    def expected_stream(self, ci):
        """loop.py's stream at env chunk ci: start; a<k>.<n> after an intervention; s<k> after moving on (its target
        stage taken from the coordinator's own moved-on record at that chunk, which the retrace checks pin down)"""
        st = "start"
        for x in self.log:
            if x["chunk"] > ci or (x["chunk"] == ci and x["event"] == "intervene" and x["chunk"] + x["chunks"] > ci):
                break
            if x["event"] == "intervene":
                st = f"a{x['stage']}.{x['attempt']}"
            elif x["event"] == "stage_done":
                to = next((e["to"] for e in self.mem().moved_on if e["chunk"] == x["chunk"]), "?")
                st = f"s{to}"
        return st

    def act(self, ep_obs, memory, chunk_index):
        if chunk_index not in self.ok:
            raise Divergence(f"the coordinator asked GR00T for env chunk {chunk_index}, where loop.py "
                             f"{'retraced' if chunk_index < len(self.chunks) else 'had stopped'}")
        want = self.expected_stream(chunk_index)
        if memory.stream != want or memory.gr00t_instruction is not None:
            raise Divergence(f"GR00T's input differs at env chunk {chunk_index}: stream {memory.stream!r} vs loop.py "
                             f"{want!r}, instruction {memory.gr00t_instruction!r} vs None")
        return {k: np.array(v) for k, v in self.chunks[chunk_index]["actions"].items()}


def check(pid, pseed):
    L = pickle.load(gzip.open(os.path.join(paths.RECOVERY, "loop", "retrace", f"{pid}_p{pseed}.pkl.gz")))
    out_l, chunks_l = L["out"], L["chunks"]
    ret_l = [(x["chunk"], x["chunks"]) for x in out_l["log"] if x["event"] == "intervene"]
    in_retrace = {c for a, n in ret_l for c in range(a, a + n)}
    gr00t = [i for i in range(L["start_chunk"], len(chunks_l)) if i not in in_retrace]
    point, rec = load_point(pid)
    ep = envkit.Episode(point["task"], point["seed"], horizon=10 ** 6)
    co = configs.build("oracle_retrace", ep, 5581, tag=f"loop/{pid}/p{pseed}")
    co.executor = ReplayExecutor(chunks_l, gr00t, out_l["log"], lambda: co.memory)
    res = dict(id=pid, pseed=pseed, task=point["task"], loop=dict(retraces=ret_l, chunks=len(chunks_l),
                                                                  success=out_l["success"], fixed=out_l["fixed"],
                                                                  fixed_after=out_l["fixed_after"]))
    diff = None
    try:
        r, _ = run_from_point(co, point, rec)
    except Divergence as x:
        diff, r = str(x), None
    m = co.memory
    ret_c = [(iv["chunk"], iv["result"].get("chunks")) for iv in m.interventions if iv["kind"] == "retrace"]
    res["coordinator"] = dict(retraces=ret_c, chunks=len(ep.chunks), success=None if r is None else r["success"],
                              fixed=None if r is None else r["fixed"], fixed_after=None if r is None else r["fixed_after"],
                              triggers=[(t.kind, t.stage, t.chunk) for t in m.triggers])
    res["loop"]["events"] = [(x["chunk"], x["event"], x.get("stage")) for x in out_l["log"]]
    if diff is None:
        if ret_c != ret_l:
            diff = f"retraces differ: coordinator {ret_c} vs loop.py {ret_l}"
        elif len(ep.chunks) != len(chunks_l):
            diff = f"run length differs: coordinator {len(ep.chunks)} chunks vs loop.py {len(chunks_l)}"
        elif (r["success"], r["fixed"], r["fixed_after"]) != (out_l["success"], out_l["fixed"], out_l["fixed_after"]):
            diff = (f"outcome differs: coordinator success {r['success']} fixed {r['fixed']} after {r['fixed_after']} vs "
                    f"loop.py {out_l['success']} {out_l['fixed']} after {out_l['fixed_after']}")
        elif r["success_t"] != out_l["success_step"]:
            diff = f"success time differs: coordinator {r['success_t']} vs loop.py {out_l['success_step']}"
    res.update(equal=diff is None, first_difference=diff)
    ep.close()
    return res


if __name__ == "__main__":
    for a in sys.argv[1:]:
        pid, s = a.rsplit(":", 1)
        try:
            print(json.dumps(check(pid, int(s)), default=str), flush=True)
        except Exception as x:
            import traceback
            print(json.dumps(dict(id=pid, pseed=int(s), error=f"{type(x).__name__}: {x}",
                                  trace=traceback.format_exc()[-800:])), flush=True)

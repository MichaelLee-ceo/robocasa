"""Offline checks of the VLM layer: backends (against stub HTTP servers, no model), the registry, answer checking,
memory -> request building, and the VLM modules with a mock VLM. python -I coordinator/tests/test_vlm.py"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from coordinator import context as C, prompts as P  # noqa: E402
from coordinator.detection.confirm import VLMConfirmer  # noqa: E402
from coordinator.memory import Memory  # noqa: E402
from coordinator.recovery.vlm import VLMRecovery  # noqa: E402
from coordinator.schema import Intervention, SubGoal, Trigger  # noqa: E402
from coordinator.tests._util import obs  # noqa: E402
from coordinator.tracking.vlm import VLMTracker  # noqa: E402
from coordinator.vlm import VLMRequest, check, make_vlm, parse_json  # noqa: E402
from coordinator.vlm.hf_server import HFServerVLM  # noqa: E402
from coordinator.vlm.mock import MockVLM  # noqa: E402
from coordinator.vlm.openai_compat import OpenAICompatVLM  # noqa: E402


def stub_server(reply, status=200, port=0):
    """a local HTTP server answering every POST with `reply(body)`; returns (port, received bodies, server)"""
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            got.append(dict(path=self.path, body=body, auth=self.headers.get("Authorization")))
            out = json.dumps(reply(body)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1], got, srv


SCHEMA = {"type": "object", "required": ["a", "k"], "properties": {"a": {"type": "boolean"},
                                                                  "k": {"type": "string", "enum": ["x", "y"]}}}


def test_parse_and_check():
    assert parse_json('<think>hmm {"no": 1}</think>\n```json\n{"a": true, "k": "x"}\n```') == {"a": True, "k": "x"}
    assert check({"a": True, "k": "x"}, SCHEMA) == []
    assert check({"a": 1, "k": "z"}, SCHEMA) == ["answer.a: expected boolean, got int", "answer.k: 'z' not one of ['x', 'y']"]
    assert check({"a": True}, SCHEMA) == ["answer: missing k"]
    r = MockVLM('{"a": true, "k": "z"}').ask(VLMRequest(parts=[("text", "q")], schema=SCHEMA))
    assert not r.ok and "schema" in r.error
    r = MockVLM("no json here").ask(VLMRequest(parts=[("text", "q")]))
    assert not r.ok and "unparseable" in r.error
    print("parse and check: ok")


def test_registry():
    v = make_vlm("vllm:Qwen/Qwen3.8-27B-FP8@8102")
    assert isinstance(v, OpenAICompatVLM) and v.base_url == "http://127.0.0.1:8102/v1" and v.model == "Qwen/Qwen3.8-27B-FP8"
    assert v.thinking_flag is False and v.temperature == 0 and v.price == (0.0, 0.0)
    v = make_vlm("ollama:qwen2.5vl:7b")
    assert v.model == "qwen2.5vl:7b" and v.base_url == "http://127.0.0.1:11434/v1" and v.thinking_flag is None
    v = make_vlm("sglang:m@gpu-box:30001")
    assert v.base_url == "http://gpu-box:30001/v1"
    v = make_vlm("hf:qwen3.5-9b@8105")
    assert isinstance(v, HFServerVLM) and v.url.endswith(":8105/answer") and v.name == "qwen3.5-9b"
    assert isinstance(make_vlm("mock"), MockVLM)
    c = make_vlm("claude:claude-opus-5-5", effort="low")
    assert c.price == (4.0, 20.0) and c.effort == "low" and c.cost(1000, 1000) == 0.024
    try:
        make_vlm("nonsense:x")
        raise AssertionError("an unknown provider must fail")
    except ValueError:
        pass
    print("registry: ok")


def test_openai_compat():
    port, got, srv = stub_server(lambda b: {"model": b["model"], "choices": [{"message": {"content": '{"a": false, "k": "y"}'},
                                                                              "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 120, "completion_tokens": 9}})
    v = make_vlm(f"vllm:test-model@{port}", image_size=128)
    req = VLMRequest(parts=[("text", "look:"), ("image", obs(1).images["video.robot0_eye_in_hand"])], system="sys",
                     schema=SCHEMA, tag="t")
    r = v.ask(req)
    assert r.ok and r.data == {"a": False, "k": "y"} and r.input_tokens == 120 and r.cost_usd == 0.0, r
    b = got[0]["body"]
    assert got[0]["path"] == "/v1/chat/completions" and b["messages"][0] == {"role": "system", "content": "sys"}
    content = b["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "look:"} and content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert b["response_format"]["json_schema"]["schema"] == SCHEMA and b["chat_template_kwargs"] == {"enable_thinking": False}
    assert b["temperature"] == 0 and got[0]["auth"] is None
    v2 = OpenAICompatVLM("m", f"http://127.0.0.1:{port}/v1", structured=False)
    v2.ask(req)
    assert "response_format" not in got[1]["body"] and "chat_template_kwargs" not in got[1]["body"]
    srv.shutdown(); srv.server_close()
    r = make_vlm(f"vllm:m@{port}", retry_s=0).ask(req)   # server gone
    assert not r.ok and "cannot reach" in r.error
    # a server restart: the call waits and gets its answer when the server is back
    import time
    later = []
    threading.Timer(3, lambda: later.append(stub_server(lambda b: {"choices": [{"message": {"content": '{"a": true, "k": "y"}'},
                                                                                "finish_reason": "stop"}]}, port=port))).start()
    t0 = time.time()
    r = make_vlm(f"vllm:m@{port}", retry_s=60).ask(req)
    assert r.ok and r.data["k"] == "y" and 3 <= time.time() - t0 < 30, (r.error, time.time() - t0)
    later[0][2].shutdown(); later[0][2].server_close()
    print("OpenAI-compatible backend: ok (retries while the server is down)")


def test_hf_server():
    port, got, srv = stub_server(lambda b: {"text": 'sure: {"a": true, "k": "x"}', "seconds": 1.0, "input_tokens": 50,
                                            "output_tokens": 5})
    r = make_vlm(f"hf:local@{port}").ask(VLMRequest(parts=[("text", "q"), ("image", obs(2).images["video.robot0_eye_in_hand"])],
                                                    system="s", schema=SCHEMA))
    assert r.ok and r.data["k"] == "x" and got[0]["body"]["parts"][1]["type"] == "image", r
    srv.shutdown(); srv.server_close()
    port, got, srv = stub_server(lambda b: {"error": "CUDA out of memory"}, status=500)
    r = make_vlm(f"hf:local@{port}").ask(VLMRequest(parts=[("text", "q")]))
    assert not r.ok and "500" in r.error and "out of memory" in r.error, r.error
    srv.shutdown(); srv.server_close()
    print("HF server backend: ok")


def memory_mid_task():
    """a memory in the middle of a 3-sub-goal task: sub-goal 1 done, sub-goal 2 on its second attempt after a retrace"""
    m = Memory("Boil the water.", [SubGoal(0, "Put the kettle on the stove.", "the kettle is on a burner"),
                                   SubGoal(1, "Turn on the burner."), SubGoal(2, "Wait.")], expected={0: 30, 1: 20, 2: 10})
    m.begin_stage(0, obs(0))
    for c in range(1, 41):
        o = obs(c, hand=(0.3 + 0.002 * c, 0, 0.6))
        m.observe(o)
        if c == 12:
            m.complete_stage(0, o); m.begin_stage(1, o)
        if c == 34:
            m.end_attempt(1, o, "timeout -> retrace")
            m.log_intervention(Intervention("retrace", 1), o, dict(arrived=True))
            m.begin_stage(1, obs(36), why="retrace")
    m.add_note(obs(30), "the kettle lid is open", "tracker")
    return m


def test_context():
    m, now = memory_mid_task(), obs(41)
    r = C.progress_request(m, now)
    t = r.text()
    assert t.startswith("A robot is doing a kitchen task in simulation. Task instruction: \"Boil the water.\"\n"
                        "The task has these stages, in order:\n  1. Put the kettle on the stove."), t[:300]  # the pilot's intro
    assert "  1. done at 9.6 s (complete when: the kettle is on a burner)" in t, t[:900]
    assert "  2. CURRENT: attempt 2, 4.0 s into it (usually done within 16 s)" in t and "  3. not started" in t
    assert "t = 32.8 s (chunk 41), wrist camera:" in t                                 # the pilot's frame lines
    assert "the kettle lid is open" in t and "Robot sensors now: gripper open" in t
    labels = [p[1] for p in r.parts if p[0] == "text" and p[1].startswith("---")]
    assert labels[0] == "--- when sub-goal 2 first began; when sub-goal 1 was done ---" and labels[-1] == "--- now ---", labels
    assert r.n_images() == 3 * len(labels) and r.schema is P.PROGRESS_SCHEMA and r.parts[-1][1] == P.PROGRESS_Q
    trig = Trigger("timeout", 1, 41)
    c = C.confirm_request(m, now, trig)
    ct = c.text()
    assert "Attempts at sub-goal 2:" in ct and "ended at" in ct and "timeout -> retrace" in ct and "retrace (arrived back)" in ct
    assert "Alarm at 32.8 s on sub-goal 2: this attempt has taken" in ct and "when this attempt began" in ct
    assert any(p[1].startswith("--- during this attempt") for p in c.parts if p[0] == "text"), [p[1] for p in c.parts if p[0] == "text"]
    rq = C.recovery_request(m, now, trig, ("retrace", "resample", "continue"))
    assert rq.schema["properties"]["action"]["enum"] == ["retrace", "resample", "continue"]
    assert "- retrace: drive the robot back" in rq.text() and "reinstruct" not in rq.parts[-1][1], rq.parts[-1][1]
    one_cam = C.progress_request(m, now, C.ContextConfig(cameras=("video.robot0_eye_in_hand",), n_recent=1))
    assert one_cam.n_images() == len([p for p in one_cam.parts if p[0] == "text" and p[1].startswith("---")])
    print("context: ok", f"(progress request: {r.n_images()} images, {len(t)} chars of text)")


def test_vlm_modules():
    m, now = memory_mid_task(), obs(41)
    s = VLMTracker(MockVLM("garbage")).update(m, now)
    assert s.done is None and s.error
    trig = Trigger("timeout", 1, 41)
    cf = VLMConfirmer(MockVLM({"failure": True, "kind": "no_progress", "reason": "hovering", "confidence": 0.8, "note": ""}))
    assert cf.confirm(m, now, trig) == (True, "no_progress: hovering") and trig.detail["vlm_kind"] == "no_progress"
    assert cf.confirm(m, now, Trigger("failure_point", 1, 41))[0] is True and len(cf.vlm.requests) == 1
    rec = VLMRecovery(MockVLM({"action": "reinstruct", "instruction": "", "reason": "try words"}))
    iv = rec.decide(m, now, trig)
    assert iv.kind == "reinstruct" and iv.instruction == "Turn on the burner." and iv.reason == "try words"
    iv = VLMRecovery(MockVLM("garbage")).decide(m, now, trig)
    assert iv.kind == "retrace" and iv.reason.startswith("VLM error, fallback")
    assert VLMRecovery(MockVLM({}), budget=1).decide(m, now, trig).kind == "stop"
    try:
        VLMRecovery(MockVLM({}), options=("backoff",))
        raise AssertionError("backoff is not executable yet")
    except ValueError:
        pass
    print("VLM modules with a mock VLM: ok")


def item(progress, advancing=True, previous_ok=True, conf=0.9):
    return {"done": progress == "done", "progress": progress, "advancing": advancing, "previous_ok": previous_ok,
            "evidence": "seen", "confidence": conf}


def test_todo_tracker():
    """the to-do list tracker: plan, item check, progress dots, done streak, the stall detector"""
    from coordinator.detection.progress import ProgressStall
    m, now = memory_mid_task(), obs(41)
    # the to-do list text and the item window (uniform over this attempt)
    r = C.item_request(m, now)
    t = r.text()
    assert "  [x] 1. Put the kettle on the stove. (done at 9.6 s)" in t and "  [>] 2. Turn on the burner." in t, t[:900]
    assert "attempt 1: 9.6-27.2 s, ended: timeout -> retrace (the robot was then driven back" in t, t[:1200]
    assert "now: attempt 2, 4.0 s into it (this item usually takes under 16 s)" in t and "  [ ] 3. Wait." in t
    labels = [p[1] for p in r.parts if p[0] == "text" and p[1].startswith("---")]
    # uniform over this attempt (the retry began at chunk 36, after the rewind), both ends included
    assert labels[0].startswith("--- this attempt began (the robot was driven back here): t = 28.8 s (chunk 36)"), labels
    assert len(labels) == 6 and labels[-1].startswith("--- now: t = 32.8 s"), labels
    assert r.n_images() == 3 * len(labels) and r.schema is P.ITEM_SCHEMA and r.parts[-1][1] == P.ITEM_Q
    m3 = Memory("Boil.", [SubGoal(0, "a"), SubGoal(1, "b")])
    m3.begin_stage(0, obs(0))
    for c in range(1, 61):
        m3.observe(obs(c))
    m3.complete_stage(0, obs(20)); m3.begin_stage(1, obs(20))
    for c in range(21, 61):
        m3.observe(obs(c))
    l3 = [p[1] for p in C.item_request(m3, obs(60)).parts if p[0] == "text" and p[1].startswith("---")]
    assert len(l3) == 10 and "item 1 was done, item 2 began" in l3[0] and "(chunk 20)" in l3[0] and "(chunk 60)" in l3[-1], l3
    assert C.item_request(m3, obs(60)).n_images() == 30
    rr = C.item_request(m, now, C.ContextConfig(window="recent"))
    lr = [p[1] for p in rr.parts if p[0] == "text" and p[1].startswith("---")]
    assert len(lr) == 10 and "(chunk 23)" in lr[0] and lr[-1].startswith("--- now"), lr     # 10 chunk ends, 2 apart
    assert rr.n_images() == 30
    # the plan: the VLM's items become sub-goals with done conditions and the any-order flag
    plan = {"items": [{"text": "Pick the kettle and place it on a burner.", "done": "kettle on a burner, let go",
                       "any_order": False}, {"text": "Turn the burner on.", "done": "flame on"}]}
    g = VLMTracker(MockVLM(plan)).make_plan("Boil.", now)
    assert [x.text for x in g] == ["Pick the kettle and place it on a burner.", "Turn the burner on."]
    assert g[0].check == "kettle on a burner, let go" and g[1].any_order is False
    # done twice in a row -> ticked (no other check)
    tr = VLMTracker(MockVLM([item("handling"), item("done"), item("done")]), done_streak=2)
    s = tr.update(m, obs(41)); assert s.current == 1 and m.dots[-1]["progress"] == "handling"
    s = tr.update(m, obs(42)); assert s.current == 1, "one 'done' is not enough"
    s = tr.update(m, obs(43)); assert s.current == 2 and s.done == [True, True, False]
    assert [x["module"] for x in m.vlm_calls[-3:]] == ["tracker"] * 3
    m.move_on(2, obs(43))                        # as the loop does
    s = VLMTracker(MockVLM(item("handling", previous_ok=False))).update(m, obs(48))
    assert s.done == [True, False, False] and s.current == 2, "previous_ok false reads the previous item undone"
    assert "  [x] 2. Turn on the burner. (done at 34.4 s)" in C.todo_block(m, obs(48)), C.todo_block(m, obs(48))
    # the stall detector: n non-advancing dots in a row (fires once per n)
    m2 = memory_mid_task()
    st = ProgressStall(n=3, min_chunks=4)
    tr2 = VLMTracker(MockVLM(item("reaching", advancing=False)))
    fired = []
    for c in range(41, 48):
        stt = tr2.update(m2, obs(c))
        fired += [(c, x.kind) for x in st.check(m2, obs(c), stt)]
    assert fired == [(43, "no_progress"), (46, "no_progress")], fired
    print("to-do list tracker with a mock VLM: ok", f"(item request: {r.n_images()} images, {len(t)} chars of text)")


def test_any_order():
    """an any-order run: the open items are judged together, a later one ticked first (out of order) is recorded and
    the current item restarts its attempt there; ticks then skip the items already done"""
    plan = [SubGoal(i, f"Place item {i + 1}.", f"{i + 1} placed", any_order=True) for i in range(3)] + \
        [SubGoal(3, "Close the lid.", "lid shut")]
    m = Memory("Pack.", plan)
    m.begin_stage(0, obs(0))
    for c in range(1, 6):
        m.observe(obs(c))
    assert m.open_items() == [0, 1, 2] and m.open_items(3) == [3]
    t = C.todo_block(m, obs(5))
    assert "the items marked [>] may be done in any order" in t and "  [>] 2. Place item 2." in t and "  [ ] 4." in t, t

    def ans(done):
        return {"done": done, "progress": "handling", "advancing": True, "previous_ok": True, "evidence": "e",
                "confidence": 0.9}
    tr = VLMTracker(MockVLM([ans({"1": False, "2": False, "3": True}), ans({"1": False, "item 3": "true"}),
                             ans({"1": True, "2": False}), ans({"1": True}), ans({"2": True}), ans({"2": True})]))
    s = tr.update(m, obs(6)); assert s.extra_done == [] and s.current == 0
    s = tr.update(m, obs(7)); assert s.extra_done == [2] and s.current == 0 and s.done[2], s
    m.tick_out_of_order(2, obs(7))                       # as the loop does
    assert m.is_done(2) and m.current == 0 and m.open_items() == [0, 1] and m.back_to == m.entry[0]
    assert m.stages[0].attempts[-1]["why"] == "item 3 done first" and m.stages[0].attempts[-2]["end"]["how"] == "item 3 done first"
    assert "  [x] 3. Place item 3. (done at 5.6 s)" in C.todo_block(m, obs(8))
    s = tr.update(m, obs(8)); assert s.current == 0
    s = tr.update(m, obs(9)); assert s.current == 1 and not s.extra_done
    m.move_on(1, obs(9))
    s = tr.update(m, obs(10)); s = tr.update(m, obs(11))
    assert s.current == 3, "item 3 is done already: the next item is 4"
    m.move_on(3, obs(11))
    assert m.open_items() == [3] and len(m.moved_on) == 3 and m.moved_on[0]["out_of_order"]
    from coordinator.tracking.vlm import done_flags
    assert done_flags({"done": True}, [1, 2]) == {1: True, 2: False} and done_flags({"done": {"2": "false"}}, [1]) == {1: False}
    # the grasp-and-release guard: a placement is ticked only after the gripper closed and opened in this attempt

    class Trace:
        path = [dict(grip=0.0)]
    m4 = Memory("Pack.", [SubGoal(i, f"Place item {i + 1}.", any_order=True) for i in range(2)], trace=Trace())
    m4.begin_stage(0, obs(0))
    tr4 = VLMTracker(MockVLM(ans({"1": True, "2": False})))
    Trace.path += [dict(grip=0.0)] * 3
    assert tr4.update(m4, obs(1)).current == 0 and tr4.update(m4, obs(2)).current == 0, "no grasp yet: no tick"
    assert "gripper has not grasped and released" in m4.notes[-1]["text"]
    Trace.path += [dict(grip=1.0)] * 3 + [dict(grip=0.0)]
    assert tr4.update(m4, obs(3)).current == 1 and len(m4.notes) == 1, "grasped and released: ticked"
    print("any-order items (out-of-order ticks, grasp-and-release guard): ok")


if __name__ == "__main__":
    test_parse_and_check(); test_registry(); test_openai_compat(); test_hf_server(); test_context(); test_vlm_modules(); test_todo_tracker(); test_any_order()
    print("all VLM checks passed")

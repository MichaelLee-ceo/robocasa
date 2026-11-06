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


def stub_server(reply, status=200):
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

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
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
    r = make_vlm(f"vllm:m@{port}").ask(req)   # server gone
    assert not r.ok and "cannot reach" in r.error
    print("OpenAI-compatible backend: ok")


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
    ans = {"stages_done": [True, True, False], "event": "fixture_short", "event_detail": "knob only part-way",
           "evidence": "knob turned", "confidence": 0.9}
    tr = VLMTracker(MockVLM(ans), done_streak=2)
    s = tr.update(m, now)
    assert s.done == [True, False, False] and s.current == 1, "a first 'done' must wait for the streak"
    s = tr.update(m, obs(42))
    assert s.done == [True, True, False] and s.current == 2 and s.note == "fixture_short: knob only part-way"
    assert len(m.vlm_calls) == 2 and tr.vlm.requests[0].schema is P.PROGRESS_SCHEMA
    s = VLMTracker(MockVLM({"stages_done": [True], "event": "none", "event_detail": "", "evidence": "", "confidence": 1})).update(m, now)
    assert s.done is None and "1 flags for 3" in s.error
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


if __name__ == "__main__":
    test_parse_and_check(); test_registry(); test_openai_compat(); test_hf_server(); test_context(); test_vlm_modules()
    print("all VLM checks passed")

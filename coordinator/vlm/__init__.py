"""Vision-language model backends behind one interface (base.py): `make_vlm(spec)` builds one from a short spec.
The backends subclass the pilot's (coordinator/vlm/pilot.py, loaded by pilot.py) and keep its Query interface (answer).

| spec                                        | backend                                   | where                          |
|---------------------------------------------|-------------------------------------------|--------------------------------|
| claude:claude-opus-5-5  (also anthropic:)   | ClaudeVLM (official SDK)                  | Anthropic API                  |
| openai:<model>                              | OpenAICompatVLM, key OPENAI_API_KEY       | OpenAI API                     |
| gemini:<model>                              | OpenAICompatVLM, key GEMINI_API_KEY       | Gemini's OpenAI endpoint       |
| vllm:<model>@<port or url>                  | OpenAICompatVLM (default port 8000)       | local vLLM server              |
| sglang:<model>@<port or url>                | OpenAICompatVLM (default port 30000)      | local SGLang server            |
| ollama:<model>@<port or url>                | OpenAICompatVLM (default port 11434)      | local Ollama                   |
| lmstudio:<model>@<port or url>              | OpenAICompatVLM (default port 1234)       | local LM Studio                |
| hf:<name>@<port>                            | HFServerVLM (default port 8101)           | coordinator/vlm/vlm_server.py      |
| mock                                        | MockVLM                                   | tests                          |
Keyword arguments go to the backend (e.g. effort="low", thinking=True, structured=False, image_size=224, price=(..)).
A model name may contain ':' (Ollama tags): only the first ':' separates the provider.

    from coordinator.vlm import make_vlm, VLMRequest
    vlm = make_vlm("claude:claude-opus-5-5")
    res = vlm.ask(VLMRequest(parts=[("text", "What is on the burner?"), ("image", frame)], schema={...}))"""
from .base import VLM, VLMError, VLMRequest, VLMResponse, encode_jpeg, parse_json, check  # noqa: F401

LOCAL = {"vllm": 8000, "sglang": 30000, "ollama": 11434, "lmstudio": 1234}
CLOUD = {"openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
         "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY")}


def _url(where, default_port):
    if not where:
        return f"http://127.0.0.1:{default_port}/v1"
    if where.isdigit():
        return f"http://127.0.0.1:{where}/v1"
    if where.startswith("http"):
        return where
    return f"http://{where}/v1"   # host:port


def make_vlm(spec, **kw):
    """a VLM from a spec string (table above); a VLM instance is returned as is"""
    if isinstance(spec, VLM):
        return spec
    if spec == "mock":
        from .mock import MockVLM
        return MockVLM(**kw)
    provider, _, rest = spec.partition(":")
    model, _, where = rest.rpartition("@") if "@" in rest else (rest, "", "")
    if provider in ("claude", "anthropic"):
        from .anthropic_api import ClaudeVLM
        return ClaudeVLM(model or "claude-opus-5-5", **kw)
    from .openai_compat import OpenAICompatVLM
    if provider in CLOUD:
        url, key = CLOUD[provider]
        return OpenAICompatVLM(model, _url(where, 0) if where else url, api_key_env=kw.pop("api_key_env", key), **kw)
    if provider in LOCAL:
        kw.setdefault("temperature", 0)
        kw.setdefault("price", (0.0, 0.0))
        if provider in ("vllm", "sglang"):
            kw.setdefault("thinking", False)
        return OpenAICompatVLM(model, _url(where, LOCAL[provider]), **kw)
    if provider == "hf":
        from .hf_server import HFServerVLM
        port = int(where) if where else 8101
        return HFServerVLM(port=port, name=model or "hf-local", **kw)
    raise ValueError(f"unknown VLM spec {spec!r} (see coordinator/vlm/__init__.py)")

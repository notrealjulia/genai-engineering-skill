"""Minimal provider-neutral LLM client used by all examples.

Set LLM_PROVIDER=openai (default) or anthropic, and optionally LLM_MODEL.
To use another provider, replace the body of call_llm(). The examples depend only on
its signature: an OpenAI-style message list in, plain text out. call_llm_tool() is the
native tool-calling variant, used where structured output matters.
"""
import json
import os

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
DEFAULT_MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
MODEL = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))

_client = None


def _get_client():
    global _client
    if _client is None:
        if PROVIDER == "anthropic":
            import anthropic
            _client = anthropic.Anthropic()
        else:
            from openai import OpenAI
            _client = OpenAI()
    return _client


def _anthropic(messages, model, max_tokens, **extra):
    # The system prompt is a separate parameter. No temperature: current Claude models
    # reject non-default sampling parameters (400), so sampling stays at the default, which
    # still varies enough for self-consistency. Adaptive thinking is on by default and its
    # tokens count against max_tokens, so leave headroom or the text can come back empty.
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    kwargs = {"system": system} if system else {}
    return _get_client().messages.create(model=model or MODEL, messages=chat,
                                         max_tokens=max(max_tokens, 8192), **kwargs, **extra)


def _as_messages(messages):
    return [{"role": "user", "content": messages}] if isinstance(messages, str) else messages


def call_llm(messages, temperature: float = 0.0, model: str | None = None,
             max_tokens: int = 1024) -> str:
    """Send chat messages ([{"role": ..., "content": ...}]) and return the reply text.

    Accepts a plain string as shorthand for a single user message. `temperature` is
    ignored on Anthropic (see _anthropic).
    """
    messages = _as_messages(messages)
    if PROVIDER == "anthropic":
        resp = _anthropic(messages, model, max_tokens)
        return "".join(b.text for b in resp.content if b.type == "text")

    resp = _get_client().chat.completions.create(model=model or MODEL, messages=messages,
                                                 temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content


def call_llm_logprobs(messages, temperature: float = 0.0, model: str | None = None,
                      max_tokens: int = 1024) -> tuple[str, list[float] | None]:
    """Like call_llm(), plus one log probability per generated token, or None when the
    provider doesn't return them (Anthropic's API doesn't): callers need a fallback."""
    if PROVIDER != "openai":
        return call_llm(messages, temperature, model, max_tokens), None
    resp = _get_client().chat.completions.create(
        model=model or MODEL, messages=_as_messages(messages), temperature=temperature,
        max_tokens=max_tokens, logprobs=True)
    choice = resp.choices[0]
    tokens = choice.logprobs.content if choice.logprobs else None
    return choice.message.content, [t.logprob for t in tokens] if tokens else None


def call_llm_tool(messages, tool: dict, temperature: float = 0.0, model: str | None = None,
                  max_tokens: int = 2048) -> dict | None:
    """Request one native tool call and return its arguments (a dict), or None.

    tool = {"name", "description", "parameters": <JSON Schema>}. The arguments come back as
    structured data, so nothing has to be scraped out of prose. For a provider without
    tool calling, ask for JSON with call_llm() and parse it with agents_common.parse_json.
    """
    messages = _as_messages(messages)
    if PROVIDER == "anthropic":
        # Current Claude models (Opus/Sonnet 5.5, Fable 5.1) reject a forced tool_choice
        # with a 400: ask for the tool in the prompt, and callers handle "no call" (None).
        last = {**messages[-1], "content": messages[-1]["content"]
                + f"\n\nReply by calling the {tool['name']} tool."}
        resp = _anthropic(messages[:-1] + [last], model, max_tokens, tools=[{
            "name": tool["name"], "description": tool["description"],
            "input_schema": tool["parameters"]}])
        return next((b.input for b in resp.content if b.type == "tool_use"), None)

    resp = _get_client().chat.completions.create(
        model=model or MODEL, messages=messages, temperature=temperature,
        max_tokens=max_tokens, tools=[{"type": "function", "function": tool}],
        tool_choice={"type": "function", "function": {"name": tool["name"]}})
    calls = resp.choices[0].message.tool_calls or []
    try:  # API-produced JSON, but a truncated reply (max_tokens) can still cut it off
        return json.loads(calls[0].function.arguments) if calls else None
    except json.JSONDecodeError:
        return None


# --- Embeddings (used by the RAG examples) -------------------------------------------
# EMBED_PROVIDER=openai (default) uses the OpenAI embeddings API.
# EMBED_PROVIDER=local uses sentence-transformers on your machine (pip install sentence-transformers).
# Anthropic has no embeddings endpoint; pair Claude with any embedding provider
# (e.g. Voyage AI, OpenAI, or a local model) by replacing embed().
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai").lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small" if EMBED_PROVIDER == "openai"
                        else "all-MiniLM-L6-v2")
_local_embedder = None


def embed(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Return one embedding vector per input text, in order."""
    global _local_embedder
    if EMBED_PROVIDER == "local":
        if _local_embedder is None:
            from sentence_transformers import SentenceTransformer
            _local_embedder = SentenceTransformer(EMBED_MODEL)
        return _local_embedder.encode(texts, batch_size=batch_size).tolist()

    from openai import OpenAI
    client = OpenAI()
    vectors = []
    for i in range(0, len(texts), batch_size):  # batch to respect request size limits
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(d.embedding for d in resp.data)
    return vectors

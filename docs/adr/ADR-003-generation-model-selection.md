# ADR-003: Generation Model Selection


## Context

VaultMind requires a language model to generate citation-grounded answers from retrieved document chunks. The model receives a structured prompt containing numbered excerpts from the DBMS textbook and must produce a factual answer with inline [n] citations and a source list.

Requirements for the generation model:

- Reliable instruction-following with a structured system prompt.
- Accurate citation of provided context without fabricating sources.
- Sufficient context window to fit 5 retrieved chunks (typically 500–800 tokens of context plus prompt overhead).
- Low cost or zero cost, given frequent evaluation runs during development.
- Low latency for interactive use and iterative testing.

Several models and hosting options were considered.

## Decision

Use **Groq (llama-3.1-8b-instant)** as the primary backend for demo, evaluation, and iteration.

Support **Ollama (qwen2.5:7b)** as a local alternative for offline development where no external API is available.

Both are configured through the `VAULTMIND_LLM_BACKEND` environment variable with no code changes required to switch between them.

## Rationale

Cost is a binding constraint for a student project with 20+ evaluation runs per session. Both paid options (Claude Haiku, GPT-4o-mini) incur per-token charges that accumulate quickly during iterative RAG debugging.

Groq's free tier provides 14,400 requests per day at approximately 600 tokens per second — sufficient for evaluation, demo, and rapid iteration. The llama-3.1-8b model, when given a well-structured system prompt, low temperature (0.1), and explicit context, produces reliable grounded answers for RAG tasks where the model is not required to reason from scratch but only to synthesise and cite provided passages.

The primary tradeoff accepted is slightly weaker instruction-following compared to Claude Haiku or GPT-4o-mini. This is mitigated by:

- Explicit numbered context format `[1]`, `[2]`, ... with page and section labels.
- A strict system prompt that forbids outside knowledge and requires inline citations.
- Low temperature (0.1) to suppress creative deviation from the source material.
- A guardrails layer that blocks out-of-scope queries before the model is invoked.

## Consequences

### Positive

- Zero API cost during development, evaluation, and demo.
- High throughput (~600 tokens/second on Groq) reduces iteration time.
- Consistent structured output with low temperature on RAG-style prompts.
- Offline fallback available via Ollama with no external dependency.
- Easy backend switching via a single environment variable.

### Negative

- Weaker instruction-following than Claude Haiku or GPT-4o-mini for edge cases.
- Groq free tier has daily request limits (14,400 req/day); production deployment would require a paid tier or a different provider.
- Ollama requires local GPU/CPU resources and a separate model download.

## Alternatives Considered

### Claude Haiku (Anthropic)

Best instruction-following of the options evaluated and the most reliable for citation formatting. Rejected due to cost (~$0.00025 per 1k input tokens), which becomes significant across 20+ eval runs per session with multi-chunk context prompts.

### GPT-4o-mini (OpenAI)

Similar capability and cost profile to Claude Haiku. Rejected for the same cost reason. Retained as a supported backend in `generate.py` for users who have an OpenAI key.

### Groq (llama-3.3-70b-versatile)

Stronger reasoning than 8b-instant, still free on Groq's tier. Available as a drop-in model swap in `generate.py`. Not selected as the default because 8b-instant is sufficient for context-grounded RAG tasks and has lower latency.

### Ollama (qwen2.5:7b) — Local

Zero external dependency, full offline capability. Suitable for development without internet access. Not selected as the primary default because it requires a local GPU or CPU inference setup that is not available in all environments.

## Future Work

- Evaluate llama-3.3-70b-versatile on a standard DBMS question set to quantify the quality gap versus 8b-instant.
- Add JSON mode enforcement for structured answer output (answer text + citations as separate fields) once the Streamlit UI requires a machine-parseable response format.
- Reassess model selection if Groq changes its free-tier limits or a strong open-weight model becomes available locally without GPU requirements.

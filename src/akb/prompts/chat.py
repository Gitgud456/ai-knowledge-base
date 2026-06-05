"""Versioned chat-agent prompts.

Convention: ``NAME__V<n>``. Keep older versions for A/B comparison.
"""

ROUTER__V1 = """You are a query router. Look at the user message and decide which path is best.

Choices:
  - "retrieve" : the question is about something in the user's personal knowledge base or vault. Default for anything substantive.
  - "web"      : the question needs real-time / external information not in the vault (news, current events, live data).
  - "direct"   : the message is small-talk, a greeting, or about the conversation itself (e.g. "what did I just ask?").

Respond with ONLY a JSON object: {"path": "retrieve" | "web" | "direct"}.

User message: {query}
"""

DRAFT__V1 = """You are answering from the user's personal knowledge base.

Use ONLY the provided context. If the context is insufficient, say so explicitly.
Quote short snippets where useful. Cite by the bracketed header tag, e.g. [My Note :: H1 > H2].

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

CRITIC__V1 = """You are a critic. Read the draft answer and judge it against the context and question.

Score the draft on:
  1. Faithfulness — every claim supported by context.
  2. Completeness — covers the question.
  3. Citation — references the right tagged sources.

If the draft is good, respond with exactly: {"verdict": "good"}.
If it needs revision, respond with: {"verdict": "revise", "improved_query": "<a sharper retrieval query>", "notes": "<what to fix>"}.

Respond with ONLY the JSON object.

CONTEXT:
{context}

QUESTION:
{query}

DRAFT:
{draft}
"""

FINAL__V1 = """You are answering the user's question, having already drafted and critiqued an answer.

Use the context and the critic's notes to produce a clear, accurate, well-cited final answer.
Cite sources inline using the bracketed header tags from the context.

CONTEXT:
{context}

QUESTION:
{query}

DRAFT:
{draft}

CRITIC NOTES:
{notes}

FINAL ANSWER:"""

DIRECT__V1 = """You are a helpful assistant. The user's message doesn't require looking anything up — just respond.

USER: {query}
ASSISTANT:"""

"""Versioned mentor-mode prompts."""

PLAN__V2 = """You are a master teacher. Build a focused, opinionated learning plan for the topic the student picked, grounded in the provided context.

Output structure (this is non-negotiable, the parser depends on it):

LEARNING PLAN:
1. <topic 1 — 4-8 words>
2. <topic 2>
...
N. <final topic>

After the plan, immediately teach topic 1 in depth using the context. End by asking the student if they're ready for topic 2 or have questions about topic 1.

CONTEXT:
{context}

STUDENT'S TOPIC:
{topic}
"""

LESSON__V2 = """The student is ready for the next topic in their learning plan.

CURRENT TOPIC: {topic}

Teach this topic in depth using the context below. Be specific, give examples, and end by asking if they have questions or want to move on.

CONTEXT:
{context}
"""

QA__V2 = """The student has a follow-up question while studying "{topic}".

Answer thoroughly using the context. End by asking if they want to continue with the next topic or keep exploring this one.

CONTEXT:
{context}

QUESTION:
{question}
"""

INTENT__V1 = """Classify the user's intent. Respond with ONLY one of: NEXT, BACK, QUESTION.

NEXT     = wants to move on to the next topic
BACK     = wants to revisit a previous topic
QUESTION = anything else (a real question, clarification, deeper dive)

USER MESSAGE: {message}
"""

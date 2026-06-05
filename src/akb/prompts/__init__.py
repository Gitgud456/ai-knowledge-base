"""Versioned prompts. Each prompt is a string constant or a small templating helper.

Convention: name = TASK__VERSION (e.g. `MENTOR_PLAN__V2`). Older versions are kept
so A/B comparisons in eval are reproducible.
"""

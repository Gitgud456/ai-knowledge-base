"""``MentorState.commit_reply`` is what restores conversational continuity."""

from __future__ import annotations

from akb.agents.mentor import MentorState


def test_commit_reply_appends_assistant_turn() -> None:
    s = MentorState(topic="x", plan=["a", "b"], history=[{"role": "user", "content": "hi"}])
    s.commit_reply("the assistant's reply")
    assert s.history[-1] == {"role": "assistant", "content": "the assistant's reply"}


def test_commit_reply_empty_no_op() -> None:
    s = MentorState(topic="x", plan=[], history=[{"role": "user", "content": "hi"}])
    s.commit_reply("")
    assert s.history == [{"role": "user", "content": "hi"}]


def test_commit_reply_returns_self_for_chaining() -> None:
    s = MentorState(topic="x", plan=[])
    assert s.commit_reply("ok") is s

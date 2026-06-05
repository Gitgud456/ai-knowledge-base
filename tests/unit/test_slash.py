"""Slash command parser — small but the contract has to be tight."""

from __future__ import annotations

from akb.agents.slash import parse


def test_plain_text_no_command() -> None:
    sc = parse("regular question without slash")
    assert sc.cmd is None
    assert sc.query == "regular question without slash"
    assert sc.force_path is None


def test_help_variants() -> None:
    for line in ("/help", "/h", "/?"):
        sc = parse(line)
        assert sc.cmd == "help"
        assert sc.show_help is True


def test_search_forces_kb_path() -> None:
    sc = parse("/search ARP spoofing")
    assert sc.cmd == "search"
    assert sc.query == "ARP spoofing"
    assert sc.force_path == "retrieve"


def test_web_forces_web_path() -> None:
    sc = parse("/web latest python release")
    assert sc.force_path == "web"


def test_cite_short_circuits_synthesis() -> None:
    sc = parse("/cite ARP")
    assert sc.cite_only is True
    assert sc.force_path == "retrieve"


def test_dry_run_flag() -> None:
    sc = parse("/dry-run something")
    assert sc.dry_run is True


def test_unknown_slash_falls_back_to_plain_text() -> None:
    sc = parse("/zoomies hello")
    assert sc.cmd is None
    assert sc.query == "/zoomies hello"


def test_command_only_no_query() -> None:
    sc = parse("/search")
    assert sc.cmd == "search"
    assert sc.query == ""

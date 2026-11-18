"""Tests for the Kanban worker surface (tools/kanban_worker_tools.py).

One test per verb, plus the resolved-surface assertions:
  - hand_off / review_result / gate_result / open_round / decompose_sprint / sprint_report
    each do exactly one board transition and report where the card landed.
  - The six verbs live in the ``kanban_worker`` toolset, never in ``kanban``.
  - ``model_tools._select_tool_names`` appends ``kanban_worker`` (not ``kanban``)
    for a dispatcher-spawned worker, so the 15 built-in schemas stay out of the
    worker's prompt.
  - ``HERMES_KANBAN_WORKER_TOOLS`` narrows the visible verbs per role.
"""
from __future__ import annotations

import json
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def board(tmp_path, monkeypatch):
    """An isolated board + a factory for running cards."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKER_TOOLS", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        yield {"kb": kb, "conn": conn, "home": home}
    finally:
        conn.close()


def _running(board, title="unit", assignee="coder", body="", **kw):
    """Create a card and hand its (id, run_id) back to the test."""
    kb, conn = board["kb"], board["conn"]
    tid = kb.create_task(conn, title=title, assignee=assignee, body=body, **kw)
    if kb.get_task(conn, tid).status != "running":
        kb.claim_task(conn, tid)
    return tid, kb._current_run_id(conn, tid)


def _as_worker(monkeypatch, tid, run_id, verbs=""):
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    if verbs:
        monkeypatch.setenv("HERMES_KANBAN_WORKER_TOOLS", verbs)
    else:
        monkeypatch.delenv("HERMES_KANBAN_WORKER_TOOLS", raising=False)


def _payload(out: str) -> dict:
    data = json.loads(out)
    assert data.get("ok") is True, data
    return data


GATE_BODY = """## Original instruction
Add subtract() to calc.py and cover it with a test.
## Verification
command: python -m pytest -q
module: calc.py
## Branches
integration: integration
round: 1
"""


# ---------------------------------------------------------------------------
# 1. hand_off (coder)
# ---------------------------------------------------------------------------

def test_hand_off_completes_the_card(board, monkeypatch):
    tid, run_id = _running(board)
    _as_worker(monkeypatch, tid, run_id, "hand_off")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_hand_off({"summary": "implemented subtract() and its test"}))
    assert out["task_id"] == tid
    assert out["status"] == "done"
    task = board["kb"].get_task(board["conn"], tid)
    assert task.status == "done"
    run = board["kb"].latest_run(board["conn"], tid)
    assert run.metadata["stage"] == "implemented"


def test_hand_off_with_blocked_reason_blocks_needs_input(board, monkeypatch):
    tid, run_id = _running(board)
    _as_worker(monkeypatch, tid, run_id, "hand_off")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_hand_off({
        "summary": "could not finish", "blocked_reason": "the spec omits the rounding rule"}))
    assert out["status"] == "blocked"
    task = board["kb"].get_task(board["conn"], tid)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"


def test_hand_off_requires_a_summary(board, monkeypatch):
    tid, run_id = _running(board)
    _as_worker(monkeypatch, tid, run_id, "hand_off")
    from tools import kanban_worker_tools as kwt

    data = json.loads(kwt._handle_hand_off({}))
    assert "error" in data
    assert "summary" in data["error"]
    assert board["kb"].get_task(board["conn"], tid).status == "running"


# ---------------------------------------------------------------------------
# 2. review_result (reviewer)
# ---------------------------------------------------------------------------

def test_review_result_pass(board, monkeypatch):
    tid, run_id = _running(board, assignee="reviewer")
    _as_worker(monkeypatch, tid, run_id, "review_result")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_review_result({
        "verdict": "pass", "findings": "subtract() handles negatives; tests cover the edge."}))
    assert out["status"] == "done"
    run = board["kb"].latest_run(board["conn"], tid)
    assert run.metadata["review_outcome"] == "pass"
    assert (run.summary or "").startswith("REVIEW PASS")
    comments = board["kb"].list_comments(board["conn"], tid)
    assert any("subtract() handles negatives" in (c.body or "") for c in comments)


def test_review_result_accepts_the_project_alias(board, monkeypatch):
    """AGENTS.md says ``approved``; the verb must not need a translation layer."""
    tid, run_id = _running(board, assignee="reviewer")
    _as_worker(monkeypatch, tid, run_id, "review_result")
    from tools import kanban_worker_tools as kwt

    _payload(kwt._handle_review_result({"verdict": "approved", "findings": "ok"}))
    run = board["kb"].latest_run(board["conn"], tid)
    assert run.metadata["review_outcome"] == "pass"


def test_review_result_fail_still_completes_the_card(board, monkeypatch):
    """Red is a done review card, not a card sent backwards."""
    tid, run_id = _running(board, assignee="reviewer")
    _as_worker(monkeypatch, tid, run_id, "review_result")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_review_result({
        "verdict": "fail", "findings": "no test for subtract(0, 0)"}))
    assert out["status"] == "done"
    run = board["kb"].latest_run(board["conn"], tid)
    assert run.metadata["review_outcome"] == "fail"
    assert (run.summary or "").startswith("REVIEW FAIL")


def test_review_result_blocked(board, monkeypatch):
    tid, run_id = _running(board, assignee="reviewer")
    _as_worker(monkeypatch, tid, run_id, "review_result")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_review_result({
        "verdict": "blocked", "findings": "diff is not present in the workspace"}))
    assert out["status"] == "blocked"
    assert board["kb"].get_task(board["conn"], tid).block_kind == "needs_input"


def test_review_result_rejects_an_unknown_verdict(board, monkeypatch):
    tid, run_id = _running(board, assignee="reviewer")
    _as_worker(monkeypatch, tid, run_id, "review_result")
    from tools import kanban_worker_tools as kwt

    data = json.loads(kwt._handle_review_result({"verdict": "lgtm", "findings": "x"}))
    assert "error" in data
    for word in ("pass", "fail", "blocked"):
        assert word in data["error"]
    assert board["kb"].get_task(board["conn"], tid).status == "running"


# ---------------------------------------------------------------------------
# 3. gate_result (integrator)
# ---------------------------------------------------------------------------

def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q", "-b", "integration")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "wt/unit")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n")
    git("commit", "-qam", "unit: subtract")
    git("checkout", "-q", "integration")
    return repo, git


def test_gate_result_pass_merges_the_feature_branch(board, monkeypatch, tmp_path):
    repo, git = _git_repo(tmp_path)
    tid, run_id = _running(
        board, assignee="integrator", workspace_kind="dir", workspace_path=str(repo),
        body="## Branches\nfeature: wt/unit\nintegration: integration\n")
    _as_worker(monkeypatch, tid, run_id, "gate_result")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_gate_result({"result": "pass", "detail": "pytest -q: 4 passed"}))
    assert out["status"] == "done"
    assert out["merged"] is True
    assert "subtract" in (repo / "calc.py").read_text()
    run = board["kb"].latest_run(board["conn"], tid)
    assert (run.summary or "").startswith("GATE PASS")
    assert run.metadata["gate_result"] == "pass"


def test_gate_result_fail_blocks_and_reverts(board, monkeypatch, tmp_path):
    repo, git = _git_repo(tmp_path)
    base = subprocess.run(["git", "-C", str(repo), "rev-parse", "integration"],
                          capture_output=True, text=True, check=True).stdout.strip()
    # The integrator merged, then the unit command failed: the verb must undo it.
    subprocess.run(["git", "-C", str(repo), "merge", "--no-ff", "-q", "-m", "merge", "wt/unit"],
                   check=True, capture_output=True)
    body = ("## Branches\nfeature: wt/unit\nintegration: integration\nrollback_ref: " + base + "\n")
    tid, run_id = _running(board, assignee="integrator", workspace_kind="dir",
                           workspace_path=str(repo), body=body)
    _as_worker(monkeypatch, tid, run_id, "gate_result")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_gate_result({
        "result": "fail", "detail": "pytest -q: 1 failed (test_subtract)"}))
    assert out["status"] == "blocked"
    assert out["reverted"] is True
    assert board["kb"].get_task(board["conn"], tid).block_kind == "needs_input"
    assert "subtract" not in (repo / "calc.py").read_text()
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "integration"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == base


# ---------------------------------------------------------------------------
# 4. open_round (integrator, per unit)
# ---------------------------------------------------------------------------

def test_open_round_creates_the_next_triplet(board, monkeypatch):
    tid, run_id = _running(board, title="U1 — subtract / gate", assignee="integrator",
                           body=GATE_BODY)
    _as_worker(monkeypatch, tid, run_id, "open_round")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_open_round({"self_failure": "REVIEW FAIL — no test for 0 - 0"}))
    assert out["status"] == "done"
    assert out["round"] == 2
    kb, conn = board["kb"], board["conn"]
    cards = out["cards"]
    assert len(cards) == 3
    c, r, g = (cards["code"], cards["review"], cards["gate"])
    assert kb.parent_ids(conn, r) == [c]
    assert kb.parent_ids(conn, g) == [r]
    assert kb.parent_ids(conn, c) == []
    for cid, role in ((c, "coder"), (r, "reviewer"), (g, "integrator")):
        task = kb.get_task(conn, cid)
        assert task.assignee == role
        assert "Add subtract() to calc.py" in task.body          # verbatim instruction
        assert "## What failed in round 1" in task.body
        assert "REVIEW FAIL — no test for 0 - 0" in task.body      # the evidence
        assert "round: 2" in task.body


def test_open_round_at_the_cap_blocks_instead(board, monkeypatch):
    body = GATE_BODY.replace("round: 1", "round: 3")
    tid, run_id = _running(board, title="U1 — subtract / gate", assignee="integrator", body=body)
    _as_worker(monkeypatch, tid, run_id, "open_round")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_open_round({"self_failure": "still red"}))
    assert out["status"] == "blocked"
    assert out["cards"] == []
    assert board["kb"].get_task(board["conn"], tid).block_kind == "needs_input"


# ---------------------------------------------------------------------------
# 5. decompose_sprint (integrator, sprint card)
# ---------------------------------------------------------------------------

SPRINT_UNITS = [
    {"title": "Unit 1 — subtract", "files": "calc.py",
     "acceptance": ["subtract(a, b) returns a - b", "negative operands work"],
     "command": "python -m pytest -q tests/test_calc.py"},
    {"title": "Unit 2 — multiply", "files": "calc.py",
     "acceptance": ["multiply(a, b) returns a * b"],
     "command": "python -m pytest -q tests/test_calc.py"},
]


def test_decompose_sprint_builds_the_graph(board, monkeypatch):
    tid, run_id = _running(board, title="S1 — sprint", assignee="integrator",
                           body="## Goal\ncalculator basics\n\n## Integration\nbranch: integration\n"
                                "full suite: python -m pytest -q\n")
    _as_worker(monkeypatch, tid, run_id, "decompose_sprint")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_decompose_sprint({"units": SPRINT_UNITS}))
    kb, conn = board["kb"], board["conn"]
    assert len(out["units"]) == 2
    assert len(out["unit_cards"]) == 6
    closure = out["closure_card"]
    # K1 is the parent of the sprint card -> S1 waits, no respawn.
    assert kb.parent_ids(conn, tid) == [closure]
    sprint = kb.get_task(conn, tid)
    assert sprint.status in {"blocked", "todo"}
    assert kb.get_task(conn, closure).assignee == "integrator"
    # Each unit is C -> R -> G with the unit's criteria and command in C's body.
    for unit in out["units"]:
        c, r, g = unit["code"], unit["review"], unit["gate"]
        assert kb.parent_ids(conn, r) == [c]
        g_parents = kb.parent_ids(conn, g)
        assert r in g_parents and closure in g_parents
        c_task = kb.get_task(conn, c)
        assert "subtract" in c_task.body or "multiply" in c_task.body
        assert "python -m pytest -q tests/test_calc.py" in c_task.body
        assert "round: 1" in c_task.body


def test_decompose_sprint_rejects_an_empty_unit(board, monkeypatch):
    tid, run_id = _running(board, title="S1 — sprint", assignee="integrator")
    _as_worker(monkeypatch, tid, run_id, "decompose_sprint")
    from tools import kanban_worker_tools as kwt

    bad = [{"title": "Unit 1", "files": "calc.py", "acceptance": [], "command": "pytest -q"}]
    data = json.loads(kwt._handle_decompose_sprint({"units": bad}))
    assert "error" in data
    assert "acceptance" in data["error"]
    assert board["kb"].get_task(board["conn"], tid).status == "running"


# ---------------------------------------------------------------------------
# 6. sprint_report (integrator, closure card)
# ---------------------------------------------------------------------------

def _sprint_with_closure(board, monkeypatch):
    tid, run_id = _running(board, title="S1 — sprint", assignee="integrator")
    _as_worker(monkeypatch, tid, run_id, "decompose_sprint")
    from tools import kanban_worker_tools as kwt
    out = _payload(kwt._handle_decompose_sprint({"units": SPRINT_UNITS[:1]}))
    return tid, out["closure_card"]


def test_sprint_report_pass_closes_the_sprint(board, monkeypatch):
    s1, k1 = _sprint_with_closure(board, monkeypatch)
    kb, conn = board["kb"], board["conn"]
    kb.claim_task(conn, k1)
    _as_worker(monkeypatch, k1, kb._current_run_id(conn, k1), "sprint_report")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_sprint_report({
        "result": "pass", "detail": "1 unit, suite green: 4 passed"}))
    assert out["sprint_task"] == s1
    assert out["status"] == "done"
    assert kb.get_task(conn, s1).status == "done"
    assert kb.get_task(conn, k1).status == "done"
    run = kb.latest_run(conn, s1)
    assert (run.summary or "").startswith("SPRINT PASS")
    assert run.metadata["sprint_result"] == "pass"


def test_sprint_report_gap_blocks_the_sprint(board, monkeypatch):
    s1, k1 = _sprint_with_closure(board, monkeypatch)
    kb, conn = board["kb"], board["conn"]
    kb.claim_task(conn, k1)
    _as_worker(monkeypatch, k1, kb._current_run_id(conn, k1), "sprint_report")
    from tools import kanban_worker_tools as kwt

    out = _payload(kwt._handle_sprint_report({
        "result": "gap", "detail": "unit green, app import fails: ModuleNotFoundError"}))
    assert out["sprint_task"] == s1
    assert out["status"] == "blocked"
    sprint = kb.get_task(conn, s1)
    assert sprint.status == "blocked"
    assert sprint.block_kind == "needs_input"
    run = kb.latest_run(conn, s1)
    assert (run.summary or "").startswith("SPRINT GAP")


# ---------------------------------------------------------------------------
# The agent-facing path: every verb is dispatched through the tool registry
# ---------------------------------------------------------------------------

_VERB_CALLS = {
    "hand_off": ("coder", "", {"summary": "implemented, pytest -q: 4 passed"}),
    "review_result": ("reviewer", "", {"verdict": "pass", "findings": "criteria met"}),
    "gate_result": ("integrator", "", {"result": "pass", "detail": "unit command green"}),
    "open_round": ("integrator", GATE_BODY, {"self_failure": "REVIEW FAIL — no test for 0 - 0"}),
    "decompose_sprint": ("integrator", "## Integration\nbranch: integration\nfull suite: pytest -q\n",
                         {"units": SPRINT_UNITS[:1]}),
    "sprint_report": ("integrator", "", {"result": "pass", "detail": "suite green: 4 passed"}),
}


@pytest.mark.parametrize("verb", sorted(_VERB_CALLS))
def test_every_verb_dispatches_through_the_registry(board, monkeypatch, verb):
    """The exact path an agent's tool call takes: registry.dispatch(name, args)."""
    assignee, body, args = _VERB_CALLS[verb]
    tid, run_id = _running(board, title=f"unit — {verb}", assignee=assignee, body=body)
    _as_worker(monkeypatch, tid, run_id, verb)
    from tools.registry import registry

    assert registry.get_entry(verb) is not None, f"{verb} is not registered"
    out = registry.dispatch(verb, args)
    data = json.loads(out if isinstance(out, str) else json.dumps(out))
    assert data.get("ok") is True, data
    assert data.get("task_id", data.get("sprint_task")) == tid
    status = board["kb"].get_task(board["conn"], tid).status
    assert status != "running", f"{verb} left the card running ({status})"


def test_each_verb_schema_is_small_and_self_describing():
    """The whole point: six tiny schemas instead of 15 routing schemas (~26.8 KB).

    ``tools/AGENTS.md`` asks for contracts rather than tool counts; this budget is the
    deliberate exception — the surface cost IS the feature here, so it gets a guard.
    """
    from tools.registry import registry

    total = 0
    for verb in sorted(VERBS):
        schema = registry.get_schema(verb)
        assert schema is not None, f"{verb} has no schema"
        assert schema["name"] == verb
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["required"], f"{verb} declares no required argument"
        total += len(json.dumps(schema, separators=(",", ":")))
    assert total < 6000, f"worker verbs cost {total} chars — the budget was ~3.6 KB"


# ---------------------------------------------------------------------------
# Surface: toolset membership, model_tools swap, per-role filtering
# ---------------------------------------------------------------------------

VERBS = {"hand_off", "review_result", "gate_result", "open_round",
         "decompose_sprint", "sprint_report"}


def test_the_six_verbs_form_their_own_toolset():
    from toolsets import resolve_toolset
    import tools.kanban_worker_tools  # noqa: F401  (registration)
    assert set(resolve_toolset("kanban_worker")) == VERBS
    assert not (set(resolve_toolset("kanban")) & VERBS)


def test_worker_gets_kanban_worker_not_the_builtin_surface(monkeypatch, tmp_path):
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.setenv("HERMES_KANBAN_WORKER_TOOLS", "hand_off")
    import model_tools as M

    names = M._select_tool_names(["file"], [], quiet_mode=True)
    assert VERBS <= names
    assert not {n for n in names if n.startswith("kanban_")}


def test_role_verbs_are_filtered_by_child_env(monkeypatch, tmp_path):
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.setenv("HERMES_KANBAN_WORKER_TOOLS", "hand_off")
    from tools import kanban_worker_tools as kwt
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    visible = {n for n in VERBS if kwt.verb_visible(n)}
    assert visible == {"hand_off"}


def test_no_worker_verbs_without_a_dispatcher_task(monkeypatch, tmp_path):
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_WORKER_TOOLS", "hand_off")
    from tools import kanban_worker_tools as kwt
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    assert not any(kwt.verb_visible(n) for n in VERBS)


def test_kanban_list_reports_review_outcome(board, monkeypatch):
    """kanban_list(status='done') must tell green from red in one call."""
    tid, run_id = _running(board, assignee="reviewer")
    _as_worker(monkeypatch, tid, run_id, "review_result")
    from tools import kanban_worker_tools as kwt
    from tools import kanban_tools as kt

    _payload(kwt._handle_review_result({"verdict": "fail", "findings": "missing test"}))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    rows = json.loads(kt._handle_list({"status": "done"}))["tasks"]
    assert [r["review_outcome"] for r in rows if r["id"] == tid] == ["fail"]

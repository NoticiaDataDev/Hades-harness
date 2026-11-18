"""Kanban worker verbs — one purpose-built tool per role transition.

The built-in ``kanban`` surface (15 tools, ~26.8 KB of JSON schemas) is sized for a
lead/orchestrator that routes a board. A dispatcher worker needs *one* verb that
performs its role's single legal transition and owns the graph shape, because a weak
local model gets graph shapes wrong and a broken graph fails **silently** — the cards
simply never run.

| verb | role | transition |
|---|---|---|
| ``hand_off`` | coder | complete ``metadata.stage=implemented``; non-empty ``blocked_reason`` -> block ``needs_input`` |
| ``review_result`` | reviewer | ``pass``/``fail`` -> complete + ``metadata.review_outcome``; ``blocked`` -> block |
| ``gate_result`` | integrator (unit) | ``pass`` -> merge + complete ``GATE PASS``; ``fail`` -> revert + block |
| ``open_round`` | integrator (unit) | reads its own card body, creates ``C->R->G`` for round N+1, completes the gate card |
| ``decompose_sprint`` | integrator (sprint) | reads its own card, creates the unit triplets + closure card, parks itself on the closure card |
| ``sprint_report`` | integrator (closure) | completes or blocks the sprint card with the gap |

Rules this module enforces rather than trusting the model with:

- **``complete`` != success.** ``review_result`` completes on red too; the verdict is the
  summary prefix (``REVIEW FAIL — …``) plus ``metadata.review_outcome``, which the
  notifier, ``kanban_list`` and the gate card's injected parent handoff all read.
- **The machine never sends a card backwards.** Red work is a *new* card (``open_round``).
- **``open_round`` never authors acceptance criteria.** Round N+1 carries the original
  instruction verbatim; only the failure evidence is appended.
- Terminal verbs pass ``expected_run_id`` so a stale or retried worker cannot
  double-complete a card a live successor owns.

Visibility is per role: the ``kanban_worker`` toolset is appended by
``model_tools._select_tool_names`` for dispatcher workers, and each verb's ``check_fn``
opens only when ``HERMES_KANBAN_WORKER_TOOLS`` (exported by the dispatcher from
``kanban.worker_tools`` in the assignee profile) names it. With the env unset the
profile's role name picks the default lane.

Two caveats worth writing down:

- ``check_fn`` here answers *surface*, which ``tools/AGENTS.md`` says it should not. The
  exception is deliberate and mirrors the built-in ``kanban`` gate (``_check_kanban_mode``):
  the dispatcher spawns **one process per card**, so the process is the session. A worker
  hosted multiplexed inside a gateway process would see another card's lane — the fix
  there is to carry the lane through the session ContextVar the way the toolset resolver
  does, not to widen this gate.
- ``kanban.worker_tools`` in the assignee profile is the intended source of the lane:
  ``allowed_verbs`` reads the env pin first, then that config key, then the role name
  (``coder`` / ``reviewer`` / ``integrator``). A profile that pins nothing still gets the
  role lane, so no config is strictly required.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Optional

import tools.kanban_tools as _kt
from tools.registry import no_cache_check_fn, registry

# --------------------------------------------------------------------------- Verbs

_ROLE_VERBS: dict[str, set[str]] = {
    "coder": {"hand_off"},
    "reviewer": {"review_result"},
    "integrator": {"gate_result", "open_round", "decompose_sprint", "sprint_report"},
}
_ALL_VERBS = frozenset(set().union(*_ROLE_VERBS.values()))

MAX_ROUND = 3


def _env_verbs() -> Optional[set[str]]:
    """``HERMES_KANBAN_WORKER_TOOLS`` as a set, or None when the dispatcher did not pin it."""
    raw = os.environ.get("HERMES_KANBAN_WORKER_TOOLS")
    if raw is None:
        return None
    return {p for p in re.split(r"[\s,]+", raw.strip()) if p}


def _config_verbs() -> Optional[set[str]]:
    """``kanban.worker_tools`` from the assignee profile's config.yaml, or None.

    Read through ``load_config_readonly``: the tool resolver asks for this once per
    schema build, and the value is never mutated. ``worker_tools: hand_off`` (a bare
    string) is accepted as one verb — YAML makes that easy to write by accident.
    """
    try:
        from hermes_cli.config import load_config_readonly

        raw = (load_config_readonly().get("kanban") or {}).get("worker_tools")
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = re.split(r"[\s,]+", raw.strip())
    if not isinstance(raw, (list, tuple, set)):
        return None
    return {str(v).strip() for v in raw if str(v).strip()} or None


def allowed_verbs() -> Optional[set[str]]:
    """The verbs this worker may see: env pin, else profile config, else role lane, else all.

    The dispatcher may pin the lane per spawned process (``HERMES_KANBAN_WORKER_TOOLS``);
    otherwise the assignee profile's ``kanban.worker_tools`` decides, and with neither the
    role name does. Never consulted for a human/lead session — ``verb_visible`` gates on
    ``HERMES_KANBAN_TASK`` first.
    """
    pinned = _env_verbs() or _config_verbs()
    if pinned is not None:
        return pinned
    role = (os.environ.get("HERMES_PROFILE") or "").strip().lower()
    return set(_ROLE_VERBS[role]) if role in _ROLE_VERBS else None


def verb_visible(name: str) -> bool:
    """True only for a dispatcher-spawned worker whose lane includes *name*."""
    if _kt._is_delegated_child_context():
        return False
    if not os.environ.get("HERMES_KANBAN_TASK"):
        return False
    if not _kt._is_dispatcher_owned_worker():
        return False
    allowed = allowed_verbs()
    return allowed is None or name in allowed


def _preamble(tool: str, args: dict) -> tuple[str, Optional[int]]:
    """Worker mutation preamble: ownership, run identity, lane membership."""
    _kt._reject_delegated_child_mutation(tool)
    tid = _kt._require_task_id(args)
    _kt._enforce_worker_task_ownership(tid)
    run_id = _kt._worker_run_id(tid)
    if os.environ.get("HERMES_KANBAN_TASK") and run_id is None:
        raise _kt._Reject(
            f"{tool} refused: this worker cannot resolve its HERMES_KANBAN_RUN_ID, so it "
            "cannot prove ownership of the card's current run. Re-run through the "
            "dispatcher so the run id is pinned.")
    allowed = allowed_verbs()
    if allowed is not None and tool not in allowed:
        raise _kt._Reject(
            f"{tool} is not in this worker's lane ({', '.join(sorted(allowed))}). "
            "Use the verb your role owns; a human/lead session routes the board.")
    return tid, run_id


# --------------------------------------------------------------------------- Body convention

# The block every worker body carries (see the lead's card template):
#
#   ## Original instruction
#   ...verbatim...
#   ## Verification
#   command: <one shell command>
#   module: <path>
#   ## Branches
#   feature: wt/<task-id>
#   integration: integration
#   round: 1
#
# ``open_round`` re-emits ``## Original instruction`` + ``## Verification`` untouched and
# appends ``## What failed in round N``. Only the round number is rewritten.

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")


def _sections(body: Optional[str]) -> dict[str, str]:
    """``{"branches": "...", "original instruction": "..."}`` (headers lowercased)."""
    out: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []
    for line in (body or "").splitlines():
        m = _SECTION_RE.match(line)
        if m:
            if current is not None:
                out[current] = "\n".join(buf).strip("\n")
            current, buf = m.group(1).strip().lower(), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip("\n")
    return out


def _kv(text: str) -> dict[str, str]:
    """Parse ``key: value`` lines into a dict (keys lowercased)."""
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip().lower()] = value.strip()
    return out


def _branches(body: Optional[str]) -> dict[str, str]:
    """Parse the ``key: value`` lines of ``## Branches``."""
    return _kv(_sections(body).get("branches", ""))


def _round_of(body: Optional[str]) -> int:
    try:
        return max(1, int(_branches(body).get("round") or 1))
    except (TypeError, ValueError):
        raise _kt._Reject("## Branches must carry an integer `round: <n>` line")


def _base_title(title: str) -> str:
    """Strip a trailing ``— round N — <role>`` so round 2's cards read like round 1's."""
    return re.sub(r"\s+—\s+round\s+\d+\s+—\s+\w+\s*$", "", title or "").strip() or (title or "unit")


# --------------------------------------------------------------------------- Git helpers

def _git(workspace: Optional[str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(workspace), *args],
                          capture_output=True, text=True, timeout=120, check=False)


def _merge_feature(workspace: Optional[str], feature: str, integration: str) -> bool:
    """Best-effort ``git checkout <integration> && git merge --no-ff <feature>``."""
    if not (workspace and feature and integration and os.path.isdir(workspace)):
        return False
    try:
        co = _git(workspace, "checkout", "-q", integration)
        if co.returncode != 0:
            return False
        merge = _git(workspace, "merge", "--no-ff", "-q", "-m", f"integrate {feature}", feature)
        return merge.returncode == 0
    except Exception:
        return False


def _revert_integration(workspace: Optional[str], integration: str, ref: str) -> bool:
    """Best-effort ``git checkout <integration> && git reset --hard <ref>``."""
    if not (workspace and integration and ref and os.path.isdir(workspace)):
        return False
    try:
        co = _git(workspace, "checkout", "-q", integration)
        if co.returncode != 0:
            return False
        return _git(workspace, "reset", "--hard", ref).returncode == 0
    except Exception:
        return False


def _comment(kb, conn, tid: str, text: Optional[str]) -> None:
    """Post findings/evidence on the card's thread so the next worker reads them."""
    if text and str(text).strip():
        kb.add_comment(conn, tid, author=_kt._persisted_identity(), body=_kt._redact(text))


def _evidence_summary(prefix: str, detail: str) -> str:
    """``"REVIEW PASS — …"`` capped at 200 chars: the first 200 chars are the verdict the
    notifier and the parent-handoff injection read."""
    return f"{prefix} — {detail}".strip()[:200]


# --------------------------------------------------------------------------- 1. hand_off

@_kt._kanban_handler("hand_off")
def _handle_hand_off(args: dict, **kw) -> str:
    """Coder: the unit is implemented, or it needs input that is not on the card."""
    tid, run_id = _preamble("hand_off", args)
    summary = _kt._redact(_kt._require_text(
        args, "summary", "summary is required — say what was implemented and how it was verified"))
    blocked_reason = str(args.get("blocked_reason") or "").strip()
    with _kt._board(args.get("board")) as (kb, conn):
        if blocked_reason:
            ok = kb.block_task(conn, tid, reason=_kt._redact(blocked_reason),
                               kind="needs_input", expected_run_id=run_id)
            _kt._check(ok, f"could not block {tid} (unknown id or not in running/ready)")
            return _kt._ok_landed(kb, conn, tid, "blocked", block_kind="needs_input")
        metadata = _kt._stamp_worker_session_metadata(tid, {"stage": "implemented"})
        ok = kb.complete_task(conn, tid, summary=summary, metadata=metadata,
                              expected_run_id=run_id)
        _kt._check(ok, f"could not complete {tid} (unknown id, stale run, or unsatisfied parent)")
        return _kt._ok_landed(kb, conn, tid, "done", stage="implemented")


# --------------------------------------------------------------------------- 2. review_result

# The project's own law (AGENTS.md) says ``approved | changes_required``; the harness says
# ``pass | fail | blocked``. Accepting both removes the translation layer.
_VERDICTS = {
    "pass": "pass", "approved": "pass",
    "fail": "fail", "changes_required": "fail",
    "blocked": "blocked",
}


def _verdict(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value not in _VERDICTS:
        raise _kt._Reject(
            f"verdict must be one of: pass (alias approved), fail (alias changes_required), "
            f"blocked. Got {raw!r}. Nothing changed.")
    return _VERDICTS[value]


@_kt._kanban_handler("review_result")
def _handle_review_result(args: dict, **kw) -> str:
    """Reviewer: a verdict on one unit. Red completes the card too — the machine never
    sends a card backwards; the integrator's gate opens a new round."""
    tid, run_id = _preamble("review_result", args)
    verdict = _verdict(args.get("verdict"))
    findings = _kt._redact(_kt._require_text(
        args, "findings", "findings are required — name the observable behaviour that failed"))
    with _kt._board(args.get("board")) as (kb, conn):
        _comment(kb, conn, tid, findings)
        if verdict == "blocked":
            ok = kb.block_task(conn, tid, reason=findings, kind="needs_input",
                               expected_run_id=run_id)
            _kt._check(ok, f"could not block {tid} (unknown id or not in running/ready)")
            return _kt._ok_landed(kb, conn, tid, "blocked", block_kind="needs_input",
                                  review_outcome=verdict)
        metadata = _kt._stamp_worker_session_metadata(
            tid, {"review_outcome": verdict, "findings": findings})
        ok = kb.complete_task(
            conn, tid, summary=_evidence_summary(f"REVIEW {verdict.upper()}", findings),
            metadata=metadata, expected_run_id=run_id)
        _kt._check(ok, f"could not complete {tid} (unknown id, stale run, or unsatisfied parent)")
        return _kt._ok_landed(kb, conn, tid, "done", review_outcome=verdict)


# --------------------------------------------------------------------------- 3. gate_result

@_kt._kanban_handler("gate_result")
def _handle_gate_result(args: dict, **kw) -> str:
    """Integrator, per unit: green merges the feature branch into the integration branch;
    red reverts it (when the body names a ``rollback_ref``) and blocks for a human."""
    tid, run_id = _preamble("gate_result", args)
    result = str(args.get("result") or "").strip().lower()
    if result not in {"pass", "fail"}:
        raise _kt._Reject(f"result must be 'pass' or 'fail', got {args.get('result')!r}")
    detail = _kt._redact(_kt._require_text(
        args, "detail", "detail is required — the unit command's output or the failure evidence"))
    with _kt._board(args.get("board")) as (kb, conn):
        task = _kt._existing_task(kb, conn, tid)
        branches = _branches(task.body)
        workspace = task.workspace_path
        if result == "pass":
            merged = _merge_feature(workspace, branches.get("feature", ""),
                                    branches.get("integration", ""))
            metadata = _kt._stamp_worker_session_metadata(
                tid, {"gate_result": "pass", "merged": merged})
            ok = kb.complete_task(
                conn, tid, summary=_evidence_summary("GATE PASS", detail),
                metadata=metadata, expected_run_id=run_id)
            _kt._check(ok, f"could not complete {tid} (unknown id, stale run, or unsatisfied parent)")
            return _kt._ok_landed(kb, conn, tid, "done", gate_result="pass", merged=merged)
        reverted = _revert_integration(workspace, branches.get("integration", ""),
                                       branches.get("rollback_ref", ""))
        ok = kb.block_task(conn, tid, reason=_evidence_summary("GATE FAIL", detail),
                           kind="needs_input", expected_run_id=run_id)
        _kt._check(ok, f"could not block {tid} (unknown id or not in running/ready)")
        return _kt._ok_landed(kb, conn, tid, "blocked", gate_result="fail",
                              reverted=reverted, block_kind="needs_input")


# --------------------------------------------------------------------------- 4. open_round

def _round_body(original: str, verification: str, integration: str, round_no: int,
                failure: str) -> str:
    """Round N+1's card body: the original instruction verbatim + the new evidence."""
    lines = [
        "## Original instruction", original.strip(), "", "## Verification",
    ]
    if verification.strip():
        lines.append(verification.strip())
    lines += ["", "## Branches", "integration: " + (integration or "integration"),
              f"round: {round_no}", "", f"## What failed in round {round_no - 1}",
              failure.strip(), ""]
    return "\n".join(lines)


@_kt._kanban_handler("open_round")
def _handle_open_round(args: dict, **kw) -> str:
    """Integrator, per unit: red at the gate opens round N+1 as three *new* cards and
    completes the gate card. Round 3 blocks instead of creating a fourth."""
    tid, run_id = _preamble("open_round", args)
    failure = _kt._redact(_kt._require_text(
        args, "self_failure", "self_failure is required — paste the evidence that failed"))
    with _kt._board(args.get("board")) as (kb, conn):
        task = _kt._existing_task(kb, conn, tid)
        sections = _sections(task.body)
        original = sections.get("original instruction", "").strip()
        _kt._check(original, "this card's body has no '## Original instruction' block; "
                             "open_round cannot re-word the unit — block instead")
        round_no = _round_of(task.body)
        integration = _branches(task.body).get("integration", "integration")
        if round_no >= MAX_ROUND:
            ok = kb.block_task(
                conn, tid, kind="needs_input", expected_run_id=run_id,
                reason=f"round {round_no} still red after {MAX_ROUND} rounds: {failure}")
            _kt._check(ok, f"could not block {tid} (unknown id or not in running/ready)")
            return _kt._ok_landed(kb, conn, tid, "blocked", round=round_no, cards=[],
                                  block_kind="needs_input")
        next_round = round_no + 1
        base = _base_title(task.title)
        body = _round_body(original, sections.get("verification", ""), integration,
                           next_round, failure)
        created_by = _kt._persisted_identity()
        board = args.get("board")
        code = kb.create_task(
            conn, title=f"{base} — round {next_round} — code", body=body, assignee="coder",
            created_by=created_by, board=board, creator_task_id=tid)
        review = kb.create_task(
            conn, title=f"{base} — round {next_round} — review", body=body,
            assignee="reviewer", parents=(code,), created_by=created_by, board=board,
            creator_task_id=tid)
        gate = kb.create_task(
            conn, title=f"{base} — round {next_round} — gate", body=body,
            assignee="integrator", parents=(review,), created_by=created_by, board=board,
            creator_task_id=tid)
        completed = kb.complete_task(
            conn, tid, summary=_evidence_summary(f"ROUND {next_round} OPENED", failure),
            metadata=_kt._stamp_worker_session_metadata(
                tid, {"round": next_round, "round_cards": [code, review, gate]}),
            expected_run_id=run_id)
        _kt._check(completed, f"could not complete {tid} (unknown id, stale run, or unsatisfied parent)")
        return _kt._ok_landed(kb, conn, tid, "done", round=next_round,
                              cards={"code": code, "review": review, "gate": gate})


# --------------------------------------------------------------------------- 5. decompose_sprint

def _unit_body(unit: dict, integration: str, workspace: str = "") -> str:
    title = str(unit["title"]).strip()
    files = str(unit["files"]).strip()
    acceptance = unit["acceptance"]
    if isinstance(acceptance, str):
        acceptance = [acceptance]
    bullets = "\n".join(f"- {str(a).strip()}" for a in acceptance if str(a).strip())
    command = str(unit["command"]).strip()
    instruction = f"{title}\nfiles: {files}\nacceptance:\n{bullets}\ncommand: {command}"
    lines = [
        "## Original instruction", instruction, "",
        "## Verification", f"command: {command}", f"module: {files}", "",
    ]
    if workspace:
        lines += ["## Workspace",
                  f"root: {workspace}",
                  "Write the files above at these exact paths, relative to the workspace root.",
                  ""]
    lines += ["## Branches", f"integration: {integration}", "round: 1", ""]
    return "\n".join(lines)


def _validate_units(units: Any) -> list[dict]:
    _kt._check(isinstance(units, list) and units, "units must be a non-empty list of unit objects")
    clean = []
    for i, unit in enumerate(units, 1):
        _kt._check(isinstance(unit, dict), f"unit {i} must be an object")
        for key in ("title", "files", "command"):
            _kt._check(str(unit.get(key) or "").strip(),
                       f"unit {i} is missing '{key}' (a unit needs a title, its files and the "
                       "one command that proves it)")
        acceptance = unit.get("acceptance")
        if isinstance(acceptance, str):
            acceptance = [acceptance]
        acceptance = [str(a).strip() for a in (acceptance or []) if str(a).strip()]
        _kt._check(acceptance,
                   f"unit {i} has no 'acceptance' criteria — the lead owns them, the integrator "
                   "must not invent them")
        clean.append({**unit, "acceptance": acceptance})
    return clean


@_kt._kanban_handler("decompose_sprint")
def _handle_decompose_sprint(args: dict, **kw) -> str:
    """Integrator, sprint card: create the ``C->R->G`` triplet per unit plus the closure
    card, then park the sprint card on the closure card."""
    tid, run_id = _preamble("decompose_sprint", args)
    units = _validate_units(args.get("units"))
    with _kt._board(args.get("board")) as (kb, conn):
        task = _kt._existing_task(kb, conn, tid)
        _kt._check(not kb.child_ids(conn, tid),
                   f"{tid} already has children — this sprint was decomposed before")
        sections = _sections(task.body)
        integration_block = sections.get("integration", "")
        branch = "integration"
        suite = ""
        declared_ws = ""
        for line in integration_block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                k = key.strip().lower()
                if k == "branch" and value.strip():
                    branch = value.strip()
                if k == "full suite" and value.strip():
                    suite = value.strip()
                if k == "workspace" and value.strip():
                    declared_ws = value.strip()
        # The workspace decides what "verified" means: the coder writes there, the reviewer
        # runs the command there, the gate merges there. Scratch is the wrong default for a
        # unit — every child would get a private, empty directory, and a green would prove
        # nothing about the code the pipeline produced. So the integrator must name the
        # workspace before any unit card exists: in '## Integration', or by creating the
        # sprint card with a dir workspace.
        workspace = declared_ws or (
            task.workspace_path
            if (task.workspace_kind == "dir" and task.workspace_path) else "")
        _kt._check(bool(workspace),
                   f"{tid} names no workspace: add 'workspace: /absolute/path' to its "
                   f"'## Integration' section, or create the sprint card with a dir workspace. "
                   f"Refusing to create unit cards — each would get a private scratch directory, "
                   f"the reviewer would test a different tree than the coder wrote, and the gate "
                   f"would merge nothing.")
        _kt._check(os.path.isabs(workspace),
                   f"workspace must be an absolute path, got {workspace!r}")
        os.makedirs(workspace, exist_ok=True)
        workspace = os.path.realpath(workspace)
        created_by = _kt._persisted_identity()
        board = args.get("board")
        made: list[dict] = []
        for index, unit in enumerate(units, 1):
            body = _unit_body(unit, branch, workspace)
            base = str(unit["title"]).strip()
            code = kb.create_task(
                conn, title=f"{base} — code", body=body, assignee="coder",
                created_by=created_by, board=board, creator_task_id=tid,
                workspace_kind="dir", workspace_path=workspace)
            review = kb.create_task(
                conn, title=f"{base} — review", body=body, assignee="reviewer",
                parents=(code,), created_by=created_by, board=board, creator_task_id=tid,
                workspace_kind="dir", workspace_path=workspace)
            gate = kb.create_task(
                conn, title=f"{base} — gate", body=body, assignee="integrator",
                parents=(review,), created_by=created_by, board=board, creator_task_id=tid,
                workspace_kind="dir", workspace_path=workspace)
            made.append({"title": base, "code": code, "review": review, "gate": gate})
        closure_body = "\n".join([
            "## Sprint",
            f"card: {tid}",   # the sprint card this close-out closes (sprint_report reads it)
            "",
            "## Sprint close-out",
            f"branch: {branch}",
            f"full suite: {suite or '<the project suite>'}",
            f"workspace: {workspace}",
            "",
            "Every unit's gate is complete. Verify the *combination*: run the full suite and",
            "start the project. Report a GAP only if the project is broken after every unit",
            "passed; otherwise report pass.",
            "",
            "Call sprint_report(result=\"pass\"|\"gap\", detail=<the evidence>).",
        ])
        closure = kb.create_task(
            conn, title=f"{_base_title(task.title)} — close-out", body=closure_body,
            assignee="integrator", parents=tuple(u["gate"] for u in made),
            created_by=created_by, board=board, creator_task_id=tid,
            workspace_kind="dir", workspace_path=workspace)
        # The close-out is a child of every gate, never their parent: linking it the other
        # way round made the closure ready the moment it was created, so it reported on a
        # sprint whose units had not started yet.
        kb.link_tasks(conn, parent_id=closure, child_id=tid, expected_child_run_id=run_id)
        parked = kb.block_task(
            conn, tid, kind="dependency", expected_run_id=run_id,
            reason=f"decomposed into {len(made)} unit(s); resumes when {closure} closes the sprint")
        _kt._check(parked or kb.get_task(conn, tid).status == "todo",
                   f"could not park {tid} on {closure}")
        landed = kb.get_task(conn, tid)
        return _kt._ok(task_id=tid, status=landed.status if landed else "todo",
                       units=made, unit_cards=[c for u in made for c in (u["code"], u["review"], u["gate"])],
                       closure_card=closure, integration_branch=branch)


# --------------------------------------------------------------------------- 6. sprint_report

@_kt._kanban_handler("sprint_report")
def _handle_sprint_report(args: dict, **kw) -> str:
    """Integrator, closure card: green closes the sprint card, a gap blocks it for the lead."""
    tid, run_id = _preamble("sprint_report", args)
    result = str(args.get("result") or "").strip().lower()
    if result not in {"pass", "gap"}:
        raise _kt._Reject(f"result must be 'pass' or 'gap', got {args.get('result')!r}")
    detail = _kt._redact(_kt._require_text(
        args, "detail", "detail is required — the full-suite output or the gap's evidence"))
    with _kt._board(args.get("board")) as (kb, conn):
        task = _kt._existing_task(kb, conn, tid)
        sections = _sections(task.body)
        # The closure card links every unit gate *and* the sprint card as children, so the
        # sprint card must be named explicitly, not inferred from child order.
        sprint_tid = _kv(sections.get("sprint", "")).get("card") or (
            kb.child_ids(conn, tid) or [tid])[0]
        _comment(kb, conn, sprint_tid, detail)
        if result == "gap":
            reason = _evidence_summary("SPRINT GAP", detail)
            # The sprint card waits in `todo` behind the closure card, and block_task only
            # accepts running/ready: close the closure card first (the lead's wake), let the
            # dependency clear, then park the sprint card for a human.
            if sprint_tid != tid:
                closed = kb.complete_task(
                    conn, tid, summary=reason,
                    metadata=_kt._stamp_worker_session_metadata(tid, {"sprint_result": "gap"}),
                    expected_run_id=run_id)
                _kt._check(closed, f"could not complete the closure card {tid}")
                kb.recompute_ready(conn)
            ok = kb.block_task(
                conn, sprint_tid, reason=reason, kind="needs_input",
                expected_run_id=run_id if sprint_tid == tid else None)
            _kt._check(ok, f"could not block the sprint card {sprint_tid}")
        else:
            # The closure card is the sprint card's parent: close it first, then the
            # sprint card, or the parent gate refuses the completion.
            if sprint_tid != tid:
                closed = kb.complete_task(
                    conn, tid, summary=_evidence_summary("SPRINT PASS", detail),
                    metadata=_kt._stamp_worker_session_metadata(tid, {"sprint_result": "pass"}),
                    expected_run_id=run_id)
                _kt._check(closed, f"could not complete the closure card {tid}")
            ok = kb.complete_task(
                conn, sprint_tid, summary=_evidence_summary("SPRINT PASS", detail),
                metadata=_kt._stamp_worker_session_metadata(sprint_tid, {"sprint_result": "pass"}),
                expected_run_id=None)
            _kt._check(ok, f"could not complete the sprint card {sprint_tid}")
        landed = kb.get_task(conn, sprint_tid)
        return _kt._ok(sprint_task=sprint_tid, status=landed.status if landed else result,
                       sprint_result=result)


# --------------------------------------------------------------------------- Schemas

def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Kanban tool schema (inner shape registry.get_schema returns); every verb takes the
    same optional trailing ``board`` the built-in kanban tools do."""
    return {"name": name, "description": description, "parameters": {
        "type": "object",
        "properties": {**properties, "board": {
            "type": "string",
            "description": "Board slug. Omit for the active board (the dispatcher's pin)."}},
        "required": required}}


_STR = {"type": "string"}

HAND_OFF_SCHEMA = _schema(
    "hand_off",
    "Hand your card on. Default: the unit is implemented (tests written and green). "
    "If something outside your control stops you, pass blocked_reason instead and a "
    "human will be asked. One call ends your run.",
    {"summary": {**_STR, "description": "What you did and the command you ran to verify it."},
     "blocked_reason": {**_STR, "description": "Only when you truly cannot proceed; asks a human."}},
    ["summary"])

REVIEW_RESULT_SCHEMA = _schema(
    "review_result",
    "Record your verdict on this unit. pass (alias approved): the acceptance criteria are "
    "observably met. fail (alias changes_required): they are not — findings must name the "
    "behaviour that failed; a new round will be opened. blocked: you cannot judge it.",
    {"verdict": {"type": "string", "enum": ["pass", "fail", "blocked", "approved", "changes_required"],
                 "description": "pass/approved, fail/changes_required, or blocked."},
     "findings": {**_STR, "description": "Structured findings; each names a behaviour and the evidence."}},
    ["verdict", "findings"])

GATE_RESULT_SCHEMA = _schema(
    "gate_result",
    "Integrator only, per unit. pass: the unit's command and the project's import/smoke check "
    "are green on the integration branch — the handler merges your feature branch and closes "
    "the gate. fail: paste the failure and the handler reverts the integration branch and "
    "blocks the gate.",
    {"result": {"type": "string", "enum": ["pass", "fail"]},
     "detail": {**_STR, "description": "The unit command's output, or the failure evidence."}},
    ["result", "detail"])

OPEN_ROUND_SCHEMA = _schema(
    "open_round",
    "Integrator only, per unit gate: the review came back red — open the next round as three "
    "new cards (code -> review -> gate) carrying this card's original instruction verbatim "
    "plus the failure evidence, and close this gate card. Round 3 blocks instead; it never "
    "creates a fourth.",
    {"self_failure": {**_STR, "description": "The evidence that failed, exactly as reported."}},
    ["self_failure"])

DECOMPOSE_SPRINT_SCHEMA = _schema(
    "decompose_sprint",
    "Integrator only, on the sprint card. Create the code -> review -> gate triplet for every "
    "unit in the sprint spec plus the close-out card, then park this card until the close-out "
    "runs. Units may be split or merged; criteria may not be added or weakened.",
    {"units": {"type": "array", "description": "One object per unit.", "items": {
        "type": "object",
        "properties": {
            "title": {**_STR, "description": "Short unit title."},
            "files": {**_STR, "description": "Paths this unit touches."},
            "acceptance": {"type": "array", "items": _STR,
                           "description": "Observable behaviours, verbatim from the sprint spec."},
            "command": {**_STR, "description": "The exact command that proves the unit."},
        },
        "required": ["title", "files", "acceptance", "command"]}}},
    ["units"])

SPRINT_REPORT_SCHEMA = _schema(
    "sprint_report",
    "Integrator only, on the close-out card. pass: every unit passed and the project's full "
    "suite plus startup check are green — the sprint closes. gap: the units were green but the "
    "combined project is broken; explain the gap and the sprint is blocked for the lead.",
    {"result": {"type": "string", "enum": ["pass", "gap"]},
     "detail": {**_STR, "description": "The full-suite output, or the gap's evidence."}},
    ["result", "detail"])


# --------------------------------------------------------------------------- Registration

_TOOLS = (
    ("hand_off", HAND_OFF_SCHEMA, _handle_hand_off, "🤝"),
    ("review_result", REVIEW_RESULT_SCHEMA, _handle_review_result, "👀"),
    ("gate_result", GATE_RESULT_SCHEMA, _handle_gate_result, "🚦"),
    ("open_round", OPEN_ROUND_SCHEMA, _handle_open_round, "🔄"),
    ("decompose_sprint", DECOMPOSE_SPRINT_SCHEMA, _handle_decompose_sprint, "🗂"),
    ("sprint_report", SPRINT_REPORT_SCHEMA, _handle_sprint_report, "🏁"),
)


def _check_fn(name: str):
    @no_cache_check_fn
    def _visible_for_role() -> bool:
        return verb_visible(name)
    return _visible_for_role


for _name, _schema_def, _handler, _emoji in _TOOLS:
    registry.register(name=_name, toolset="kanban_worker", schema=_schema_def,
                      handler=_handler, emoji=_emoji, check_fn=_check_fn(_name))

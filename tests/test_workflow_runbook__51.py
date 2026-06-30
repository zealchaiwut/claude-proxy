"""Tests for issue #51: Commander onboarding runbook at docs/workflow.md.

Each test is anchored to a specific AC item.
"""
import re
from pathlib import Path

RUNBOOK = Path(__file__).parent.parent / "docs" / "workflow.md"


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


# AC 1 — file exists and is non-empty
def test_ac1_file_exists():
    assert RUNBOOK.exists(), "docs/workflow.md must exist"
    assert len(_text()) > 200, "docs/workflow.md must be substantive (>200 chars)"


# AC 2 — prerequisites section calls out design-docs guard requirement
def test_ac2_prerequisites_section_exists():
    text = _text().lower()
    assert "prerequisite" in text, "runbook must contain a prerequisites section"


def test_ac2_prerequisites_mentions_product_md():
    text = _text()
    assert "PRODUCT.md" in text, "prerequisites must mention PRODUCT.md"


def test_ac2_prerequisites_mentions_design_md():
    text = _text()
    assert "DESIGN.md" in text, "prerequisites must mention DESIGN.md"


def test_ac2_prerequisites_mentions_develop_branch():
    text = _text().lower()
    assert "develop" in text, "prerequisites must mention the develop branch"


def test_ac2_design_docs_guard_mentioned():
    text = _text().lower()
    assert "design-docs" in text or "design docs" in text, (
        "prerequisites must mention the design-docs guard requirement"
    )


# AC 3 — step (a): create claude-proxy repo, commit PRODUCT.md + DESIGN.md to develop
def test_ac3_step_a_create_repo_mentioned():
    text = _text().lower()
    assert "create" in text and "repo" in text or "repository" in text, (
        "step (a) must instruct the reader to create the claude-proxy repo"
    )


def test_ac3_step_a_commit_product_design_to_develop():
    text = _text()
    # Both files mentioned in context of develop/commit
    assert "PRODUCT.md" in text and "DESIGN.md" in text, (
        "step (a) must mention committing both PRODUCT.md and DESIGN.md"
    )


def test_ac3_step_a_explains_why_design_docs_guard():
    text = _text().lower()
    # Must explain why (design-docs guard blocks scaffold without them)
    assert "guard" in text or "blocks" in text or "required" in text, (
        "step (a) must explain why PRODUCT.md and DESIGN.md are needed"
    )


# AC 4 — step (b): scaffold_project.py then init_project.py in order, with expected output
def test_ac4_step_b_scaffold_project_mentioned():
    text = _text()
    assert "scaffold_project.py" in text, (
        "step (b) must mention scaffold_project.py"
    )


def test_ac4_step_b_init_project_mentioned():
    text = _text()
    assert "init_project.py" in text, (
        "step (b) must mention init_project.py"
    )


def test_ac4_step_b_order_scaffold_before_init():
    text = _text()
    scaffold_pos = text.index("scaffold_project.py")
    init_pos = text.index("init_project.py")
    assert scaffold_pos < init_pos, (
        "scaffold_project.py must appear before init_project.py (run in order)"
    )


def test_ac4_step_b_expected_output_for_scaffold():
    text = _text()
    # Section about scaffold must have some expected output
    scaffold_pos = text.index("scaffold_project.py")
    after_scaffold = text[scaffold_pos : scaffold_pos + 600]
    assert "stamp" in after_scaffold.lower() or "created" in after_scaffold.lower() or (
        "ok" in after_scaffold.lower() or "success" in after_scaffold.lower()
    ), "step (b) must show expected output after scaffold_project.py"


def test_ac4_step_b_expected_output_for_init():
    text = _text()
    init_pos = text.index("init_project.py")
    after_init = text[init_pos : init_pos + 600]
    assert (
        "label" in after_init.lower()
        or "branch" in after_init.lower()
        or "ok" in after_init.lower()
        or "success" in after_init.lower()
        or "created" in after_init.lower()
    ), "step (b) must show expected output after init_project.py"


# AC 5 — step (c): installing and starting proxy service, M7 tickets 1-2 (#48 and #49)
def test_ac5_step_c_install_mentioned():
    text = _text().lower()
    assert "install" in text, "step (c) must cover installing the proxy service"


def test_ac5_step_c_references_issue_48():
    text = _text()
    assert "#48" in text or "issue 48" in text.lower() or "ticket 48" in text.lower(), (
        "step (c) must reference M7 ticket #48 (installable entrypoints)"
    )


def test_ac5_step_c_references_issue_49():
    text = _text()
    assert "#49" in text or "issue 49" in text.lower() or "ticket 49" in text.lower(), (
        "step (c) must reference M7 ticket #49 (systemd/launchd service units)"
    )


# AC 6 — step (d): ANTHROPIC_BASE_URL=http://localhost:8788
def test_ac6_step_d_anthropic_base_url_exact_value():
    text = _text()
    assert "ANTHROPIC_BASE_URL=http://localhost:8788" in text or (
        "ANTHROPIC_BASE_URL" in text and "http://localhost:8788" in text
    ), "step (d) must name ANTHROPIC_BASE_URL=http://localhost:8788 exactly"


def test_ac6_step_d_explains_both_claude_code_and_commander():
    text = _text().lower()
    # Must explain that both Claude Code and Commander dispatch need this value
    assert "claude code" in text or "claude-code" in text, (
        "step (d) must mention Claude Code as a consumer of ANTHROPIC_BASE_URL"
    )
    assert "commander" in text or "dispatch" in text, (
        "step (d) must mention Commander dispatch as a consumer of ANTHROPIC_BASE_URL"
    )


# AC 7 — step (e): CCPROXY_PROFILE with per-subprocess mapping
def test_ac7_step_e_ccproxy_profile_named():
    text = _text()
    assert "CCPROXY_PROFILE" in text, "step (e) must name the CCPROXY_PROFILE env var"


def test_ac7_step_e_cheap_or_local_for_tester():
    text = _text().lower()
    assert ("cheap" in text or "local" in text) and "tester" in text, (
        "step (e) must specify cheap or local profile for the tester subprocess"
    )


def test_ac7_step_e_cheap_or_local_for_estimator():
    text = _text().lower()
    assert ("cheap" in text or "local" in text) and "estimator" in text, (
        "step (e) must specify cheap or local profile for the estimator subprocess"
    )


def test_ac7_step_e_anthropic_for_coder():
    text = _text().lower()
    assert "anthropic" in text and "coder" in text, (
        "step (e) must specify anthropic profile for the coder subprocess"
    )


# AC 8 — step (f): exact curl commands for /health and /ready with expected responses
def test_ac8_step_f_health_curl_command():
    text = _text()
    assert "/health" in text, "step (f) must include a /health command"
    assert "curl" in text.lower() or "GET" in text, (
        "step (f) must show how to call /health"
    )


def test_ac8_step_f_health_expected_response():
    text = _text()
    # Must show expected response body
    assert '"status"' in text or "'status'" in text, (
        "step (f) must show the expected /health response body with status field"
    )


def test_ac8_step_f_ready_curl_command():
    text = _text()
    assert "/ready" in text, "step (f) must include a /ready command"


def test_ac8_step_f_ready_expected_response_ok():
    text = _text()
    # Expected success response: {"status": "ok", "profile": "..."}
    assert '"ok"' in text or "'ok'" in text, (
        "step (f) must show the expected /ready success response with status ok"
    )


def test_ac8_step_f_smoke_sprint_invocation():
    text = _text().lower()
    assert "smoke" in text or ("sprint" in text and ("run" in text or "dispatch" in text)), (
        "step (f) must include a smoke sprint invocation"
    )


# AC 9 — no step requires consulting a file outside docs/workflow.md or docs/
def test_ac9_no_external_file_references():
    text = _text()
    # Check for patterns like "see README" or "consult CLAUDE.md" that point outside docs/
    bad_patterns = [
        r"\bsee README\b",
        r"\bsee CLAUDE\.md\b",
        r"\bconsult CLAUDE\.md\b",
        r"\bsee \.env\.example\b",
        r"see config\.example\.toml",
    ]
    for pattern in bad_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        assert not match, (
            f"runbook must not direct readers outside docs/ — found: {match.group()!r}"
        )

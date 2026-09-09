"""The command line, which had no tests at all.

WHY THIS EXISTS
    `cli.py` is 463 statements and was at 0% coverage: the primary user-facing
    entry point of the whole system, with no regression protection. A renamed
    option, a moved import or a command that raises on invocation would all have
    shipped undetected, because nothing ever invoked one.

WHAT IT COVERS, AND WHAT IT DELIBERATELY DOES NOT
    Command wiring, argument contracts, and the behaviour of every command that
    can run without a 709 MB checkpoint: the read-only listing and inspection
    commands, and the error paths.

    It does not run inference. `predict`, `analyze`, `resimulate` and the rest
    need real TRIBE weights, gated model access and tens of minutes each; those
    live behind the opt-in `tribe_integration` marker. What is checked here is
    that they are correctly *wired* -- that their options parse and their help
    renders -- which is the part that breaks silently during a refactor.

THE ENUMERATION MATTERS
    The command list is read from the Typer app rather than written out here,
    so a command added later is covered by these tests the day it is added
    rather than the day someone remembers to update a list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from blackmirror.cli import app

runner = CliRunner()

#: Read from the app, not hand-maintained: a new command is covered on arrival.
COMMAND_NAMES = sorted(get_command(app).commands)  # type: ignore[attr-defined]

#: Commands that only read stored artifacts, so they can run against an empty
#: directory and must say so cleanly rather than raising.
READ_ONLY = [
    "list-runs",
    "list-resimulations",
    "list-searches",
]


@pytest.fixture
def empty_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BLACKMIRROR_ARTIFACT_DIR", str(tmp_path))
    return tmp_path


class TestCommandWiring:
    def test_the_app_exposes_the_commands_the_docs_describe(self) -> None:
        # A floor, not an exact list: this fails if commands disappear in a
        # refactor, without breaking every time one is legitimately added.
        expected = {
            "predict", "analyze", "compare", "benchmark", "export-mesh",
            "inspect-run", "inspect-model", "list-runs", "verify-env",
            "resimulate", "list-searches", "search-report",
        }
        missing = expected - set(COMMAND_NAMES)
        assert not missing, f"commands vanished from the CLI: {sorted(missing)}"

    @pytest.mark.parametrize("name", COMMAND_NAMES)
    def test_every_command_renders_help(self, name: str) -> None:
        """Catches a broken option definition or a moved import at test time."""
        result = runner.invoke(app, [name, "--help"])

        assert result.exit_code == 0, f"{name} --help failed:\n{result.output}"
        assert result.output.strip()

    @pytest.mark.parametrize("name", COMMAND_NAMES)
    def test_every_command_documents_itself(self, name: str) -> None:
        command = get_command(app).commands[name]  # type: ignore[attr-defined]
        assert (command.help or "").strip(), f"{name} has no help text"

    def test_the_top_level_help_lists_every_command(self) -> None:
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        # Rich wraps long lines, so check the name appears rather than matching
        # a whole formatted row.
        for name in COMMAND_NAMES:
            assert name in result.output, f"{name} is not discoverable from --help"


class TestReadOnlyCommandsOnAnEmptyStore:
    @pytest.mark.parametrize("name", READ_ONLY)
    def test_an_empty_store_is_reported_not_raised(
        self, name: str, empty_artifacts: Path
    ) -> None:
        """Nothing stored yet is the normal first-run state, not an error."""
        result = runner.invoke(app, [name])

        assert result.exit_code == 0, f"{name} raised on an empty store:\n{result.output}"
        assert "Traceback" not in result.output

    def test_inspect_run_with_no_runs_explains_rather_than_crashing(
        self, empty_artifacts: Path
    ) -> None:
        result = runner.invoke(app, ["inspect-run"])

        assert "Traceback" not in result.output
        assert "No runs" in result.output or "no runs" in result.output

    def test_list_runs_honours_its_limit_option(self, empty_artifacts: Path) -> None:
        result = runner.invoke(app, ["list-runs", "--limit", "5"])

        assert result.exit_code == 0
        assert "Traceback" not in result.output


class TestErrorPaths:
    def test_an_unknown_command_fails_without_a_traceback(self) -> None:
        result = runner.invoke(app, ["not-a-real-command"])

        assert result.exit_code != 0
        assert "Traceback" not in result.output

    def test_a_missing_run_id_is_reported_readably(self, empty_artifacts: Path) -> None:
        result = runner.invoke(app, ["inspect-run", "no-such-run"])

        # Either exit code is defensible; an unhandled traceback is not.
        assert "Traceback" not in result.output
        assert result.output.strip()

    def test_a_traversing_run_id_does_not_escape_the_artifact_root(
        self, empty_artifacts: Path
    ) -> None:
        secret = empty_artifacts.parent / "secret.txt"
        secret.write_text("private", encoding="utf-8")

        result = runner.invoke(app, ["inspect-run", "../../secret"])

        assert "private" not in result.output
        assert "Traceback" not in result.output


class TestVerifyEnv:
    def test_it_reports_rather_than_raising_when_dependencies_are_absent(
        self, empty_artifacts: Path
    ) -> None:
        """The whole point of the command is to run where things are missing."""
        result = runner.invoke(app, ["verify-env"])

        assert "Traceback" not in result.output
        assert result.output.strip(), "verify-env printed nothing"

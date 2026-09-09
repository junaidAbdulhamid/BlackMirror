"""Developer CLI.

Deliberately thin. Every command delegates to :class:`InferenceService` or a
utility module — no inference logic lives here — so that the FastAPI layer added
in a later phase calls exactly the same code paths::

    CLI     ─┐
             ├─▶ InferenceService ─▶ CorticalPredictorBackend
    FastAPI ─┘
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from blackmirror.config.settings import (
    BackendName,
    DevicePreference,
    Settings,
    get_settings,
)
from blackmirror.errors import BlackMirrorError
from blackmirror.schemas.prediction import PredictionResult
from blackmirror.storage.artifact_store import ArtifactStore
from blackmirror.utils.logging import configure_logging

app = typer.Typer(
    name="blackmirror",
    help="BlackMirror — predicted cortical response inference (Phase 1: TRIBE Core).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

BackendOption = Annotated[
    BackendName | None,
    typer.Option("--backend", "-b", help="Override the configured prediction backend."),
]
DeviceOption = Annotated[
    DevicePreference | None,
    typer.Option("--device", "-d", help="Override the compute device."),
]


def _settings(
    backend: BackendName | None = None, device: DevicePreference | None = None
) -> Settings:
    overrides: dict[str, object] = {}
    if backend is not None:
        overrides["backend"] = backend
    if device is not None:
        overrides["device"] = device
    settings = get_settings(**overrides)
    configure_logging(settings.log_level)
    return settings


def _fail(exc: BlackMirrorError) -> None:
    console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
    cause = exc.__cause__
    if cause is not None:
        console.print(f"[dim]caused by {type(cause).__name__}: {cause}[/dim]")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------


@app.command("verify-env")
def verify_env(
    backend: BackendOption = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Diagnose whether this environment can run inference, and how to fix it if not."""
    from blackmirror.diagnostics import run_environment_checks

    settings = _settings(backend)
    report = run_environment_checks(settings)

    if json_output:
        console.print_json(json.dumps(report.to_dict()))
        raise typer.Exit(code=0 if report.ready else 1)

    console.print()
    console.print("[bold]BlackMirror Environment[/bold]")
    console.print("=" * 60)

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Check", width=22)
    table.add_column("Status", width=10)
    table.add_column("Detail", overflow="fold")
    for check in report.checks:
        colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "SKIP": "dim"}[check.status]
        table.add_row(check.name, f"[{colour}]{check.status}[/{colour}]", check.detail)
    console.print(table)
    console.print()

    remedies = [c for c in report.checks if c.status in {"FAIL", "WARN"} and c.remedy]
    if remedies:
        console.print("[bold]How to fix[/bold]")
        for check in remedies:
            console.print(f"  [bold]{check.name}[/bold]: {check.remedy}")
        console.print()

    if report.ready:
        console.print("[bold green]Environment ready.[/bold green]")
    else:
        console.print("[bold red]Environment not ready.[/bold red] See remedies above.")
    console.print()
    raise typer.Exit(code=0 if report.ready else 1)


@app.command("inspect-model")
def inspect_model(backend: BackendOption = None, device: DeviceOption = None) -> None:
    """Load the model and print its provenance and configuration."""
    from blackmirror.inference.service import InferenceService

    settings = _settings(backend, device)
    try:
        service = InferenceService(settings)
        metadata = service.get_model_metadata()
    except BlackMirrorError as exc:
        _fail(exc)
        return

    console.print()
    console.print("[bold]Model[/bold]")
    console.print("=" * 60)
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold", width=22)
    table.add_column("Value", overflow="fold")
    table.add_row("Name", metadata.name)
    table.add_row("Backend", metadata.backend)
    table.add_row("Model id", metadata.model_id)
    table.add_row("Checkpoint", metadata.checkpoint or "-")
    table.add_row("Version", metadata.version or "-")
    table.add_row("License", metadata.license or "-")
    table.add_row("Source", metadata.source or "-")
    table.add_row("Modalities", ", ".join(metadata.modalities) or "-")
    table.add_row("Device", metadata.loaded_device)
    table.add_row("dtype", metadata.dtype)
    table.add_row(
        "Parameters",
        f"{metadata.parameter_count:,}" if metadata.parameter_count else "-",
    )
    table.add_row("Subject conditioning", metadata.subject_conditioning or "-")
    console.print(table)

    if metadata.feature_extractors:
        console.print()
        console.print("[bold]Frozen feature extractors[/bold]")
        for modality, extractor in sorted(metadata.feature_extractors.items()):
            console.print(f"  {modality:<12} {extractor}")

    if metadata.extra:
        console.print()
        console.print("[bold]Configuration[/bold]")
        for key, value in sorted(metadata.extra.items()):
            console.print(f"  {key:<24} {value}")

    if metadata.is_synthetic:
        console.print()
        console.print(
            "[bold yellow]WARNING:[/bold yellow] this backend produces SYNTHETIC data, "
            "not brain predictions."
        )
    console.print()


@app.command("predict")
def predict(
    content: Annotated[Path, typer.Argument(help="Path to the stimulus file.")],
    backend: BackendOption = None,
    device: DeviceOption = None,
    reuse_cache: Annotated[
        bool, typer.Option("--reuse-cache", help="Return an identical completed run if one exists.")
    ] = False,
    plot: Annotated[
        bool, typer.Option("--plot/--no-plot", help="Write the sanity diagnostic plot.")
    ] = True,
) -> None:
    """Run inference on a stimulus and persist a complete run directory."""
    from blackmirror.inference.service import InferenceService

    settings = _settings(backend, device)
    if reuse_cache:
        settings = settings.model_copy(update={"reuse_cached_runs": True})

    try:
        service = InferenceService(settings)
        result = service.predict(content)
    except BlackMirrorError as exc:
        _fail(exc)
        return

    if plot:
        from blackmirror.diagnostics import write_sanity_plot

        write_sanity_plot(result, settings)

    _print_result(result)


@app.command("inspect-run")
def inspect_run(
    run_id: Annotated[
        str | None, typer.Argument(help="Run id. Defaults to the most recent run.")
    ] = None,
) -> None:
    """Print the summary for a stored run."""
    settings = _settings()
    store = ArtifactStore(settings.artifact_dir)

    if run_id is None:
        runs = store.list_runs()
        if not runs:
            console.print(f"[yellow]No runs found under {store.runs_dir}[/yellow]")
            raise typer.Exit(code=1)
        run_id = runs[0]

    try:
        result = store.read_manifest(run_id)
    except BlackMirrorError as exc:
        _fail(exc)
        return
    _print_result(result)


@app.command("list-runs")
def list_runs(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum runs to list.")] = 20,
) -> None:
    """List stored runs, newest first."""
    settings = _settings()
    store = ArtifactStore(settings.artifact_dir)
    runs = store.list_runs()[:limit]

    if not runs:
        console.print(f"[yellow]No runs found under {store.runs_dir}[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Run id")
    table.add_column("Stimulus")
    table.add_column("Backend")
    table.add_column("Shape")
    table.add_column("Status")
    for rid in runs:
        try:
            result = store.read_manifest(rid)
        except BlackMirrorError:
            table.add_row(rid, "[red]unreadable manifest[/red]", "-", "-", "-")
            continue
        label = result.model.backend + (" (synthetic)" if result.model.is_synthetic else "")
        table.add_row(
            rid,
            result.stimulus.filename,
            label,
            str(list(result.prediction.shape)),
            result.status,
        )
    console.print(table)


@app.command("benchmark")
def benchmark(
    run_id: Annotated[
        str | None, typer.Argument(help="Run id. Defaults to the most recent run.")
    ] = None,
) -> None:
    """Write a performance baseline record for a stored run."""
    from blackmirror.benchmarks import build_benchmark, write_benchmark

    settings = _settings()
    store = ArtifactStore(settings.artifact_dir)

    if run_id is None:
        runs = store.list_runs()
        if not runs:
            console.print(f"[yellow]No runs found under {store.runs_dir}[/yellow]")
            raise typer.Exit(code=1)
        run_id = runs[0]

    try:
        result = store.read_manifest(run_id)
        path = write_benchmark(result, settings.benchmarks_dir)
    except BlackMirrorError as exc:
        _fail(exc)
        return

    console.print_json(json.dumps(build_benchmark(result)))
    console.print(f"\n[bold green]Wrote[/bold green] {path}\n")


@app.command("analyze")
def analyze(
    run_id: Annotated[
        str | None, typer.Argument(help="Run id. Defaults to the most recent run.")
    ] = None,
) -> None:
    """Derive and persist Phase 3 neural analytics for a completed run."""
    from blackmirror.analytics.pipeline import analyze_run

    settings = _settings()
    store = ArtifactStore(settings.artifact_dir)
    if run_id is None:
        runs = store.list_runs()
        if not runs:
            console.print(f"[yellow]No runs found under {store.runs_dir}[/yellow]")
            raise typer.Exit(code=1)
        run_id = runs[0]
    try:
        result = analyze_run(settings.artifact_dir, run_id)
    except (BlackMirrorError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(result.model_dump_json())


@app.command("analyze-content")
def analyze_content_command(
    run_id: Annotated[
        str | None, typer.Argument(help="Run id. Defaults to the most recent run.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Recompute even when a valid cached result exists.")
    ] = False,
    no_visual: Annotated[
        bool, typer.Option("--no-visual", help="Skip CLIP keyframe description.")
    ] = False,
    no_ocr: Annotated[
        bool, typer.Option("--no-ocr", help="Skip on-screen text detection.")
    ] = False,
) -> None:
    """Derive and persist Phase 4 multimodal content intelligence for a run."""
    from blackmirror.content.pipeline import ContentAnalysisConfig, analyze_content

    settings = _settings()
    store = ArtifactStore(settings.artifact_dir)
    if run_id is None:
        runs = store.list_runs()
        if not runs:
            console.print(f"[yellow]No runs found under {store.runs_dir}[/yellow]")
            raise typer.Exit(code=1)
        run_id = runs[0]

    config = ContentAnalysisConfig(
        enable_visual_semantics=not no_visual,
        enable_ocr=not no_ocr,
    )
    try:
        result = analyze_content(
            run_id, artifact_root=settings.artifact_dir, config=config, force=force
        )
    except (BlackMirrorError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc

    _print_content_summary(result)


def _print_content_summary(result: object) -> None:
    """Human-readable summary of a content analysis."""
    metrics = result.metrics  # type: ignore[attr-defined]
    metadata = result.metadata  # type: ignore[attr-defined]

    console.print()
    console.print("[bold]Content Intelligence[/bold]")
    console.print("=" * 60)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold", width=26)
    table.add_column("Value", overflow="fold")
    rows: list[tuple[str, str]] = [
        ("Stimulus", result.stimulus_filename),  # type: ignore[attr-defined]
        ("Duration", f"{metrics.duration_seconds:.2f} s"),
        ("Modalities", ", ".join(result.modalities_analysed) or "-"),  # type: ignore[attr-defined]
        ("Shots", str(metrics.shot_count)),
        ("Mean shot duration", _seconds(metrics.mean_shot_duration)),
        ("Cuts / minute", _number(metrics.cuts_per_minute)),
        ("Scenes", str(metrics.scene_count)),
        ("Speech", _percent(metrics.speech_fraction)),
        ("Music", _percent(metrics.music_fraction)),
        ("Silence", _percent(metrics.silence_fraction)),
        ("Words", f"{metrics.word_count} ({_number(metrics.words_per_minute)} wpm)"),
        ("Text overlays", str(metrics.text_overlay_count)),
        ("Calls to action", str(metrics.cta_count)),
        ("Content events", str(metrics.content_event_count)),
        ("Neural associations", str(len(result.associations))),  # type: ignore[attr-defined]
        ("Analysis version", metadata.analysis_version),
        ("Runtime", f"{metadata.stage_seconds.get('total', 0.0):.1f} s"),
    ]
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)

    if metadata.warnings:
        console.print()
        console.print("[bold yellow]Warnings[/bold yellow]")
        for warning in metadata.warnings:
            console.print(f"  - {warning}")

    console.print()
    console.print(f"[dim]{metadata.interpretation_notice}[/dim]")
    console.print()


def _seconds(value: float | None) -> str:
    return f"{value:.2f} s" if value is not None else "-"


def _number(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"


def _percent(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "-"


@app.command("clear-content-cache")
def clear_content_cache() -> None:
    """Delete regenerable content-analysis intermediates.

    Decoded audio and OCR frames are cached under the model cache directory,
    keyed by stimulus hash, so they are shared across runs and never duplicated
    into the artifact store. Everything here can be regenerated from the source
    media, so clearing it costs only time.
    """
    from blackmirror.content.workspace import clear_all

    settings = _settings()
    reclaimed = clear_all(settings.model_cache_dir)
    console.print(
        f"Reclaimed [bold]{reclaimed / 1e6:.1f} MB[/bold] from "
        f"{settings.model_cache_dir / 'content_workspace'}"
    )


@app.command("compare")
def compare(
    reference_run_id: Annotated[str, typer.Argument(help="Reference (A) run id.")],
    candidate_run_ids: Annotated[
        list[str], typer.Argument(help="One or more candidate (B/N) run ids.")
    ],
    tolerance_seconds: Annotated[
        float,
        typer.Option(
            "--timeline-tolerance",
            min=0.0,
            help="Maximum timestamp difference for exact observed-sample matching.",
        ),
    ] = 1e-6,
    alignment_method: Annotated[
        str,
        typer.Option(
            "--alignment",
            help="exact_observed_intersection or opt-in linear_candidate_to_reference.",
        ),
    ] = "exact_observed_intersection",
    max_interpolation_gap_seconds: Annotated[
        float,
        typer.Option(
            "--max-interpolation-gap", min=0.001, help="No interpolation across larger gaps."
        ),
    ] = 10.0,
) -> None:
    """Create a controlled candidate-minus-reference neural comparison."""
    from blackmirror.comparison.pipeline import compare_runs

    settings = _settings()
    try:
        result = compare_runs(
            settings.artifact_dir,
            reference_run_id,
            tuple(candidate_run_ids),
            tolerance_seconds=tolerance_seconds,
            alignment_method=alignment_method,
            max_interpolation_gap_seconds=max_interpolation_gap_seconds,
        )
    except (BlackMirrorError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(result.model_dump_json())


@app.command("bind-resimulation")
def bind_resimulation(
    resimulation_id: Annotated[str, typer.Argument(help="Name for this measured pass.")],
    experiment_id: Annotated[str, typer.Option("--experiment", help="Scored experiment id.")],
    optimization_key: Annotated[str, typer.Option("--optimization", help="Phase 7 request key.")],
    proposed_variant_id: Annotated[str, typer.Option("--candidate", help="Approved spec id.")],
    variant_media: Annotated[
        Path,
        typer.Option("--media", exists=True, dir_okay=False, help="The variant you built."),
    ],
    source_media: Annotated[
        Path | None,
        typer.Option(
            "--source",
            exists=True,
            dir_okay=False,
            help="Defaults to the parent run's stimulus.",
        ),
    ] = None,
    adapter_id: Annotated[str, typer.Option("--adapter")] = "user-supplied",
    max_attempts: Annotated[int, typer.Option("--max-attempts", min=1, max=20)] = 3,
    output: Annotated[
        Path | None, typer.Option("--out", help="Write the request JSON here instead of stdout.")
    ] = None,
) -> None:
    """Derive a valid Phase 8 request from what Phase 6 and Phase 7 already stored.

    The objective set, both of its hashes, the per-hypothesis direction map and
    the media content hashes are all read from those records rather than typed,
    which is what keeps a re-simulation measuring the objective that was
    actually approved.
    """
    from blackmirror.resimulation.request_builder import RequestBuildError, build_request

    settings = _settings()
    try:
        request = build_request(
            settings.artifact_dir,
            resimulation_id=resimulation_id,
            experiment_id=experiment_id,
            optimization_request_key=optimization_key,
            proposed_variant_id=proposed_variant_id,
            variant_media=variant_media,
            source_media=source_media,
            adapter_id=adapter_id,
            max_attempts=max_attempts,
        )
    except (RequestBuildError, BlackMirrorError, FileNotFoundError, OSError, ValueError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    payload = request.model_dump_json(indent=2)
    if output is not None:
        output.write_text(payload, encoding="utf-8")
        console.print(f"Wrote {output}")
    else:
        console.print_json(payload)


@app.command("resimulate")
def resimulate(
    request_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Phase 8 request JSON.")
    ],
) -> None:
    """Run or idempotently resume one approved, user-bound Phase 8 candidate.

    Build the request with `bind-resimulation` first. This runs TRIBE inference
    and the full analytics, content, scoring and comparison chain, so it takes
    hours, not seconds.
    """
    from blackmirror.inference.service import InferenceService
    from blackmirror.resimulation.backend import ExistingPipelineBackend
    from blackmirror.resimulation.orchestrator import ResimulationOrchestrator
    from blackmirror.resimulation.schemas import ResimulationRequest

    settings = _settings()
    try:
        request = ResimulationRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
        backend = ExistingPipelineBackend(settings.artifact_dir, InferenceService(settings))
        result = ResimulationOrchestrator(settings.artifact_dir, backend).start(request)
    except (BlackMirrorError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(result.model_dump_json())


@app.command("resume-resimulation")
def resume_resimulation(resimulation_id: Annotated[str, typer.Argument()]) -> None:
    """Verify all upstream hashes and resume after the last durable Phase 8 stage."""
    from blackmirror.inference.service import InferenceService
    from blackmirror.resimulation.backend import ExistingPipelineBackend
    from blackmirror.resimulation.orchestrator import ResimulationOrchestrator

    settings = _settings()
    try:
        backend = ExistingPipelineBackend(settings.artifact_dir, InferenceService(settings))
        result = ResimulationOrchestrator(settings.artifact_dir, backend).resume(resimulation_id)
    except (BlackMirrorError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(result.model_dump_json())


@app.command("stop-resimulation")
def stop_resimulation(
    resimulation_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason")] = "cancelled",
) -> None:
    """Ask a running pass to stop at its next stage boundary.

    This is a durable request, not a kill. The worker checks for it between
    stages so that a partially written artifact is never left behind, which
    means a stop takes effect when the current stage finishes.
    """
    from blackmirror.resimulation.schemas import StopReason
    from blackmirror.resimulation.storage import ResimulationStore

    settings = _settings()
    try:
        stop_reason = StopReason(reason)
        if stop_reason not in {
            StopReason.CANCELLED,
            StopReason.TIMEOUT,
            StopReason.RESOURCE_LIMIT,
        }:
            raise ValueError(
                "stop reason must be one of: cancelled, timeout, resource_limit"
            )
        store = ResimulationStore(settings.artifact_dir)
        current = store.read(resimulation_id)
        if current.status.value == "completed":
            raise ValueError("a completed resimulation cannot be stopped retroactively")
        store.request_stop(resimulation_id, stop_reason)
    except (BlackMirrorError, FileNotFoundError, OSError, ValueError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Stop requested for {resimulation_id!r}; it takes effect at the next stage boundary."
    )


@app.command("list-resimulations")
def list_resimulations(
    experiment_id: Annotated[str | None, typer.Option("--experiment")] = None,
) -> None:
    """Show every recorded pass, its stage, and its measured outcome."""
    from blackmirror.api.resimulation_loader import ResimulationLoader

    settings = _settings()
    states = ResimulationLoader(settings.artifact_dir).list(experiment_id)
    if not states:
        console.print("No re-simulations recorded.")
        return
    table = Table(title="Phase 8 re-simulations")
    for column in ("id", "experiment", "stage", "status", "outcome"):
        table.add_column(column)
    for state in states:
        outcomes = ", ".join(
            f"{delta.objective_id}:{delta.outcome.value}" for delta in state.objective_deltas
        )
        table.add_row(
            state.request.resimulation_id,
            state.request.experiment_id,
            state.current_stage.name,
            state.status.value,
            outcomes or "not measured",
        )
    console.print(table)


@app.command("list-searches")
def list_searches() -> None:
    """Show every recorded Phase 9 search and what it observed."""
    from blackmirror.api.search_loader import SearchLoader

    settings = _settings()
    loader = SearchLoader(settings.artifact_dir)
    ids = loader.ids()
    if not ids:
        console.print("No searches recorded.")
        return
    table = Table(title="Phase 9 searches")
    for column in ("id", "strategy", "status", "best", "improvement", "evals"):
        table.add_column(column)
    for search_id in ids:
        summary = loader.summary(search_id)
        raw_metrics = summary.get("metrics")
        metrics: dict[str, object] = raw_metrics if isinstance(raw_metrics, dict) else {}
        best = metrics.get("best_fitness")
        gain = metrics.get("absolute_improvement")
        resolvable = metrics.get("improvement_is_resolvable")
        # A gain inside the noise floor is marked, not printed as if it counted.
        gain_text = "—" if not isinstance(gain, int | float) else f"{gain:+.4f}"
        if isinstance(gain, int | float) and resolvable is False:
            gain_text += " (inside noise)"
        table.add_row(
            search_id,
            str(summary.get("strategy") or "—"),
            "running" if summary.get("running") else str(summary.get("status") or "—"),
            "—" if not isinstance(best, int | float) else f"{best:.4f}",
            gain_text,
            str(metrics.get("evaluations", 0)),
        )
    console.print(table)


@app.command("search-report")
def search_report(
    search_id: Annotated[str, typer.Argument(help="Search id.")],
) -> None:
    """Print a finished search's report, including what it does not establish."""
    from blackmirror.api.search_loader import SearchLoader

    settings = _settings()
    loader = SearchLoader(settings.artifact_dir)
    try:
        report = loader.report(search_id)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    if report is None:
        console.print(
            f"[yellow]Search {search_id!r} has not finished, so it has no report "
            f"yet.[/yellow]"
        )
        raise typer.Exit(code=1)
    console.print_json(json.dumps(report, default=str))


@app.command("search-trajectory")
def search_trajectory(
    search_id: Annotated[str, typer.Argument(help="Search id.")],
) -> None:
    """Show best-so-far against evaluation count, the sample-efficiency curve."""
    from blackmirror.api.search_loader import SearchLoader

    settings = _settings()
    loader = SearchLoader(settings.artifact_dir)
    try:
        points = loader.trajectory(search_id)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    if not points:
        console.print("Nothing evaluated yet.")
        return
    table = Table(title=f"Trajectory — {search_id}")
    for column in ("eval", "gen", "candidate", "fitness", "best so far", ""):
        table.add_column(column)
    for point in points:
        fitness = point.get("fitness")
        best_so_far = point.get("best_fitness")
        table.add_row(
            str(point.get("evaluation_number")),
            str(point.get("generation")),
            str(point.get("candidate_id")),
            "—" if not isinstance(fitness, int | float) else f"{fitness:.4f}",
            "—" if not isinstance(best_so_far, int | float) else f"{best_so_far:.4f}",
            "new best" if point.get("is_new_best") else "",
        )
    console.print(table)


@app.command("export-mesh")
def export_mesh(
    space: Annotated[str, typer.Option("--space", help="Surface template.")] = "fsaverage5",
    force: Annotated[bool, typer.Option("--force", help="Re-export if already present.")] = False,
) -> None:
    """Export the cortical surface mesh predictions map onto (vertices, faces, mapping)."""
    from blackmirror.cortical.mesh import export_surface_mesh

    settings = _settings()
    try:
        manifest = export_surface_mesh(space, settings.mesh_dir, force=force)
    except BlackMirrorError as exc:
        _fail(exc)
        return

    console.print()
    console.print(f"[bold green]Exported {space} mesh[/bold green] -> {settings.mesh_dir / space}")
    for name, info in manifest["arrays"].items():
        console.print(f"  {name:<32} shape={info['shape']} dtype={info['dtype']}")
    console.print()


def _print_result(result: PredictionResult) -> None:
    console.print()
    console.print("=" * 60)
    console.print("[bold]BLACKMIRROR — PREDICTED CORTICAL RESPONSE[/bold]")
    console.print("=" * 60)

    console.print()
    console.print("[bold]Stimulus[/bold]")
    console.print(f"  File:      {result.stimulus.filename}")
    console.print(f"  Type:      {result.stimulus.media_type.value}")
    duration = result.stimulus.duration_seconds
    console.print(f"  Duration:  {f'{duration:.2f}s' if duration else 'unknown'}")
    console.print(f"  SHA-256:   {result.stimulus.sha256}")

    console.print()
    console.print("[bold]Model[/bold]")
    console.print(f"  Model:     {result.model.name}")
    console.print(f"  Device:    {result.model.loaded_device}")
    console.print(f"  dtype:     {result.model.dtype}")
    if result.model.is_synthetic:
        console.print("  [bold yellow]SYNTHETIC OUTPUT — not a brain prediction[/bold yellow]")

    console.print()
    console.print("[bold]Prediction[/bold]")
    for line in result.summary_lines():
        console.print(f"  {line}")

    if result.validation.warnings:
        console.print()
        console.print("[bold yellow]Validation warnings[/bold yellow]")
        for warning in result.validation.warnings:
            console.print(f"  - {warning}")

    console.print()
    console.print("[bold]Performance[/bold]")
    p = result.performance
    console.print(f"  Preprocessing:  {p.preprocessing_seconds:.2f}s")
    console.print(f"  Model load:     {p.model_load_seconds:.2f}s")
    console.print(f"  Inference:      {p.inference_seconds:.2f}s")
    console.print(f"  Postprocessing: {p.postprocessing_seconds:.2f}s")
    console.print(f"  Validation:     {p.validation_seconds:.2f}s")
    console.print(f"  Persistence:    {p.persistence_seconds:.2f}s")
    console.print(f"  Total:          {p.total_seconds:.2f}s")
    if p.realtime_factor is not None:
        console.print(f"  Realtime factor: {p.realtime_factor:.1f}x content duration")

    console.print()
    console.print("[bold]Artifacts[/bold]")
    console.print(f"  {result.artifacts.run_dir}")

    console.print()
    console.print(f"[dim]{result.interpretation_notice}[/dim]")
    console.print()
    console.print("[bold green]SUCCESS[/bold green]")
    console.print("=" * 60)
    console.print()


def main() -> None:
    try:
        app()
    except BlackMirrorError as exc:  # safety net for anything raised outside a command
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

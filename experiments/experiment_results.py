"""
Experiment result capture — dataclass and persistence utilities.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ExperimentResult:
    """Structured result for a single experiment run."""
    experiment_id: str
    phase: int
    model_name: str
    agent_config: str               # "commander-only" or "commander-subagents"
    prompt: str
    model_settings: dict
    output: str                     # final return value from commander.query()
    trace: list                     # list of ReasoningStep dicts
    timestamp: str                  # ISO format
    duration_seconds: float
    observations: str = ""          # filled post-run by researcher
    evaluation: dict = field(default_factory=lambda: {
        "output_quality": None,
        "consistency": None,
        "reasoning_depth": None,
        "differences": None,
    })


def save_result(result: ExperimentResult, logbook, base_dir: str = "experiments/results") -> Path:
    """
    Persist an experiment result to disk.

    Creates a folder per run containing:
    - result.json  — full structured result
    - trace.txt    — plain-text logbook export
    - trace.html   — HTML logbook export

    Returns the path of the created folder.
    """
    base = Path(base_dir)
    run_dir = base / result.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write structured JSON result
    (run_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Write logbook trace exports
    (run_dir / "trace.txt").write_text(
        logbook.console.export_text(),
        encoding="utf-8",
    )
    (run_dir / "trace.html").write_text(
        logbook.console.export_html(inline_styles=True),
        encoding="utf-8",
    )

    return run_dir

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from core.Agent import ReasoningStep
from typing import Any, Dict, List

class Logbook:
    def __init__(self, debug: bool = False):
        self.console = Console(record=True)
        self.width = 90
        self.debug = debug
        self.trace = []

    def reset(self) -> None:
        self.trace = []
        self.turn = 0

    def record_step(self, turn: int, reasoning: ReasoningStep, stack_depth: int) -> None:
       
        step = {
            "turn": turn,
            "reasoning": reasoning,
            "stack_depth": stack_depth,
        }

        self.trace.append(step)
        self._log_step(reasoning, stack_depth, turn)

    def _log_step(self, step: ReasoningStep, depth: int, turn: int) -> None:
        
        title = f"TURN {turn} | AGENT: {step.agent} | DEPTH: {depth}"
        table = Table(show_header=False, box=None, padding=(0, 1))
        
        # Input section
        table.add_row(Text("INPUT", style="bold cyan"), "")
        table.add_row("", step.prompt)

        table.add_row("", "")  # spacer

        # Output section
        table.add_row(Text("OUTPUT", style="bold green"), "")
        table.add_row("Thought", step.thought)
        table.add_row("Action", step.action)
        table.add_row("Target", str(step.target))
        table.add_row("Content", step.content)
        table.add_row(
            "Parse error",
            "[red]True[/red]" if step.parse_error else "[green]False[/green]",
        )

        if step.parse_error:
            table.add_row("", "")
            table.add_row(Text("RAW MODEL OUTPUT", style="bold yellow"), "")
            table.add_row("", step.raw or "")

        self.console.print(
            Panel(
                table,
                title=title,
                border_style="red" if step.parse_error else "green",
                width=self.width,
            )
        )
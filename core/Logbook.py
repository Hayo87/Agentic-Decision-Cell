from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing import List
from dataclasses import dataclass

@dataclass
class ReasoningStep:
    agent: str
    prompt: str
    thought: str
    action: str
    line: str
    parse_error: bool
    raw: str


class Logbook:
    def __init__(self, debug: bool = False):
        self.console = Console(record=True, markup=False)
        self.width = 90
        self.debug = debug
        self.trace: List[ReasoningStep] = []

    def record_step(self, step: ReasoningStep) -> None:
       
        self.trace.append(step)
        self._log_step(step)

    def _log_step(self, step: ReasoningStep) -> None:
        
        title = f"| AGENT: {step.agent} |"
        table = Table(show_header=False, box=None, padding=(0, 1))
        
        # Input section
        table.add_row(Text("INPUT", style="bold cyan"), "")
        table.add_row("", step.prompt)

        table.add_row("", "")

        # Output section
        table.add_row(Text("OUTPUT", style="bold green"), "")
        table.add_row("Thought", step.thought)
        table.add_row("Action", step.action)
        table.add_row("Action line", str(step.line))
        table.add_row("Parse error", str(step.parse_error))

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
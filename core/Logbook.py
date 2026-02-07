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

@dataclass
class ReviewStep:
    agent: str
    draft_decision: str
    review_questions: str
    human_feedback: str
    verdict: str          # "approved", "changes", or "rejected"
    iteration: int        # review round number


class Logbook:
    def __init__(self, debug: bool = False):
        self.console = Console(record=True, markup=False)
        self.width = 90
        self.debug = debug
        self.trace: List[ReasoningStep] = []

    def record_step(self, step: ReasoningStep) -> None:
        self.trace.append(step)
        self._log_step(step)

    def record_review(self, review: ReviewStep) -> None:
        self.trace.append(review)
        self._log_review(review)

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

    def _log_review(self, review: ReviewStep) -> None:
        title = f"| HUMAN REVIEW #{review.iteration} — {review.agent} |"
        table = Table(show_header=False, box=None, padding=(0, 1))

        table.add_row(Text("DRAFT DECISION", style="bold cyan"), "")
        table.add_row("", review.draft_decision)
        table.add_row("", "")

        table.add_row(Text("REVIEW QUESTIONS", style="bold yellow"), "")
        table.add_row("", review.review_questions)
        table.add_row("", "")

        table.add_row(Text("HUMAN FEEDBACK", style="bold magenta"), "")
        table.add_row("", review.human_feedback)
        table.add_row("", "")

        table.add_row(Text("VERDICT", style="bold white"), review.verdict.upper())

        if review.verdict == "approved":
            border = "green"
        elif review.verdict == "changes":
            border = "yellow"
        else:
            border = "red"

        self.console.print(
            Panel(
                table,
                title=title,
                border_style=border,
                width=self.width,
            )
        )
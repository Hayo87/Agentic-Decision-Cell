import re
import warnings
from core.Bots import load_chat_agent
from core.Logbook import Logbook, ReasoningStep, ReviewStep

# Fallback error messages used when a YAML config is missing the errors section
_DEFAULT_ERRORS = {
    "parse_autocorrect": (
        "FORMAT ERROR. Rewrite using EXACTLY:\n"
        "Thought: <one sentence>\n"
        "Action: <valid_action>\n"
        "Action Line: <plain text>\n\n"
        "Your previous response:\n{raw}"
    ),
    "unknown_tool": 'TOOL ERROR: "{action}" not recognized.',
    "no_answer": "TIMEOUT: agent did not reach Action: finish.",
}

class Agent:
    """
    Class for a LLM agent, load YAML config, applies system prompt, and queries an LLM backend returning a parsed action.
    """

    def __init__(self, config: dict, logbook: Logbook, tools: list):
        """
        Load the agent config, set metadata, and initialize the LLM client.
        """
        # Store metadata
        self.config = config
        self.logbook = logbook
        self.tools = {t.name: t for t in tools}
        self.name = config["name"]
        self.role = config["role"]

        # Merge provided errors over defaults so missing keys never crash
        self.errors = {**_DEFAULT_ERRORS, **config.get("errors", {})}

        # Warn early if the YAML config is missing any expected error key
        missing = [k for k in _DEFAULT_ERRORS if k not in config.get("errors", {})]
        if missing:
            warnings.warn(
                f"Agent '{self.name}': YAML config missing error keys {missing}. "
                f"Using built-in defaults.",
                stacklevel=2,
            )

        # LLM inference function (forward optional model_settings from config)
        self.infer = load_chat_agent(
            config["provider"], config["model"], config["system_prompt"],
            model_settings=config.get("model_settings"),
        )

    def query(self, question: str, max_steps: int = 8,
              require_human_review: bool = False) -> str:
        q = question
        prev_parse_error = False
        human_approved = False
        review_iteration = 0

        for _ in range(max_steps):
            raw = self.infer(q)
            step = self.parse(q, raw)

            self.logbook.record_step(step)

            if step.parse_error:
                if prev_parse_error:
                    break

                prev_parse_error = True
                q = self.errors["parse_autocorrect"].format(raw=raw)
                continue
            else:
                prev_parse_error = False

            action_lower = step.action.strip().lower()

            # --- HITL gate: intercept request_human_review ---
            if action_lower == "request_human_review":
                if not require_human_review:
                    observation = self.errors["unknown_tool"].format(
                        action=step.action)
                    q = f"Observation: {observation}"
                    continue

                review_iteration += 1
                draft = step.line

                review_tool = self.tools.get("request_human_review")
                if not review_tool:
                    q = ("Observation: ERROR — request_human_review tool "
                         "not available.")
                    continue

                human_response = str(review_tool.invoke(draft))
                verdict = self._parse_review_verdict(human_response)

                review_questions = (
                    "1. Is this decision substantively correct?\n"
                    "2. Which assumptions require adjustment?\n"
                    "3. Should this direction be approved, revised, "
                    "or rejected?"
                )
                self.logbook.record_review(ReviewStep(
                    agent=self.name,
                    draft_decision=draft,
                    review_questions=review_questions,
                    human_feedback=human_response,
                    verdict=verdict,
                    iteration=review_iteration,
                ))

                if verdict == "approved":
                    human_approved = True
                    q = (
                        "Observation: HUMAN REVIEW APPROVED. "
                        "Your draft has been approved. "
                        "Now issue Action: final_decision with your "
                        "final decision."
                    )
                elif verdict == "changes":
                    feedback = (human_response.split(":", 1)[1].strip()
                                if ":" in human_response
                                else human_response)
                    q = (
                        f"Observation: HUMAN REVIEW — CHANGES REQUESTED. "
                        f"Feedback: {feedback}. "
                        f"Revise your draft and submit again using "
                        f"Action: request_human_review."
                    )
                else:  # rejected
                    feedback = (human_response.split(":", 1)[1].strip()
                                if ":" in human_response
                                else human_response)
                    q = (
                        f"Observation: HUMAN REVIEW — REJECTED. "
                        f"Reason: {feedback}. "
                        f"Propose an alternative CoA and submit using "
                        f"Action: request_human_review."
                    )
                continue

            # --- Governance guard on finish/final_decision ---
            if action_lower in ("finish", "final_decision"):
                if require_human_review and not human_approved:
                    q = (
                        "Observation: GOVERNANCE VIOLATION — You cannot "
                        "issue a final decision without human approval. "
                        "You MUST first use Action: request_human_review "
                        "to submit your draft decision for human review."
                    )
                    continue
                return f"{step.action}: {step.line}"

            tool = self.tools.get(step.action)
            if not tool:
                observation = self.errors["unknown_tool"].format(
                    action=step.action)
            else:
                observation = str(tool.invoke(step.line))

            q = f"Observation: {observation}"

        return self.errors["no_answer"]

    def parse(self, question: str, response: str, ) -> ReasoningStep:
        
        def pick(key: str) -> str:
            m = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.*?)\s*$", response)
            return m.group(1).strip() if m else ""

        thought = pick("Thought")
        action  = pick("Action")
        line    = pick("Action Line")
    
        parse_error = not (thought and action and line)

        return ReasoningStep(
            agent=self.name,
            prompt=question,
            thought=thought or "Missing Thought",
            action=action or "Missing Action",
            line=line or "Missing Action Line",
            parse_error=parse_error,
            raw=response,
        )
    
    def _parse_review_verdict(self, human_response: str) -> str:
        """Parse the human review response into a verdict category."""
        r = human_response.strip().lower()
        if r.startswith("approved") or r == "approve":
            return "approved"
        elif r.startswith("changes") or r.startswith("revise"):
            return "changes"
        elif r.startswith("rejected") or r.startswith("reject"):
            return "rejected"
        # Unrecognized input treated as change request (safe fallback)
        return "changes"

    def reset(self):
        self.infer = load_chat_agent(
            self.config["provider"], self.config["model"], self.config["system_prompt"],
            model_settings=self.config.get("model_settings"),
        )
import re
from core.Bots import load_chat_agent
from core.Logbook import Logbook, ReasoningStep

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
        self.errors = config.get("errors", {})

        # LLM inference function
        self.infer = load_chat_agent(config["provider"], config["model"], config["system_prompt"])

    def query(self, question: str, max_steps: int = 8) -> str:
        q = question
        prev_parse_error = False

        for _ in range(max_steps):
            raw = self.infer(q)
            step = self.parse(q, raw)

            self.logbook.record_step(step)

            if step.parse_error:
                if prev_parse_error:
                    # second parse error in a row is stop
                    break

                prev_parse_error = True
                q = self.errors["parse_autocorrect"].format(raw=raw)
                continue
            else:
                prev_parse_error = False

            if step.action.strip().lower() == "finish":
                return f"{step.action}: {step.line}"

            tool = self.tools.get(step.action)
            if not tool:
                observation = self.errors["unknown_tool"].format(action= step.action)
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
    
    def reset(self):
        self.infer = load_chat_agent(self.config["provider"], self.config["model"], self.config["system_prompt"])
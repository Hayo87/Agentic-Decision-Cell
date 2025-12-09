import yaml
from core.Bots import load_chat_agent
from typing import Dict, Optional

class Agent:
    """
    Class for a LLM agent, load YAML config, applies system prompt, and queries an LLM backend returning a parsed action. 
    """

    def __init__(self, agent_name: str, provider: str, model: str, debug=False):
        """
        Load the agent YAML, set metadata, and initialize the LLM client.
        """

        # Set debug mode
        self.debug = debug

        # Store model and provider
        self.model = model
        self.provider = provider

        # Load agent YAML file
        with open(f"agents/system_prompts/{agent_name}.yaml", "r", encoding="utf-8") as f:
            agent_config = yaml.safe_load(f)

        # Set agent meta data 
        self.name = agent_config["name"]
        self.role = agent_config["role"]
        self.system_prompt = agent_config["system_prompt"]

        # LLM inference function
        self.infer = load_chat_agent(provider, model, self.system_prompt)

    def query(self, question: str) -> Dict[str, Optional[str]]:
        """
        Send a question to the LLM and return raw text output.
        """
        if self.debug:
            print("\n--- Agent Query ---")
            print(f"Agent: {self.name}")
            print(f"Question:\n{question}")

        raw_response = self.infer(question)
        response = self.parse(raw_response)

        if self.debug:
            print(f"RAW RESPONSE:\n{raw_response}")
            print(f"PARSED RESPONSE: {response}")

        return response
        
    def get_descriptor(self):
        """
        Return a short human-readable description of this agent.
        """
        return f"{self.name}: {self.role}"

    def parse(self, raw: str) -> Dict[str, Optional[str]]:
        """
        Parse the first Action[...] line from LLM output.
        """

        lines = [l.strip() for l in raw.splitlines() if l.strip()]

        action_line = None
        for line in lines:
            if line.startswith("Action["):
                action_line = line
                break

        if not action_line:
            raise ValueError(f"No Action[...] line found in:\n{raw}")

        # Remove leading "Action[" and ending "]"
        inside = action_line[len("Action[") : action_line.index("]")]

        # inside

        if ":" in inside:
            action_type, subtype = inside.split(":", 1)
            action_type = action_type.strip()
            subtype = subtype.strip()
        else:
            action_type = inside.strip()
            subtype = None

        # Get content after the closing bracket 
        after = action_line[action_line.index("]") + 1:].lstrip(":").strip()

        return {
            "type": action_type,   # e.g. "delegate" or "finish"
            "target": subtype,     # e.g. "Intel" or None
            "content": after,      # everything after ]
        }
    
    def reset(self):
        """Clear the internal LLM message history so the agent starts fresh."""
        self.infer = load_chat_agent(self.provider, self.model, self.system_prompt)
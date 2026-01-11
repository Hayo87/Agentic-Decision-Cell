import yaml
from core.Bots import load_chat_agent
from typing import Dict, Optional, Union

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

    def parse(self, raw: str) -> Dict[str, Union[str, None, bool]]:
        
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        get = lambda k: next((l.split(":", 1)[1].strip() for l in lines if l.startswith(k + ":")), "")

        raw_thought = get("Thought")
        raw_action  = get("Action")
        raw_target  = get("Target")
        raw_content = get("Content")

        parse_error = not (raw_thought and raw_action and raw_content)

        thought  = raw_thought or "Agent failed to produce proper Thought."
        action   = raw_action or "Agent failed to produce proper Action"
        content  = raw_content or "Agent failed to produce proper Content"
        target   = raw_target or None

        if isinstance(target, str) and target.lower() in {"", "none", "null"}:
            target = None

        return {
            "thought": thought,
            "type": action,
            "target": target,
            "content": content,
            "parse_error": parse_error,
            "raw": raw
        }
    
    def reset(self):
        """Clear the internal LLM message history so the agent starts fresh."""
        self.infer = load_chat_agent(self.provider, self.model, self.system_prompt)
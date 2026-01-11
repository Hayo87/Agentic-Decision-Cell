from typing import List, Dict, Any
from core.Agent import Agent
from core.Tooling import ToolHandler

class DecisionCell:
    """
    Coordinates one or more Agents for ReAct-style reasoning.
    Keeps a call stack and delegates tool/actions to an ActionHandler.
    """

    def __init__(
        self,
        agents: List[Agent],
        commander: Agent,
        tool_handler: ToolHandler,
        max_turns: int = 50,
        debug: bool = False,
    ):

        """
        Init orchestrator with agents and shared settings.
        """
        # Store settings
        self.debug = debug
        self.max_turns = max_turns
        self.tool_handler = tool_handler

        # Store agents and entry point
        self.agents = agents
        self.agents_by_name = {a.name: a for a in agents}
        self.entry_agent = commander

        # Internal state
        self.stack = []      
        self.trace = []      

    def run(self, objective: str) -> Dict[str, Any]:
        """
        Run the full reasoning loop for a given objective.

        Returns a dict like:
        {
          "result": final_answer_or_error,
          "trace":  list_of_steps_for_debug_or_logging
        }
        """
        # Reset state for a new run
        self.reset()

        # Push initial frame on stack
        self.stack.append({
            "agent": self.entry_agent,
            "question": objective,
            "observation": None,
        })

        turns = 0

        # Main reasoning loop
        while self.stack and turns < self.max_turns:
            turns += 1

            # Get current frame
            frame = self.stack[-1]
            agent = frame["agent"]                
            if frame["observation"] is not None:
                prompt = f"Observation: {frame['observation']}"
            else:
                prompt = frame["question"]
            
            # Ask agent
            response = agent.query(prompt)

            # Unpack results
            action_type = response["type"]
            target = response["target"]
            content = response["content"] 

            # Log 
            self.trace.append({
                "agent": agent.name,
                "question": frame["question"],
                "observation": frame["observation"],
                "response": response,
                "stack_depth": len(self.stack),
            })


            # If parse error inform parent
            if response.get("parse_error"):
                response["type"] = "finish"
                response["target"] = None
                response["content"] = "Agent was unable to answer the question."

            # Process the action
            match action_type:
            
                case "finish":
                    self.stack.pop()

                    if self.stack:
                        # Pass result to parent
                        self.stack[-1]["observation"] = content
                        continue
                    else:
                        # No parent, final output
                        return {
                            "result": content,
                            "trace": self.trace,
                        }

                case "delegate":
                    try:
                        target_agent = self.agents_by_name[target]
                        self.stack.append({
                            "agent": target_agent,
                            "question": content,
                            "observation": None
                        })
                        continue
                    except Exception:
                        frame["observation"] = f"[AGENT_ERROR] Unknown delegate target: {target}"
                        continue
                        
                # Tool call 
                case _:
                    # placeholder: tool or other actions
                    result = self.tool_handler.handle(action_type, target, agent.name, content)
                    frame["observation"] = result
                    continue


            if done:
                return {
                    "result": result,
                    "trace": self.trace,
                }

        # Max turns hit or stack empty without return
        return {
            "result": "FAILED_OR_INCOMPLETE",
            "trace": self.trace,
        }

    def reset(self) -> None:
        """
        Clear internal state (stack, trace, cached observations)
        so the same Orchestrator can be reused for a new run.
        """

        # Reset internal state
        self.stack = []
        self.trace = []

        # Reset agents
        for a in self.agents:
            a.reset()            
        self.entry_agent.reset()




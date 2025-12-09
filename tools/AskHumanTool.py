from tools.BaseTool import BaseTool

class AskHumanTool(BaseTool):
    def __init__(self, agent: str):
        self.agent = agent

    def supports(self, action: str, target: str, agent: str) -> bool:
        return action == "ask_human" and agent == self.agent

    def run(self, content: str) -> str:
        """
        Ask the notebook user and return their answer.
        """
        print(f"\n[{self.agent}] is asking a human:")
        print(content)
        answer = input("Your answer for the agent: ")
        return answer
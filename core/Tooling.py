from tools.BaseTool import BaseTool

class ToolRegistry:
    """Minimal registry that finds a tool by asking each tool if it supports the request."""

    def __init__(self, tools: list[BaseTool]):
        self.tools = tools

    def get(self, action: str, target: str, agent: str):
        """
        Return the first tool that declares support for (action, target, agent).
        """
        for tool in self.tools:
            if tool.supports(action, target, agent):
                return tool
        return None


class ToolHandler:
    def __init__(self, registry: ToolRegistry):
        """
        Initialize the handler with a ToolRegistry.
        The registry is used to look up tools at runtime.
        """
        self.registry = registry

    def handle(self, action: str, target: str, agent: str, content: str) -> str:
        """
        Handle a tool call produced by the agent.

        Parameters:
            action  - the requested operation (e.g. "search")
            target  - the tool name or tool domain (e.g. "knowledge_base")
            agent   - the calling agent/context if needed
            content - the input payload for the tool

        Returns:
            A string observation to feed back into the agent loop.
        """
        
        # Find the correct tool
        tool = self.registry.get(action, target, agent)
        if tool is None:
            return (
                f"ERROR: Unknown tool '{action}:{target}'. "
                "This tool is not available for your role. "
                "Do NOT try this tool again. If you still require information, "
                "use Action[ask_human] instead."
            )

        # Run tool and return result
        return tool.run(content)

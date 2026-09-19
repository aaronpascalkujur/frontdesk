from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Base class for all Frontdesk tools.
    Every tool must implement name, description, parameters, and execute.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool's unique identifier (e.g., 'get_time')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a short description for the LLM to understand what this tool does."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        Return a JSON Schema dictionary defining the tool's input parameters.
        Example:
        {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
        """

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Execute the tool with the given arguments.
        Args:
            **kwargs: Arguments defined in the parameters schema
        Returns:
            str: Result of the tool execution
        """

"""
Example Agent Implementations for PromptCraft-Hybrid.

This module contains example implementations of the PromptCraft agent framework,
demonstrating best practices for agent development and showcasing the capabilities
of the base framework.

Available Examples:
- TextProcessorAgent: Text processing and analysis agent
- (More examples can be added as needed)

These examples serve as:
- Reference implementations for new agent development
- Testing fixtures for the agent framework
- Documentation of framework capabilities
- Integration test subjects

Usage:
    ```python
    from src.agents.examples import TextProcessorAgent
    from src.agents.registry import agent_registry

    # Agent is automatically registered via decorator
    agent = agent_registry.get_agent("text_processor", {"agent_id": "text_processor"})
    ```
"""

from .text_processor_agent import TextProcessorAgent

__all__ = ["TextProcessorAgent"]

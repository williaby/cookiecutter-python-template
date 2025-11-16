"""
Core Business Logic Module for PromptCraft-Hybrid.

This module contains the core business logic components that drive the PromptCraft
system's intelligent query processing, enhanced retrieval, and resilient execution
capabilities. It serves as the foundation for the system's cognitive architecture.

The core module provides:
- Query counseling and intelligent routing
- HyDE-enhanced retrieval processing
- Zen MCP integration with resilience patterns
- Central coordination for agent interactions

Architecture:
    The core module implements the central intelligence layer that connects
    user queries with appropriate agents and processing pipelines. It provides
    the cognitive capabilities that make PromptCraft an intelligent AI workbench.

Key Components:
    - query_counselor: Intelligent query analysis and routing
    - hyde_processor: HyDE-enhanced semantic retrieval
    - zen_mcp_error_handling: Resilient external service integration

Dependencies:
    - src.agents: For agent discovery and coordination
    - src.config: For configuration management
    - src.utils: For logging, encryption, and resilience utilities
    - External services: AI models, vector databases, MCP servers

Called by:
    - src/main.py: Primary application entry point
    - src/agents: Agent implementations for enhanced processing
    - FastAPI endpoints: Request handling and processing

Complexity: Varies by component - O(1) for routing, O(n) for processing pipelines
"""

# TODO: Import query_counselor module when implementation is complete
# TODO: Import hyde_processor module when implementation is complete
# TODO: Import zen_mcp_error_handling module (already implemented)
# TODO: Export key components for external usage
# TODO: Add module-level configuration and initialization

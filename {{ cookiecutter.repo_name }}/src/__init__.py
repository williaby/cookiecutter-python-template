"""
PromptCraft-Hybrid: Zen-Powered AI Workbench

This is the main source package for PromptCraft-Hybrid, a sophisticated AI workbench
that transforms queries into accurate, context-aware outputs through intelligent
orchestration and multi-agent collaboration.

System Architecture:
    PromptCraft implements a hybrid architecture with four progressive user journeys:

    1. Journey 1: Quick Enhancement - Basic prompt improvement
    2. Journey 2: Power Templates - Template-based prompt generation
    3. Journey 3: Light IDE Integration - Local development integration
    4. Journey 4: Full Automation - Complete execution automation

Core Components:
    agents/: Multi-agent system framework with specialized AI agents
    core/: Core business logic including query processing and HyDE retrieval
    config/: Configuration management with environment-specific settings
    security/: Security middleware, authentication, and audit logging
    utils/: Shared utilities for encryption, logging, and resilience patterns
    ui/: Gradio interface components for user interaction
    ingestion/: Knowledge processing pipeline for document ingestion
    mcp_integration/: MCP server integration for agent orchestration

Key Technologies:
    - Zen MCP Server: Real-time user interactions and agent orchestration
    - Prefect: Background workflow orchestration
    - External Qdrant: Vector database for semantic search (192.168.1.16:6333)
    - Azure AI: LLM services integration
    - FastAPI: High-performance web API framework
    - Gradio: Interactive web UI components

C.R.E.A.T.E. Framework:
    The system implements the C.R.E.A.T.E. prompt engineering methodology:
    - Context: Role, persona, background, goals
    - Request: Core task, deliverable specifications
    - Examples: Few-shot examples and demonstrations
    - Augmentations: Frameworks, evidence, reasoning prompts
    - Tone & Format: Voice, style, structural formatting
    - Evaluation: Quality checks and verification

Deployment:
    - Docker containerization with multi-stage builds
    - Ubuntu VM deployment (192.168.1.205:7860)
    - Cloudflare tunnel for secure remote access
    - External Qdrant vector database on Unraid NAS

Development Philosophy:
    1. Reuse First: Leverage existing tools and patterns
    2. Configure Don't Build: Use Zen MCP and external services
    3. Focus on Unique Value: Build only PromptCraft-specific logic

Time Complexity: N/A (package initialization)
Space Complexity: O(1) - minimal memory for package setup
"""

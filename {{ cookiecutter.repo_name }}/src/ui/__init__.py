"""
User Interface Module for PromptCraft-Hybrid.

This module provides interactive user interface components for the PromptCraft system,
primarily using Gradio to create web-based interfaces that enable users to interact
with the AI workbench across the four progressive journeys.

Key Components:
    - Gradio interface components for web-based interaction
    - Journey-specific UI elements (Quick Enhancement, Power Templates, etc.)
    - Interactive prompt composition interfaces
    - Real-time feedback and validation displays
    - Agent selection and configuration interfaces

Progressive Journey Interfaces:
    Journey 1: Quick Enhancement
        - Simple prompt input/output interface
        - Basic enhancement suggestions and improvements
        - Immediate feedback on prompt quality

    Journey 2: Power Templates
        - Template selection and customization interface
        - C.R.E.A.T.E. framework-based template builder
        - Knowledge base integration for template suggestions

    Journey 3: Light IDE Integration
        - Local development environment integration
        - File-based prompt management interface
        - Git integration for prompt versioning

    Journey 4: Full Automation
        - Complete workflow automation interface
        - Multi-agent coordination visualization
        - Execution monitoring and control panels

Architecture:
    The UI module provides a web-based interface layer that connects users
    to the PromptCraft core functionality. It implements:

    - Gradio-based interactive components
    - Real-time WebSocket communication for live updates
    - Responsive design for multiple device types
    - Integration with FastAPI backend services
    - Security-aware input validation and sanitization

Gradio Integration:
    The system uses Gradio for rapid prototyping and deployment of
    interactive machine learning interfaces:
    - Custom component development
    - Theme customization for PromptCraft branding
    - Multi-modal input support (text, files, images)
    - Real-time collaboration features

Usage:
    UI components are typically imported by the main application
    or specific interface modules:

    >>> from src.ui import journey_interfaces
    >>> from src.ui import gradio_components
    >>> from src.ui import interactive_elements

Dependencies:
    - gradio: For interactive web interface components
    - src.core: For business logic integration and query processing
    - src.agents: For agent interaction and coordination
    - src.config: For UI configuration and theming

Called by:
    - src/main.py: For UI initialization and FastAPI integration
    - Gradio application: For component rendering and event handling
    - Web browser clients: For user interaction and interface rendering

Deployment:
    - Accessible at http://192.168.1.205:7860 (development)
    - Cloudflare tunnel integration for secure remote access
    - Docker containerization for consistent deployment

Time Complexity: O(1) for component initialization, O(n) for dynamic content rendering
Space Complexity: O(k) where k is the number of active UI components and user sessions
"""

"""Command-line interface for {{ cookiecutter.project_name }}.

Provides commands for common operations and demonstrates Click best practices
with structured logging integration.
"""

import sys
from typing import Any

import click

from {{ cookiecutter.project_slug }}.core.config import settings
from {{ cookiecutter.project_slug }}.utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
@click.version_option(version="{{ cookiecutter.version }}", prog_name="{{ cookiecutter.cli_tool_name }}")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """{{ cookiecutter.project_name }} - {{ cookiecutter.project_short_description }}."""
    # Store debug flag in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    if debug:
        logger.debug("Debug mode enabled")


@cli.command()
@click.option(
    "--name",
    "-n",
    type=str,
    default="World",
    help="Name to greet",
)
@click.pass_context
def hello(ctx: click.Context, name: str) -> None:
    """Greet the user with a personalized message."""
    try:
        debug = ctx.obj.get("debug", False) if ctx.obj else False

        logger.info(
            "Processing hello command",
            name=name,
            debug=debug,
        )

        message = f"Hello, {name}!"
        click.echo(message)

        logger.info("Command completed successfully", result=message)

    except Exception as e:
        logger.error("Command failed", error=str(e), exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Display current configuration settings.

    Shows configuration values from environment variables or defaults.
    """
    try:
        debug = ctx.obj.get("debug", False) if ctx.obj else False

        logger.info("Retrieving configuration")

        click.echo("Current Configuration:")
        click.echo(f"  Project: {{ cookiecutter.project_name }}")
        click.echo(f"  Version: {{ cookiecutter.version }}")
        click.echo(f"  Debug: {debug}")
        click.echo(f"  Log Level: {settings.log_level}")

        logger.info("Configuration displayed successfully")

    except Exception as e:
        logger.error("Failed to display configuration", error=str(e), exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()

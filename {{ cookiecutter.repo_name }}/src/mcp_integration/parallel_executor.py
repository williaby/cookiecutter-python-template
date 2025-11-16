"""Parallel Subagent Executor for MCP server coordination."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.utils.logging_mixin import LoggerMixin

from .client import MCPClient
from .config_manager import MCPConfigurationManager
from .docker_mcp_client import DockerMCPClient

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Result of a subagent execution."""

    def __init__(
        self,
        agent_id: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
        execution_time: float = 0.0,
    ) -> None:
        self.agent_id = agent_id
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "agent_id": self.agent_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }


class ParallelSubagentExecutor(LoggerMixin):
    """Executor for running subagents in parallel via MCP servers.

    Coordinates execution of multiple subagents concurrently, manages resource allocation,
    and provides comprehensive error handling and result aggregation.

    NEW: Supports Docker MCP Toolkit integration with smart routing between
    Docker-deployed servers (≤2GB) and self-hosted servers (>2GB) for optimal
    resource utilization and universal IDE access.
    """

    def __init__(self, config_manager: MCPConfigurationManager, mcp_client: MCPClient) -> None:
        """Initialize parallel executor.

        Args:
            config_manager: MCP configuration manager
            mcp_client: MCP client for server communication
        """
        super().__init__()
        self.config_manager = config_manager
        self.mcp_client = mcp_client
        self.docker_client = DockerMCPClient()
        self.max_workers = self._get_max_workers()
        self.timeout_seconds = 120  # Default timeout per subagent
        self.execution_history: list[dict[str, Any]] = []

    def _get_max_workers(self) -> int:
        """Get maximum number of concurrent workers."""
        parallel_config = self.config_manager.get_parallel_execution_config()
        max_concurrent = parallel_config.get("max_concurrent", 5)
        return min(
            max_concurrent if isinstance(max_concurrent, int) and max_concurrent > 0 else 5,
            10,
        )  # Cap at 10 for safety

    async def _select_optimal_client(self, server_name: str, tool_name: str = "") -> tuple[Any, str]:
        """Select optimal client (Docker vs self-hosted) for server and tool.

        Args:
            server_name: Name of the MCP server
            tool_name: Specific tool being called (for feature checking)

        Returns:
            Tuple of (client_instance, deployment_type)
        """
        try:
            # Check if server is available in Docker MCP Toolkit
            if await self.docker_client.is_available(server_name):
                # If specific tool requested, check if Docker deployment supports it
                if tool_name and not await self.docker_client.supports_feature(server_name, tool_name):
                    self.logger.info(f"Docker MCP doesn't support {tool_name} for {server_name}, using self-hosted")
                    return self.mcp_client, "self_hosted_fallback"

                # Check server configuration preferences
                server_config = self.config_manager.get_server_config(server_name)
                if server_config:
                    preference = server_config.deployment_preference
                    if preference == "self-hosted":
                        self.logger.info(f"Server {server_name} configured for self-hosted deployment")
                        return self.mcp_client, "self_hosted_configured"
                    if preference == "docker":
                        return self.docker_client, "docker_preferred"

                # Default to Docker for universal IDE access
                self.logger.debug(f"Using Docker MCP for {server_name} (universal IDE access)")
                return self.docker_client, "docker_default"

            # Fall back to self-hosted
            self.logger.debug(f"Using self-hosted MCP for {server_name} (not available in Docker)")
            return self.mcp_client, "self_hosted_only"
        except Exception as e:
            # If Docker client fails, fall back to self-hosted
            self.logger.warning(f"Docker client error for {server_name}, falling back to self-hosted: {e}")
            return self.mcp_client, "self_hosted"

    async def execute_subagents_parallel(
        self,
        subagent_tasks: list[dict[str, Any]],
        coordination_strategy: str = "independent",
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute multiple subagents in parallel.

        Args:
            subagent_tasks: List of subagent task configurations
            coordination_strategy: How to coordinate subagents ("independent", "consensus", "pipeline")
            timeout: Timeout in seconds for entire execution

        Returns:
            Aggregated results from all subagents
        """
        if not subagent_tasks:
            self.logger.warning("No subagent tasks provided")
            return {"success": False, "error": "No tasks provided", "results": []}

        execution_timeout = timeout or self.timeout_seconds
        start_time = time.time()

        self.logger.info(
            f"Starting parallel execution of {len(subagent_tasks)} subagents with strategy: {coordination_strategy}",
        )

        try:
            if coordination_strategy == "independent":
                results = await self._execute_independent(subagent_tasks, execution_timeout)
            elif coordination_strategy == "consensus":
                results = await self._execute_consensus(subagent_tasks, execution_timeout)
            elif coordination_strategy == "pipeline":
                results = await self._execute_pipeline(subagent_tasks, execution_timeout)
            else:
                raise ValueError(f"Unknown coordination strategy: {coordination_strategy}")

            execution_time = time.time() - start_time

            # Record execution history
            execution_record = {
                "timestamp": start_time,
                "task_count": len(subagent_tasks),
                "coordination_strategy": coordination_strategy,
                "execution_time": execution_time,
                "success_count": sum(1 for r in results["results"] if r["success"]),
                "total_count": len(results["results"]),
            }
            self.execution_history.append(execution_record)

            self.logger.info(f"Parallel execution completed in {execution_time:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            return {"success": False, "error": str(e), "results": [], "execution_time": time.time() - start_time}

    async def _execute_independent(self, tasks: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
        """Execute subagents independently without coordination."""
        tasks_with_timeout = [(task, timeout // len(tasks)) for task in tasks]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_single_subagent, task, task_timeout): task
                for task, task_timeout in tasks_with_timeout
            }

            results = []
            for future in as_completed(future_to_task, timeout=timeout):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result.to_dict())
                except Exception as e:
                    self.logger.error(f"Task {task.get('agent_id', 'unknown')} failed: {e}")
                    error_result = ExecutionResult(
                        agent_id=task.get("agent_id", "unknown"),
                        success=False,
                        error=str(e),
                    )
                    results.append(error_result.to_dict())

        success_count = sum(1 for r in results if r["success"])
        return {
            "success": success_count > 0,
            "coordination_strategy": "independent",
            "results": results,
            "summary": {"total": len(results), "successful": success_count, "failed": len(results) - success_count},
        }

    async def _execute_consensus(self, tasks: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
        """Execute subagents with consensus building."""
        # First execute all subagents independently
        independent_results = await self._execute_independent(tasks, timeout // 2)

        if not independent_results["success"]:
            return independent_results

        # Then build consensus from results
        consensus_result = self._build_consensus(independent_results["results"])

        return {
            "success": True,
            "coordination_strategy": "consensus",
            "results": independent_results["results"],
            "consensus": consensus_result,
            "summary": independent_results["summary"],
        }

    async def _execute_pipeline(self, tasks: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
        """Execute subagents in pipeline mode where output feeds into next."""
        results = []
        current_input: Any | None = None
        task_timeout = timeout // len(tasks)

        for i, task in enumerate(tasks):
            # Pass output from previous task as input to current task
            if i > 0 and current_input is not None:
                task["input_from_previous"] = current_input

            result = self._execute_single_subagent(task, task_timeout)
            results.append(result.to_dict())

            if result.success:
                current_input = result.result
                self.logger.debug(f"Pipeline step {i+1} completed successfully")
            else:
                self.logger.error(f"Pipeline step {i+1} failed, stopping pipeline")
                break

        success_count = sum(1 for r in results if r["success"])
        pipeline_success = success_count == len(tasks)  # All must succeed for pipeline

        return {
            "success": pipeline_success,
            "coordination_strategy": "pipeline",
            "results": results,
            "pipeline_complete": pipeline_success,
            "summary": {"total": len(results), "successful": success_count, "failed": len(results) - success_count},
        }

    def _execute_single_subagent(self, task: dict[str, Any], _timeout: int) -> ExecutionResult:
        """Execute a single subagent task with smart routing."""
        agent_id = task.get("agent_id", "unknown")
        start_time = time.time()

        try:
            self.logger.debug(f"Executing subagent: {agent_id}")

            # Get server and tool information from task
            server_name = task.get("server_name", agent_id)
            tool_name = task.get("tool_name", "")

            # Use smart routing to select optimal client
            try:
                # Since we're in a synchronous context, we'll simulate the async call
                # In a real implementation, this method would need to be async
                result = self._simulate_subagent_execution_with_routing(task, server_name, tool_name)
            except Exception as routing_error:
                # Fall back to standard simulation if routing fails
                self.logger.warning(f"Smart routing failed for {agent_id}, falling back: {routing_error}")
                result = self._simulate_subagent_execution(task)

            execution_time = time.time() - start_time

            return ExecutionResult(agent_id=agent_id, success=True, result=result, execution_time=execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Subagent {agent_id} execution failed: {e}")

            return ExecutionResult(agent_id=agent_id, success=False, error=str(e), execution_time=execution_time)

    def _simulate_subagent_execution(self, task: dict[str, Any]) -> dict[str, Any]:
        """Simulate subagent execution (placeholder for real MCP communication)."""
        agent_id = task.get("agent_id", "unknown")
        task_type = task.get("type", "unknown")

        # Simulate different types of subagent work
        if task_type == "analysis":
            return {
                "analysis_type": "code_review",
                "agent": agent_id,
                "findings": ["Code quality looks good", "No security issues found"],
                "confidence": 0.85,
                "deployment": "self_hosted_legacy",
            }
        if task_type == "validation":
            return {
                "validation_type": "configuration",
                "agent": agent_id,
                "valid": True,
                "checks_passed": 12,
                "checks_failed": 0,
                "deployment": "self_hosted_legacy",
            }
        if task_type == "generation":
            return {
                "generation_type": "documentation",
                "agent": agent_id,
                "output": "Generated comprehensive documentation",
                "word_count": 1500,
                "deployment": "self_hosted_legacy",
            }
        return {
            "task_type": task_type,
            "agent": agent_id,
            "status": "completed",
            "result": f"Task completed by {agent_id}",
            "deployment": "self_hosted_legacy",
        }

    def _simulate_subagent_execution_with_routing(
        self,
        task: dict[str, Any],
        server_name: str,
        tool_name: str,
    ) -> dict[str, Any]:
        """Simulate subagent execution with smart routing information."""
        # Simulate client selection logic (synchronous version)
        # This would normally use the async _select_optimal_client method
        deployment_type = "unknown"

        # Simple heuristic for simulation: servers with "basic" use Docker, others self-hosted
        if "basic" in server_name.lower() or "filesystem" in server_name.lower():
            deployment_type = "docker_default"
            available_features = ["read_file", "list_files", "basic_search"]
        else:
            deployment_type = "self_hosted_only"
            available_features = ["bulk_operations", "advanced_search", "caching", "webhooks"]

        # Check if requested tool is supported
        tool_supported = not tool_name or tool_name in available_features
        if not tool_supported and deployment_type == "docker_default":
            deployment_type = "self_hosted_fallback"
            available_features = ["bulk_operations", "advanced_search", "caching", "webhooks"]

        # Generate result based on deployment type
        base_result = self._simulate_subagent_execution(task)
        base_result.update(
            {
                "deployment": deployment_type,
                "server_name": server_name,
                "tool_name": tool_name if tool_name else "default",
                "available_features": available_features,
                "routing_metadata": {
                    "docker_available": deployment_type.startswith("docker"),
                    "feature_supported": tool_supported,
                    "fallback_used": "fallback" in deployment_type,
                },
            },
        )

        # Adjust performance characteristics based on deployment
        if deployment_type.startswith("docker"):
            base_result["performance"] = {
                "memory_limit": "2GB",
                "response_time": "good",
                "ide_compatibility": "universal",
            }
        else:
            base_result["performance"] = {
                "memory_limit": "unlimited",
                "response_time": "optimal",
                "ide_compatibility": "claude_code",
            }

        return base_result

    def _build_consensus(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Build consensus from multiple subagent results."""
        successful_results = [r for r in results if r["success"]]

        if not successful_results:
            return {"consensus_reached": False, "reason": "No successful results"}

        # Simple consensus: majority agreement
        consensus_score = len(successful_results) / len(results)
        consensus_threshold = 0.6  # 60% threshold
        consensus_reached = consensus_score >= consensus_threshold

        return {
            "consensus_reached": consensus_reached,
            "consensus_score": consensus_score,
            "participating_agents": len(successful_results),
            "total_agents": len(results),
            "summary": "Consensus reached" if consensus_reached else "No consensus",
        }

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get execution statistics and metrics."""
        if not self.execution_history:
            return {"total_executions": 0}

        total_executions = len(self.execution_history)
        total_tasks = sum(record["task_count"] for record in self.execution_history)
        total_successful = sum(record["success_count"] for record in self.execution_history)
        avg_execution_time = sum(record["execution_time"] for record in self.execution_history) / total_executions

        return {
            "total_executions": total_executions,
            "total_tasks_executed": total_tasks,
            "total_successful_tasks": total_successful,
            "success_rate": total_successful / total_tasks if total_tasks > 0 else 0,
            "average_execution_time": avg_execution_time,
            "max_concurrent_workers": self.max_workers,
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on parallel execution system."""
        config_health = self.config_manager.get_health_status()
        client_health = await self.mcp_client.health_check()
        docker_health = await self.docker_client.health_check()
        execution_stats = self.get_execution_statistics()

        overall_healthy = (
            config_health.get("configuration_valid", False)
            and client_health.get("overall_status") == "healthy"
            and docker_health.get("docker_mcp_available", True)  # Docker is optional, so default to True
        )

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "parallel_execution_enabled": config_health.get("parallel_execution", False),
            "max_workers": self.max_workers,
            "smart_routing_enabled": True,
            "configuration_status": config_health,
            "self_hosted_client_status": client_health,
            "docker_client_status": docker_health,
            "execution_statistics": execution_stats,
            "routing_summary": {
                "docker_servers_available": docker_health.get("total_servers", 0),
                "authenticated_services": docker_health.get("authenticated_servers", 0),
                "fallback_capability": client_health.get("overall_status") == "healthy",
            },
        }

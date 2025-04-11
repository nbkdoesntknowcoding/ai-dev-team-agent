"""
Task scheduler for the multi-agent development system.

This module manages task creation, scheduling, prioritization, and delegation
among multiple agents. It handles dependencies between tasks, optimizes resource
allocation, tracks progress, and ensures that the system can scale across many
concurrent agents working together to accomplish goals.
"""

import asyncio
import heapq
import json
import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union, cast
import time
import threading
import copy

from pydantic import BaseModel, Field, validator

# Set up logging
logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """States a task can be in during its lifecycle."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELED = "canceled"
    RETRYING = "retrying"


class TaskPriority(int, Enum):
    """Numeric task priorities with higher values indicating higher priority."""
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 20


class TaskType(str, Enum):
    """Types of tasks that can be scheduled."""
    NORMAL = "normal"
    PERIODIC = "periodic"
    REVISION = "revision"
    COORDINATION = "coordination"
    VALIDATION = "validation"
    HUMAN_FEEDBACK = "human_feedback"
    INTEGRATION = "integration"
    ANALYSIS = "analysis"


class TaskResult(BaseModel):
    """Result of a task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TaskDefinition(BaseModel):
    """Definition of a task to be scheduled."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    agent_id: str
    agent_type: str
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    retry_config: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    task_type: TaskType = TaskType.NORMAL
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    category: str = "default"
    
    @validator("deadline", pre=True, always=True)
    def validate_deadline(cls, v):
        """Validate deadline format."""
        if v is None:
            return None
        
        # If it's already a string, assume it's an ISO format
        if isinstance(v, str):
            try:
                datetime.fromisoformat(v)
                return v
            except ValueError:
                raise ValueError("Deadline must be in ISO format")
        
        # If it's a datetime, convert to ISO
        if isinstance(v, datetime):
            return v.isoformat()
        
        # If it's a timedelta, interpret as relative to now
        if isinstance(v, timedelta):
            return (datetime.now() + v).isoformat()
        
        raise ValueError("Deadline must be datetime, timedelta, or ISO string")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "action": self.action,
            "params": self.params,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
            "deadline": self.deadline,
            "retry_config": self.retry_config,
            "tags": self.tags,
            "task_type": self.task_type.value,
            "created_at": self.created_at,
            "category": self.category
        }


class TaskInstance(BaseModel):
    """Instance of a task in the scheduler."""
    definition: TaskDefinition
    state: TaskState = TaskState.PENDING
    current_agent: Optional[str] = None
    scheduled_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    attempt: int = 0
    max_attempts: int = 3
    retry_delay: float = 60.0  # seconds
    last_error: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "definition": self.definition.to_dict(),
            "state": self.state.value,
            "current_agent": self.current_agent,
            "scheduled_time": self.scheduled_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "retry_delay": self.retry_delay,
            "last_error": self.last_error,
            "results": self.results,
            "metadata": self.metadata
        }


class PeriodicTaskConfig(BaseModel):
    """Configuration for a periodic task."""
    interval_seconds: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_executions: Optional[int] = None
    execution_count: int = 0
    active: bool = True
    skip_if_previous_running: bool = True
    next_execution: Optional[str] = None
    timezone: str = "UTC"
    days_of_week: Optional[List[int]] = None  # 0=Monday, 6=Sunday


class TaskListener:
    """Interface for task event listeners."""
    
    async def on_task_created(self, task: TaskInstance) -> None:
        """Called when a task is created."""
        pass
    
    async def on_task_scheduled(self, task: TaskInstance) -> None:
        """Called when a task is scheduled."""
        pass
    
    async def on_task_started(self, task: TaskInstance) -> None:
        """Called when a task starts execution."""
        pass
    
    async def on_task_completed(self, task: TaskInstance, result: TaskResult) -> None:
        """Called when a task completes successfully."""
        pass
    
    async def on_task_failed(self, task: TaskInstance, error: str) -> None:
        """Called when a task fails."""
        pass
    
    async def on_task_retrying(self, task: TaskInstance) -> None:
        """Called when a task is being retried."""
        pass
    
    async def on_task_canceled(self, task: TaskInstance) -> None:
        """Called when a task is canceled."""
        pass


class AgentRegistry:
    """Registry of available agents and their capabilities."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_types: Dict[str, List[str]] = defaultdict(list)
        self.capabilities: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.lock = threading.RLock()
    
    def register_agent(
        self, 
        agent_id: str, 
        agent_type: str, 
        name: str,
        capabilities: Dict[str, List[str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an agent with the registry.
        
        Args:
            agent_id: Unique ID of the agent
            agent_type: Type of the agent
            name: Human-readable name of the agent
            capabilities: Dictionary of action categories and supported actions
            metadata: Additional agent metadata
        """
        with self.lock:
            self.agents[agent_id] = {
                "id": agent_id,
                "type": agent_type,
                "name": name,
                "capabilities": capabilities,
                "metadata": metadata or {},
                "status": "available",
                "last_seen": datetime.now().isoformat()
            }
            
            self.agent_types[agent_type].append(agent_id)
            
            # Register capabilities
            for category, actions in capabilities.items():
                for action in actions:
                    self.capabilities[category][action].append(agent_id)
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            agent_type = agent["type"]
            
            # Remove from agents dict
            del self.agents[agent_id]
            
            # Remove from agent types
            if agent_id in self.agent_types[agent_type]:
                self.agent_types[agent_type].remove(agent_id)
            
            # Remove from capabilities
            for category, actions in agent["capabilities"].items():
                for action in actions:
                    if agent_id in self.capabilities[category][action]:
                        self.capabilities[category][action].remove(agent_id)
            
            return True
    
    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update an agent's status.
        
        Args:
            agent_id: ID of the agent
            status: New status (e.g., "available", "busy", "offline")
            
        Returns:
            True if status was updated, False if agent not found
        """
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            self.agents[agent_id]["status"] = status
            self.agents[agent_id]["last_seen"] = datetime.now().isoformat()
            return True
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent information or None if not found
        """
        with self.lock:
            return self.agents.get(agent_id)
    
    def find_agents_for_task(
        self, 
        action: str, 
        category: str = "default", 
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """Find agents capable of executing a task.
        
        Args:
            action: Action to execute
            category: Action category
            agent_type: Optional agent type to filter by
            tags: Optional tags to match
            
        Returns:
            List of agent IDs
        """
        with self.lock:
            # Find agents with matching capability
            capable_agents = self.capabilities[category].get(action, [])
            
            # Filter by agent type if specified
            if agent_type:
                capable_agents = [
                    agent_id for agent_id in capable_agents
                    if self.agents.get(agent_id, {}).get("type") == agent_type
                ]
            
            # Filter by tags if specified
            if tags:
                filtered_agents = []
                for agent_id in capable_agents:
                    agent_tags = self.agents.get(agent_id, {}).get("metadata", {}).get("tags", [])
                    if any(tag in agent_tags for tag in tags):
                        filtered_agents.append(agent_id)
                capable_agents = filtered_agents
            
            # Filter by availability
            available_agents = [
                agent_id for agent_id in capable_agents
                if self.agents.get(agent_id, {}).get("status") == "available"
            ]
            
            return available_agents or capable_agents  # Fall back to any capable agent if none available
    
    def get_all_agents(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all registered agents.
        
        Args:
            agent_type: Optional agent type to filter by
            
        Returns:
            List of agent information dictionaries
        """
        with self.lock:
            if agent_type:
                return [
                    self.agents[agent_id] for agent_id in self.agent_types.get(agent_type, [])
                    if agent_id in self.agents
                ]
            else:
                return list(self.agents.values())


class TaskScheduler:
    """Scheduler for managing and executing tasks across multiple agents."""
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        shared_memory: Any = None,
        max_concurrent_tasks: int = 10,
        default_timeout: int = 300,
        task_execution_hook: Optional[Callable[[TaskInstance], Awaitable[TaskResult]]] = None,
        listeners: Optional[List[TaskListener]] = None,
        periodic_check_interval: int = 5,
    ):
        """Initialize the task scheduler.
        
        Args:
            agent_registry: Registry of available agents
            shared_memory: Optional shared memory interface
            max_concurrent_tasks: Maximum number of tasks to run concurrently
            default_timeout: Default task timeout in seconds
            task_execution_hook: Optional hook for executing tasks
            listeners: Optional list of task event listeners
            periodic_check_interval: Interval for checking periodic tasks (seconds)
        """
        self.agent_registry = agent_registry
        self.shared_memory = shared_memory
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.task_execution_hook = task_execution_hook
        self.listeners = listeners or []
        self.periodic_check_interval = periodic_check_interval
        
        # Task storage
        self.tasks: Dict[str, TaskInstance] = {}
        self.pending_tasks: List[Tuple[int, float, str]] = []  # priority queue: (priority, timestamp, task_id)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskInstance] = {}
        self.periodic_tasks: Dict[str, PeriodicTaskConfig] = {}
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # task_id -> set of tasks that depend on it
        self.blocked_tasks: Dict[str, Set[str]] = defaultdict(set)  # task_id -> set of tasks blocking it
        
        # Task history (only keeps the most recent N results)
        self.task_history_limit = 1000
        self.task_history: deque = deque(maxlen=self.task_history_limit)
        
        # Locking
        self._lock = threading.RLock()
        self._periodic_task_lock = threading.RLock()
        
        # Management
        self._running = False
        self._scheduler_task = None
        self._periodic_checker_task = None
        
        logger.info("Task scheduler initialized")
    
    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._periodic_checker_task = asyncio.create_task(self._periodic_task_checker())
        
        logger.info("Task scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return
        
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            self._scheduler_task = None
        
        if self._periodic_checker_task:
            self._periodic_checker_task.cancel()
            self._periodic_checker_task = None
        
        logger.info("Task scheduler stopped")
    
    async def schedule_task(self, task_def: Union[TaskDefinition, Dict[str, Any]]) -> str:
        """Schedule a new task.
        
        Args:
            task_def: Task definition or dictionary
            
        Returns:
            ID of the scheduled task
        """
        # Convert dictionary to TaskDefinition if needed
        if isinstance(task_def, dict):
            # Handle priority conversion
            if "priority" in task_def and isinstance(task_def["priority"], (str, int)):
                if isinstance(task_def["priority"], str):
                    priorities = {
                        "low": TaskPriority.LOW,
                        "medium": TaskPriority.MEDIUM,
                        "high": TaskPriority.HIGH,
                        "critical": TaskPriority.CRITICAL
                    }
                    task_def["priority"] = priorities.get(task_def["priority"].lower(), TaskPriority.MEDIUM)
                else:
                    # Convert integer to enum
                    task_def["priority"] = TaskPriority(task_def["priority"])
            
            # Handle task_type conversion
            if "task_type" in task_def and isinstance(task_def["task_type"], str):
                task_def["task_type"] = TaskType(task_def["task_type"])
            
            task_def = TaskDefinition(**task_def)
        
        # Create task instance
        task = TaskInstance(definition=task_def)
        
        # Store the task
        with self._lock:
            self.tasks[task.definition.id] = task
            
            # Add to pending queue
            heapq.heappush(
                self.pending_tasks,
                (-task.definition.priority.value, time.time(), task.definition.id)
            )
            
            # Update dependency graph
            for dep_id in task.definition.dependencies:
                self.dependency_graph[dep_id].add(task.definition.id)
                self.blocked_tasks[task.definition.id].add(dep_id)
            
            # Set state to BLOCKED if it has dependencies
            if task.definition.dependencies:
                task.state = TaskState.BLOCKED
            
            # Handle periodic task configuration
            if task.definition.task_type == TaskType.PERIODIC:
                periodic_config = task.definition.params.get("periodic_config", {})
                self._setup_periodic_task(task.definition.id, periodic_config)
        
        # Store in shared memory if available
        if self.shared_memory:
            await self.shared_memory.store(
                key=f"task_{task.definition.id}",
                value=task.to_dict(),
                category="tasks"
            )
        
        # Notify listeners
        for listener in self.listeners:
            try:
                await listener.on_task_created(task)
            except Exception as e:
                logger.error(f"Error notifying listener of task creation: {str(e)}")
        
        logger.info(f"Scheduled task {task.definition.id}: {task.definition.title}")
        return task.definition.id
    
    def _setup_periodic_task(self, task_id: str, config: Dict[str, Any]) -> None:
        """Set up a periodic task with the given configuration.
        
        Args:
            task_id: ID of the task
            config: Periodic task configuration
        """
        with self._periodic_task_lock:
            interval_seconds = config.get("interval_seconds", 3600)  # Default: hourly
            
            # Parse start_time if provided
            start_time = config.get("start_time")
            if start_time:
                # Convert to ISO if it's a datetime object
                if isinstance(start_time, datetime):
                    start_time = start_time.isoformat()
            else:
                # Default: start now
                start_time = datetime.now().isoformat()
            
            # Create periodic task config
            periodic_config = PeriodicTaskConfig(
                interval_seconds=interval_seconds,
                start_time=start_time,
                end_time=config.get("end_time"),
                max_executions=config.get("max_executions"),
                skip_if_previous_running=config.get("skip_if_previous_running", True),
                timezone=config.get("timezone", "UTC"),
                days_of_week=config.get("days_of_week"),
                active=config.get("active", True)
            )
            
            # Calculate next execution time
            start_dt = datetime.fromisoformat(start_time)
            now = datetime.now()
            
            # If start time is in the future, use that
            if start_dt > now:
                next_execution = start_dt.isoformat()
            else:
                # Otherwise, calculate next occurrence based on interval
                delta = timedelta(seconds=interval_seconds)
                next_time = now + delta
                next_execution = next_time.isoformat()
            
            periodic_config.next_execution = next_execution
            
            # Store the configuration
            self.periodic_tasks[task_id] = periodic_config
            
            logger.info(f"Set up periodic task {task_id} with interval {interval_seconds}s, next execution at {next_execution}")
    
    async def _periodic_task_checker(self) -> None:
        """Background task that checks for periodic tasks to execute."""
        while self._running:
            try:
                await self._check_periodic_tasks()
                await asyncio.sleep(self.periodic_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic task checker: {str(e)}")
                await asyncio.sleep(self.periodic_check_interval)
    
    async def _check_periodic_tasks(self) -> None:
        """Check for periodic tasks that need to be executed."""
        now = datetime.now()
        tasks_to_execute = []
        
        with self._periodic_task_lock:
            for task_id, config in list(self.periodic_tasks.items()):
                # Skip inactive tasks
                if not config.active:
                    continue
                
                # Skip if max executions reached
                if config.max_executions and config.execution_count >= config.max_executions:
                    continue
                
                # Skip if end time passed
                if config.end_time and datetime.fromisoformat(config.end_time) < now:
                    continue
                
                # Check if it's time to execute
                if config.next_execution and datetime.fromisoformat(config.next_execution) <= now:
                    # Check if previous instance is still running
                    if config.skip_if_previous_running:
                        original_task = self.tasks.get(task_id)
                        if original_task and original_task.state == TaskState.RUNNING:
                            logger.info(f"Skipping periodic task {task_id} because previous instance is still running")
                            
                            # Update next execution time
                            next_time = now + timedelta(seconds=config.interval_seconds)
                            config.next_execution = next_time.isoformat()
                            continue
                    
                    # Check if this day is allowed
                    if config.days_of_week:
                        # Convert to Python's day of week (0 = Monday)
                        day_of_week = now.weekday()
                        if day_of_week not in config.days_of_week:
                            # Skip this day, calculate next execution on an allowed day
                            next_time = now + timedelta(days=1)
                            for _ in range(7):  # Try all possible days
                                if next_time.weekday() in config.days_of_week:
                                    break
                                next_time += timedelta(days=1)
                            
                            # Set time of day to match the original next_execution
                            original_time = datetime.fromisoformat(config.next_execution)
                            next_time = next_time.replace(
                                hour=original_time.hour,
                                minute=original_time.minute,
                                second=original_time.second
                            )
                            
                            config.next_execution = next_time.isoformat()
                            continue
                    
                    # Add to execution list
                    tasks_to_execute.append(task_id)
                    
                    # Update execution count and next execution time
                    config.execution_count += 1
                    next_time = now + timedelta(seconds=config.interval_seconds)
                    config.next_execution = next_time.isoformat()
        
        # Create new instances of the periodic tasks
        for task_id in tasks_to_execute:
            try:
                # Get the original task definition
                original_task = self.tasks.get(task_id)
                if not original_task:
                    # Try to load from shared memory
                    if self.shared_memory:
                        task_data = await self.shared_memory.retrieve(
                            key=f"task_{task_id}",
                            category="tasks"
                        )
                        if task_data:
                            original_task = TaskInstance(**task_data)
                    
                    if not original_task:
                        logger.error(f"Cannot find original task for periodic execution: {task_id}")
                        continue
                
                # Create a new task instance with the same definition
                # but a new ID and reset state
                new_task_def = copy.deepcopy(original_task.definition)
                new_task_def.id = f"{task_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                new_task_def.created_at = datetime.now().isoformat()
                
                # Add reference to the parent task
                new_task_def.params["parent_task_id"] = task_id
                
                # Schedule the new task
                await self.schedule_task(new_task_def)
                
                logger.info(f"Created new instance of periodic task {task_id}: {new_task_def.id}")
                
            except Exception as e:
                logger.error(f"Error creating new instance of periodic task {task_id}: {str(e)}")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that assigns tasks to agents."""
        while self._running:
            try:
                # Check if we can run more tasks
                running_count = len(self.running_tasks)
                if running_count >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.5)
                    continue
                
                # How many slots are available
                available_slots = self.max_concurrent_tasks - running_count
                
                # Get up to N highest priority tasks
                tasks_to_run = []
                
                with self._lock:
                    # Get blocked task IDs
                    blocked_task_ids = set()
                    for task_id, dependencies in self.blocked_tasks.items():
                        # Check if any dependencies are not completed
                        if any(
                            dep_id in self.tasks and self.tasks[dep_id].state not in [TaskState.SUCCEEDED, TaskState.CANCELED]
                            for dep_id in dependencies
                        ):
                            blocked_task_ids.add(task_id)
                    
                    # Collect tasks to run
                    while len(tasks_to_run) < available_slots and self.pending_tasks:
                        # Get highest priority task
                        _, _, task_id = heapq.heappop(self.pending_tasks)
                        
                        # Skip if task is not in pending state anymore
                        if task_id not in self.tasks:
                            continue
                        
                        task = self.tasks[task_id]
                        if task.state != TaskState.PENDING and task.state != TaskState.SCHEDULED:
                            continue
                        
                        # Skip if task is blocked by dependencies
                        if task_id in blocked_task_ids:
                            task.state = TaskState.BLOCKED
                            continue
                        
                        # Skip if task has a deadline that has passed
                        if task.definition.deadline:
                            deadline = datetime.fromisoformat(task.definition.deadline)
                            if deadline < datetime.now():
                                logger.warning(f"Task {task_id} has missed its deadline")
                                task.state = TaskState.FAILED
                                task.last_error = "Deadline missed"
                                continue
                        
                        # Add to tasks to run
                        tasks_to_run.append(task_id)
                
                # Start tasks
                for task_id in tasks_to_run:
                    await self._start_task(task_id)
                
                # If no tasks were started, sleep for a bit
                if not tasks_to_run:
                    await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                await asyncio.sleep(1)
    
    async def _start_task(self, task_id: str) -> None:
        """Start execution of a task.
        
        Args:
            task_id: ID of the task to start
        """
        with self._lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found, cannot start")
                return
            
            task = self.tasks[task_id]
            
            # Skip if task is already running or completed
            if task.state in [TaskState.RUNNING, TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELED]:
                return
            
            # Mark task as running
            task.state = TaskState.RUNNING
            task.start_time = datetime.now().isoformat()
            
            # Find an agent to execute the task
            agent_id = task.definition.agent_id
            
            # If no specific agent is assigned, find a suitable one
            if agent_id == "auto" or not agent_id:
                agents = self.agent_registry.find_agents_for_task(
                    action=task.definition.action,
                    category=task.definition.category,
                    agent_type=task.definition.agent_type,
                    tags=task.definition.tags
                )
                
                if agents:
                    agent_id = agents[0]  # Take the first available agent
                else:
                    logger.warning(
                        f"No suitable agent found for task {task_id} "
                        f"({task.definition.action}/{task.definition.category})"
                    )
                    task.state = TaskState.FAILED
                    task.last_error = "No suitable agent found"
                    return
            
            # Verify agent exists
            agent = self.agent_registry.get_agent(agent_id)
            if not agent:
                logger.warning(f"Agent {agent_id} not found, cannot execute task {task_id}")
                task.state = TaskState.FAILED
                task.last_error = f"Agent {agent_id} not found"
                return
            
            # Assign task to agent
            task.current_agent = agent_id
            
            # Update agent status
            self.agent_registry.update_agent_status(agent_id, "busy")
            
            # Create task to execute the agent task
            execution_task = asyncio.create_task(
                self._execute_task(task_id)
            )
            
            # Store running task
            self.running_tasks[task_id] = execution_task
            
            # Notify listeners
            for listener in self.listeners:
                try:
                    await listener.on_task_started(task)
                except Exception as e:
                    logger.error(f"Error notifying listener of task start: {str(e)}")
        
        # Update shared memory if available
        if self.shared_memory:
            try:
                await self.shared_memory.store(
                    key=f"task_{task_id}",
                    value=task.to_dict(),
                    category="tasks"
                )
            except Exception as e:
                logger.error(f"Error storing task state in shared memory: {str(e)}")
        
        logger.info(f"Started task {task_id} on agent {task.current_agent}")
    
    async def _execute_task(self, task_id: str) -> None:
        """Execute a task and process its result.
        
        Args:
            task_id: ID of the task to execute
        """
        task = None
        agent_id = None
        
        try:
            with self._lock:
                if task_id not in self.tasks:
                    logger.warning(f"Task {task_id} not found, cannot execute")
                    return
                
                task = self.tasks[task_id]
                agent_id = task.current_agent
            
            start_time = time.time()
            result = None
            
            # Execute the task
            if self.task_execution_hook:
                result = await self.task_execution_hook(task)
            else:
                result = await self._default_task_execution(task)
            
            execution_time = time.time() - start_time
            
            # Process the result
            await self._process_task_result(task, result, execution_time)
            
        except asyncio.CancelledError:
            logger.warning(f"Task {task_id} execution was cancelled")
            if task:
                with self._lock:
                    task.state = TaskState.CANCELED
                    task.end_time = datetime.now().isoformat()
                
                # Notify listeners
                for listener in self.listeners:
                    try:
                        await listener.on_task_canceled(task)
                    except Exception as e:
                        logger.error(f"Error notifying listener of task cancellation: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {str(e)}")
            if task:
                with self._lock:
                    task.state = TaskState.FAILED
                    task.last_error = str(e)
                    task.end_time = datetime.now().isoformat()
                
                # Check if we should retry
                if task.attempt < task.max_attempts:
                    await self._retry_task(task)
                else:
                    # Notify listeners
                    for listener in self.listeners:
                        try:
                            await listener.on_task_failed(task, str(e))
                        except Exception as listener_error:
                            logger.error(f"Error notifying listener of task failure: {str(listener_error)}")
        
        finally:
            # Clean up
            with self._lock:
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            
            # Update agent status
            if agent_id:
                self.agent_registry.update_agent_status(agent_id, "available")
    
    async def _default_task_execution(self, task: TaskInstance) -> TaskResult:
        """Default implementation of task execution.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
            
        Raises:
            Exception: If task execution fails
        """
        # This is a placeholder for the actual execution logic
        # In a real implementation, this would call the agent API or service
        
        agent_id = task.current_agent
        if not agent_id:
            raise ValueError(f"No agent assigned to task {task.definition.id}")
        
        # Get agent info
        agent = self.agent_registry.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # In a real implementation, this would call the agent's API
        # For now, we just simulate a successful execution
        start_time = time.time()
        
        # Simulate work
        await asyncio.sleep(0.5)
        
        execution_time = time.time() - start_time
        
        # Simulate a result
        result = TaskResult(
            task_id=task.definition.id,
            success=True,
            result=f"Simulated result for task {task.definition.id}",
            execution_time=execution_time
        )
        
        return result
    
    async def _process_task_result(
        self, 
        task: TaskInstance, 
        result: TaskResult,
        execution_time: float
    ) -> None:
        """Process the result of a task execution.
        
        Args:
            task: Task that was executed
            result: Result of the execution
            execution_time: How long the execution took in seconds
        """
        with self._lock:
            # Update task state
            if result.success:
                task.state = TaskState.SUCCEEDED
            else:
                task.state = TaskState.FAILED
                task.last_error = result.error or "Unknown error"
            
            # Record the result
            task.results.append(result.dict())
            task.end_time = datetime.now().isoformat()
            
            # Move to completed tasks if successful
            if task.state == TaskState.SUCCEEDED:
                # Check if any tasks are blocked by this one
                dependent_tasks = self.dependency_graph.get(task.definition.id, set())
                
                # Unblock dependent tasks
                for dep_task_id in dependent_tasks:
                    if dep_task_id in self.blocked_tasks:
                        self.blocked_tasks[dep_task_id].discard(task.definition.id)
                        
                        # If all dependencies are satisfied, move to pending
                        if not self.blocked_tasks[dep_task_id]:
                            if dep_task_id in self.tasks:
                                dep_task = self.tasks[dep_task_id]
                                if dep_task.state == TaskState.BLOCKED:
                                    dep_task.state = TaskState.PENDING
                                    
                                    # Add to pending queue
                                    heapq.heappush(
                                        self.pending_tasks,
                                        (-dep_task.definition.priority.value, time.time(), dep_task_id)
                                    )
            
            # Add to task history
            self.task_history.append((
                task.definition.id,
                task.definition.title,
                task.state.value,
                datetime.now().isoformat()
            ))
            
            # Move to completed tasks if done
            if task.state in [TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELED]:
                # Keep original in tasks dict for dependency tracking
                # but also add to completed_tasks for history
                self.completed_tasks[task.definition.id] = copy.deepcopy(task)
        
        # Update shared memory if available
        if self.shared_memory:
            try:
                await self.shared_memory.store(
                    key=f"task_{task.definition.id}",
                    value=task.to_dict(),
                    category="tasks"
                )
                
                # Also store the result separately
                await self.shared_memory.store(
                    key=f"task_result_{task.definition.id}",
                    value=result.dict(),
                    category="task_results"
                )
            except Exception as e:
                logger.error(f"Error storing task result in shared memory: {str(e)}")
        
        # Notify listeners
        if task.state == TaskState.SUCCEEDED:
            for listener in self.listeners:
                try:
                    await listener.on_task_completed(task, result)
                except Exception as e:
                    logger.error(f"Error notifying listener of task completion: {str(e)}")
        elif task.state == TaskState.FAILED:
            for listener in self.listeners:
                try:
                    await listener.on_task_failed(task, result.error or "Unknown error")
                except Exception as e:
                    logger.error(f"Error notifying listener of task failure: {str(e)}")
        
        logger.info(
            f"Task {task.definition.id} completed with status {task.state.value} "
            f"in {execution_time:.2f}s"
        )
    
    async def _retry_task(self, task: TaskInstance) -> None:
        """Retry a failed task.
        
        Args:
            task: Task to retry
        """
        with self._lock:
            # Increment attempt counter
            task.attempt += 1
            
            # Reset state
            task.state = TaskState.RETRYING
            task.start_time = None
            task.end_time = None
            
            # Calculate retry delay (with exponential backoff)
            retry_delay = task.retry_delay * (2 ** (task.attempt - 1))
            
            # Schedule for retry after delay
            asyncio.create_task(self._schedule_retry(task.definition.id, retry_delay))
        
        # Notify listeners
        for listener in self.listeners:
            try:
                await listener.on_task_retrying(task)
            except Exception as e:
                logger.error(f"Error notifying listener of task retry: {str(e)}")
        
        logger.info(
            f"Task {task.definition.id} scheduled for retry (attempt {task.attempt}/{task.max_attempts}) "
            f"with delay {retry_delay:.2f}s"
        )
    
    async def _schedule_retry(self, task_id: str, delay: float) -> None:
        """Schedule a task for retry after a delay.
        
        Args:
            task_id: ID of the task to retry
            delay: Delay in seconds before retry
        """
        try:
            # Wait for the retry delay
            await asyncio.sleep(delay)
            
            with self._lock:
                if task_id not in self.tasks:
                    logger.warning(f"Task {task_id} not found, cannot retry")
                    return
                
                task = self.tasks[task_id]
                
                # Skip if task is no longer in retrying state
                if task.state != TaskState.RETRYING:
                    return
                
                # Set back to pending
                task.state = TaskState.PENDING
                
                # Add to pending queue
                heapq.heappush(
                    self.pending_tasks,
                    (-task.definition.priority.value, time.time(), task_id)
                )
            
            # Update shared memory if available
            if self.shared_memory:
                try:
                    await self.shared_memory.store(
                        key=f"task_{task_id}",
                        value=task.to_dict(),
                        category="tasks"
                    )
                except Exception as e:
                    logger.error(f"Error storing task state in shared memory: {str(e)}")
            
            logger.info(f"Task {task_id} is now ready for retry")
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error scheduling retry for task {task_id}: {str(e)}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled, False otherwise
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # Skip if task is already completed
            if task.state in [TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELED]:
                return False
            
            # Cancel running task
            if task.state == TaskState.RUNNING and task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
            
            # Update task state
            task.state = TaskState.CANCELED
            task.end_time = datetime.now().isoformat()
        
        # Update shared memory if available
        if self.shared_memory:
            try:
                await self.shared_memory.store(
                    key=f"task_{task_id}",
                    value=task.to_dict(),
                    category="tasks"
                )
            except Exception as e:
                logger.error(f"Error storing task state in shared memory: {str(e)}")
        
        # Notify listeners
        for listener in self.listeners:
            try:
                await listener.on_task_canceled(task)
            except Exception as e:
                logger.error(f"Error notifying listener of task cancellation: {str(e)}")
        
        logger.info(f"Cancelled task {task_id}")
        return True
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task information or None if not found
        """
        with self._lock:
            if task_id in self.tasks:
                return self.tasks[task_id].to_dict()
            
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].to_dict()
        
        # If not in memory, try to load from shared memory
        if self.shared_memory:
            try:
                task_data = await self.shared_memory.retrieve(
                    key=f"task_{task_id}",
                    category="tasks"
                )
                if task_data:
                    return task_data
            except Exception as e:
                logger.error(f"Error retrieving task from shared memory: {str(e)}")
        
        return None
    
    async def get_task_results(self, task_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get results of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of task results or None if not found
        """
        task_data = await self.get_task(task_id)
        if task_data:
            return task_data.get("results", [])
        
        # Try to get just the results if available in shared memory
        if self.shared_memory:
            try:
                result_data = await self.shared_memory.retrieve(
                    key=f"task_result_{task_id}",
                    category="task_results"
                )
                if result_data:
                    return [result_data]
            except Exception as e:
                logger.error(f"Error retrieving task result from shared memory: {str(e)}")
        
        return None
    
    async def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get all pending tasks.
        
        Returns:
            List of pending task information
        """
        pending_tasks = []
        
        with self._lock:
            for _, _, task_id in self.pending_tasks:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.state in [TaskState.PENDING, TaskState.SCHEDULED, TaskState.BLOCKED]:
                        pending_tasks.append(task.to_dict())
        
        return pending_tasks
    
    async def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get all running tasks.
        
        Returns:
            List of running task information
        """
        running_tasks = []
        
        with self._lock:
            for task_id in self.running_tasks:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    running_tasks.append(task.to_dict())
        
        return running_tasks
    
    async def get_completed_tasks(
        self, 
        limit: int = 100, 
        offset: int = 0, 
        state: Optional[TaskState] = None
    ) -> List[Dict[str, Any]]:
        """Get completed tasks.
        
        Args:
            limit: Maximum number of tasks to return
            offset: Offset for pagination
            state: Optional state filter
            
        Returns:
            List of completed task information
        """
        completed_tasks = []
        
        with self._lock:
            tasks = list(self.completed_tasks.values())
            
            # Filter by state if specified
            if state:
                tasks = [task for task in tasks if task.state == state]
            
            # Sort by end time (newest first)
            tasks.sort(key=lambda x: x.end_time or "", reverse=True)
            
            # Apply pagination
            tasks = tasks[offset:offset+limit]
            
            # Convert to dicts
            completed_tasks = [task.to_dict() for task in tasks]
        
        return completed_tasks
    
    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status or None if not found
        """
        with self._lock:
            if task_id in self.tasks:
                return self.tasks[task_id].state.value
            
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].state.value
        
        # If not in memory, try to load from shared memory
        if self.shared_memory:
            try:
                task_data = await self.shared_memory.retrieve(
                    key=f"task_{task_id}",
                    category="tasks"
                )
                if task_data:
                    return task_data.get("state")
            except Exception as e:
                logger.error(f"Error retrieving task from shared memory: {str(e)}")
        
        return None
    
    async def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get task execution history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of task history entries
        """
        with self._lock:
            # Convert deque to list and format for output
            history = list(self.task_history)[-limit:]
            
            return [
                {
                    "task_id": entry[0],
                    "title": entry[1],
                    "state": entry[2],
                    "timestamp": entry[3]
                }
                for entry in history
            ]
    
    async def schedule_revision(self, revision: Dict[str, Any]) -> str:
        """Schedule a revision task.
        
        Args:
            revision: Revision task details
            
        Returns:
            ID of the scheduled task
        """
        # Create task definition from revision
        task_def = TaskDefinition(
            title=f"Revision: {revision.get('description', 'Untitled')}",
            description=revision.get("description", ""),
            agent_id=revision.get("agent_id", ""),
            agent_type=revision.get("agent_type", ""),
            action="revise",
            params={
                "original_task_id": revision.get("original_task_id"),
                "feedback_ids": revision.get("feedback_ids", []),
                "requirements": revision.get("requirements", {})
            },
            priority=TaskPriority.HIGH,
            dependencies=[],
            task_type=TaskType.REVISION,
            category="revisions"
        )
        
        # Schedule the task
        return await self.schedule_task(task_def)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.
        
        Returns:
            Dictionary of scheduler statistics
        """
        with self._lock:
            # Count tasks by state
            tasks_by_state = defaultdict(int)
            for task in self.tasks.values():
                tasks_by_state[task.state.value] += 1
            
            for task in self.completed_tasks.values():
                tasks_by_state[task.state.value] += 1
            
            # Get periodic tasks
            with self._periodic_task_lock:
                periodic_tasks = {}
                for task_id, config in self.periodic_tasks.items():
                    periodic_tasks[task_id] = {
                        "interval_seconds": config.interval_seconds,
                        "next_execution": config.next_execution,
                        "execution_count": config.execution_count,
                        "active": config.active
                    }
            
            return {
                "running": self._running,
                "total_tasks": len(self.tasks) + len(self.completed_tasks),
                "pending_tasks": len(self.pending_tasks),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "tasks_by_state": dict(tasks_by_state),
                "periodic_tasks": periodic_tasks,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "timestamp": datetime.now().isoformat()
            }
    
    def __del__(self) -> None:
        """Clean up resources when the scheduler is destroyed."""
        try:
            self.stop()
        except:
            pass


# Example task listeners that integrate with various notification systems

class SlackTaskListener(TaskListener):
    """Task listener that sends notifications to Slack."""
    
    def __init__(self, webhook_url: str, channel: str = "#tasks"):
        """Initialize the Slack task listener.
        
        Args:
            webhook_url: Slack webhook URL
            channel: Slack channel to send notifications to
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.session = None
    
    async def _ensure_session(self) -> None:
        """Ensure an aiohttp session is available."""
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()
    
    async def _send_to_slack(self, message: str, blocks: Optional[List[Dict[str, Any]]] = None) -> None:
        """Send a message to Slack.
        
        Args:
            message: Message text
            blocks: Optional message blocks
        """
        await self._ensure_session()
        
        try:
            import aiohttp
            
            payload = {
                "channel": self.channel,
                "text": message
            }
            
            if blocks:
                payload["blocks"] = blocks
            
            async with self.session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Error sending to Slack: {response.status} {await response.text()}")
        
        except Exception as e:
            logger.error(f"Error sending to Slack: {str(e)}")
    
    async def on_task_created(self, task: TaskInstance) -> None:
        """Called when a task is created."""
        await self._send_to_slack(
            f"Task created: {task.definition.title}",
            [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Task created:* {task.definition.title}\n"
                           f"*ID:* {task.definition.id}\n"
                           f"*Priority:* {task.definition.priority.name}\n"
                           f"*Agent:* {task.definition.agent_id or 'Auto-assign'}"
                }
            }]
        )
    
    async def on_task_started(self, task: TaskInstance) -> None:
        """Called when a task starts execution."""
        await self._send_to_slack(
            f"Task started: {task.definition.title}",
            [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Task started:* {task.definition.title}\n"
                           f"*ID:* {task.definition.id}\n"
                           f"*Agent:* {task.current_agent}"
                }
            }]
        )
    
    async def on_task_completed(self, task: TaskInstance, result: TaskResult) -> None:
        """Called when a task completes successfully."""
        await self._send_to_slack(
            f"Task completed: {task.definition.title}",
            [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Task completed:* {task.definition.title}\n"
                           f"*ID:* {task.definition.id}\n"
                           f"*Agent:* {task.current_agent}\n"
                           f"*Execution time:* {result.execution_time:.2f}s"
                }
            }]
        )
    
    async def on_task_failed(self, task: TaskInstance, error: str) -> None:
        """Called when a task fails."""
        await self._send_to_slack(
            f"Task failed: {task.definition.title}",
            [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Task failed:* {task.definition.title}\n"
                           f"*ID:* {task.definition.id}\n"
                           f"*Agent:* {task.current_agent}\n"
                           f"*Error:* {error}"
                }
            }]
        )
    
    async def close(self) -> None:
        """Close resources."""
        if self.session:
            await self.session.close()
            self.session = None


class LoggingTaskListener(TaskListener):
    """Task listener that logs task events."""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize the logging task listener.
        
        Args:
            log_level: Log level
        """
        self.logger = logging.getLogger(__name__ + ".TaskLogger")
        self.logger.setLevel(log_level)
    
    async def on_task_created(self, task: TaskInstance) -> None:
        """Called when a task is created."""
        self.logger.info(
            f"Task created: {task.definition.id} - {task.definition.title} "
            f"(Priority: {task.definition.priority.name})"
        )
    
    async def on_task_scheduled(self, task: TaskInstance) -> None:
        """Called when a task is scheduled."""
        self.logger.info(
            f"Task scheduled: {task.definition.id} - {task.definition.title}"
        )
    
    async def on_task_started(self, task: TaskInstance) -> None:
        """Called when a task starts execution."""
        self.logger.info(
            f"Task started: {task.definition.id} - {task.definition.title} "
            f"(Agent: {task.current_agent})"
        )
    
    async def on_task_completed(self, task: TaskInstance, result: TaskResult) -> None:
        """Called when a task completes successfully."""
        self.logger.info(
            f"Task completed: {task.definition.id} - {task.definition.title} "
            f"(Agent: {task.current_agent}, Time: {result.execution_time:.2f}s)"
        )
    
    async def on_task_failed(self, task: TaskInstance, error: str) -> None:
        """Called when a task fails."""
        self.logger.error(
            f"Task failed: {task.definition.id} - {task.definition.title} "
            f"(Agent: {task.current_agent}, Error: {error})"
        )
    
    async def on_task_retrying(self, task: TaskInstance) -> None:
        """Called when a task is being retried."""
        self.logger.warning(
            f"Task retrying: {task.definition.id} - {task.definition.title} "
            f"(Attempt: {task.attempt}/{task.max_attempts})"
        )
    
    async def on_task_canceled(self, task: TaskInstance) -> None:
        """Called when a task is canceled."""
        self.logger.warning(
            f"Task canceled: {task.definition.id} - {task.definition.title}"
        )


class WebhookTaskListener(TaskListener):
    """Task listener that sends notifications to a webhook."""
    
    def __init__(self, webhook_url: str):
        """Initialize the webhook task listener.
        
        Args:
            webhook_url: Webhook URL
        """
        self.webhook_url = webhook_url
        self.session = None
    
    async def _ensure_session(self) -> None:
        """Ensure an aiohttp session is available."""
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()
    
    async def _send_to_webhook(self, event: str, data: Dict[str, Any]) -> None:
        """Send a notification to the webhook.
        
        Args:
            event: Event name
            data: Event data
        """
        await self._ensure_session()
        
        try:
            import aiohttp
            
            payload = {
                "event": event,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            async with self.session.post(self.webhook_url, json=payload) as response:
                if response.status not in [200, 201, 202, 204]:
                    logger.error(f"Error sending to webhook: {response.status} {await response.text()}")
        
        except Exception as e:
            logger.error(f"Error sending to webhook: {str(e)}")
    
    async def on_task_created(self, task: TaskInstance) -> None:
        """Called when a task is created."""
        await self._send_to_webhook("task.created", task.to_dict())
    
    async def on_task_scheduled(self, task: TaskInstance) -> None:
        """Called when a task is scheduled."""
        await self._send_to_webhook("task.scheduled", task.to_dict())
    
    async def on_task_started(self, task: TaskInstance) -> None:
        """Called when a task starts execution."""
        await self._send_to_webhook("task.started", task.to_dict())
    
    async def on_task_completed(self, task: TaskInstance, result: TaskResult) -> None:
        """Called when a task completes successfully."""
        await self._send_to_webhook("task.completed", {
            "task": task.to_dict(),
            "result": result.dict()
        })
    
    async def on_task_failed(self, task: TaskInstance, error: str) -> None:
        """Called when a task fails."""
        await self._send_to_webhook("task.failed", {
            "task": task.to_dict(),
            "error": error
        })
    
    async def on_task_retrying(self, task: TaskInstance) -> None:
        """Called when a task is being retried."""
        await self._send_to_webhook("task.retrying", task.to_dict())
"""
Workflow engine for the multi-agent development system.

This module provides a workflow orchestration system that coordinates complex
multi-step processes across different agents. It supports sequential, parallel,
and conditional execution, handles state persistence, and provides monitoring
and visualization of workflow execution.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union, cast
import threading
from collections import defaultdict, deque
import copy

from pydantic import BaseModel, Field, validator

# Set up logging
logger = logging.getLogger(__name__)


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    TASK = "task"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SEQUENCE = "sequence"
    WAIT = "wait"
    LOOP = "loop"
    HUMAN_INPUT = "human_input"
    APPROVAL = "approval"
    NOTIFICATION = "notification"
    CODE_EXECUTION = "code_execution"
    SUBWORKFLOW = "subworkflow"


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELED = "canceled"
    WAITING = "waiting"


class StepStatus(str, Enum):
    """Status of a workflow step execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


class WorkflowTriggerType(str, Enum):
    """Types of workflow triggers."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    API = "api"
    TASK_COMPLETION = "task_completion"
    CODE_CHANGE = "code_change"
    PERIODIC = "periodic"


class StepInput(BaseModel):
    """Input for a workflow step."""
    source: str  # "static", "previous_step", "workflow_variable", "global_variable"
    value: Any
    key: Optional[str] = None  # For when the source is another step or a variable


class StepOutput(BaseModel):
    """Output from a workflow step."""
    status: StepStatus
    result: Any
    error: Optional[str] = None
    execution_time: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class WorkflowStep(BaseModel):
    """Definition of a single step in a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    step_type: WorkflowStepType
    agent_id: Optional[str] = None
    action: Optional[str] = None
    inputs: Dict[str, StepInput] = Field(default_factory=dict)
    condition: Optional[str] = None  # For conditional steps, Python expression
    retry_config: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    next_steps: List[str] = Field(default_factory=list)  # Step IDs of next steps
    on_failure: Optional[str] = None  # Step ID to go to on failure
    on_timeout: Optional[str] = None  # Step ID to go to on timeout
    metadata: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)  # Step-specific configuration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "step_type": self.step_type.value,
            "agent_id": self.agent_id,
            "action": self.action,
            "inputs": {
                k: v.dict() for k, v in self.inputs.items()
            },
            "condition": self.condition,
            "retry_config": self.retry_config,
            "timeout_seconds": self.timeout_seconds,
            "next_steps": self.next_steps,
            "on_failure": self.on_failure,
            "on_timeout": self.on_timeout,
            "metadata": self.metadata,
            "config": self.config
        }


class WorkflowStepExecution(BaseModel):
    """Execution instance of a workflow step."""
    step_definition: WorkflowStep
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    outputs: Optional[StepOutput] = None
    attempt: int = 1
    max_attempts: int = 1
    logs: List[str] = Field(default_factory=list)
    inputs_resolved: Dict[str, Any] = Field(default_factory=dict)
    next_steps_resolved: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_definition": self.step_definition.to_dict(),
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "outputs": self.outputs.dict() if self.outputs else None,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "logs": self.logs,
            "inputs_resolved": self.inputs_resolved,
            "next_steps_resolved": self.next_steps_resolved
        }
    
    def add_log(self, message: str) -> None:
        """Add a log message with timestamp."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")


class WorkflowTrigger(BaseModel):
    """Trigger configuration for a workflow."""
    trigger_type: WorkflowTriggerType
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trigger_type": self.trigger_type.value,
            "config": self.config,
            "enabled": self.enabled
        }


class WorkflowDefinition(BaseModel):
    """Definition of a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    version: str = "1.0.0"
    steps: Dict[str, WorkflowStep]
    start_step_id: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    triggers: List[WorkflowTrigger] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    timeout_seconds: Optional[int] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    created_by: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "start_step_id": self.start_step_id,
            "variables": self.variables,
            "triggers": [t.to_dict() for t in self.triggers],
            "tags": self.tags,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "metadata": self.metadata
        }


class WorkflowExecution(BaseModel):
    """Execution instance of a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    step_executions: Dict[str, WorkflowStepExecution] = Field(default_factory=dict)
    current_step_ids: List[str] = Field(default_factory=list)
    completed_step_ids: List[str] = Field(default_factory=list)
    failed_step_ids: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    trigger_info: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "step_executions": {k: v.to_dict() for k, v in self.step_executions.items()},
            "current_step_ids": self.current_step_ids,
            "completed_step_ids": self.completed_step_ids,
            "failed_step_ids": self.failed_step_ids,
            "variables": self.variables,
            "metrics": self.metrics,
            "trigger_info": self.trigger_info,
            "logs": self.logs
        }
    
    def add_log(self, message: str) -> None:
        """Add a log message with timestamp."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")


class WorkflowListener:
    """Interface for workflow event listeners."""
    
    async def on_workflow_started(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is started."""
        pass
    
    async def on_workflow_completed(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is completed successfully."""
        pass
    
    async def on_workflow_failed(self, execution: WorkflowExecution, error: str) -> None:
        """Called when a workflow fails."""
        pass
    
    async def on_workflow_paused(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is paused."""
        pass
    
    async def on_workflow_resumed(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is resumed."""
        pass
    
    async def on_workflow_canceled(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is canceled."""
        pass
    
    async def on_step_started(self, execution: WorkflowExecution, step_id: str) -> None:
        """Called when a step is started."""
        pass
    
    async def on_step_completed(self, execution: WorkflowExecution, step_id: str, output: StepOutput) -> None:
        """Called when a step is completed successfully."""
        pass
    
    async def on_step_failed(self, execution: WorkflowExecution, step_id: str, error: str) -> None:
        """Called when a step fails."""
        pass


class WorkflowEngine:
    """Engine for executing and managing workflows."""
    
    def __init__(
        self,
        task_scheduler: Any = None,
        agent_registry: Any = None,
        shared_memory: Any = None,
        max_concurrent_workflows: int = 10,
        max_concurrent_steps: int = 50,
        default_timeout: int = 3600,
        listeners: Optional[List[WorkflowListener]] = None,
        step_handlers: Optional[Dict[WorkflowStepType, Callable]] = None,
        workflow_store_path: Optional[str] = None,
        enable_persistence: bool = True,
    ):
        """Initialize the workflow engine.
        
        Args:
            task_scheduler: Task scheduler for executing tasks
            agent_registry: Registry of available agents
            shared_memory: Optional shared memory interface
            max_concurrent_workflows: Maximum number of workflows to run concurrently
            max_concurrent_steps: Maximum number of steps to run concurrently
            default_timeout: Default workflow timeout in seconds
            listeners: Optional list of workflow event listeners
            step_handlers: Optional custom handlers for step types
            workflow_store_path: Path to store workflow definitions and executions
            enable_persistence: Whether to enable persistence of workflow state
        """
        self.task_scheduler = task_scheduler
        self.agent_registry = agent_registry
        self.shared_memory = shared_memory
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_concurrent_steps = max_concurrent_steps
        self.default_timeout = default_timeout
        self.listeners = listeners or []
        self.workflow_store_path = workflow_store_path
        self.enable_persistence = enable_persistence
        
        # Workflow storage
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.running_steps: Dict[str, asyncio.Task] = {}
        
        # Set up step handlers
        self.step_handlers = {
            WorkflowStepType.TASK: self._handle_task_step,
            WorkflowStepType.CONDITIONAL: self._handle_conditional_step,
            WorkflowStepType.PARALLEL: self._handle_parallel_step,
            WorkflowStepType.SEQUENCE: self._handle_sequence_step,
            WorkflowStepType.WAIT: self._handle_wait_step,
            WorkflowStepType.LOOP: self._handle_loop_step,
            WorkflowStepType.HUMAN_INPUT: self._handle_human_input_step,
            WorkflowStepType.APPROVAL: self._handle_approval_step,
            WorkflowStepType.NOTIFICATION: self._handle_notification_step,
            WorkflowStepType.CODE_EXECUTION: self._handle_code_execution_step,
            WorkflowStepType.SUBWORKFLOW: self._handle_subworkflow_step,
        }
        
        # Override with custom handlers if provided
        if step_handlers:
            self.step_handlers.update(step_handlers)
        
        # Trigger tracking
        self.event_triggers: Dict[str, List[str]] = defaultdict(list)  # event -> list of workflow IDs
        self.scheduled_triggers: Dict[str, Dict[str, Any]] = {}  # workflow_id -> schedule info
        
        # Locking
        self._workflow_lock = threading.RLock()
        self._step_lock = threading.RLock()
        
        # Management
        self._running = False
        self._trigger_checker_task = None
        
        # Execution history (limited to recent executions)
        self.execution_history_limit = 1000
        self.execution_history = deque(maxlen=self.execution_history_limit)
        
        logger.info("Workflow engine initialized")
        
        # Load stored workflows if persistence is enabled
        if self.enable_persistence and self.workflow_store_path:
            self._load_stored_workflows()
    
    def _load_stored_workflows(self) -> None:
        """Load stored workflow definitions from disk."""
        if not self.workflow_store_path:
            return
        
        import os
        from pathlib import Path
        
        definitions_path = Path(self.workflow_store_path) / "definitions"
        if definitions_path.exists():
            for file_path in definitions_path.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        workflow_data = json.load(f)
                        
                    # Convert steps to the correct format
                    if "steps" in workflow_data:
                        steps = {}
                        for step_id, step_data in workflow_data["steps"].items():
                            if "step_type" in step_data and isinstance(step_data["step_type"], str):
                                step_data["step_type"] = WorkflowStepType(step_data["step_type"])
                            
                            # Convert inputs
                            if "inputs" in step_data:
                                step_data["inputs"] = {
                                    k: StepInput(**v) if isinstance(v, dict) else v
                                    for k, v in step_data["inputs"].items()
                                }
                            
                            steps[step_id] = WorkflowStep(**step_data)
                        workflow_data["steps"] = steps
                    
                    # Convert triggers
                    if "triggers" in workflow_data:
                        triggers = []
                        for trigger_data in workflow_data["triggers"]:
                            if "trigger_type" in trigger_data and isinstance(trigger_data["trigger_type"], str):
                                trigger_data["trigger_type"] = WorkflowTriggerType(trigger_data["trigger_type"])
                            triggers.append(WorkflowTrigger(**trigger_data))
                        workflow_data["triggers"] = triggers
                    
                    # Create workflow definition
                    workflow = WorkflowDefinition(**workflow_data)
                    self.workflow_definitions[workflow.id] = workflow
                    
                    # Register any triggers
                    self._register_workflow_triggers(workflow)
                    
                    logger.info(f"Loaded workflow definition: {workflow.name} ({workflow.id})")
                except Exception as e:
                    logger.error(f"Error loading workflow from {file_path}: {str(e)}")
    
    def _register_workflow_triggers(self, workflow: WorkflowDefinition) -> None:
        """Register triggers for a workflow."""
        for trigger in workflow.triggers:
            if not trigger.enabled:
                continue
            
            if trigger.trigger_type == WorkflowTriggerType.EVENT:
                event_type = trigger.config.get("event_type")
                if event_type:
                    self.event_triggers[event_type].append(workflow.id)
            
            elif trigger.trigger_type == WorkflowTriggerType.SCHEDULED:
                schedule = trigger.config.get("schedule")
                if schedule:
                    # Store for scheduled execution
                    self.scheduled_triggers[workflow.id] = {
                        "schedule": schedule,
                        "next_run": self._calculate_next_run(schedule),
                        "trigger": trigger
                    }
            
            elif trigger.trigger_type == WorkflowTriggerType.PERIODIC:
                interval_seconds = trigger.config.get("interval_seconds")
                if interval_seconds:
                    # Store for periodic execution
                    self.scheduled_triggers[workflow.id] = {
                        "interval": interval_seconds,
                        "next_run": datetime.now() + timedelta(seconds=interval_seconds),
                        "trigger": trigger
                    }
            
            elif trigger.trigger_type == WorkflowTriggerType.TASK_COMPLETION:
                # We'll handle this dynamically when tasks complete
                pass
    
    def _calculate_next_run(self, schedule: Dict[str, Any]) -> datetime:
        """Calculate the next run time based on a schedule."""
        # Simplified implementation - a real one would use cron-like parsing
        now = datetime.now()
        
        # Check for specific time of day
        if "hour" in schedule and "minute" in schedule:
            hour = schedule["hour"]
            minute = schedule["minute"]
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If the time is in the past, schedule for tomorrow
            if next_run < now:
                next_run = next_run + timedelta(days=1)
            
            return next_run
        
        # Default: run in 1 hour
        return now + timedelta(hours=1)
    
    def start(self) -> None:
        """Start the workflow engine."""
        if self._running:
            return
        
        self._running = True
        self._trigger_checker_task = asyncio.create_task(self._trigger_checker_loop())
        
        logger.info("Workflow engine started")
    
    def stop(self) -> None:
        """Stop the workflow engine."""
        if not self._running:
            return
        
        self._running = False
        
        if self._trigger_checker_task:
            self._trigger_checker_task.cancel()
            self._trigger_checker_task = None
        
        # Cancel all running workflows
        for workflow_id, task in list(self.running_workflows.items()):
            task.cancel()
            self.running_workflows.pop(workflow_id, None)
        
        # Cancel all running steps
        for step_id, task in list(self.running_steps.items()):
            task.cancel()
            self.running_steps.pop(step_id, None)
        
        logger.info("Workflow engine stopped")
    
    async def _trigger_checker_loop(self) -> None:
        """Background task that checks for workflow triggers."""
        while self._running:
            try:
                await self._check_scheduled_triggers()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trigger checker loop: {str(e)}")
                await asyncio.sleep(10)
    
    async def _check_scheduled_triggers(self) -> None:
        """Check for scheduled triggers that need to be executed."""
        now = datetime.now()
        workflows_to_trigger = []
        
        # Find triggers that need to be executed
        for workflow_id, info in list(self.scheduled_triggers.items()):
            if "next_run" in info and info["next_run"] <= now:
                workflows_to_trigger.append({
                    "workflow_id": workflow_id, 
                    "trigger_info": info
                })
                
                # Update next run time
                if "interval" in info:
                    # For periodic triggers
                    info["next_run"] = now + timedelta(seconds=info["interval"])
                elif "schedule" in info:
                    # For scheduled triggers
                    info["next_run"] = self._calculate_next_run(info["schedule"])
        
        # Trigger workflows
        for trigger_info in workflows_to_trigger:
            workflow_id = trigger_info["workflow_id"]
            info = trigger_info["trigger_info"]
            
            trigger_data = {
                "type": "scheduled" if "schedule" in info else "periodic",
                "time": now.isoformat(),
                "details": info.get("schedule") if "schedule" in info else {"interval": info.get("interval")}
            }
            
            await self.execute_workflow(
                workflow_id=workflow_id,
                trigger_info=trigger_data
            )
    
    async def register_workflow(self, workflow: Union[WorkflowDefinition, Dict[str, Any]]) -> str:
        """Register a new workflow definition.
        
        Args:
            workflow: The workflow definition to register
            
        Returns:
            ID of the registered workflow
        """
        # Convert dictionary to WorkflowDefinition if needed
        if isinstance(workflow, dict):
            # Convert step types
            if "steps" in workflow:
                for step_data in workflow["steps"].values():
                    if "step_type" in step_data and isinstance(step_data["step_type"], str):
                        step_data["step_type"] = WorkflowStepType(step_data["step_type"])
            
            # Convert trigger types
            if "triggers" in workflow:
                for trigger_data in workflow["triggers"]:
                    if "trigger_type" in trigger_data and isinstance(trigger_data["trigger_type"], str):
                        trigger_data["trigger_type"] = WorkflowTriggerType(trigger_data["trigger_type"])
            
            workflow = WorkflowDefinition(**workflow)
        
        # Validate the workflow
        self._validate_workflow(workflow)
        
        # Store the workflow
        with self._workflow_lock:
            self.workflow_definitions[workflow.id] = workflow
            
            # Register any triggers
            self._register_workflow_triggers(workflow)
        
        # Persist the workflow if enabled
        if self.enable_persistence and self.workflow_store_path:
            await self._persist_workflow(workflow)
        
        logger.info(f"Registered workflow: {workflow.name} ({workflow.id})")
        return workflow.id
    
    def _validate_workflow(self, workflow: WorkflowDefinition) -> None:
        """Validate a workflow definition.
        
        Args:
            workflow: The workflow to validate
            
        Raises:
            ValueError: If the workflow is invalid
        """
        # Check that the start step exists
        if workflow.start_step_id not in workflow.steps:
            raise ValueError(f"Start step {workflow.start_step_id} not found in workflow steps")
        
        # Check that all next_steps exist
        for step_id, step in workflow.steps.items():
            for next_step_id in step.next_steps:
                if next_step_id not in workflow.steps:
                    raise ValueError(f"Next step {next_step_id} for step {step_id} not found in workflow steps")
            
            # Check on_failure and on_timeout if set
            if step.on_failure and step.on_failure not in workflow.steps:
                raise ValueError(f"Failure step {step.on_failure} for step {step_id} not found in workflow steps")
            
            if step.on_timeout and step.on_timeout not in workflow.steps:
                raise ValueError(f"Timeout step {step.on_timeout} for step {step_id} not found in workflow steps")
        
        # Check for cycles (simplified check - a more thorough check would use graph algorithms)
        visited = set()
        
        def check_cycles(step_id: str, path: List[str]) -> None:
            if step_id in path:
                raise ValueError(f"Cycle detected in workflow: {' -> '.join(path + [step_id])}")
            
            if step_id in visited:
                return
            
            visited.add(step_id)
            step = workflow.steps.get(step_id)
            if not step:
                return
            
            for next_step_id in step.next_steps:
                check_cycles(next_step_id, path + [step_id])
        
        check_cycles(workflow.start_step_id, [])
    
    async def _persist_workflow(self, workflow: WorkflowDefinition) -> None:
        """Persist a workflow definition to storage.
        
        Args:
            workflow: The workflow to persist
        """
        if not self.workflow_store_path:
            return
        
        import os
        from pathlib import Path
        
        # Create directory if it doesn't exist
        definitions_path = Path(self.workflow_store_path) / "definitions"
        os.makedirs(definitions_path, exist_ok=True)
        
        # Write workflow to file
        file_path = definitions_path / f"{workflow.id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(workflow.to_dict(), f, indent=2)
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        input_variables: Optional[Dict[str, Any]] = None,
        trigger_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            input_variables: Optional input variables for the workflow
            trigger_info: Optional trigger information
            
        Returns:
            ID of the workflow execution
            
        Raises:
            ValueError: If the workflow is not found
        """
        # Get the workflow definition
        workflow_def = self.workflow_definitions.get(workflow_id)
        if not workflow_def:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Check if we can start more workflows
        with self._workflow_lock:
            if len(self.running_workflows) >= self.max_concurrent_workflows:
                raise ValueError("Maximum number of concurrent workflows reached")
        
        # Create execution instance
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            variables=dict(workflow_def.variables),  # Clone workflow variables
            trigger_info=trigger_info or {}
        )
        
        # Add any input variables
        if input_variables:
            execution.variables.update(input_variables)
        
        # Store execution
        with self._workflow_lock:
            self.workflow_executions[execution.id] = execution
        
        # Start execution
        execution.status = WorkflowStatus.RUNNING
        execution.add_log(f"Starting workflow {workflow_def.name}")
        
        # Create task to execute the workflow
        workflow_task = asyncio.create_task(
            self._execute_workflow(execution.id)
        )
        
        # Store running workflow
        with self._workflow_lock:
            self.running_workflows[execution.id] = workflow_task
        
        # Persist execution state if enabled
        if self.enable_persistence and self.workflow_store_path:
            await self._persist_execution_state(execution)
        
        # Notify listeners
        for listener in self.listeners:
            try:
                await listener.on_workflow_started(execution)
            except Exception as e:
                logger.error(f"Error notifying listener of workflow start: {str(e)}")
        
        logger.info(f"Started workflow execution: {execution.id} (workflow: {workflow_def.name})")
        return execution.id
    
    async def _persist_execution_state(self, execution: WorkflowExecution) -> None:
        """Persist a workflow execution state to storage.
        
        Args:
            execution: The execution to persist
        """
        if not self.workflow_store_path:
            return
        
        import os
        from pathlib import Path
        
        # Create directory if it doesn't exist
        executions_path = Path(self.workflow_store_path) / "executions"
        os.makedirs(executions_path, exist_ok=True)
        
        # Write execution to file
        file_path = executions_path / f"{execution.id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(execution.to_dict(), f, indent=2)
    
    async def _execute_workflow(self, execution_id: str) -> None:
        """Execute a workflow and process its result.
        
        Args:
            execution_id: ID of the execution to run
        """
        execution = None
        
        try:
            with self._workflow_lock:
                if execution_id not in self.workflow_executions:
                    logger.warning(f"Workflow execution {execution_id} not found, cannot execute")
                    return
                
                execution = self.workflow_executions[execution_id]
                workflow_def = self.workflow_definitions.get(execution.workflow_id)
                
                if not workflow_def:
                    raise ValueError(f"Workflow definition {execution.workflow_id} not found")
                
                # Start with the initial step
                execution.current_step_ids = [workflow_def.start_step_id]
            
            # Execute the workflow
            # Set timeout for the entire workflow if specified
            timeout = workflow_def.timeout_seconds or self.default_timeout
            
            try:
                # Set a timeout for the entire workflow
                await asyncio.wait_for(
                    self._execute_workflow_steps(execution, workflow_def),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                execution.add_log(f"Workflow execution timed out after {timeout} seconds")
                execution.status = WorkflowStatus.FAILED
                
                # Notify listeners of failure
                for listener in self.listeners:
                    try:
                        await listener.on_workflow_failed(
                            execution, 
                            f"Workflow timed out after {timeout} seconds"
                        )
                    except Exception as e:
                        logger.error(f"Error notifying listener of workflow timeout: {str(e)}")
            
            # Check final status
            if execution.status == WorkflowStatus.RUNNING:
                # If we got here without another status, we completed successfully
                execution.status = WorkflowStatus.COMPLETED
                execution.end_time = datetime.now().isoformat()
                
                execution.add_log("Workflow completed successfully")
                
                # Calculate metrics
                if execution.start_time:
                    start_time = datetime.fromisoformat(execution.start_time)
                    end_time = datetime.fromisoformat(execution.end_time)
                    duration = (end_time - start_time).total_seconds()
                    execution.metrics["duration_seconds"] = duration
                
                execution.metrics["total_steps"] = len(execution.step_executions)
                execution.metrics["successful_steps"] = len(execution.completed_step_ids)
                execution.metrics["failed_steps"] = len(execution.failed_step_ids)
                
                # Notify listeners of success
                for listener in self.listeners:
                    try:
                        await listener.on_workflow_completed(execution)
                    except Exception as e:
                        logger.error(f"Error notifying listener of workflow completion: {str(e)}")
            
            # Add to execution history
            self.execution_history.append({
                "id": execution.id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "metrics": execution.metrics
            })
        
        except Exception as e:
            logger.error(f"Error executing workflow {execution_id}: {str(e)}")
            
            if execution:
                execution.status = WorkflowStatus.FAILED
                execution.end_time = datetime.now().isoformat()
                execution.add_log(f"Workflow failed with error: {str(e)}")
                
                # Notify listeners of failure
                for listener in self.listeners:
                    try:
                        await listener.on_workflow_failed(execution, str(e))
                    except Exception as listener_error:
                        logger.error(f"Error notifying listener of workflow failure: {str(listener_error)}")
        
        finally:
            # Clean up
            with self._workflow_lock:
                if execution_id in self.running_workflows:
                    del self.running_workflows[execution_id]
            
            # Persist final state if enabled
            if execution and self.enable_persistence and self.workflow_store_path:
                await self._persist_execution_state(execution)
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution, workflow_def: WorkflowDefinition) -> None:
        """Execute the steps of a workflow.
        
        Args:
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        # Continue executing steps until we're done or failed
        while execution.status == WorkflowStatus.RUNNING and execution.current_step_ids:
            # Get the current step IDs to execute
            step_ids = list(execution.current_step_ids)
            execution.current_step_ids = []
            
            # Execute steps (potentially in parallel)
            if len(step_ids) == 1:
                # Single step - execute directly
                next_step_ids = await self._execute_step(step_ids[0], execution, workflow_def)
                execution.current_step_ids.extend(next_step_ids)
            else:
                # Multiple steps - execute in parallel
                tasks = [
                    self._execute_step(step_id, execution, workflow_def)
                    for step_id in step_ids
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and add next steps
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # Step execution failed
                        logger.error(f"Parallel step {step_ids[i]} failed: {str(result)}")
                        execution.failed_step_ids.append(step_ids[i])
                    else:
                        # Add next steps
                        execution.current_step_ids.extend(result)
            
            # Check if we have no more steps to execute
            if not execution.current_step_ids:
                break
    
    async def _execute_step(
        self, 
        step_id: str, 
        execution: WorkflowExecution, 
        workflow_def: WorkflowDefinition
    ) -> List[str]:
        """Execute a single workflow step.
        
        Args:
            step_id: ID of the step to execute
            execution: The workflow execution instance
            workflow_def: The workflow definition
            
        Returns:
            List of next step IDs to execute
            
        Raises:
            ValueError: If the step is not found or has an invalid type
        """
        # Get the step definition
        step_def = workflow_def.steps.get(step_id)
        if not step_def:
            raise ValueError(f"Step {step_id} not found in workflow {workflow_def.id}")
        
        # Check if we've already executed this step
        if step_id in execution.step_executions:
            step_exec = execution.step_executions[step_id]
            if step_exec.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]:
                # Step already completed, return its next steps
                return step_exec.next_steps_resolved
            if step_exec.status == StepStatus.FAILED:
                # Step already failed, check if there's an on_failure step
                if step_def.on_failure:
                    return [step_def.on_failure]
                return []
        
        # Check if we can run more steps
        with self._step_lock:
            if len(self.running_steps) >= self.max_concurrent_steps:
                # Too many concurrent steps, return this step to try again later
                return [step_id]
        
        # Create step execution if it doesn't exist
        step_exec = execution.step_executions.get(step_id)
        if not step_exec:
            max_attempts = step_def.retry_config.get("max_attempts", 1)
            step_exec = WorkflowStepExecution(
                step_definition=step_def,
                max_attempts=max_attempts
            )
            execution.step_executions[step_id] = step_exec
        
        # Resolve inputs
        await self._resolve_step_inputs(step_exec, execution, workflow_def)
        
        # Mark as running
        step_exec.status = StepStatus.RUNNING
        step_exec.start_time = datetime.now().isoformat()
        step_exec.add_log(f"Starting step {step_def.name}")
        
        # Notify listeners
        for listener in self.listeners:
            try:
                await listener.on_step_started(execution, step_id)
            except Exception as e:
                logger.error(f"Error notifying listener of step start: {str(e)}")
        
        # Create task to execute the step
        step_task = asyncio.create_task(
            self._execute_step_with_timeout(step_exec, execution, workflow_def)
        )
        
        # Store running step
        with self._step_lock:
            self.running_steps[f"{execution.id}:{step_id}"] = step_task
        
        try:
            # Wait for step execution to complete
            await step_task
        finally:
            # Clean up
            with self._step_lock:
                self.running_steps.pop(f"{execution.id}:{step_id}", None)
        
        # Check result
        if step_exec.status == StepStatus.COMPLETED:
            # Add to completed steps
            execution.completed_step_ids.append(step_id)
            
            # Get next steps
            if step_exec.next_steps_resolved:
                return step_exec.next_steps_resolved
            
            # Default to step's next_steps
            return step_def.next_steps
        
        elif step_exec.status == StepStatus.FAILED:
            # Add to failed steps
            execution.failed_step_ids.append(step_id)
            
            # Check if we should retry
            if step_exec.attempt < step_exec.max_attempts:
                # Retry the step
                step_exec.attempt += 1
                step_exec.status = StepStatus.PENDING
                step_exec.add_log(f"Retrying step (attempt {step_exec.attempt}/{step_exec.max_attempts})")
                
                # Add back to current steps to retry
                return [step_id]
            
            # Check if there's an on_failure step
            if step_def.on_failure:
                return [step_def.on_failure]
        
        # No next steps
        return []
    
    async def _resolve_step_inputs(self, step_exec: WorkflowStepExecution, execution: WorkflowExecution, workflow_def: WorkflowDefinition) -> None:
        """Resolve the inputs for a step.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        resolved_inputs = {}
        
        for input_name, input_def in step_exec.step_definition.inputs.items():
            value = None
            
            if input_def.source == "static":
                # Static value
                value = input_def.value
            
            elif input_def.source == "previous_step" and input_def.key:
                # Value from a previous step
                step_id, output_key = input_def.key.split(".", 1)
                if step_id in execution.step_executions:
                    previous_step = execution.step_executions[step_id]
                    if previous_step.outputs and output_key in previous_step.outputs.result:
                        value = previous_step.outputs.result[output_key]
            
            elif input_def.source == "workflow_variable" and input_def.key:
                # Value from workflow variables
                if input_def.key in execution.variables:
                    value = execution.variables[input_def.key]
            
            elif input_def.source == "global_variable" and input_def.key:
                # Value from global variables (e.g., from shared memory)
                if self.shared_memory:
                    value = await self.shared_memory.retrieve(
                        key=input_def.key,
                        category="global_variables"
                    )
            
            resolved_inputs[input_name] = value
        
        step_exec.inputs_resolved = resolved_inputs
    
    async def _execute_step_with_timeout(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Execute a step with timeout handling.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get timeout
        timeout = step_def.timeout_seconds or 300  # Default: 5 minutes
        
        try:
            # Execute with timeout
            start_time = time.time()
            
            try:
                # Get the appropriate handler for the step type
                handler = self.step_handlers.get(step_def.step_type)
                if not handler:
                    raise ValueError(f"No handler found for step type {step_def.step_type}")
                
                # Execute the handler with timeout
                await asyncio.wait_for(
                    handler(step_exec, execution, workflow_def),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                # Handle success
                if step_exec.status != StepStatus.FAILED:  # Handler might have already set it to FAILED
                    step_exec.status = StepStatus.COMPLETED
                    step_exec.end_time = datetime.now().isoformat()
                    
                    # Create output if not already set
                    if not step_exec.outputs:
                        step_exec.outputs = StepOutput(
                            status=step_exec.status,
                            result={},
                            execution_time=execution_time
                        )
                    
                    step_exec.add_log(f"Step completed successfully in {execution_time:.2f}s")
                    
                    # Notify listeners
                    for listener in self.listeners:
                        try:
                            await listener.on_step_completed(execution, step_def.id, step_exec.outputs)
                        except Exception as e:
                            logger.error(f"Error notifying listener of step completion: {str(e)}")
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                
                # Handle timeout
                step_exec.status = StepStatus.FAILED
                step_exec.end_time = datetime.now().isoformat()
                
                error_message = f"Step timed out after {timeout} seconds"
                step_exec.outputs = StepOutput(
                    status=StepStatus.FAILED,
                    result=None,
                    error=error_message,
                    execution_time=execution_time
                )
                
                step_exec.add_log(error_message)
                
                # Check if there's an on_timeout step
                if step_def.on_timeout:
                    step_exec.next_steps_resolved = [step_def.on_timeout]
                
                # Notify listeners
                for listener in self.listeners:
                    try:
                        await listener.on_step_failed(execution, step_def.id, error_message)
                    except Exception as e:
                        logger.error(f"Error notifying listener of step timeout: {str(e)}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle error
            step_exec.status = StepStatus.FAILED
            step_exec.end_time = datetime.now().isoformat()
            
            error_message = str(e)
            step_exec.outputs = StepOutput(
                status=StepStatus.FAILED,
                result=None,
                error=error_message,
                execution_time=execution_time
            )
            
            step_exec.add_log(f"Step failed with error: {error_message}")
            
            # Notify listeners
            for listener in self.listeners:
                try:
                    await listener.on_step_failed(execution, step_def.id, error_message)
                except Exception as listener_error:
                    logger.error(f"Error notifying listener of step failure: {str(listener_error)}")
    
    # Step type handlers
    
    async def _handle_task_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a task step.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Check for required fields
        if not step_def.action:
            raise ValueError(f"Task step {step_def.id} is missing required action field")
        
        # Check if we have a task scheduler
        if not self.task_scheduler:
            raise ValueError("Task scheduler is required to execute task steps")
        
        # Create task definition
        task_def = {
            "title": step_def.name,
            "description": step_def.description,
            "agent_id": step_def.agent_id or "auto",
            "agent_type": step_def.metadata.get("agent_type", ""),
            "action": step_def.action,
            "params": step_exec.inputs_resolved,
            "priority": step_def.metadata.get("priority", "medium"),
            "tags": step_def.metadata.get("tags", []),
            "category": step_def.metadata.get("category", "default")
        }
        
        # Schedule the task
        step_exec.add_log(f"Scheduling task: {step_def.action}")
        task_id = await self.task_scheduler.schedule_task(task_def)
        
        # Wait for task completion
        step_exec.add_log(f"Waiting for task {task_id} to complete")
        
        # Poll for completion
        max_wait_time = 3600  # 1 hour maximum wait
        poll_interval = 1.0  # Start with 1 second
        max_poll_interval = 10.0  # Maximum 10 second interval
        wait_time = 0
        
        while wait_time < max_wait_time:
            # Get task status
            status = await self.task_scheduler.get_task_status(task_id)
            
            if status in ["completed", "succeeded"]:
                # Task completed successfully
                result = await self.task_scheduler.get_task_results(task_id)
                
                step_exec.outputs = StepOutput(
                    status=StepStatus.COMPLETED,
                    result=result[0] if result else {},
                    execution_time=0.0
                )
                
                step_exec.add_log(f"Task {task_id} completed successfully")
                return
            
            elif status in ["failed", "rejected", "canceled"]:
                # Task failed
                task_info = await self.task_scheduler.get_task(task_id)
                error_message = "Task failed"
                if task_info and "last_error" in task_info:
                    error_message = task_info["last_error"]
                
                step_exec.outputs = StepOutput(
                    status=StepStatus.FAILED,
                    result=None,
                    error=error_message,
                    execution_time=0.0
                )
                
                step_exec.add_log(f"Task {task_id} failed: {error_message}")
                return
            
            # Wait before checking again
            await asyncio.sleep(poll_interval)
            wait_time += poll_interval
            
            # Exponential backoff with maximum
            poll_interval = min(poll_interval * 1.5, max_poll_interval)
        
        # If we get here, we've waited too long
        raise TimeoutError(f"Timed out waiting for task {task_id} to complete")
    
    async def _handle_conditional_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a conditional step.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Check for condition
        if not step_def.condition:
            raise ValueError(f"Conditional step {step_def.id} is missing required condition field")
        
        # Create evaluation context
        eval_context = {
            "inputs": step_exec.inputs_resolved,
            "variables": execution.variables,
            "step": step_exec,
            "execution": {
                "id": execution.id,
                "workflow_id": execution.workflow_id,
                "status": execution.status,
                "start_time": execution.start_time,
                "current_step_ids": execution.current_step_ids,
                "completed_step_ids": execution.completed_step_ids,
                "failed_step_ids": execution.failed_step_ids
            }
        }
        
        # Safely evaluate the condition
        try:
            result = eval(step_def.condition, {"__builtins__": {}}, eval_context)
            condition_result = bool(result)
        except Exception as e:
            step_exec.add_log(f"Error evaluating condition: {str(e)}")
            raise ValueError(f"Error evaluating condition in step {step_def.id}: {str(e)}")
        
        # Set the output
        step_exec.outputs = StepOutput(
            status=StepStatus.COMPLETED,
            result={"condition_result": condition_result},
            execution_time=0.0
        )
        
        # Determine next steps based on condition result
        if condition_result:
            # Condition is true, use the step's next_steps
            step_exec.next_steps_resolved = step_def.next_steps
            step_exec.add_log(f"Condition evaluated to True, proceeding to next steps: {', '.join(step_def.next_steps)}")
        else:
            # Condition is false, skip next steps
            true_branch = step_def.next_steps
            false_branch = step_def.config.get("false_branch", [])
            
            step_exec.next_steps_resolved = false_branch
            step_exec.add_log(
                f"Condition evaluated to False, skipping steps: {', '.join(true_branch)}, " +
                (f"proceeding to: {', '.join(false_branch)}" if false_branch else "no alternative steps specified")
            )
    
    async def _handle_parallel_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a parallel step that spawns multiple parallel steps.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get parallel steps from config
        parallel_steps = step_def.config.get("steps", [])
        if not parallel_steps:
            step_exec.add_log("No parallel steps defined, skipping")
            return
        
        # Check that all steps exist
        for parallel_step_id in parallel_steps:
            if parallel_step_id not in workflow_def.steps:
                raise ValueError(f"Parallel step {parallel_step_id} not found in workflow steps")
        
        # Set resolved next steps to the parallel steps
        step_exec.next_steps_resolved = parallel_steps
        step_exec.add_log(f"Spawning {len(parallel_steps)} parallel steps: {', '.join(parallel_steps)}")
    
    async def _handle_sequence_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a sequence step that defines a series of steps to be executed in order.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get sequence steps from config
        sequence_steps = step_def.config.get("steps", [])
        if not sequence_steps:
            step_exec.add_log("No sequence steps defined, skipping")
            return
        
        # Check that all steps exist
        for seq_step_id in sequence_steps:
            if seq_step_id not in workflow_def.steps:
                raise ValueError(f"Sequence step {seq_step_id} not found in workflow steps")
        
        # Modify the steps to form a sequence
        # We'll return the first step in the sequence, and update next_steps for each step
        with self._workflow_lock:
            # Save the original next steps of the last step in the sequence
            last_step_id = sequence_steps[-1]
            last_step = workflow_def.steps.get(last_step_id)
            original_next_steps = list(last_step.next_steps) if last_step else []
            
            # Clear next steps for all steps in the sequence
            for i, seq_step_id in enumerate(sequence_steps):
                seq_step = workflow_def.steps.get(seq_step_id)
                if seq_step:
                    if i < len(sequence_steps) - 1:
                        # Point to next step in sequence
                        seq_step.next_steps = [sequence_steps[i + 1]]
                    else:
                        # Last step points to original next steps of the sequence step
                        seq_step.next_steps = step_def.next_steps
        
        # Set resolved next steps to the first step in the sequence
        if sequence_steps:
            step_exec.next_steps_resolved = [sequence_steps[0]]
            step_exec.add_log(f"Starting sequence of {len(sequence_steps)} steps: {', '.join(sequence_steps)}")
        else:
            step_exec.next_steps_resolved = []
    
    async def _handle_wait_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a wait step that pauses for a specified duration.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get wait duration from config or inputs
        duration_seconds = step_def.config.get("duration_seconds", 0)
        
        # Check for duration in inputs
        if "duration_seconds" in step_exec.inputs_resolved:
            try:
                duration_seconds = int(step_exec.inputs_resolved["duration_seconds"])
            except (ValueError, TypeError):
                pass
        
        if duration_seconds <= 0:
            step_exec.add_log("Invalid wait duration, skipping")
            return
        
        step_exec.add_log(f"Waiting for {duration_seconds} seconds")
        await asyncio.sleep(duration_seconds)
        
        step_exec.outputs = StepOutput(
            status=StepStatus.COMPLETED,
            result={"waited_seconds": duration_seconds},
            execution_time=float(duration_seconds)
        )
    
    async def _handle_loop_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a loop step that executes steps repeatedly.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get loop configuration
        loop_steps = step_def.config.get("steps", [])
        iterations = step_def.config.get("iterations", 1)
        loop_variable = step_def.config.get("loop_variable", "loop_index")
        
        # Check for iterations in inputs
        if "iterations" in step_exec.inputs_resolved:
            try:
                iterations = int(step_exec.inputs_resolved["iterations"])
            except (ValueError, TypeError):
                pass
        
        # Check that all steps exist
        for loop_step_id in loop_steps:
            if loop_step_id not in workflow_def.steps:
                raise ValueError(f"Loop step {loop_step_id} not found in workflow steps")
        
        # Handle loop based on current iteration
        current_iteration = step_exec.metadata.get("current_iteration", 0)
        
        if current_iteration == 0:
            # First iteration - set up loop
            step_exec.metadata["current_iteration"] = 1
            step_exec.metadata["max_iterations"] = iterations
            
            # Set the loop variable in workflow variables
            execution.variables[loop_variable] = 1
            
            if loop_steps:
                # Start the loop with the first step
                step_exec.next_steps_resolved = [loop_steps[0]]
                step_exec.add_log(f"Starting loop with {iterations} iterations, first step: {loop_steps[0]}")
                
                # Modify the steps to loop back to this step after completion
                with self._workflow_lock:
                    # The last step in the loop should loop back to this step
                    last_loop_step_id = loop_steps[-1]
                    last_loop_step = workflow_def.steps.get(last_loop_step_id)
                    if last_loop_step:
                        last_loop_step.next_steps = [step_def.id]
                
                # We'll re-execute this step for each iteration
                return
            else:
                step_exec.add_log("No loop steps defined, skipping loop")
        else:
            # Check if we've completed all iterations
            if current_iteration >= iterations:
                # Loop complete, restore original next steps for last loop step
                with self._workflow_lock:
                    last_loop_step_id = loop_steps[-1]
                    last_loop_step = workflow_def.steps.get(last_loop_step_id)
                    if last_loop_step:
                        # Restore next steps to the loop step's next steps
                        last_loop_step.next_steps = step_def.next_steps
                
                step_exec.outputs = StepOutput(
                    status=StepStatus.COMPLETED,
                    result={"iterations_completed": iterations},
                    execution_time=0.0
                )
                
                step_exec.add_log(f"Loop completed after {iterations} iterations")
                
                # Continue with the original next steps
                step_exec.next_steps_resolved = step_def.next_steps
            else:
                # Increment iteration counter
                current_iteration += 1
                step_exec.metadata["current_iteration"] = current_iteration
                
                # Update the loop variable
                execution.variables[loop_variable] = current_iteration
                
                # Continue the loop with the first step
                step_exec.next_steps_resolved = [loop_steps[0]]
                step_exec.add_log(f"Starting iteration {current_iteration}/{iterations}")
    
    async def _handle_human_input_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a human input step that waits for human input.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get human input configuration
        input_type = step_def.config.get("input_type", "text")
        prompt = step_def.config.get("prompt", "Please provide input")
        options = step_def.config.get("options", [])
        timeout_minutes = step_def.config.get("timeout_minutes", 60)
        
        # Create a human input request
        input_request = {
            "workflow_execution_id": execution.id,
            "step_id": step_def.id,
            "input_type": input_type,
            "prompt": prompt,
            "options": options,
            "expires_at": (datetime.now() + timedelta(minutes=timeout_minutes)).isoformat()
        }
        
        # Store the request in shared memory if available
        input_request_id = str(uuid.uuid4())
        if self.shared_memory:
            await self.shared_memory.store(
                key=f"human_input_request_{input_request_id}",
                value=input_request,
                category="human_input_requests"
            )
        
        # Set the step to waiting state
        step_exec.status = StepStatus.WAITING
        step_exec.add_log(f"Waiting for human input: {prompt}")
        
        # Store the input request ID in the step metadata
        step_exec.metadata["input_request_id"] = input_request_id
        
        # The actual waiting and input handling is done by the workflow engine's
        # event handling mechanism. We'll raise an exception to pause execution.
        raise RuntimeError("WAITING_FOR_HUMAN_INPUT")
    
    async def _handle_approval_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of an approval step that waits for human approval.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get approval configuration
        description = step_def.config.get("description", "Please review and approve")
        approvers = step_def.config.get("approvers", [])
        timeout_minutes = step_def.config.get("timeout_minutes", 1440)  # Default: 24 hours
        
        # Create an approval request
        approval_request = {
            "workflow_execution_id": execution.id,
            "step_id": step_def.id,
            "description": description,
            "approvers": approvers,
            "expires_at": (datetime.now() + timedelta(minutes=timeout_minutes)).isoformat()
        }
        
        # Store the request in shared memory if available
        approval_request_id = str(uuid.uuid4())
        if self.shared_memory:
            await self.shared_memory.store(
                key=f"approval_request_{approval_request_id}",
                value=approval_request,
                category="approval_requests"
            )
        
        # Set the step to waiting state
        step_exec.status = StepStatus.WAITING
        step_exec.add_log(f"Waiting for approval: {description}")
        
        # Store the approval request ID in the step metadata
        step_exec.metadata["approval_request_id"] = approval_request_id
        
        # The actual waiting and approval handling is done by the workflow engine's
        # event handling mechanism. We'll raise an exception to pause execution.
        raise RuntimeError("WAITING_FOR_APPROVAL")
    
    async def _handle_notification_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a notification step that sends a notification.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get notification configuration
        notification_type = step_def.config.get("type", "info")
        channel = step_def.config.get("channel", "default")
        recipients = step_def.config.get("recipients", [])
        subject = step_def.config.get("subject", "Workflow Notification")
        message = step_def.config.get("message", "")
        
        # Format subject and message with variables
        try:
            subject = subject.format(**execution.variables)
        except (KeyError, ValueError) as e:
            step_exec.add_log(f"Error formatting subject: {str(e)}")
        
        try:
            message = message.format(**execution.variables)
        except (KeyError, ValueError) as e:
            step_exec.add_log(f"Error formatting message: {str(e)}")
        
        # Create a notification
        notification = {
            "workflow_execution_id": execution.id,
            "step_id": step_def.id,
            "type": notification_type,
            "channel": channel,
            "recipients": recipients,
            "subject": subject,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store the notification in shared memory if available
        if self.shared_memory:
            notification_id = str(uuid.uuid4())
            await self.shared_memory.store(
                key=f"notification_{notification_id}",
                value=notification,
                category="notifications"
            )
        
        step_exec.add_log(f"Sent notification: {subject}")
        
        # Create output
        step_exec.outputs = StepOutput(
            status=StepStatus.COMPLETED,
            result={"notification_sent": True, "subject": subject},
            execution_time=0.0
        )
    
    async def _handle_code_execution_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a code execution step that executes code.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get code configuration
        code = step_def.config.get("code", "")
        language = step_def.config.get("language", "python")
        timeout_seconds = step_def.config.get("timeout_seconds", 30)
        
        # Check if code is provided in inputs
        if "code" in step_exec.inputs_resolved:
            code = step_exec.inputs_resolved["code"]
        
        if not code:
            raise ValueError("No code provided for execution")
        
        step_exec.add_log(f"Executing {language} code")
        
        # Execute the code
        if language.lower() == "python":
            # Execute Python code
            try:
                # Create a restricted execution environment
                local_vars = {
                    "inputs": step_exec.inputs_resolved,
                    "variables": execution.variables,
                    "results": {}
                }
                
                # Execute the code with timeout
                exec_task = asyncio.create_task(
                    self._execute_python_code(code, local_vars)
                )
                
                try:
                    await asyncio.wait_for(exec_task, timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Code execution timed out after {timeout_seconds} seconds")
                
                # Get the results
                results = local_vars.get("results", {})
                
                # Update workflow variables with any values set by the code
                for key, value in local_vars.get("variables", {}).items():
                    execution.variables[key] = value
                
                step_exec.outputs = StepOutput(
                    status=StepStatus.COMPLETED,
                    result=results,
                    execution_time=0.0
                )
                
                step_exec.add_log("Code executed successfully")
                
            except Exception as e:
                raise RuntimeError(f"Error executing Python code: {str(e)}")
        else:
            raise ValueError(f"Unsupported code language: {language}")
    
    async def _execute_python_code(self, code: str, local_vars: Dict[str, Any]) -> None:
        """Execute Python code in a separate thread to allow for async execution.
        
        Args:
            code: Python code to execute
            local_vars: Local variables for execution
        """
        # Execute in a separate thread to not block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: exec(code, {"__builtins__": __builtins__}, local_vars)
        )
    
    async def _handle_subworkflow_step(
        self,
        step_exec: WorkflowStepExecution,
        execution: WorkflowExecution,
        workflow_def: WorkflowDefinition
    ) -> None:
        """Handle execution of a subworkflow step that executes another workflow.
        
        Args:
            step_exec: The step execution instance
            execution: The workflow execution instance
            workflow_def: The workflow definition
        """
        step_def = step_exec.step_definition
        
        # Get subworkflow configuration
        subworkflow_id = step_def.config.get("workflow_id")
        
        # Check if workflow ID is provided in inputs
        if "workflow_id" in step_exec.inputs_resolved:
            subworkflow_id = step_exec.inputs_resolved["workflow_id"]
        
        if not subworkflow_id:
            raise ValueError("No subworkflow ID provided")
        
        # Get variables to pass to subworkflow
        variables = {}
        for key, value in step_exec.inputs_resolved.items():
            # Filter out special keys
            if key != "workflow_id":
                variables[key] = value
        
        # Add variables from config
        config_variables = step_def.config.get("variables", {})
        variables.update(config_variables)
        
        step_exec.add_log(f"Executing subworkflow: {subworkflow_id}")
        
        # Execute the subworkflow
        try:
            sub_execution_id = await self.execute_workflow(
                workflow_id=subworkflow_id,
                input_variables=variables,
                trigger_info={
                    "type": "subworkflow",
                    "parent_workflow_id": execution.workflow_id,
                    "parent_execution_id": execution.id,
                    "parent_step_id": step_def.id
                }
            )
            
            step_exec.metadata["subworkflow_execution_id"] = sub_execution_id
            
            # Wait for subworkflow to complete
            await self._wait_for_subworkflow(sub_execution_id, step_exec)
            
        except Exception as e:
            raise RuntimeError(f"Error executing subworkflow: {str(e)}")
    
    async def _wait_for_subworkflow(self, execution_id: str, step_exec: WorkflowStepExecution) -> None:
        """Wait for a subworkflow to complete.
        
        Args:
            execution_id: ID of the subworkflow execution
            step_exec: The parent step execution instance
        """
        # Poll for completion
        max_wait_time = 3600  # 1 hour maximum wait
        poll_interval = 1.0  # Start with 1 second
        max_poll_interval = 10.0  # Maximum 10 second interval
        wait_time = 0
        
        while wait_time < max_wait_time:
            # Get execution status
            with self._workflow_lock:
                sub_execution = self.workflow_executions.get(execution_id)
                
                if sub_execution:
                    if sub_execution.status == WorkflowStatus.COMPLETED:
                        # Subworkflow completed successfully
                        step_exec.outputs = StepOutput(
                            status=StepStatus.COMPLETED,
                            result={
                                "subworkflow_id": sub_execution.workflow_id,
                                "execution_id": execution_id,
                                "variables": sub_execution.variables
                            },
                            execution_time=0.0
                        )
                        
                        step_exec.add_log(f"Subworkflow {execution_id} completed successfully")
                        return
                    
                    elif sub_execution.status in [WorkflowStatus.FAILED, WorkflowStatus.CANCELED]:
                        # Subworkflow failed
                        error_message = "Subworkflow failed or was canceled"
                        if sub_execution.logs:
                            # Get the last error from logs
                            for log in reversed(sub_execution.logs):
                                if "failed" in log.lower() or "error" in log.lower():
                                    error_message = log
                                    break
                        
                        step_exec.outputs = StepOutput(
                            status=StepStatus.FAILED,
                            result=None,
                            error=error_message,
                            execution_time=0.0
                        )
                        
                        step_exec.add_log(f"Subworkflow {execution_id} failed: {error_message}")
                        return
            
            # Wait before checking again
            await asyncio.sleep(poll_interval)
            wait_time += poll_interval
            
            # Exponential backoff with maximum
            poll_interval = min(poll_interval * 1.5, max_poll_interval)
        
        # If we get here, we've waited too long
        raise TimeoutError(f"Timed out waiting for subworkflow {execution_id} to complete")
    
    async def handle_human_input(
        self, 
        input_request_id: str, 
        input_value: Any
    ) -> bool:
        """Handle human input for a waiting step.
        
        Args:
            input_request_id: ID of the input request
            input_value: The input value provided
            
        Returns:
            True if the input was successfully handled, False otherwise
        """
        # Find the waiting workflow and step
        for execution_id, execution in self.workflow_executions.items():
            if execution.status == WorkflowStatus.WAITING:
                for step_id, step_exec in execution.step_executions.items():
                    if (step_exec.status == StepStatus.WAITING and 
                            step_exec.metadata.get("input_request_id") == input_request_id):
                        
                        # Found the waiting step
                        workflow_def = self.workflow_definitions.get(execution.workflow_id)
                        if not workflow_def:
                            logger.error(f"Workflow definition {execution.workflow_id} not found")
                            return False
                        
                        # Set the output
                        step_exec.outputs = StepOutput(
                            status=StepStatus.COMPLETED,
                            result={"input_value": input_value},
                            execution_time=0.0
                        )
                        
                        step_exec.status = StepStatus.COMPLETED
                        step_exec.end_time = datetime.now().isoformat()
                        
                        # Set next steps
                        step_def = workflow_def.steps.get(step_id)
                        if step_def:
                            step_exec.next_steps_resolved = step_def.next_steps
                        
                        step_exec.add_log(f"Received human input: {input_value}")
                        
                        # Resume the workflow execution
                        execution.status = WorkflowStatus.RUNNING
                        execution.current_step_ids.extend(step_exec.next_steps_resolved)
                        
                        # Create task to continue execution
                        workflow_task = asyncio.create_task(
                            self._execute_workflow(execution_id)
                        )
                        
                        # Store running workflow
                        with self._workflow_lock:
                            self.running_workflows[execution_id] = workflow_task
                        
                        return True
        
        logger.warning(f"No waiting step found for input request {input_request_id}")
        return False
    
    async def handle_approval(
        self, 
        approval_request_id: str, 
        approved: bool, 
        comment: Optional[str] = None
    ) -> bool:
        """Handle approval decision for a waiting step.
        
        Args:
            approval_request_id: ID of the approval request
            approved: Whether the request was approved
            comment: Optional comment with the decision
            
        Returns:
            True if the approval was successfully handled, False otherwise
        """
        # Find the waiting workflow and step
        for execution_id, execution in self.workflow_executions.items():
            if execution.status == WorkflowStatus.WAITING:
                for step_id, step_exec in execution.step_executions.items():
                    if (step_exec.status == StepStatus.WAITING and 
                            step_exec.metadata.get("approval_request_id") == approval_request_id):
                        
                        # Found the waiting step
                        workflow_def = self.workflow_definitions.get(execution.workflow_id)
                        if not workflow_def:
                            logger.error(f"Workflow definition {execution.workflow_id} not found")
                            return False
                        
                        # Get step definition
                        step_def = workflow_def.steps.get(step_id)
                        if not step_def:
                            logger.error(f"Step definition {step_id} not found")
                            return False
                        
                        # Handle the approval decision
                        if approved:
                            # Approval granted
                            step_exec.outputs = StepOutput(
                                status=StepStatus.COMPLETED,
                                result={
                                    "approved": True,
                                    "comment": comment
                                },
                                execution_time=0.0
                            )
                            
                            step_exec.status = StepStatus.COMPLETED
                            step_exec.end_time = datetime.now().isoformat()
                            
                            # Use the step's next steps
                            step_exec.next_steps_resolved = step_def.next_steps
                            
                            step_exec.add_log(f"Approval granted: {comment or 'No comment'}")
                        else:
                            # Approval denied
                            step_exec.outputs = StepOutput(
                                status=StepStatus.FAILED,
                                result={
                                    "approved": False,
                                    "comment": comment
                                },
                                error="Approval denied",
                                execution_time=0.0
                            )
                            
                            step_exec.status = StepStatus.FAILED
                            step_exec.end_time = datetime.now().isoformat()
                            
                            # Use on_failure steps if defined
                            if step_def.on_failure:
                                step_exec.next_steps_resolved = [step_def.on_failure]
                            else:
                                step_exec.next_steps_resolved = []
                            
                            step_exec.add_log(f"Approval denied: {comment or 'No comment'}")
                        
                        # Resume the workflow execution
                        execution.status = WorkflowStatus.RUNNING
                        execution.current_step_ids.extend(step_exec.next_steps_resolved)
                        
                        # Create task to continue execution
                        workflow_task = asyncio.create_task(
                            self._execute_workflow(execution_id)
                        )
                        
                        # Store running workflow
                        with self._workflow_lock:
                            self.running_workflows[execution_id] = workflow_task
                        
                        return True
        
        logger.warning(f"No waiting step found for approval request {approval_request_id}")
        return False
    
    async def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> List[str]:
        """Handle an event by triggering any associated workflows.
        
        Args:
            event_type: Type of the event
            event_data: Event data
            
        Returns:
            List of workflow execution IDs that were triggered by this event
        """
        triggered_workflow_ids = []
        
        # Find workflows triggered by this event
        workflow_ids = self.event_triggers.get(event_type, [])
        
        for workflow_id in workflow_ids:
            try:
                # Execute the workflow with event data
                execution_id = await self.execute_workflow(
                    workflow_id=workflow_id,
                    input_variables={"event_data": event_data},
                    trigger_info={
                        "type": "event",
                        "event_type": event_type,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                triggered_workflow_ids.append(execution_id)
                
                logger.info(f"Triggered workflow {workflow_id} by event {event_type}")
            except Exception as e:
                logger.error(f"Error triggering workflow {workflow_id} by event {event_type}: {str(e)}")
        
        return triggered_workflow_ids
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow.
        
        Args:
            execution_id: ID of the workflow execution to cancel
            
        Returns:
            True if workflow was cancelled, False otherwise
        """
        with self._workflow_lock:
            if execution_id not in self.workflow_executions:
                return False
            
            execution = self.workflow_executions[execution_id]
            
            # Skip if workflow is already completed
            if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELED]:
                return False
            
            # Cancel running workflow task
            if execution_id in self.running_workflows:
                self.running_workflows[execution_id].cancel()
                del self.running_workflows[execution_id]
            
            # Cancel running steps
            for step_id in list(execution.current_step_ids):
                step_key = f"{execution_id}:{step_id}"
                if step_key in self.running_steps:
                    self.running_steps[step_key].cancel()
                    del self.running_steps[step_key]
            
            # Update workflow state
            execution.status = WorkflowStatus.CANCELED
            execution.end_time = datetime.now().isoformat()
            execution.add_log("Workflow canceled")
            
            # Clear current steps
            execution.current_step_ids = []
        
        # Notify listeners
        for listener in self.listeners:
            try:
                await listener.on_workflow_canceled(execution)
            except Exception as e:
                logger.error(f"Error notifying listener of workflow cancellation: {str(e)}")
        
        # Persist final state if enabled
        if self.enable_persistence and self.workflow_store_path:
            await self._persist_execution_state(execution)
        
        logger.info(f"Canceled workflow execution: {execution_id}")
        return True
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow.
        
        Args:
            execution_id: ID of the workflow execution to pause
            
        Returns:
            True if workflow was paused, False otherwise
        """
        with self._workflow_lock:
            if execution_id not in self.workflow_executions:
                return False
            
            execution = self.workflow_executions[execution_id]
            
            # Skip if workflow is not running
            if execution.status != WorkflowStatus.RUNNING:
                return False
            
            # Cancel running workflow task but keep the state
            if execution_id in self.running_workflows:
                self.running_workflows[execution_id].cancel()
                del self.running_workflows[execution_id]
            
            # Update workflow state
            execution.status = WorkflowStatus.PAUSED
            execution.add_log("Workflow paused")
        
        # Notify listeners
        for listener in self.listeners:
            try:
                await listener.on_workflow_paused(execution)
            except Exception as e:
                logger.error(f"Error notifying listener of workflow pause: {str(e)}")
        
        # Persist state if enabled
        if self.enable_persistence and self.workflow_store_path:
            await self._persist_execution_state(execution)
        
        logger.info(f"Paused workflow execution: {execution_id}")
        return True
    
    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume a paused workflow.
        
        Args:
            execution_id: ID of the workflow execution to resume
            
        Returns:
            True if workflow was resumed, False otherwise
        """
        with self._workflow_lock:
            if execution_id not in self.workflow_executions:
                return False
            
            execution = self.workflow_executions[execution_id]
            
            # Skip if workflow is not paused
            if execution.status != WorkflowStatus.PAUSED:
                return False
            
            # Update workflow state
            execution.status = WorkflowStatus.RUNNING
            execution.add_log("Workflow resumed")
            
            # Create task to continue execution
            workflow_task = asyncio.create_task(
                self._execute_workflow(execution_id)
            )
            
            # Store running workflow
            self.running_workflows[execution_id] = workflow_task
        
        # Notify listeners
        for listener in self.listeners:
            try:
                await listener.on_workflow_resumed(execution)
            except Exception as e:
                logger.error(f"Error notifying listener of workflow resume: {str(e)}")
        
        logger.info(f"Resumed workflow execution: {execution_id}")
        return True
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Workflow information or None if not found
        """
        with self._workflow_lock:
            workflow = self.workflow_definitions.get(workflow_id)
            if workflow:
                return workflow.to_dict()
        
        return None
    
    async def get_workflow_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a workflow execution.
        
        Args:
            execution_id: ID of the workflow execution
            
        Returns:
            Workflow execution information or None if not found
        """
        with self._workflow_lock:
            execution = self.workflow_executions.get(execution_id)
            if execution:
                return execution.to_dict()
        
        # If not in memory and persistence is enabled, try to load from storage
        if self.enable_persistence and self.workflow_store_path:
            try:
                import os
                from pathlib import Path
                
                # Check if execution file exists
                executions_path = Path(self.workflow_store_path) / "executions"
                file_path = executions_path / f"{execution_id}.json"
                
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"Error loading execution {execution_id} from storage: {str(e)}")
        
        return None
    
    async def get_step_execution(
        self, 
        execution_id: str, 
        step_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get information about a step execution.
        
        Args:
            execution_id: ID of the workflow execution
            step_id: ID of the step
            
        Returns:
            Step execution information or None if not found
        """
        with self._workflow_lock:
            execution = self.workflow_executions.get(execution_id)
            if execution and step_id in execution.step_executions:
                return execution.step_executions[step_id].to_dict()
        
        return None
    
    async def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Get information about all workflows.
        
        Returns:
            List of workflow information
        """
        with self._workflow_lock:
            return [workflow.to_dict() for workflow in self.workflow_definitions.values()]
    
    async def get_workflow_executions(
        self, 
        workflow_id: Optional[str] = None, 
        status: Optional[WorkflowStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get information about workflow executions.
        
        Args:
            workflow_id: Optional workflow ID to filter by
            status: Optional status to filter by
            limit: Maximum number of executions to return
            offset: Offset for pagination
            
        Returns:
            List of workflow execution information
        """
        executions = []
        
        with self._workflow_lock:
            # Filter and sort executions
            filtered_executions = [
                execution for execution in self.workflow_executions.values()
                if (workflow_id is None or execution.workflow_id == workflow_id) and
                    (status is None or execution.status == status)
            ]
            
            # Sort by start time (newest first)
            filtered_executions.sort(key=lambda x: x.start_time, reverse=True)
            
            # Apply pagination
            paginated_executions = filtered_executions[offset:offset+limit]
            
            # Convert to dictionaries
            executions = [execution.to_dict() for execution in paginated_executions]
        
        return executions
    
    async def get_workflow_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get workflow execution history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of workflow execution history entries
        """
        with self._workflow_lock:
            # Convert history to list
            history = list(self.execution_history)[-limit:]
            return history
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.
        
        Returns:
            Dictionary of engine statistics
        """
        with self._workflow_lock:
            # Count executions by status
            executions_by_status = defaultdict(int)
            for execution in self.workflow_executions.values():
                executions_by_status[execution.status.value] += 1
            
            return {
                "total_workflows": len(self.workflow_definitions),
                "total_executions": len(self.workflow_executions),
                "running_executions": len(self.running_workflows),
                "running_steps": len(self.running_steps),
                "executions_by_status": dict(executions_by_status),
                "max_concurrent_workflows": self.max_concurrent_workflows,
                "max_concurrent_steps": self.max_concurrent_steps,
                "trigger_count": {
                    "event": sum(len(workflows) for workflows in self.event_triggers.values()),
                    "scheduled": len([t for t in self.scheduled_triggers.values() if "schedule" in t]),
                    "periodic": len([t for t in self.scheduled_triggers.values() if "interval" in t])
                },
                "timestamp": datetime.now().isoformat()
            }


    # Example workflow listeners that integrate with notification systems

class LoggingWorkflowListener(WorkflowListener):
    """Workflow listener that logs workflow events."""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize the logging workflow listener.
        
        Args:
            log_level: Log level
        """
        self.logger = logging.getLogger(__name__ + ".WorkflowLogger")
        self.logger.setLevel(log_level)
    
    async def on_workflow_started(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is started."""
        self.logger.info(f"Workflow started: {execution.id} (workflow: {execution.workflow_id})")
    
    async def on_workflow_completed(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is completed successfully."""
        duration = "unknown"
        if execution.start_time and execution.end_time:
            try:
                start = datetime.fromisoformat(execution.start_time)
                end = datetime.fromisoformat(execution.end_time)
                duration = f"{(end - start).total_seconds():.2f}s"
            except (ValueError, TypeError):
                pass
        
        self.logger.info(
            f"Workflow completed: {execution.id} (workflow: {execution.workflow_id}, "
            f"steps: {len(execution.completed_step_ids)}, duration: {duration})"
        )
    
    async def on_workflow_failed(self, execution: WorkflowExecution, error: str) -> None:
        """Called when a workflow fails."""
        self.logger.error(
            f"Workflow failed: {execution.id} (workflow: {execution.workflow_id}, error: {error})"
       )
   
    async def on_workflow_paused(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is paused."""
        self.logger.info(f"Workflow paused: {execution.id} (workflow: {execution.workflow_id})")
    
    async def on_workflow_resumed(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is resumed."""
        self.logger.info(f"Workflow resumed: {execution.id} (workflow: {execution.workflow_id})")
    
    async def on_workflow_canceled(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is canceled."""
        self.logger.warning(f"Workflow canceled: {execution.id} (workflow: {execution.workflow_id})")
    
    async def on_step_started(self, execution: WorkflowExecution, step_id: str) -> None:
        """Called when a step is started."""
        step_name = ""
        if step_id in execution.step_executions:
            step_name = execution.step_executions[step_id].step_definition.name
        
        self.logger.info(
            f"Step started: {step_id} ({step_name}) in workflow {execution.id}"
        )
    
    async def on_step_completed(self, execution: WorkflowExecution, step_id: str, output: StepOutput) -> None:
        """Called when a step is completed successfully."""
        step_name = ""
        if step_id in execution.step_executions:
            step_name = execution.step_executions[step_id].step_definition.name
        
        self.logger.info(
            f"Step completed: {step_id} ({step_name}) in workflow {execution.id} "
            f"in {output.execution_time:.2f}s"
        )
    
    async def on_step_failed(self, execution: WorkflowExecution, step_id: str, error: str) -> None:
        """Called when a step fails."""
        step_name = ""
        if step_id in execution.step_executions:
            step_name = execution.step_executions[step_id].step_definition.name
        
        self.logger.error(
            f"Step failed: {step_id} ({step_name}) in workflow {execution.id}: {error}"
        )


class WebhookWorkflowListener(WorkflowListener):
    """Workflow listener that sends notifications to a webhook."""
    
    def __init__(self, webhook_url: str):
        """Initialize the webhook workflow listener.
        
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
    
    async def on_workflow_started(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is started."""
        await self._send_to_webhook("workflow.started", {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id
        })
    
    async def on_workflow_completed(self, execution: WorkflowExecution) -> None:
        """Called when a workflow is completed successfully."""
        await self._send_to_webhook("workflow.completed", {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "metrics": execution.metrics
        })
    
    async def on_workflow_failed(self, execution: WorkflowExecution, error: str) -> None:
        """Called when a workflow fails."""
        await self._send_to_webhook("workflow.failed", {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "error": error
        })
    
    async def on_step_failed(self, execution: WorkflowExecution, step_id: str, error: str) -> None:
        """Called when a step fails."""
        await self._send_to_webhook("step.failed", {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "step_id": step_id,
            "error": error
        })
    
    async def close(self) -> None:
        """Close resources."""
        if self.session:
            await self.session.close()
            self.session = None
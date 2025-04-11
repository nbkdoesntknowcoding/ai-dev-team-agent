"""
Orchestrator package for the multi-agent development system.

This package provides workflow orchestration, task scheduling, and coordination
services for managing complex multi-agent interactions and development processes.
"""

from orchestrator.workflow_engine import (
    # Core classes
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowExecution,
    WorkflowStepExecution,
    WorkflowListener,
    
    # Type definitions
    WorkflowStepType,
    WorkflowStatus,
    StepStatus,
    WorkflowTriggerType,
    
    # Input/Output models
    StepInput,
    StepOutput,
    
    # Trigger related
    WorkflowTrigger,
    
    # Example listeners
    LoggingWorkflowListener,
    WebhookWorkflowListener,
)

from orchestrator.task_scheduler import (
    # Core classes
    TaskScheduler,
    AgentRegistry,
    TaskListener,
    
    # Type definitions
    TaskState,
    TaskPriority,
    TaskType,
    
    # Task models
    TaskDefinition,
    TaskInstance,
    TaskResult,
    
    # Example listeners
    LoggingTaskListener,
    SlackTaskListener,
    WebhookTaskListener,
)

# Version information
__version__ = "0.1.0"
__author__ = "AI Learning Platform Team"
__email__ = "team@ailearningplatform.com"
__license__ = "Proprietary"

# Convenience re-exports for the most commonly used classes
__all__ = [
    # Workflow engine components
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowStepType",
    "WorkflowStatus",
    "WorkflowTrigger",
    "WorkflowTriggerType",
    
    # Task scheduler components
    "TaskScheduler",
    "AgentRegistry",
    "TaskDefinition",
    "TaskResult",
    "TaskPriority",
    "TaskType",
    "TaskState",
    
    # Listeners
    "LoggingWorkflowListener",
    "LoggingTaskListener",
]
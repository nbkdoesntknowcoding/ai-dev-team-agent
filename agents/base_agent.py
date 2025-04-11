"""
Base agent implementation for the multi-agent development system.

This module provides the foundational Agent classes that specific agent types
will inherit from. It handles communication with LLMs, maintains context,
and provides a standardized interface for all agents.
"""

import uuid
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union, Literal, TypedDict, cast
from enum import Enum
import logging
from pydantic import BaseModel, Field, validator

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
try:
    from langchain_google_genai import GoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Agent tools
from langchain.agents import Tool
from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent

# Performance monitoring
from contextlib import contextmanager
import time

# Set up logging
logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class AgentRole(str, Enum):
    """Standardized agent role types."""
    PROJECT_MANAGER = "project_manager"
    ARCHITECTURE_DESIGNER = "architecture_designer"
    UI_DEVELOPER = "ui_developer"
    FRONTEND_LOGIC = "frontend_logic"
    FRONTEND_INTEGRATION = "frontend_integration"
    API_DEVELOPER = "api_developer"
    DATABASE_DESIGNER = "database_designer"
    BACKEND_LOGIC = "backend_logic"
    INFRASTRUCTURE = "infrastructure"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    CODE_REVIEWER = "code_reviewer"
    TEST_DEVELOPER = "test_developer"
    UX_TESTER = "ux_tester"
    RESEARCHER = "researcher"
    DOCUMENTATION = "documentation"
    HUMAN_INTERFACE = "human_interface"


class TaskStatus(str, Enum):
    """Possible statuses for a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    REVISED = "revised"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskRequirement(BaseModel):
    """A specific requirement for a task."""
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    acceptance_criteria: Optional[List[str]] = None


class TaskContext(BaseModel):
    """Context information for a task."""
    related_files: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    notes: Optional[str] = None
    constraints: Optional[List[str]] = None
    resources: Optional[List[str]] = None


class Task(BaseModel):
    """A task to be executed by an agent."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    agent_type: str
    requirements: Union[str, List[TaskRequirement], Dict[str, Any]]
    context: Optional[TaskContext] = None
    expected_output: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    requires_human_review: bool = True
    max_retries: int = 3
    timeout_seconds: Optional[int] = None


class TaskResult(BaseModel):
    """The result of a task execution."""
    agent_id: str
    agent_name: str
    task_id: str
    result: Any
    status: TaskStatus
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    execution_time: float
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentState(BaseModel):
    """State management for agents."""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: str
    current_task: Optional[Task] = None
    last_execution_time: Optional[float] = None
    total_tasks_completed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    memory_keys: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    def update_stats(self, status: TaskStatus) -> None:
        """Update agent statistics based on task completion status."""
        self.total_tasks_completed += 1
        if status in [TaskStatus.COMPLETED]:
            self.successful_tasks += 1
        elif status in [TaskStatus.FAILED, TaskStatus.REJECTED]:
            self.failed_tasks += 1


class AgentMemory:
    """Memory management for agents."""
    
    def __init__(self, max_items: int = 50):
        """Initialize agent memory."""
        self.memory: List[Dict[str, Any]] = []
        self.max_items = max_items
    
    def add(self, item: Dict[str, Any]) -> None:
        """Add an item to memory."""
        self.memory.append(item)
        
        # Trim if we exceed max items
        if len(self.memory) > self.max_items:
            self.memory = self.memory[-self.max_items:]
    
    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent items from memory."""
        return self.memory[-n:] if self.memory else []
    
    def search(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """Search for items in memory by key-value pair."""
        return [item for item in self.memory if item.get(key) == value]

    def clear(self) -> None:
        """Clear all items from memory."""
        self.memory = []


@contextmanager
def timer():
    """Context manager for timing execution."""
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed = end - start


def default_system_prompt(agent_name: str, agent_type: str) -> str:
    """Generate a default system prompt for an agent."""
    return (
        f"You are {agent_name}, a specialized AI agent working as part of a "
        f"software development team. Your role is: {agent_type}. "
        f"You collaborate with other AI agents to build software solutions. "
        f"Always think step-by-step and explain your reasoning clearly. "
        f"Be thorough but concise in your responses. "
        f"When you create code, make sure it's well-structured, properly commented, "
        f"and follows best practices for the relevant language or framework."
    )


class BaseAgent:
    """Base class for all agents in the multi-agent development system."""
    
    def __init__(
        self,
        name: str,
        agent_type: str,
        model_provider: ModelProvider = ModelProvider.ANTHROPIC,
        model_name: str = "claude-3-sonnet-20240229",
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        shared_memory: Any = None,
        verbose: bool = False,
        task_scheduler: Any = None,
        workflow_engine: Any = None,
        code_tools: Any = None,
        feedback_processor: Any = None,
        review_interface: Any = None,
        **kwargs
    ):
        """Initialize the base agent.
        
        Args:
            name: Human-readable name for this agent
            agent_type: Type of agent (corresponding to its role)
            model_provider: LLM provider to use
            model_name: Specific model to use
            tools: List of LangChain tools available to the agent
            system_prompt: Custom system prompt (defaults to auto-generated)
            max_tokens: Maximum tokens for model response
            temperature: Temperature for model sampling (0-1)
            shared_memory: Shared memory interface for agent communication
            verbose: Whether to enable verbose logging
        """
        self.name = name
        self.state = AgentState(agent_type=agent_type)
        self.memory = AgentMemory()
        self.verbose = verbose
        self.tools = tools or []
        self.shared_memory = shared_memory
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.task_scheduler = task_scheduler
        self.workflow_engine = workflow_engine
        self.code_tools = code_tools
        self.feedback_processor = feedback_processor
        self.review_interface = review_interface
        
        # Set up the LLM based on provider
        self.llm = self._setup_llm(model_provider, model_name)
        
        # Define system prompt for this agent
        self.system_prompt = system_prompt or self._get_system_prompt()
        
        # Performance tracking
        self.performance_metrics = {
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "total_token_usage": 0,
            "total_calls": 0,
        }
        
        logger.info(f"Initialized agent: {self.name} ({agent_type})")
    
    def _setup_llm(self, model_provider: ModelProvider, model_name: str) -> BaseChatModel:
        """Set up the language model based on provider.
        
        Args:
            model_provider: The LLM provider to use
            model_name: The specific model to use
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If the model provider is unsupported
        """
        if model_provider == ModelProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif model_provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif model_provider == ModelProvider.GOOGLE and GOOGLE_AVAILABLE:
            return GoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        else:
            supported = [ModelProvider.ANTHROPIC, ModelProvider.OPENAI]
            if GOOGLE_AVAILABLE:
                supported.append(ModelProvider.GOOGLE)
            
            raise ValueError(
                f"Unsupported model provider: {model_provider}. "
                f"Supported providers: {', '.join([p.value for p in supported])}"
            )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent type.
        
        This should be overridden by subclasses to provide role-specific prompts.
        """
        return default_system_prompt(self.name, self.state.agent_type)
    
    async def execute_task(self, task: Union[Task, Dict[str, Any]]) -> TaskResult:
        """Execute a given task and return the result.
        
        Args:
            task: The task to execute (either a Task object or dict representation)
            
        Returns:
            TaskResult containing the execution results
        """
        # Convert dict to Task if necessary
        if isinstance(task, dict):
            task = Task(**task)
        
        start_time = time.time()
        execution_time = 0.0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Update agent state
        self.state.current_task = task
        
        if self.verbose:
            logger.info(f"Agent {self.name} executing task: {task.task_id}")
            logger.info(f"Task description: {task.description}")
        
        try:
            # Create messages for the model
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._format_task(task))
            ]
            
            # Add context from memory if available
            context_message = self._get_context_for_task(task)
            if context_message:
                messages.append(HumanMessage(content=context_message))
            
            # Execute with timeout if specified
            if task.timeout_seconds:
                try:
                    response = await asyncio.wait_for(
                        self.llm.agenerate([messages]), 
                        timeout=task.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Task execution timed out after {task.timeout_seconds} seconds")
            else:
                response = await self.llm.agenerate([messages])
            
            result = response.generations[0][0].text
            
            # Try to extract token usage
            try:
                if hasattr(response, 'llm_output') and response.llm_output:
                    if isinstance(response.llm_output, dict) and 'token_usage' in response.llm_output:
                        token_usage = response.llm_output['token_usage']
            except:
                pass
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics["total_execution_time"] += execution_time
            self.performance_metrics["total_token_usage"] += token_usage.get("total_tokens", 0)
            self.performance_metrics["total_calls"] += 1
            self.performance_metrics["average_execution_time"] = (
                self.performance_metrics["total_execution_time"] / 
                self.performance_metrics["total_calls"]
            )
            
            # Update agent state
            self.state.last_execution_time = execution_time
            
            # Update agent memory
            self.memory.add({
                "task_id": task.task_id,
                "description": task.description,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": TaskStatus.COMPLETED
            })
            
            # Update shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=task.task_id,
                    value={
                        "task": task.dict(),
                        "result": result,
                        "agent": self.name,
                        "timestamp": datetime.now().isoformat(),
                        "status": TaskStatus.COMPLETED
                    },
                    category="task_results"
                )
            
            # Update agent stats
            self.state.update_stats(TaskStatus.COMPLETED)
            
            task_result = TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=task.task_id,
                result=result,
                status=TaskStatus.COMPLETED,
                execution_time=execution_time,
                token_usage=token_usage
            )
            
            if self.verbose:
                logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            
            return task_result
            
        except Exception as e:
            # Handle failures
            execution_time = time.time() - start_time
            error_message = str(e)
            
            if self.verbose:
                logger.error(f"Error executing task {task.task_id}: {error_message}")
            
            # Update agent memory
            self.memory.add({
                "task_id": task.task_id,
                "description": task.description,
                "error": error_message,
                "timestamp": datetime.now().isoformat(),
                "status": TaskStatus.FAILED
            })
            
            # Update shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"{task.task_id}_error",
                    value={
                        "task": task.dict(),
                        "error": error_message,
                        "agent": self.name,
                        "timestamp": datetime.now().isoformat(),
                        "status": TaskStatus.FAILED
                    },
                    category="task_results"
                )
            
            # Update agent stats
            self.state.update_stats(TaskStatus.FAILED)
            
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=task.task_id,
                result=None,
                status=TaskStatus.FAILED,
                execution_time=execution_time,
                error=error_message,
                token_usage=token_usage
            )
    
    def _format_task(self, task: Task) -> str:
        """Format a task into a prompt for the model.
        
        Args:
            task: The task to format
            
        Returns:
            Formatted prompt string
        """
        # Format requirements based on type
        if isinstance(task.requirements, list):
            formatted_requirements = "\n".join([
                f"- {req.description} (Priority: {req.priority.value})" +
                (f"\n  Acceptance Criteria: {', '.join(req.acceptance_criteria)}" 
                 if req.acceptance_criteria else "")
                for req in task.requirements
            ])
        elif isinstance(task.requirements, dict):
            formatted_requirements = json.dumps(task.requirements, indent=2)
        else:
            formatted_requirements = str(task.requirements)
        
        # Build the prompt
        prompt = (
            f"# Task: {task.description}\n\n"
            f"## Requirements:\n{formatted_requirements}\n\n"
        )
        
        # Add context if available
        if task.context:
            ctx = task.context
            prompt += "## Context:\n"
            
            if ctx.notes:
                prompt += f"Notes: {ctx.notes}\n"
            
            if ctx.related_files:
                prompt += f"Related Files: {', '.join(ctx.related_files)}\n"
                
            if ctx.dependencies:
                prompt += f"Dependencies: {', '.join(ctx.dependencies)}\n"
                
            if ctx.constraints:
                prompt += f"Constraints: {', '.join(ctx.constraints)}\n"
                
            if ctx.resources:
                prompt += f"Resources: {', '.join(ctx.resources)}\n"
            
            prompt += "\n"
        
        # Add expected output format if specified
        if task.expected_output:
            prompt += f"## Expected Output:\n{task.expected_output}\n\n"
        
        # Add instructions based on agent role
        prompt += (
            f"Please complete this task based on your role as {self.state.agent_type}. "
            f"Be thorough but concise in your response. "
            f"If you generate code, ensure it is well-structured, follows best practices, "
            f"and includes appropriate comments and documentation."
        )
        
        return prompt
    
    def _get_context_for_task(self, task: Task) -> Optional[str]:
        """Get relevant context for a task from agent memory and shared memory.
        
        Args:
            task: The task to get context for
            
        Returns:
            Context string or None if no relevant context
        """
        context_parts = []
        
        # Get related tasks from memory
        recent_tasks = self.memory.get_recent(3)
        if recent_tasks:
            context_parts.append("## Recent Tasks:")
            for i, task_data in enumerate(recent_tasks, 1):
                context_parts.append(f"{i}. Task: {task_data.get('description', 'Unknown')}")
                if 'result' in task_data:
                    # Truncate long results
                    result = task_data['result']
                    if isinstance(result, str) and len(result) > 500:
                        result = result[:500] + "... [truncated]"
                    context_parts.append(f"   Result: {result}")
            context_parts.append("")
        
        # If shared memory is available, check for related information
        if self.shared_memory and task.context and task.context.related_files:
            context_parts.append("## Related Files:")
            for file_key in task.context.related_files:
                file_data = self.shared_memory.retrieve(file_key, "code")
                if file_data:
                    # Add file information (truncate if necessary)
                    file_content = file_data.get("content", "")
                    if len(file_content) > 1000:
                        file_content = file_content[:1000] + "... [truncated]"
                    
                    context_parts.append(f"File: {file_key}")
                    context_parts.append("```")
                    context_parts.append(file_content)
                    context_parts.append("```")
            context_parts.append("")
        
        # If there's context, join it all together
        if context_parts:
            return "\n".join(context_parts)
        
        return None
    
    async def process_feedback(self, task_id: str, feedback: Dict[str, Any]) -> Optional[TaskResult]:
        """Process feedback on a completed task.
        
        Args:
            task_id: ID of the task that received feedback
            feedback: Feedback data including approval status and comments
            
        Returns:
            TaskResult for a revised task if the original was rejected, None otherwise
        """
        # Find the original task in memory
        original_task_data = next((t for t in self.memory.search("task_id", task_id)), None)
        
        if not original_task_data:
            logger.warning(f"Feedback received for unknown task: {task_id}")
            return None
        
        is_approved = feedback.get("approved", False)
        feedback_text = feedback.get("feedback", "")
        
        if is_approved:
            # Update memory to reflect approval
            original_task_data["feedback"] = feedback_text
            original_task_data["approved"] = True
            
            if self.verbose:
                logger.info(f"Positive feedback received for task {task_id}")
            
            return None
        else:
            # Task was rejected, create a revised version
            if self.verbose:
                logger.info(f"Negative feedback received for task {task_id}. Creating revised task.")
            
            # Get the original task
            original_task = self.state.current_task
            if not original_task:
                # If not in current state, try to reconstruct from memory
                try:
                    original_task = Task(
                        task_id=task_id,
                        description=original_task_data.get("description", "Unknown task"),
                        agent_type=self.state.agent_type,
                        requirements="See feedback for revision requirements"
                    )
                except:
                    logger.error(f"Could not reconstruct original task {task_id} for revision")
                    return None
            
            # Create a revised task
            revised_task = Task(
                task_id=f"{task_id}_revised",
                description=f"Revise task: {original_task.description}",
                agent_type=self.state.agent_type,
                requirements=f"Original task requirements plus feedback: {feedback_text}",
                context=TaskContext(
                    notes=f"This is a revision of task {task_id} based on feedback: {feedback_text}",
                    related_files=original_task.context.related_files if original_task.context else None
                ),
                expected_output=original_task.expected_output,
                priority=original_task.priority,
                requires_human_review=True
            )
            
            # Execute the revised task
            result = await self.execute_task(revised_task)
            
            # Update the original task data in memory
            original_task_data["feedback"] = feedback_text
            original_task_data["approved"] = False
            original_task_data["revised_task_id"] = revised_task.task_id
            
            return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent.
        
        Returns:
            Dictionary of performance metrics
        """
        success_rate = 0
        if self.state.total_tasks_completed > 0:
            success_rate = (self.state.successful_tasks / self.state.total_tasks_completed) * 100
            
        return {
            "agent_id": self.state.agent_id,
            "name": self.name,
            "type": self.state.agent_type,
            "tasks_completed": self.state.total_tasks_completed,
            "successful_tasks": self.state.successful_tasks,
            "failed_tasks": self.state.failed_tasks,
            "success_rate": success_rate,
            "average_execution_time": self.performance_metrics["average_execution_time"],
            "total_token_usage": self.performance_metrics["total_token_usage"]
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} ({self.state.agent_type})"
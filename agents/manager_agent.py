"""
Project Manager Agent for the multi-agent development system.

This agent is responsible for coordinating other agents, planning project activities,
managing requirements, and interfacing with human stakeholders. It serves as the central
orchestration point for complex development workflows.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import uuid

from pydantic import BaseModel, Field

from agents.base_agent import (
    BaseAgent, 
    Task, 
    TaskResult, 
    TaskStatus, 
    TaskPriority,
    TaskContext,
    TaskRequirement,
    AgentRole
)

# Set up logging
logger = logging.getLogger(__name__)


class ProjectRequirement(BaseModel):
    """A requirement for a project."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    priority: TaskPriority
    status: str = "pending"
    assignee: Optional[str] = None
    acceptance_criteria: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None


class ProjectPlan(BaseModel):
    """A plan for a project."""
    name: str
    description: str
    requirements: List[ProjectRequirement]
    milestones: List[Dict[str, Any]]
    task_assignments: Dict[str, List[str]] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    estimated_timeline: Optional[Dict[str, Any]] = None
    risks: Optional[List[Dict[str, Any]]] = None


class ProjectManagerAgent(BaseAgent):
    """Project Manager Agent responsible for coordination and planning."""
    
    def __init__(
        self, 
        name: str = "Project Manager", 
        agent_registry: Optional[Dict[str, BaseAgent]] = None,
        **kwargs
    ):
        """Initialize the Project Manager agent.
        
        Args:
            name: Human-readable name for this agent
            agent_registry: Registry of available agents for task delegation
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.PROJECT_MANAGER, 
            **kwargs
        )
        self.agent_registry = agent_registry or {}
        self.active_projects: Dict[str, ProjectPlan] = {}
        self.task_queue: List[Task] = []
        
        logger.info("Project Manager Agent initialized")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Project Manager."""
        return (
            f"You are {self.name}, the Project Manager of an AI development team. "
            f"Your responsibilities include:\n"
            f"1. Breaking down project requirements into manageable tasks\n"
            f"2. Assigning tasks to appropriate specialized agents\n"
            f"3. Monitoring progress and ensuring deadlines are met\n"
            f"4. Identifying potential issues and risks\n"
            f"5. Communicating with human stakeholders\n"
            f"6. Making high-level decisions about project direction\n\n"
            f"Always think step-by-step and provide clear rationale for your decisions. "
            f"When unsure about a decision, identify options and request human input. "
            f"Remember that you are coordinating a team of specialized AI agents, each with "
            f"different skills and responsibilities. Delegate tasks appropriately based on "
            f"agent specializations.\n\n"
            f"Your output should be structured, clear, and actionable. Provide specific "
            f"assignments, requirements, and acceptance criteria whenever possible. "
            f"When creating project plans, ensure that dependencies are clearly identified "
            f"and that the timeline is realistic."
        )
    
    async def create_project_plan(
        self, 
        project_name: str, 
        project_description: str,
        requirements: List[Dict[str, Any]],
        available_agents: Optional[List[str]] = None
    ) -> TaskResult:
        """Create a comprehensive project plan based on requirements.
        
        Args:
            project_name: Name of the project
            project_description: Brief description of the project
            requirements: List of project requirements
            available_agents: List of available agent types for assignment
            
        Returns:
            TaskResult containing the project plan
        """
        # Create a task for the project planning
        task = Task(
            task_id=f"create_project_plan_{project_name.lower().replace(' ', '_')}",
            description=f"Create a detailed project plan for '{project_name}'",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_name": project_name,
                "project_description": project_description,
                "requirements": requirements,
                "available_agents": available_agents or list(self.agent_registry.keys())
            },
            context=TaskContext(
                notes=(
                    "Create a comprehensive project plan including task breakdown, "
                    "dependencies, assignments, milestones, and risk assessment."
                )
            ),
            expected_output=(
                "A complete project plan with tasks, dependencies, assignments, "
                "timeline, and risks in JSON format."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the project plan
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the plan from the result
                # First, try to parse as JSON
                try:
                    plan_data = json.loads(result.result)
                except:
                    # If not valid JSON, try to extract JSON block
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        plan_data = json.loads(json_match.group(1))
                    else:
                        # Not parsable, return the raw result
                        logger.warning(f"Could not parse project plan for {project_name}")
                        return result
                
                # Create and store the project plan
                project_id = project_name.lower().replace(' ', '_')
                
                # Convert requirements to ProjectRequirement objects
                project_requirements = []
                for req in plan_data.get("requirements", []):
                    project_requirements.append(
                        ProjectRequirement(
                            id=req.get("id", str(uuid.uuid4())),
                            description=req.get("description", ""),
                            priority=req.get("priority", TaskPriority.MEDIUM),
                            acceptance_criteria=req.get("acceptance_criteria", []),
                            dependencies=req.get("dependencies", [])
                        )
                    )
                
                # Create the project plan
                project_plan = ProjectPlan(
                    name=project_name,
                    description=project_description,
                    requirements=project_requirements,
                    milestones=plan_data.get("milestones", []),
                    task_assignments=plan_data.get("task_assignments", {}),
                    dependencies=plan_data.get("dependencies", {}),
                    estimated_timeline=plan_data.get("timeline", {}),
                    risks=plan_data.get("risks", [])
                )
                
                # Store the plan
                self.active_projects[project_id] = project_plan
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"project_plan_{project_id}",
                        value=project_plan.dict(),
                        category="project_plans"
                    )
                
                logger.info(f"Created project plan for {project_name}")
                
                # Update the result with the parsed plan
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=project_plan.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"project_id": project_id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing project plan: {str(e)}")
                # Still return the original result
                return result
        
        return result
    
    async def break_down_requirements(
        self, 
        project_id: str,
        requirements: Optional[List[Dict[str, Any]]] = None
    ) -> TaskResult:
        """Break down project requirements into detailed tasks.
        
        Args:
            project_id: ID of the project
            requirements: Optional list of requirements (if not using a stored project)
            
        Returns:
            TaskResult containing the detailed task breakdown
        """
        # Check if we have the project in our active projects
        project_plan = None
        if project_id in self.active_projects:
            project_plan = self.active_projects[project_id]
        elif self.shared_memory:
            # Try to get from shared memory
            stored_plan = self.shared_memory.retrieve(
                key=f"project_plan_{project_id}",
                category="project_plans"
            )
            if stored_plan:
                project_plan = ProjectPlan(**stored_plan)
        
        # If no project plan and no requirements, we can't proceed
        if not project_plan and not requirements:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"break_down_requirements_{project_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error="No project plan or requirements provided"
            )
        
        # Use provided requirements or get from project plan
        req_to_process = requirements or [
            req.dict() for req in project_plan.requirements
        ] if project_plan else []
        
        # Create a task for breaking down requirements
        task = Task(
            task_id=f"break_down_requirements_{project_id}",
            description=f"Break down requirements for project {project_id} into detailed tasks",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "project_name": project_plan.name if project_plan else "Unknown",
                "project_description": project_plan.description if project_plan else "",
                "requirements": req_to_process
            },
            context=TaskContext(
                notes=(
                    "Break down each requirement into specific, actionable tasks. "
                    "Each task should be assigned to a specific agent type and include "
                    "detailed acceptance criteria."
                )
            ),
            expected_output=(
                "A complete task breakdown with detailed descriptions, "
                "agent assignments, and acceptance criteria in JSON format."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful and we have a project plan, update it with the tasks
        if (result.status == TaskStatus.COMPLETED and 
            result.result and 
            project_id in self.active_projects):
            try:
                # Extract the tasks from the result
                tasks_data = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    tasks_data = parsed_result.get("tasks", [])
                except:
                    # If not valid JSON, try to extract JSON block
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(1))
                        tasks_data = parsed_result.get("tasks", [])
                
                if tasks_data:
                    # Store the tasks in shared memory if available
                    if self.shared_memory:
                        self.shared_memory.store(
                            key=f"project_tasks_{project_id}",
                            value={"tasks": tasks_data},
                            category="project_tasks"
                        )
                    
                    logger.info(f"Broke down requirements for {project_id} into {len(tasks_data)} tasks")
            except Exception as e:
                logger.error(f"Error processing task breakdown: {str(e)}")
        
        return result
    
    async def assign_tasks(
        self, 
        project_id: str,
        tasks: List[Dict[str, Any]],
        available_agents: Optional[Dict[str, List[str]]] = None
    ) -> TaskResult:
        """Assign tasks to appropriate agents.
        
        Args:
            project_id: ID of the project
            tasks: List of tasks to assign
            available_agents: Dictionary of agent types to available agent IDs
            
        Returns:
            TaskResult containing the task assignments
        """
        # Create a task for assigning tasks
        task = Task(
            task_id=f"assign_tasks_{project_id}",
            description=f"Assign tasks for project {project_id} to appropriate agents",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "tasks": tasks,
                "available_agents": available_agents or {
                    agent_type: [agent.state.agent_id] 
                    for agent_type, agent in self.agent_registry.items()
                }
            },
            context=TaskContext(
                notes=(
                    "Assign each task to the most appropriate agent type based on the task "
                    "requirements and agent specialization. Consider workload balance and "
                    "dependencies between tasks."
                )
            ),
            expected_output=(
                "A complete task assignment with task IDs mapped to agent IDs, "
                "including rationale for each assignment in JSON format."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the assignments and queue tasks
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the assignments from the result
                assignments = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    assignments = parsed_result.get("assignments", {})
                except:
                    # If not valid JSON, try to extract JSON block
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(1))
                        assignments = parsed_result.get("assignments", {})
                
                if assignments:
                    # Store the assignments in shared memory if available
                    if self.shared_memory:
                        self.shared_memory.store(
                            key=f"task_assignments_{project_id}",
                            value={"assignments": assignments},
                            category="task_assignments"
                        )
                    
                    # Update project plan if available
                    if project_id in self.active_projects:
                        self.active_projects[project_id].task_assignments = assignments
                    
                    logger.info(f"Assigned {len(assignments)} tasks for project {project_id}")
            except Exception as e:
                logger.error(f"Error processing task assignments: {str(e)}")
        
        return result
    
    async def generate_workflow(
        self, 
        project_id: str,
        tasks: Optional[List[Dict[str, Any]]] = None,
        dependencies: Optional[Dict[str, List[str]]] = None
    ) -> TaskResult:
        """Generate a workflow with task dependencies for a project.
        
        Args:
            project_id: ID of the project
            tasks: Optional list of tasks (if not using stored tasks)
            dependencies: Optional dictionary of task dependencies
            
        Returns:
            TaskResult containing the workflow definition
        """
        # Try to get tasks from shared memory if not provided
        if not tasks and self.shared_memory:
            stored_tasks = self.shared_memory.retrieve(
                key=f"project_tasks_{project_id}",
                category="project_tasks"
            )
            if stored_tasks and "tasks" in stored_tasks:
                tasks = stored_tasks["tasks"]
        
        # If no tasks, we can't proceed
        if not tasks:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"generate_workflow_{project_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error="No tasks provided or available in shared memory"
            )
        
        # Get project plan if available
        project_plan = None
        if project_id in self.active_projects:
            project_plan = self.active_projects[project_id]
        elif self.shared_memory:
            stored_plan = self.shared_memory.retrieve(
                key=f"project_plan_{project_id}",
                category="project_plans"
            )
            if stored_plan:
                project_plan = ProjectPlan(**stored_plan)
        
        # Use provided dependencies or get from project plan
        deps = dependencies or (
            project_plan.dependencies if project_plan else {}
        )
        
        # Create a task for generating the workflow
        task = Task(
            task_id=f"generate_workflow_{project_id}",
            description=f"Generate a workflow for project {project_id} with task dependencies",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "project_name": project_plan.name if project_plan else "Unknown",
                "tasks": tasks,
                "existing_dependencies": deps
            },
            context=TaskContext(
                notes=(
                    "Create a workflow that orders tasks appropriately based on dependencies. "
                    "Identify parallelization opportunities and critical paths. "
                    "The workflow should be compatible with a workflow orchestration system."
                )
            ),
            expected_output=(
                "A complete workflow definition with tasks and dependencies in JSON format, "
                "compatible with a workflow orchestration system."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the workflow
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the workflow from the result
                workflow = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    workflow = parsed_result
                except:
                    # If not valid JSON, try to extract JSON block
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        workflow = json.loads(json_match.group(1))
                
                if workflow:
                    # Ensure the workflow has a unique ID
                    if "id" not in workflow:
                        workflow["id"] = f"workflow_{project_id}_{str(uuid.uuid4())[:8]}"
                    
                    # Store the workflow in shared memory if available
                    if self.shared_memory:
                        self.shared_memory.store(
                            key=f"workflow_{project_id}",
                            value=workflow,
                            category="workflows"
                        )
                    
                    logger.info(f"Generated workflow for project {project_id}")
                    
                    # Update the result with the parsed workflow
                    updated_result = TaskResult(
                        agent_id=result.agent_id,
                        agent_name=result.agent_name,
                        task_id=result.task_id,
                        result=workflow,
                        status=result.status,
                        timestamp=result.timestamp,
                        execution_time=result.execution_time,
                        token_usage=result.token_usage,
                        metadata={"workflow_id": workflow["id"]}
                    )
                    
                    return updated_result
            except Exception as e:
                logger.error(f"Error processing workflow: {str(e)}")
        
        return result
    
    async def monitor_progress(
        self, 
        project_id: str,
        completed_tasks: Optional[List[str]] = None,
        in_progress_tasks: Optional[List[str]] = None,
        pending_tasks: Optional[List[str]] = None
    ) -> TaskResult:
        """Monitor project progress and generate a status report.
        
        Args:
            project_id: ID of the project
            completed_tasks: Optional list of completed task IDs
            in_progress_tasks: Optional list of in-progress task IDs
            pending_tasks: Optional list of pending task IDs
            
        Returns:
            TaskResult containing the status report
        """
        # Try to get project information
        project_plan = None
        if project_id in self.active_projects:
            project_plan = self.active_projects[project_id]
        elif self.shared_memory:
            stored_plan = self.shared_memory.retrieve(
                key=f"project_plan_{project_id}",
                category="project_plans"
            )
            if stored_plan:
                project_plan = ProjectPlan(**stored_plan)
        
        # Try to get task information from shared memory if not provided
        if self.shared_memory:
            # Get task statuses if not provided
            if not completed_tasks or not in_progress_tasks or not pending_tasks:
                task_statuses = self.shared_memory.retrieve(
                    key=f"task_statuses_{project_id}",
                    category="task_statuses"
                )
                if task_statuses:
                    completed_tasks = completed_tasks or task_statuses.get("completed", [])
                    in_progress_tasks = in_progress_tasks or task_statuses.get("in_progress", [])
                    pending_tasks = pending_tasks or task_statuses.get("pending", [])
        
        # Create a task for monitoring progress
        task = Task(
            task_id=f"monitor_progress_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Generate a status report for project {project_id}",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "project_name": project_plan.name if project_plan else "Unknown",
                "project_description": project_plan.description if project_plan else "",
                "task_status": {
                    "completed": completed_tasks or [],
                    "in_progress": in_progress_tasks or [],
                    "pending": pending_tasks or []
                }
            },
            context=TaskContext(
                notes=(
                    "Generate a comprehensive status report for the project. "
                    "Include overall progress, milestone status, risks, and next steps. "
                    "Highlight any issues or blockers that need attention."
                )
            ),
            expected_output=(
                "A detailed status report including overall progress percentage, "
                "milestone status, risks, and recommended next steps in a structured format."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the status report
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the status report from the result
                status_report = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    status_report = parsed_result
                except:
                    # If not valid JSON, use the raw result
                    status_report = {"raw_report": result.result}
                
                if status_report:
                    # Store the status report in shared memory if available
                    if self.shared_memory:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.shared_memory.store(
                            key=f"status_report_{project_id}_{timestamp}",
                            value=status_report,
                            category="status_reports"
                        )
                    
                    logger.info(f"Generated status report for project {project_id}")
            except Exception as e:
                logger.error(f"Error processing status report: {str(e)}")
        
        return result
    
    async def handle_issue(
        self, 
        project_id: str,
        issue_description: str,
        issue_type: str,
        severity: str,
        affected_tasks: Optional[List[str]] = None
    ) -> TaskResult:
        """Handle an issue that has arisen during project execution.
        
        Args:
            project_id: ID of the project
            issue_description: Description of the issue
            issue_type: Type of issue (technical, resource, scope, etc.)
            severity: Severity level (low, medium, high, critical)
            affected_tasks: Optional list of affected task IDs
            
        Returns:
            TaskResult containing the issue resolution plan
        """
        # Create a task for handling the issue
        task = Task(
            task_id=f"handle_issue_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Handle {severity} {issue_type} issue in project {project_id}",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "issue_description": issue_description,
                "issue_type": issue_type,
                "severity": severity,
                "affected_tasks": affected_tasks or []
            },
            context=TaskContext(
                notes=(
                    "Analyze the issue and develop a resolution plan. "
                    "Consider impact on project timeline, resources, and scope. "
                    "Identify actions needed from specific agents or humans."
                )
            ),
            expected_output=(
                "A detailed issue resolution plan including impact assessment, "
                "recommended actions, assignments, and timeline adjustments if needed."
            ),
            priority=TaskPriority.HIGH if severity in ["high", "critical"] else TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the resolution plan
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the resolution plan from the result
                resolution_plan = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    resolution_plan = parsed_result
                except:
                    # If not valid JSON, use the raw result
                    resolution_plan = {"raw_plan": result.result}
                
                if resolution_plan:
                    # Store the resolution plan in shared memory if available
                    if self.shared_memory:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.shared_memory.store(
                            key=f"issue_resolution_{project_id}_{timestamp}",
                            value={
                                "issue_description": issue_description,
                                "issue_type": issue_type,
                                "severity": severity,
                                "affected_tasks": affected_tasks or [],
                                "resolution_plan": resolution_plan,
                                "timestamp": timestamp
                            },
                            category="issue_resolutions"
                        )
                    
                    logger.info(f"Generated resolution plan for {severity} issue in project {project_id}")
            except Exception as e:
                logger.error(f"Error processing resolution plan: {str(e)}")
        
        return result
    
    async def communicate_with_stakeholders(
        self, 
        project_id: str,
        message_type: str,
        content: str,
        recipient_role: str,
        requires_response: bool = False
    ) -> TaskResult:
        """Generate communication for project stakeholders.
        
        Args:
            project_id: ID of the project
            message_type: Type of message (status_update, question, issue_alert, etc.)
            content: Content or context for the message
            recipient_role: Role of the recipient (client, team_member, executive, etc.)
            requires_response: Whether a response is needed
            
        Returns:
            TaskResult containing the formatted communication
        """
        # Create a task for generating stakeholder communication
        task = Task(
            task_id=f"communicate_{project_id}_{message_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Generate {message_type} communication for {recipient_role} on project {project_id}",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "message_type": message_type,
                "content": content,
                "recipient_role": recipient_role,
                "requires_response": requires_response
            },
            context=TaskContext(
                notes=(
                    "Generate appropriate communication for the specified stakeholder. "
                    "Tailor the tone, detail level, and format to the recipient role. "
                    "Be clear, concise, and professional."
                )
            ),
            expected_output=(
                "A professionally formatted message appropriate for the recipient role, "
                "including subject line, greeting, body, and closing."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the communication
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store the communication in shared memory if available
            if self.shared_memory:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.shared_memory.store(
                    key=f"communication_{project_id}_{message_type}_{timestamp}",
                    value={
                        "project_id": project_id,
                        "message_type": message_type,
                        "content": content,
                        "recipient_role": recipient_role,
                        "requires_response": requires_response,
                        "message": result.result,
                        "timestamp": timestamp
                    },
                    category="communications"
                )
                
                logger.info(f"Generated {message_type} communication for {recipient_role} on project {project_id}")
        
        return result
    
    async def delegate_task(
        self, 
        task: Task, 
        agent_type: str
    ) -> TaskResult:
        """Delegate a task to another agent.
        
        Args:
            task: The task to delegate
            agent_type: The type of agent to delegate to
            
        Returns:
            TaskResult from the delegated agent
        """
        if not self.agent_registry or agent_type not in self.agent_registry:
            logger.error(f"Cannot delegate task: Agent type {agent_type} not found in registry")
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=task.task_id,
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Agent type {agent_type} not found in registry"
            )
        
        # Get the target agent
        target_agent = self.agent_registry[agent_type]
        
        logger.info(f"Delegating task {task.task_id} to {target_agent.name}")
        
        # Execute the task on the target agent
        return await target_agent.execute_task(task)
    
    async def create_retrospective(
        self, 
        project_id: str,
        completed_tasks: List[Dict[str, Any]],
        issues_encountered: List[Dict[str, Any]],
        timeline_adherence: Dict[str, Any]
    ) -> TaskResult:
        """Create a project retrospective analysis.
        
        Args:
            project_id: ID of the project
            completed_tasks: List of completed tasks with metadata
            issues_encountered: List of issues that arose during the project
            timeline_adherence: Data about timeline adherence
            
        Returns:
            TaskResult containing the retrospective analysis
        """
        # Get project plan if available
        project_plan = None
        if project_id in self.active_projects:
            project_plan = self.active_projects[project_id]
        elif self.shared_memory:
            stored_plan = self.shared_memory.retrieve(
                key=f"project_plan_{project_id}",
                category="project_plans"
            )
            if stored_plan:
                project_plan = ProjectPlan(**stored_plan)
        
        # Create a task for generating the retrospective
        task = Task(
            task_id=f"retrospective_{project_id}",
            description=f"Create a comprehensive retrospective for project {project_id}",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "project_name": project_plan.name if project_plan else "Unknown",
                "project_description": project_plan.description if project_plan else "",
                "completed_tasks": completed_tasks,
                "issues_encountered": issues_encountered,
                "timeline_adherence": timeline_adherence
            },
            context=TaskContext(
                notes=(
                    "Analyze the project execution and create a comprehensive retrospective. "
                    "Identify what went well, what could be improved, and lessons learned. "
                    "Provide specific recommendations for future projects."
                )
            ),
            expected_output=(
                "A detailed retrospective analysis including successes, challenges, "
                "lessons learned, and recommendations for improvement."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the retrospective
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the retrospective from the result
                retrospective = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    retrospective = parsed_result
                except:
                    # If not valid JSON, use the raw result with minimal structure
                    retrospective = {
                        "project_id": project_id,
                        "raw_retrospective": result.result,
                        "timestamp": datetime.now().isoformat()
                    }
                
                if retrospective:
                    # Store the retrospective in shared memory if available
                    if self.shared_memory:
                        self.shared_memory.store(
                            key=f"retrospective_{project_id}",
                            value=retrospective,
                            category="retrospectives"
                        )
                    
                    logger.info(f"Created retrospective for project {project_id}")
            except Exception as e:
                logger.error(f"Error processing retrospective: {str(e)}")
        
        return result
    
    async def prioritize_backlog(
        self, 
        project_id: str,
        backlog_items: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Prioritize a product backlog based on business value, effort, and dependencies.
        
        Args:
            project_id: ID of the project
            backlog_items: List of backlog items with metadata
            constraints: Optional constraints to consider (resources, deadlines, etc.)
            
        Returns:
            TaskResult containing the prioritized backlog
        """
        # Create a task for prioritizing the backlog
        task = Task(
            task_id=f"prioritize_backlog_{project_id}",
            description=f"Prioritize the product backlog for project {project_id}",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "backlog_items": backlog_items,
                "constraints": constraints or {}
            },
            context=TaskContext(
                notes=(
                    "Analyze the backlog items and prioritize them based on business value, "
                    "effort required, dependencies, and any provided constraints. "
                    "Consider both strategic goals and tactical considerations."
                )
            ),
            expected_output=(
                "A prioritized backlog with items ordered by priority, "
                "including rationale for prioritization decisions."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the prioritized backlog
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the prioritized backlog from the result
                prioritized_backlog = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    prioritized_backlog = parsed_result
                except:
                    # If not valid JSON, try to extract JSON block
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        prioritized_backlog = json.loads(json_match.group(1))
                    else:
                        # Not parsable, use the raw result
                        prioritized_backlog = {"raw_prioritization": result.result}
                
                if prioritized_backlog:
                    # Store the prioritized backlog in shared memory if available
                    if self.shared_memory:
                        self.shared_memory.store(
                            key=f"prioritized_backlog_{project_id}",
                            value=prioritized_backlog,
                            category="backlogs"
                        )
                    
                    logger.info(f"Prioritized backlog for project {project_id}")
            except Exception as e:
                logger.error(f"Error processing prioritized backlog: {str(e)}")
        
        return result
    
    async def estimate_task_effort(
        self, 
        tasks: List[Dict[str, Any]],
        estimation_method: str = "t-shirt_sizing"
    ) -> TaskResult:
        """Estimate effort for a list of tasks using specified estimation method.
        
        Args:
            tasks: List of tasks to estimate
            estimation_method: Method to use for estimation 
                               (t-shirt_sizing, fibonacci, story_points, etc.)
            
        Returns:
            TaskResult containing the task effort estimates
        """
        # Create a task for estimating effort
        task = Task(
            task_id=f"estimate_effort_{uuid.uuid4()}",
            description=f"Estimate effort for tasks using {estimation_method} method",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "tasks": tasks,
                "estimation_method": estimation_method
            },
            context=TaskContext(
                notes=(
                    f"Estimate the effort required for each task using the {estimation_method} method. "
                    f"Consider complexity, uncertainty, and amount of work required. "
                    f"Provide rationale for each estimate."
                )
            ),
            expected_output=(
                "A list of tasks with effort estimates using the specified method, "
                "including rationale for each estimate."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and return the estimates
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the estimates from the result
                estimates = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    estimates = parsed_result
                except:
                    # If not valid JSON, try to extract JSON block
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        estimates = json.loads(json_match.group(1))
                    else:
                        # Not parsable, use the raw result
                        estimates = {"raw_estimates": result.result}
                
                if estimates:
                    logger.info(f"Generated effort estimates for {len(tasks)} tasks")
                    
                    # Update the result with the parsed estimates
                    updated_result = TaskResult(
                        agent_id=result.agent_id,
                        agent_name=result.agent_name,
                        task_id=result.task_id,
                        result=estimates,
                        status=result.status,
                        timestamp=result.timestamp,
                        execution_time=result.execution_time,
                        token_usage=result.token_usage
                    )
                    
                    return updated_result
            except Exception as e:
                logger.error(f"Error processing effort estimates: {str(e)}")
        
        return result
    
    async def generate_release_notes(
        self, 
        project_id: str,
        version: str,
        completed_features: List[Dict[str, Any]],
        fixed_issues: Optional[List[Dict[str, Any]]] = None,
        known_issues: Optional[List[Dict[str, Any]]] = None,
        audience: str = "technical"
    ) -> TaskResult:
        """Generate release notes for a project version.
        
        Args:
            project_id: ID of the project
            version: Version number or identifier
            completed_features: List of completed features to include
            fixed_issues: Optional list of fixed issues to include
            known_issues: Optional list of known issues to acknowledge
            audience: Target audience for the notes (technical, end_user, executive)
            
        Returns:
            TaskResult containing the formatted release notes
        """
        # Get project plan if available
        project_plan = None
        if project_id in self.active_projects:
            project_plan = self.active_projects[project_id]
        elif self.shared_memory:
            stored_plan = self.shared_memory.retrieve(
                key=f"project_plan_{project_id}",
                category="project_plans"
            )
            if stored_plan:
                project_plan = ProjectPlan(**stored_plan)
        
        # Create a task for generating release notes
        task = Task(
            task_id=f"release_notes_{project_id}_{version}",
            description=f"Generate release notes for {project_id} version {version}",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "project_id": project_id,
                "project_name": project_plan.name if project_plan else "Unknown",
                "version": version,
                "completed_features": completed_features,
                "fixed_issues": fixed_issues or [],
                "known_issues": known_issues or [],
                "audience": audience
            },
            context=TaskContext(
                notes=(
                    f"Generate comprehensive release notes for version {version} "
                    f"targeted at a {audience} audience. Include all completed features, "
                    f"fixed issues, and known issues where applicable."
                )
            ),
            expected_output=(
                "Professional release notes with appropriate sections for features, "
                "fixes, and known issues, formatted for the target audience."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the release notes
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store the release notes in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"release_notes_{project_id}_{version}",
                    value={
                        "project_id": project_id,
                        "version": version,
                        "notes": result.result,
                        "audience": audience,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="release_notes"
                )
                
                logger.info(f"Generated release notes for {project_id} version {version}")
        
        return result
    
    async def analyze_resource_allocation(
        self, 
        agents_workload: Dict[str, List[Dict[str, Any]]],
        timeframe: str = "current"
    ) -> TaskResult:
        """Analyze resource allocation and suggest optimizations.
        
        Args:
            agents_workload: Dictionary of agent types to their assigned tasks
            timeframe: Timeframe to analyze (current, upcoming, past)
            
        Returns:
            TaskResult containing the resource allocation analysis
        """
        # Create a task for analyzing resource allocation
        task = Task(
            task_id=f"resource_analysis_{timeframe}_{uuid.uuid4()}",
            description=f"Analyze {timeframe} resource allocation and suggest optimizations",
            agent_type=str(AgentRole.PROJECT_MANAGER),
            requirements={
                "agents_workload": agents_workload,
                "timeframe": timeframe
            },
            context=TaskContext(
                notes=(
                    f"Analyze the current workload distribution across agents for the {timeframe} timeframe. "
                    f"Identify bottlenecks, underutilized resources, and suggest optimizations. "
                    f"Consider task complexity, dependencies, and deadlines."
                )
            ),
            expected_output=(
                "A comprehensive resource allocation analysis with identified issues "
                "and specific recommendations for optimization."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the analysis
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the analysis from the result
                analysis = None
                
                # First, try to parse as JSON
                try:
                    parsed_result = json.loads(result.result)
                    analysis = parsed_result
                except:
                    # If not valid JSON, use the raw result
                    analysis = {"raw_analysis": result.result}
                
                if analysis and self.shared_memory:
                    # Store the analysis in shared memory
                    self.shared_memory.store(
                        key=f"resource_analysis_{timeframe}_{datetime.now().strftime('%Y%m%d')}",
                        value=analysis,
                        category="resource_analyses"
                    )
                    
                    logger.info(f"Generated resource allocation analysis for {timeframe} timeframe")
            except Exception as e:
                logger.error(f"Error processing resource analysis: {str(e)}")
        
        return result
    
    def register_agent(self, agent_type: str, agent: BaseAgent) -> None:
        """Register an agent with the Project Manager.
        
        Args:
            agent_type: Type of the agent
            agent: The agent instance
        """
        self.agent_registry[agent_type] = agent
        logger.info(f"Registered agent: {agent.name} as {agent_type}")
    
    def get_active_projects(self) -> List[str]:
        """Get the list of active project IDs.
        
        Returns:
            List of active project IDs
        """
        return list(self.active_projects.keys())
    
    def get_project_plan(self, project_id: str) -> Optional[ProjectPlan]:
        """Get the plan for a specific project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            ProjectPlan if found, None otherwise
        """
        if project_id in self.active_projects:
            return self.active_projects[project_id]
        
        # Try to get from shared memory
        if self.shared_memory:
            stored_plan = self.shared_memory.retrieve(
                key=f"project_plan_{project_id}",
                category="project_plans"
            )
            if stored_plan:
                return ProjectPlan(**stored_plan)
        
        return None
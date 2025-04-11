"""
Multi-agent development system for AI-powered software engineering.

This package contains specialized AI agents that collaborate to design, implement,
and deploy software solutions. Each agent has unique capabilities and expertise
in different aspects of the software development lifecycle.
"""

# Version information
__version__ = "0.1.0"

# Import agent roles enum for external use
from agents.base_agent import (
    AgentRole,
    TaskStatus,
    TaskPriority,
    Task,
    TaskResult,
    TaskContext,
    TaskRequirement,
    ModelProvider,
)

# Base agent class
from agents.base_agent import BaseAgent

# Project management
from agents.manager_agent import ProjectManagerAgent, ProjectPlan, ProjectRequirement

# Architecture and design
from agents.designer_agent import ArchitectureDesignerAgent
from agents.research_agent import ResearchSpecialistAgent

# Frontend agents
try:
    from agents.frontend_agents import (
        UIComponentDeveloper,
        FrontendLogicDeveloper,
        FrontendIntegrationDeveloper,
    )
except ImportError:
    # If frontend agents are not implemented yet, create placeholder
    UIComponentDeveloper = None
    FrontendLogicDeveloper = None
    FrontendIntegrationDeveloper = None

# Backend agents
from agents.backend_agents import (
    APIDeveloper, 
    DatabaseDesigner,
    BackendLogicDeveloper,
)

# DevOps agents
try:
    from agents.devops_agents import (
        InfrastructureDeveloper,
        DeploymentSpecialist,
        SecurityAnalyst,
    )
except ImportError:
    # If DevOps agents are not implemented yet, create placeholder
    InfrastructureDeveloper = None
    DeploymentSpecialist = None
    SecurityAnalyst = None

# QA agents
try:
    from agents.qa_agents import (
        CodeReviewer,
        TestDeveloper,
        UXTester,
    )
except ImportError:
    # If QA agents are not implemented yet, create placeholder
    CodeReviewer = None
    TestDeveloper = None
    UXTester = None

# Support agents
try:
    from agents.research_agent import ResearchSpecialist
    from agents.doc_agent import DocumentationWriter
   #from agents.human_interface_agent import HumanInterfaceAgent
except ImportError:
    # If support agents are not implemented yet, create placeholder
    ResearchSpecialist = None
    DocumentationWriter = None
    #HumanInterfaceAgent = None

# Create a mapping of agent roles to agent classes for easy lookup
AGENT_CLASSES = {
    # Management
    AgentRole.PROJECT_MANAGER: ProjectManagerAgent,
    AgentRole.ARCHITECTURE_DESIGNER: ArchitectureDesignerAgent,
    
    # Frontend
    AgentRole.UI_DEVELOPER: UIComponentDeveloper,
    AgentRole.FRONTEND_LOGIC: FrontendLogicDeveloper,
    AgentRole.FRONTEND_INTEGRATION: FrontendIntegrationDeveloper,
    
    # Backend
    AgentRole.API_DEVELOPER: APIDeveloper,
    AgentRole.DATABASE_DESIGNER: DatabaseDesigner,
    AgentRole.BACKEND_LOGIC: BackendLogicDeveloper,
    
    # DevOps
    AgentRole.INFRASTRUCTURE: InfrastructureDeveloper,
    AgentRole.DEPLOYMENT: DeploymentSpecialist,
    AgentRole.SECURITY: SecurityAnalyst,
    
    # QA
    AgentRole.CODE_REVIEWER: CodeReviewer,
    AgentRole.TEST_DEVELOPER: TestDeveloper,
    AgentRole.UX_TESTER: UXTester,
    
    # Support
    AgentRole.RESEARCHER: ResearchSpecialistAgent,
    AgentRole.DOCUMENTATION: DocumentationWriter,
    #AgentRole.HUMAN_INTERFACE: HumanInterfaceAgent,
}

# Public API
__all__ = [
    # Core classes
    "BaseAgent",
    "AgentRole",
    "TaskStatus",
    "TaskPriority",
    "Task",
    "TaskResult",
    "TaskContext",
    "TaskRequirement",
    "ModelProvider",
    "AGENT_CLASSES",
    
    # Project management
    "ProjectManagerAgent",
    "ProjectPlan",
    "ProjectRequirement",
    
    # Architecture
    "ArchitectureDesignerAgent",
    
    # Frontend
    "UIComponentDeveloper",
    "FrontendLogicDeveloper",
    "FrontendIntegrationDeveloper",
    
    # Backend
    "APIDeveloper",
    "DatabaseDesigner",
    "BackendLogicDeveloper",
    
    # DevOps
    "InfrastructureDeveloper",
    "DeploymentSpecialist",
    "SecurityAnalyst",
    
    # QA
    "CodeReviewer",
    "TestDeveloper",
    "UXTester",
    
    # Support
    "ResearchSpecialist",
    "DocumentationWriter",
    "HumanInterfaceAgent",
]

def create_agent(
    agent_role: AgentRole, 
    name: str = None, 
    model_provider: ModelProvider = ModelProvider.ANTHROPIC,
    model_name: str = "claude-3-sonnet-20240229",
    **kwargs
) -> BaseAgent:
    """
    Factory function to create and configure an agent of a specific role.
    
    Args:
        agent_role: The role of the agent to create
        name: Optional name for the agent (defaults to role name)
        model_provider: LLM provider to use
        model_name: Specific model to use
        **kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        Configured agent instance
        
    Raises:
        ValueError: If the requested agent role is not implemented
    """
    agent_class = AGENT_CLASSES.get(agent_role)
    
    if agent_class is None:
        raise ValueError(f"Agent role {agent_role} is not implemented yet")
    
    # Use role name if no name provided
    if name is None:
        name = agent_role.value.replace("_", " ").title()
    
    return agent_class(
        name=name,
        model_provider=model_provider,
        model_name=model_name,
        **kwargs
    )

def list_available_agents() -> list:
    """
    List all available agent types that are currently implemented.
    
    Returns:
        List of available agent roles
    """
    return [role for role, cls in AGENT_CLASSES.items() if cls is not None]
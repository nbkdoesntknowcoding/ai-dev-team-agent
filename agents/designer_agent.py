"""
Architecture Designer Agent for the multi-agent development system.

This agent is responsible for designing system architecture, making technology decisions,
and translating requirements into technical specifications. It works closely with both
the Project Manager and the UI/UX designer to ensure the architecture supports both
functional requirements and design specifications.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple
import uuid
import base64

from pydantic import BaseModel, Field, validator

from agents.base_agent import (
    BaseAgent, 
    Task, 
    TaskResult, 
    TaskStatus, 
    TaskPriority,
    TaskContext,
    AgentRole,
    ModelProvider
)

# Set up logging
logger = logging.getLogger(__name__)


class DesignConstraint(BaseModel):
    """A constraint that the architecture must satisfy."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    rationale: Optional[str] = None
    source: Optional[str] = None  # e.g., "security", "performance", "regulatory"


class TechnologyOption(BaseModel):
    """A technology option for a component."""
    name: str
    description: str
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    maturity: str  # e.g., "experimental", "emerging", "established", "legacy"
    community_support: str  # e.g., "low", "medium", "high"
    learning_curve: str  # e.g., "low", "medium", "high"
    suitable_for: List[str] = Field(default_factory=list)


class ArchitectureComponent(BaseModel):
    """A component in the system architecture."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    responsibilities: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    technology_options: List[TechnologyOption] = Field(default_factory=list)
    selected_technology: Optional[str] = None
    interfaces: List[Dict[str, Any]] = Field(default_factory=list)
    deployment_considerations: Optional[str] = None
    scalability_considerations: Optional[str] = None
    security_considerations: Optional[str] = None


class UiComponentSpec(BaseModel):
    """Specification for a UI component."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    design_url: Optional[str] = None  # URL to design file or mockup
    design_source: Optional[str] = None  # Base64 encoded design data if embedded
    behavior_description: Optional[str] = None
    data_requirements: List[str] = Field(default_factory=list)
    states: List[str] = Field(default_factory=list)  # e.g., "loading", "error", "empty"
    accessibility_requirements: List[str] = Field(default_factory=list)
    responsive_behavior: Optional[str] = None


class SystemArchitecture(BaseModel):
    """Complete system architecture design."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    version: str = "1.0.0"
    components: List[ArchitectureComponent] = Field(default_factory=list)
    ui_components: List[UiComponentSpec] = Field(default_factory=list)
    data_flow: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[DesignConstraint] = Field(default_factory=list)
    technology_stack: Dict[str, List[str]] = Field(default_factory=dict)
    deployment_architecture: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DesignFeedback(BaseModel):
    """Feedback on a design from stakeholders."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    component_id: str
    feedback_source: str  # e.g., "UI/UX Designer", "Project Manager", "Developer"
    feedback_text: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: str = "pending"  # "pending", "addressed", "rejected"
    response: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ArchitectureDesignerAgent(BaseAgent):
    """Agent specialized in designing system architecture."""
    
    def __init__(self, name, ui_ux_designer_contact=None, model_provider=ModelProvider.ANTHROPIC, model_name="claude-3-sonnet-20240229", **kwargs):
        """Initialize the Architecture Designer agent.
        
        Args:
            name: Human-readable name for this agent
            ui_ux_designer_contact: Contact information for the UI/UX designer
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        self.ui_ux_designer_contact = None
        super().__init__(
            name=name, 
            agent_type=AgentRole.ARCHITECTURE_DESIGNER,
            model_provider=model_provider,
            model_name=model_name, 
            **kwargs
        )
        self.ui_ux_designer_contact = ui_ux_designer_contact
        
        # Track architectures and designs
        self.architectures: Dict[str, SystemArchitecture] = {}
        self.design_feedback: Dict[str, List[DesignFeedback]] = {}
        self.design_iterations: Dict[str, List[Dict[str, Any]]] = {}
        
        # Knowledge base of common patterns and solutions
        self.pattern_library: Dict[str, Dict[str, Any]] = {}
        
        # Technology evaluations
        self.technology_evaluations: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Architecture Designer Agent initialized")
        if ui_ux_designer_contact:
            logger.info(f"UI/UX Designer contact set: {ui_ux_designer_contact}")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Architecture Designer."""
        basic_prompt = (
            f"You are {self.name}, an Architecture Designer in an AI development team. "
            f"Your responsibilities include:\n"
            f"1. Designing system architecture based on project requirements\n"
            f"2. Making technology decisions and recommendations\n"
            f"3. Creating technical specifications for components\n"
            f"4. Ensuring the architecture supports both functional and non-functional requirements\n"
            f"5. Collaborating with UI/UX designers to implement their designs\n\n"
            f"Think step-by-step when designing architecture. Consider trade-offs, scalability, "
            f"security, maintainability, and performance. Justify your technology choices "
            f"and design decisions with clear reasoning.\n\n"
            f"When working with UI/UX designs, focus on translating them into technical "
            f"specifications that developers can implement. Consider the technical implications "
            f"of design choices and provide feedback when necessary."
        )
        
        if self.ui_ux_designer_contact:
            return basic_prompt + f"\n\nA UI/UX designer ({self.ui_ux_designer_contact}) "
            f"will be providing design specifications and mockups. You should incorporate "
            f"these designs into your architecture and ensure that the technical implementation "
            f"can support the intended user experience. Request design clarifications when needed."
        
        return basic_prompt
    
    async def design_system_architecture(
        self, 
        project_id: str,
        name: str,
        description: str,
        requirements: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        existing_technologies: Optional[List[str]] = None
    ) -> TaskResult:
        """Design a comprehensive system architecture based on requirements.
        
        Args:
            project_id: ID of the project
            name: Name for the architecture
            description: Brief description of the system
            requirements: List of functional and non-functional requirements
            constraints: List of constraints the architecture must satisfy
            existing_technologies: Optional list of technologies that must be used
            
        Returns:
            TaskResult containing the system architecture
        """
        # Create a task for architecture design
        task = Task(
            task_id=f"design_architecture_{project_id}",
            description=f"Design system architecture for '{name}'",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "project_id": project_id,
                "name": name,
                "description": description,
                "requirements": requirements,
                "constraints": constraints,
                "existing_technologies": existing_technologies or []
            },
            context=TaskContext(
                notes=(
                    f"Design a comprehensive system architecture for '{name}'. "
                    f"The architecture should satisfy all requirements and constraints. "
                    + (f"It must incorporate these existing technologies: {', '.join(existing_technologies)}. " 
                       if existing_technologies else "")
                )
            ),
            expected_output=(
                "A complete system architecture with components, interfaces, data flow, "
                "technology choices, and deployment considerations."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the architecture
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the architecture from the result
                architecture_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    architecture_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            architecture_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not architecture_data:
                    logger.warning(f"Could not parse architecture as JSON. Attempting to extract from text.")
                    architecture_data = self._extract_architecture_from_text(result.result, name, description)
                
                # Create design constraints from input constraints
                design_constraints = []
                for constraint in constraints:
                    design_constraints.append(
                        DesignConstraint(
                            description=constraint.get("description", ""),
                            priority=constraint.get("priority", TaskPriority.MEDIUM),
                            rationale=constraint.get("rationale", ""),
                            source=constraint.get("source", "")
                        )
                    )
                
                # Create architecture components
                architecture_components = []
                for component_data in architecture_data.get("components", []):
                    # Process technology options for this component
                    tech_options = []
                    for tech in component_data.get("technology_options", []):
                        if isinstance(tech, dict):
                            tech_options.append(TechnologyOption(**tech))
                        else:
                            # If it's just a string, create a basic option
                            tech_options.append(
                                TechnologyOption(
                                    name=tech,
                                    description=f"Option: {tech}",
                                    pros=["Identified as suitable option"],
                                    cons=[],
                                    maturity="unknown",
                                    community_support="unknown",
                                    learning_curve="unknown",
                                    suitable_for=[component_data.get("name", "This component")]
                                )
                            )
                    
                    # Create the component
                    architecture_components.append(
                        ArchitectureComponent(
                            name=component_data.get("name", "Unnamed Component"),
                            description=component_data.get("description", ""),
                            responsibilities=component_data.get("responsibilities", []),
                            dependencies=component_data.get("dependencies", []),
                            technology_options=tech_options,
                            selected_technology=component_data.get("selected_technology"),
                            interfaces=component_data.get("interfaces", []),
                            deployment_considerations=component_data.get("deployment_considerations"),
                            scalability_considerations=component_data.get("scalability_considerations"),
                            security_considerations=component_data.get("security_considerations")
                        )
                    )
                
                # Create UI component specs
                ui_components = []
                for ui_data in architecture_data.get("ui_components", []):
                    ui_components.append(
                        UiComponentSpec(
                            name=ui_data.get("name", "Unnamed UI Component"),
                            description=ui_data.get("description", ""),
                            design_url=ui_data.get("design_url"),
                            behavior_description=ui_data.get("behavior_description"),
                            data_requirements=ui_data.get("data_requirements", []),
                            states=ui_data.get("states", []),
                            accessibility_requirements=ui_data.get("accessibility_requirements", []),
                            responsive_behavior=ui_data.get("responsive_behavior")
                        )
                    )
                
                # Create the system architecture
                architecture = SystemArchitecture(
                    name=name,
                    description=description,
                    components=architecture_components,
                    ui_components=ui_components,
                    data_flow=architecture_data.get("data_flow", []),
                    constraints=design_constraints,
                    technology_stack=architecture_data.get("technology_stack", {}),
                    deployment_architecture=architecture_data.get("deployment_architecture")
                )
                
                # Store the architecture
                self.architectures[architecture.id] = architecture
                
                # Initialize feedback and iterations tracking
                self.design_feedback[architecture.id] = []
                self.design_iterations[architecture.id] = [{
                    "version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "architecture": architecture.dict()
                }]
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"architecture_{project_id}",
                        value=architecture.dict(),
                        category="architectures"
                    )
                
                logger.info(f"Created system architecture '{name}' with {len(architecture_components)} components and {len(ui_components)} UI components")
                
                # Return the architecture as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=architecture.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"architecture_id": architecture.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing system architecture: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_architecture_from_text(
        self, 
        text: str, 
        name: str, 
        description: str
    ) -> Dict[str, Any]:
        """Extract structured architecture data from unstructured text.
        
        Args:
            text: The text to extract from
            name: The name of the architecture
            description: The description of the architecture
            
        Returns:
            Structured architecture data
        """
        architecture_data = {
            "name": name,
            "description": description,
            "components": [],
            "ui_components": [],
            "data_flow": [],
            "technology_stack": {}
        }
        
        # Extract components
        component_sections = re.findall(
            r'(?i)#+\s*(?:Component|Module|Service):\s*([^\n]+)(?:\n+(.+?))?(?=\n+#+\s*(?:Component|Module|Service|Data Flow|UI Component|Technology Stack|Deployment)|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in component_sections:
            component = {
                "name": title.strip(),
                "description": "",
                "responsibilities": [],
                "dependencies": [],
                "technology_options": [],
                "interfaces": []
            }
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Overview):\s*([^\n]+)', content)
            if desc_match:
                component["description"] = desc_match.group(1).strip()
            
            # Extract responsibilities
            resp_match = re.search(
                r'(?i)(?:Responsibilities|Functions|Purpose):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if resp_match:
                responsibilities = re.findall(r'[-*]\s*([^\n]+)', resp_match.group(1))
                component["responsibilities"] = [r.strip() for r in responsibilities]
            
            # Extract dependencies
            dep_match = re.search(
                r'(?i)(?:Dependencies|Depends On):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if dep_match:
                dependencies = re.findall(r'[-*]\s*([^\n]+)', dep_match.group(1))
                component["dependencies"] = [d.strip() for d in dependencies]
            
            # Extract technology options
            tech_match = re.search(
                r'(?i)(?:Technology Options|Technologies|Tech Stack):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if tech_match:
                technologies = re.findall(r'[-*]\s*([^\n]+)', tech_match.group(1))
                component["technology_options"] = [t.strip() for t in technologies]
            
            # Extract selected technology
            sel_tech_match = re.search(r'(?i)(?:Selected Technology|Chosen Technology):\s*([^\n]+)', content)
            if sel_tech_match:
                component["selected_technology"] = sel_tech_match.group(1).strip()
            
            # Extract interfaces
            int_match = re.search(
                r'(?i)(?:Interfaces|APIs|Endpoints):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if int_match:
                interfaces = re.findall(r'[-*]\s*([^\n]+)', int_match.group(1))
                for interface in interfaces:
                    component["interfaces"].append({"description": interface.strip()})
            
            # Extract considerations
            dep_cons_match = re.search(r'(?i)Deployment Considerations:\s*([^\n]+(?:\n+[^#][^\n]+)*)', content)
            if dep_cons_match:
                component["deployment_considerations"] = dep_cons_match.group(1).strip()
            
            scale_cons_match = re.search(r'(?i)Scalability Considerations:\s*([^\n]+(?:\n+[^#][^\n]+)*)', content)
            if scale_cons_match:
                component["scalability_considerations"] = scale_cons_match.group(1).strip()
            
            sec_cons_match = re.search(r'(?i)Security Considerations:\s*([^\n]+(?:\n+[^#][^\n]+)*)', content)
            if sec_cons_match:
                component["security_considerations"] = sec_cons_match.group(1).strip()
            
            architecture_data["components"].append(component)
        
        # Extract UI components
        ui_sections = re.findall(
            r'(?i)#+\s*(?:UI Component|Frontend Component|User Interface):\s*([^\n]+)(?:\n+(.+?))?(?=\n+#+\s*(?:Component|Module|Service|Data Flow|UI Component|Technology Stack|Deployment)|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in ui_sections:
            ui_component = {
                "name": title.strip(),
                "description": "",
                "data_requirements": [],
                "states": [],
                "accessibility_requirements": []
            }
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Overview):\s*([^\n]+)', content)
            if desc_match:
                ui_component["description"] = desc_match.group(1).strip()
            
            # Extract behavior description
            behavior_match = re.search(r'(?i)(?:Behavior|Interactions|User Flow):\s*([^\n]+(?:\n+[^#][^\n]+)*)', content)
            if behavior_match:
                ui_component["behavior_description"] = behavior_match.group(1).strip()
            
            # Extract data requirements
            data_match = re.search(
                r'(?i)(?:Data Requirements|Data Needs|Required Data):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if data_match:
                data_reqs = re.findall(r'[-*]\s*([^\n]+)', data_match.group(1))
                ui_component["data_requirements"] = [d.strip() for d in data_reqs]
            
            # Extract states
            states_match = re.search(
                r'(?i)(?:States|UI States|Component States):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if states_match:
                states = re.findall(r'[-*]\s*([^\n]+)', states_match.group(1))
                ui_component["states"] = [s.strip() for s in states]
            
            # Extract accessibility requirements
            a11y_match = re.search(
                r'(?i)(?:Accessibility|A11y Requirements):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if a11y_match:
                a11y_reqs = re.findall(r'[-*]\s*([^\n]+)', a11y_match.group(1))
                ui_component["accessibility_requirements"] = [a.strip() for a in a11y_reqs]
            
            # Extract responsive behavior
            resp_match = re.search(r'(?i)(?:Responsive Behavior|Responsiveness):\s*([^\n]+(?:\n+[^#][^\n]+)*)', content)
            if resp_match:
                ui_component["responsive_behavior"] = resp_match.group(1).strip()
            
            # Extract design URL
            design_url_match = re.search(r'(?i)(?:Design URL|Mockup URL|Figma URL):\s*([^\n]+)', content)
            if design_url_match:
                ui_component["design_url"] = design_url_match.group(1).strip()
            
            architecture_data["ui_components"].append(ui_component)
        
        # Extract data flow
        data_flow_section = re.search(
            r'(?i)#+\s*(?:Data Flow|Information Flow)(?:\n+(.+?))?(?=\n+#+\s*(?:Component|Module|Service|UI Component|Technology Stack|Deployment)|\Z)',
            text,
            re.DOTALL
        )
        
        if data_flow_section and data_flow_section.group(1):
            flow_content = data_flow_section.group(1)
            
            # Try to extract structured flow descriptions
            flow_items = re.findall(r'[-*]\s*([^\n]+)', flow_content)
            
            for item in flow_items:
                # Try to parse source -> destination format
                flow_match = re.search(r'([^->]+)\s*->\s*([^:]+)(?::\s*(.+))?', item)
                if flow_match:
                    source = flow_match.group(1).strip()
                    destination = flow_match.group(2).strip()
                    description = flow_match.group(3).strip() if flow_match.group(3) else ""
                    
                    architecture_data["data_flow"].append({
                        "source": source,
                        "destination": destination,
                        "description": description
                    })
                else:
                    # Just add as generic flow description
                    architecture_data["data_flow"].append({
                        "description": item.strip()
                    })
        
        # Extract technology stack
        tech_stack_section = re.search(
            r'(?i)#+\s*(?:Technology Stack|Tech Stack)(?:\n+(.+?))?(?=\n+#+\s*(?:Component|Module|Service|Data Flow|UI Component|Deployment)|\Z)',
            text,
            re.DOTALL
        )
        
        if tech_stack_section and tech_stack_section.group(1):
            stack_content = tech_stack_section.group(1)
            
            # Look for categorized technology listings
            categories = re.findall(
                r'(?i)(?:##\s*([^:\n]+):|[-*]\s*([^:]+):)\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                stack_content
            )
            
            for cat_match in categories:
                category = (cat_match[0] or cat_match[1]).strip().lower()
                tech_list = cat_match[2]
                
                techs = re.findall(r'[-*]\s*([^\n]+)', tech_list)
                clean_techs = [t.strip() for t in techs]
                
                if category and clean_techs:
                    architecture_data["technology_stack"][category] = clean_techs
            
            # If no categories found, try to extract a simple list
            if not architecture_data["technology_stack"]:
                techs = re.findall(r'[-*]\s*([^\n]+)', stack_content)
                if techs:
                    architecture_data["technology_stack"]["general"] = [t.strip() for t in techs]
        
        # Extract deployment architecture
        deployment_section = re.search(
            r'(?i)#+\s*(?:Deployment Architecture|Deployment)(?:\n+(.+?))?(?=\n+#+\s*(?:Component|Module|Service|Data Flow|UI Component|Technology Stack)|\Z)',
            text,
            re.DOTALL
        )
        
        if deployment_section and deployment_section.group(1):
            deployment_content = deployment_section.group(1).strip()
            architecture_data["deployment_architecture"] = {
                "description": deployment_content
            }
            
            # Try to extract environments
            environments = re.findall(
                r'(?i)(?:##\s*([^:\n]+) Environment:|[-*]\s*([^:]+) Environment:)\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                deployment_content
            )
            
            if environments:
                architecture_data["deployment_architecture"]["environments"] = []
                
                for env_match in environments:
                    env_name = (env_match[0] or env_match[1]).strip()
                    env_content = env_match[2]
                    
                    environment = {
                        "name": env_name,
                        "components": []
                    }
                    
                    # Extract component deployments
                    components = re.findall(r'[-*]\s*([^\n]+)', env_content)
                    for comp in components:
                        environment["components"].append({
                            "description": comp.strip()
                        })
                    
                    architecture_data["deployment_architecture"]["environments"].append(environment)
        
        return architecture_data
    
    async def incorporate_design_specifications(
        self, 
        architecture_id: str,
        design_specs: Dict[str, Any],
        design_source: Optional[str] = None,
        design_url: Optional[str] = None
    ) -> TaskResult:
        """Incorporate UI/UX design specifications into the architecture.
        
        Args:
            architecture_id: ID of the architecture to update
            design_specs: Design specifications from the UI/UX designer
            design_source: Optional base64 encoded design file data
            design_url: Optional URL to design files
            
        Returns:
            TaskResult containing the updated architecture
        """
        # Check if architecture exists
        if architecture_id not in self.architectures:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"incorporate_design_{architecture_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Architecture with ID {architecture_id} not found"
            )
        
        # Get the current architecture
        architecture = self.architectures[architecture_id]
        
        # Create a task for incorporating the design
        task = Task(
            task_id=f"incorporate_design_{architecture_id}",
            description=f"Incorporate UI/UX design specifications into the architecture '{architecture.name}'",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "architecture_id": architecture_id,
                "architecture_name": architecture.name,
                "design_specs": design_specs,
                "has_design_source": design_source is not None,
                "has_design_url": design_url is not None
            },
            context=TaskContext(
                notes=(
                    f"Incorporate the provided UI/UX design specifications into the existing "
                    f"architecture '{architecture.name}'. Update or add UI components as needed. "
                    f"Ensure the technical implementation can support the intended user experience."
                    + (f"\nDesign files are available at: {design_url}" if design_url else "")
                )
            ),
            expected_output=(
                "Updated system architecture with UI components that incorporate the design specifications. "
                "Include any technical considerations or adjustments needed to support the designs."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, update the architecture
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the updated architecture from the result
                updated_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    updated_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            updated_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the textual analysis
                if not updated_data:
                    logger.warning(f"Could not parse updated architecture as JSON. Processing as text update.")
                    # Create a simple structure to track the updates
                    updated_data = {
                        "ui_components": design_specs.get("ui_components", []),
                        "analysis": result.result
                    }
                
                # Create or update UI components
                new_ui_components = []
                existing_ui_ids = {ui.id: ui for ui in architecture.ui_components}
                
                for ui_data in updated_data.get("ui_components", []):
                    # Check if this is an update to an existing component
                   # Check if this is an update to an existing component
                    existing_id = None
                    if "id" in ui_data and ui_data["id"] in existing_ui_ids:
                        existing_id = ui_data["id"]
                    else:
                        # Try to match by name
                        for ui_id, ui in existing_ui_ids.items():
                            if ui.name == ui_data.get("name"):
                                existing_id = ui_id
                                break
                    
                    if existing_id:
                        # Update existing component
                        ui_component = existing_ui_ids[existing_id]
                        
                        # Update fields
                        ui_component.description = ui_data.get("description", ui_component.description)
                        ui_component.behavior_description = ui_data.get("behavior_description", ui_component.behavior_description)
                        ui_component.data_requirements = ui_data.get("data_requirements", ui_component.data_requirements)
                        ui_component.states = ui_data.get("states", ui_component.states)
                        ui_component.accessibility_requirements = ui_data.get("accessibility_requirements", ui_component.accessibility_requirements)
                        ui_component.responsive_behavior = ui_data.get("responsive_behavior", ui_component.responsive_behavior)
                        
                        # Only update URLs and sources if provided
                        if design_url and not ui_component.design_url:
                            ui_component.design_url = design_url
                        if "design_url" in ui_data:
                            ui_component.design_url = ui_data["design_url"]
                        
                        if design_source and not ui_component.design_source:
                            ui_component.design_source = design_source
                        
                        new_ui_components.append(ui_component)
                    else:
                        # Create new component
                        ui_component = UiComponentSpec(
                            name=ui_data.get("name", "Unnamed UI Component"),
                            description=ui_data.get("description", ""),
                            design_url=ui_data.get("design_url") or design_url,
                            design_source=design_source,
                            behavior_description=ui_data.get("behavior_description"),
                            data_requirements=ui_data.get("data_requirements", []),
                            states=ui_data.get("states", []),
                            accessibility_requirements=ui_data.get("accessibility_requirements", []),
                            responsive_behavior=ui_data.get("responsive_behavior")
                        )
                        
                        new_ui_components.append(ui_component)
                
                # Keep any existing components not explicitly updated
                for ui_id, ui in existing_ui_ids.items():
                    if ui not in new_ui_components:
                        new_ui_components.append(ui)
                
                # Update the architecture
                architecture.ui_components = new_ui_components
                architecture.updated_at = datetime.now().isoformat()
                
                # Increment version number (minor version)
                version_parts = architecture.version.split('.')
                if len(version_parts) >= 3:
                    version_parts[1] = str(int(version_parts[1]) + 1)
                    architecture.version = '.'.join(version_parts)
                
                # Store the updated architecture
                self.architectures[architecture_id] = architecture
                
                # Track the design iteration
                self.design_iterations[architecture_id].append({
                    "version": architecture.version,
                    "timestamp": datetime.now().isoformat(),
                    "architecture": architecture.dict(),
                    "design_url": design_url,
                    "has_design_source": design_source is not None
                })
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"architecture_{architecture_id}",
                        value=architecture.dict(),
                        category="architectures"
                    )
                    
                    # Also store design iteration
                    self.shared_memory.store(
                        key=f"architecture_iteration_{architecture_id}_{architecture.version}",
                        value={
                            "version": architecture.version,
                            "timestamp": datetime.now().isoformat(),
                            "design_url": design_url,
                            "has_design_source": design_source is not None
                        },
                        category="architecture_iterations"
                    )
                
                logger.info(f"Updated architecture '{architecture.name}' with UI/UX design specifications")
                
                # Return the updated architecture
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=architecture.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "architecture_id": architecture_id,
                        "version": architecture.version
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error updating architecture with design specifications: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def process_design_feedback(
        self, 
        architecture_id: str,
        component_id: str,
        feedback_source: str,
        feedback_text: str,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> TaskResult:
        """Process feedback on a design component and update the architecture.
        
        Args:
            architecture_id: ID of the architecture
            component_id: ID of the component receiving feedback
            feedback_source: Source of the feedback (e.g., "UI/UX Designer")
            feedback_text: The feedback text
            priority: Priority of the feedback
            
        Returns:
            TaskResult containing the response to the feedback
        """
        # Check if architecture exists
        if architecture_id not in self.architectures:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"process_feedback_{architecture_id}_{component_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Architecture with ID {architecture_id} not found"
            )
        
        # Get the current architecture
        architecture = self.architectures[architecture_id]
        
        # Create the feedback record
        feedback = DesignFeedback(
            component_id=component_id,
            feedback_source=feedback_source,
            feedback_text=feedback_text,
            priority=priority
        )
        
        # Store the feedback
        self.design_feedback[architecture_id].append(feedback)
        
        # Find the component
        target_component = None
        is_ui_component = False
        
        for component in architecture.components:
            if component.id == component_id:
                target_component = component
                break
        
        if not target_component:
            for ui_component in architecture.ui_components:
                if ui_component.id == component_id:
                    target_component = ui_component
                    is_ui_component = True
                    break
        
        if not target_component:
            feedback.status = "rejected"
            feedback.response = f"Component with ID {component_id} not found in architecture"
            
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"process_feedback_{architecture_id}_{component_id}",
                result=feedback.dict(),
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=feedback.response
            )
        
        # Create a task for processing the feedback
        component_type = "UI component" if is_ui_component else "component"
        task = Task(
            task_id=f"process_feedback_{architecture_id}_{component_id}",
            description=f"Process design feedback for {component_type} '{target_component.name}'",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "architecture_id": architecture_id,
                "component_id": component_id,
                "component_name": target_component.name,
                "component_type": component_type,
                "feedback_source": feedback_source,
                "feedback_text": feedback_text,
                "priority": priority
            },
            context=TaskContext(
                notes=(
                    f"Process feedback from {feedback_source} for the {component_type} '{target_component.name}'. "
                    f"Determine how to address the feedback and what changes are needed to the architecture. "
                    f"Feedback: {feedback_text}"
                )
            ),
            expected_output=(
                "A response to the feedback that includes whether it will be addressed, "
                "what changes will be made, and any technical considerations or trade-offs."
            ),
            priority=priority
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, update the feedback and possibly the architecture
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Update the feedback record
                feedback.status = "addressed"
                feedback.response = result.result
                
                # Check if we need to update the component based on the feedback
                update_needed = "yes" in result.result.lower() or "will implement" in result.result.lower() or "updating" in result.result.lower()
                
                if update_needed:
                    logger.info(f"Updating {component_type} '{target_component.name}' based on feedback")
                    
                    # For UI components, we might want to extract specific changes
                    if is_ui_component:
                        ui_component = target_component
                        
                        # Look for specific changes in the result
                        behavior_match = re.search(r'(?i)behavior:?\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', result.result)
                        if behavior_match:
                            ui_component.behavior_description = behavior_match.group(1).strip()
                        
                        data_match = re.search(r'(?i)data requirements:?\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', result.result)
                        if data_match:
                            data_text = data_match.group(1).strip()
                            data_items = re.findall(r'[-*]\s*([^\n]+)', data_text)
                            if data_items:
                                ui_component.data_requirements = [d.strip() for d in data_items]
                        
                        states_match = re.search(r'(?i)states:?\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', result.result)
                        if states_match:
                            states_text = states_match.group(1).strip()
                            state_items = re.findall(r'[-*]\s*([^\n]+)', states_text)
                            if state_items:
                                ui_component.states = [s.strip() for s in state_items]
                    
                    # For regular components, we might update the description or tech choices
                    else:
                        component = target_component
                        
                        # Look for technology changes
                        tech_match = re.search(r'(?i)technology:?\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', result.result)
                        if tech_match:
                            tech_text = tech_match.group(1).strip()
                            
                            selected_match = re.search(r'(?i)selected:?\s*([^\n]+)', tech_text)
                            if selected_match:
                                component.selected_technology = selected_match.group(1).strip()
                    
                    # Update the architecture timestamp
                    architecture.updated_at = datetime.now().isoformat()
                
                # Store the updated architecture if needed
                if update_needed:
                    self.architectures[architecture_id] = architecture
                    
                    # Store in shared memory if available
                    if self.shared_memory:
                        self.shared_memory.store(
                            key=f"architecture_{architecture_id}",
                            value=architecture.dict(),
                            category="architectures"
                        )
                
                # Store the feedback
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"design_feedback_{feedback.id}",
                        value=feedback.dict(),
                        category="design_feedback"
                    )
                
                logger.info(f"Processed design feedback for {component_type} '{target_component.name}'")
                
                # Return the feedback response
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=feedback.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "feedback_id": feedback.id,
                        "architecture_updated": update_needed
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing design feedback: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def define_component_interfaces(
        self, 
        architecture_id: str,
        component_id: str,
        interface_requirements: List[Dict[str, Any]]
    ) -> TaskResult:
        """Define detailed interfaces for a component in the architecture.
        
        Args:
            architecture_id: ID of the architecture
            component_id: ID of the component to define interfaces for
            interface_requirements: Requirements for the interfaces
            
        Returns:
            TaskResult containing the defined interfaces
        """
        # Check if architecture exists
        if architecture_id not in self.architectures:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"define_interfaces_{component_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Architecture with ID {architecture_id} not found"
            )
        
        # Get the current architecture
        architecture = self.architectures[architecture_id]
        
        # Find the component
        target_component = None
        
        for component in architecture.components:
            if component.id == component_id:
                target_component = component
                break
        
        if not target_component:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"define_interfaces_{component_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Component with ID {component_id} not found in architecture"
            )
        
        # Create a task for defining the interfaces
        task = Task(
            task_id=f"define_interfaces_{component_id}",
            description=f"Define interfaces for component '{target_component.name}'",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "architecture_id": architecture_id,
                "component_id": component_id,
                "component_name": target_component.name,
                "component_description": target_component.description,
                "component_responsibilities": target_component.responsibilities,
                "technology": target_component.selected_technology,
                "interface_requirements": interface_requirements
            },
            context=TaskContext(
                notes=(
                    f"Define detailed interfaces for the component '{target_component.name}'. "
                    f"Consider the component's responsibilities and how it interacts with other components. "
                    f"The interfaces should be compatible with the selected technology: "
                    f"{target_component.selected_technology}"
                )
            ),
            expected_output=(
                "Detailed interface definitions including input/output parameters, data formats, "
                "error handling, and usage examples."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, update the component interfaces
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract interfaces from the result
                interfaces = []
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    parsed_result = json.loads(result.result)
                    if "interfaces" in parsed_result:
                        interfaces = parsed_result["interfaces"]
                    else:
                        interfaces = parsed_result
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            parsed_json = json.loads(json_match.group(1))
                            if "interfaces" in parsed_json:
                                interfaces = parsed_json["interfaces"]
                            else:
                                interfaces = parsed_json
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract interfaces from text
                if not interfaces:
                    interfaces = self._extract_interfaces_from_text(result.result)
                
                # Update the component's interfaces
                target_component.interfaces = interfaces
                
                # Update the architecture timestamp
                architecture.updated_at = datetime.now().isoformat()
                
                # Store the updated architecture
                self.architectures[architecture_id] = architecture
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"architecture_{architecture_id}",
                        value=architecture.dict(),
                        category="architectures"
                    )
                    
                    # Also store the component interfaces separately for easier access
                    self.shared_memory.store(
                        key=f"component_interfaces_{component_id}",
                        value=interfaces,
                        category="component_interfaces"
                    )
                
                logger.info(f"Defined {len(interfaces)} interfaces for component '{target_component.name}'")
                
                # Return the interfaces
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=interfaces,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "component_id": component_id,
                        "component_name": target_component.name
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error defining component interfaces: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_interfaces_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract interface definitions from unstructured text.
        
        Args:
            text: The text to extract from
            
        Returns:
            List of interface definitions
        """
        interfaces = []
        
        # Look for interface sections
        interface_sections = re.findall(
            r'(?i)#+\s*(?:Interface|API|Endpoint):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+\s*(?:Interface|API|Endpoint)|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in interface_sections:
            interface = {
                "name": title.strip(),
                "description": "",
                "type": "unknown",
                "parameters": [],
                "returns": [],
                "errors": [],
                "example": ""
            }
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Purpose):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if desc_match:
                interface["description"] = desc_match.group(1).strip()
            
            # Extract interface type
            type_match = re.search(r'(?i)(?:Type|Interface Type):\s*([^\n]+)', content)
            if type_match:
                interface["type"] = type_match.group(1).strip().lower()
            
            # Extract parameters
            params_match = re.search(
                r'(?i)(?:Parameters|Inputs):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if params_match:
                params_text = params_match.group(1)
                param_items = re.findall(r'[-*]\s*([^\n]+)', params_text)
                
                for param in param_items:
                    param_parts = re.search(r'`?([^`:\s]+)`?\s*(?:\(([^)]+)\))?(?::\s*(.+))?', param)
                    if param_parts:
                        param_name = param_parts.group(1).strip()
                        param_type = param_parts.group(2).strip() if param_parts.group(2) else "unknown"
                        param_desc = param_parts.group(3).strip() if param_parts.group(3) else ""
                        
                        interface["parameters"].append({
                            "name": param_name,
                            "type": param_type,
                            "description": param_desc,
                            "required": "optional" not in param.lower()
                        })
                    else:
                        # Simple parameter description
                        interface["parameters"].append({
                            "description": param.strip()
                        })
            
            # Extract returns
            returns_match = re.search(
                r'(?i)(?:Returns|Output):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if returns_match:
                returns_text = returns_match.group(1)
                return_items = re.findall(r'[-*]\s*([^\n]+)', returns_text)
                
                for ret in return_items:
                    ret_parts = re.search(r'`?([^`:\s]+)`?\s*(?:\(([^)]+)\))?(?::\s*(.+))?', ret)
                    if ret_parts:
                        ret_name = ret_parts.group(1).strip()
                        ret_type = ret_parts.group(2).strip() if ret_parts.group(2) else "unknown"
                        ret_desc = ret_parts.group(3).strip() if ret_parts.group(3) else ""
                        
                        interface["returns"].append({
                            "name": ret_name,
                            "type": ret_type,
                            "description": ret_desc
                        })
                    else:
                        # Simple return description
                        interface["returns"].append({
                            "description": ret.strip()
                        })
            
            # Extract errors
            errors_match = re.search(
                r'(?i)(?:Errors|Exceptions|Error Handling):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if errors_match:
                errors_text = errors_match.group(1)
                error_items = re.findall(r'[-*]\s*([^\n]+)', errors_text)
                
                for err in error_items:
                    err_parts = re.search(r'`?([^`:\s]+)`?\s*(?:\(([^)]+)\))?(?::\s*(.+))?', err)
                    if err_parts:
                        err_name = err_parts.group(1).strip()
                        err_code = err_parts.group(2).strip() if err_parts.group(2) else ""
                        err_desc = err_parts.group(3).strip() if err_parts.group(3) else ""
                        
                        interface["errors"].append({
                            "name": err_name,
                            "code": err_code,
                            "description": err_desc
                        })
                    else:
                        # Simple error description
                        interface["errors"].append({
                            "description": err.strip()
                        })
            
            # Extract examples
            example_match = re.search(
                r'(?i)(?:Example|Usage Example):\s*\n+(```[\s\S]*?```|(?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if example_match:
                interface["example"] = example_match.group(1).strip()
            
            interfaces.append(interface)
        
        # If no structured interfaces found, but we have code blocks, try to extract from there
        if not interfaces:
            code_blocks = re.findall(r'```(?:[a-z]*)\n([\s\S]*?)```', text)
            
            for i, code in enumerate(code_blocks):
                # Try to determine if this is an interface definition
                if (
                    "interface" in code.lower() or 
                    "class" in code.lower() or 
                    "function" in code.lower() or 
                    "def " in code.lower() or 
                    "@api" in code.lower()
                ):
                    interfaces.append({
                        "name": f"Interface {i+1}",
                        "description": "Extracted from code example",
                        "type": "code",
                        "code": code.strip()
                    })
        
        return interfaces
    
    async def evaluate_technology_option(
        self, 
        technology_name: str,
        evaluation_criteria: List[str],
        use_case: str,
        alternatives: Optional[List[str]] = None
    ) -> TaskResult:
        """Evaluate a technology option based on specific criteria.
        
        Args:
            technology_name: Name of the technology to evaluate
            evaluation_criteria: Criteria to evaluate against
            use_case: The specific use case
            alternatives: Optional list of alternative technologies to compare with
            
        Returns:
            TaskResult containing the technology evaluation
        """
        # Check if we already have an evaluation for this technology
        cached_evaluation = self.technology_evaluations.get(technology_name.lower())
        
        if cached_evaluation and set(cached_evaluation.get("criteria", [])) >= set(evaluation_criteria):
            logger.info(f"Using cached evaluation for {technology_name}")
            
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"evaluate_technology_{technology_name.lower().replace(' ', '_')}",
                result=cached_evaluation,
                status=TaskStatus.COMPLETED,
                execution_time=0.1,  # Negligible time for cache hit
                metadata={
                    "technology": technology_name,
                    "from_cache": True
                }
            )
        
        # Create a task for evaluating the technology
        task = Task(
            task_id=f"evaluate_technology_{technology_name.lower().replace(' ', '_')}",
            description=f"Evaluate technology: {technology_name} for {use_case}",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "technology": technology_name,
                "criteria": evaluation_criteria,
                "use_case": use_case,
                "alternatives": alternatives or []
            },
            context=TaskContext(
                notes=(
                    f"Evaluate {technology_name} as a technology option for {use_case}. "
                    f"Assess it based on the provided criteria and compare with alternatives "
                    f"if provided."
                )
            ),
            expected_output=(
                "A comprehensive evaluation of the technology including scores for each criterion, "
                "pros and cons, community support assessment, maturity evaluation, and "
                "specific considerations for the use case."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the evaluation
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract evaluation from the result
                evaluation = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    evaluation = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            evaluation = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the result text as is
                if not evaluation:
                    evaluation = {
                        "technology": technology_name,
                        "criteria": evaluation_criteria,
                        "use_case": use_case,
                        "evaluation": result.result,
                        "alternatives": alternatives or []
                    }
                
                # Add metadata
                evaluation.update({
                    "timestamp": datetime.now().isoformat(),
                    "criteria": evaluation_criteria
                })
                
                # Cache the evaluation
                self.technology_evaluations[technology_name.lower()] = evaluation
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"technology_evaluation_{technology_name.lower().replace(' ', '_')}",
                        value=evaluation,
                        category="technology_evaluations"
                    )
                
                logger.info(f"Evaluated technology {technology_name} for {use_case}")
                
                # Return the evaluation
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=evaluation,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "technology": technology_name,
                        "use_case": use_case
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing technology evaluation: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def map_design_to_components(
        self, 
        design_url: str,
        design_description: str,
        existing_architecture_id: Optional[str] = None
    ) -> TaskResult:
        """Map a UI/UX design to technical components.
        
        Args:
            design_url: URL to the design
            design_description: Description of the design
            existing_architecture_id: Optional ID of an existing architecture to map against
            
        Returns:
            TaskResult containing the component mapping
        """
        # Get existing architecture if specified
        existing_architecture = None
        if existing_architecture_id:
            existing_architecture = self.architectures.get(existing_architecture_id)
            
            if not existing_architecture and self.shared_memory:
                arch_data = self.shared_memory.retrieve(
                    key=f"architecture_{existing_architecture_id}",
                    category="architectures"
                )
                if arch_data:
                    existing_architecture = SystemArchitecture(**arch_data)
                    # Create a task for mapping design to components
        task = Task(
            task_id=f"map_design_to_components_{uuid.uuid4()}",
            description=f"Map UI/UX design to technical components",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "design_url": design_url,
                "design_description": design_description,
                "has_existing_architecture": existing_architecture is not None,
                "existing_components": (
                    [c.dict() for c in existing_architecture.components] if existing_architecture else []
                ),
                "existing_ui_components": (
                    [ui.dict() for ui in existing_architecture.ui_components] if existing_architecture else []
                )
            },
            context=TaskContext(
                notes=(
                    f"Analyze the UI/UX design at {design_url} and map it to technical components. "
                    f"Design description: {design_description}"
                    + (
                        f"\nMap the design to the existing architecture components where possible." 
                        if existing_architecture else ""
                    )
                )
            ),
            expected_output=(
                "A comprehensive mapping of UI/UX design elements to technical components, "
                "including required backend services, data models, and API endpoints."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process the mapping
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract mapping from the result
                mapping = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    mapping = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            mapping = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the result text as is
                if not mapping:
                    mapping = {
                        "design_url": design_url,
                        "design_description": design_description,
                        "mapping": result.result
                    }
                else:
                    # Ensure basic metadata is included
                    if "design_url" not in mapping:
                        mapping["design_url"] = design_url
                    if "design_description" not in mapping:
                        mapping["design_description"] = design_description
                
                # Add timestamp
                mapping["timestamp"] = datetime.now().isoformat()
                
                # Store in shared memory if available
                if self.shared_memory:
                    mapping_id = f"design_mapping_{uuid.uuid4()}"
                    self.shared_memory.store(
                        key=mapping_id,
                        value=mapping,
                        category="design_mappings"
                    )
                    
                    # Link to existing architecture if specified
                    if existing_architecture_id:
                        self.shared_memory.store(
                            key=f"architecture_design_mapping_{existing_architecture_id}",
                            value={
                                "architecture_id": existing_architecture_id,
                                "mapping_id": mapping_id,
                                "design_url": design_url,
                                "timestamp": datetime.now().isoformat()
                            },
                            category="architecture_design_mappings"
                        )
                
                logger.info(f"Mapped UI/UX design to technical components")
                
                # Return the mapping
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=mapping,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "mapping_id": mapping_id if self.shared_memory else None,
                        "architecture_id": existing_architecture_id
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing design-to-component mapping: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def create_technical_specification(
        self, 
        component_id: str,
        architecture_id: Optional[str] = None,
        additional_requirements: Optional[List[Dict[str, Any]]] = None
    ) -> TaskResult:
        """Create a detailed technical specification for a component.
        
        Args:
            component_id: ID of the component
            architecture_id: Optional ID of the architecture containing the component
            additional_requirements: Optional additional requirements for the specification
            
        Returns:
            TaskResult containing the technical specification
        """
        # Find the component
        component = None
        is_ui_component = False
        architecture = None
        
        # If architecture ID is provided, look there first
        if architecture_id:
            architecture = self.architectures.get(architecture_id)
            
            if not architecture and self.shared_memory:
                arch_data = self.shared_memory.retrieve(
                    key=f"architecture_{architecture_id}",
                    category="architectures"
                )
                if arch_data:
                    architecture = SystemArchitecture(**arch_data)
            
            if architecture:
                # Look for the component
                for comp in architecture.components:
                    if comp.id == component_id:
                        component = comp
                        break
                
                if not component:
                    for ui_comp in architecture.ui_components:
                        if ui_comp.id == component_id:
                            component = ui_comp
                            is_ui_component = True
                            break
        
        # If not found and we have shared memory, try to find it there
        if not component and self.shared_memory:
            # Try to find a UI component
            ui_comp_data = self.shared_memory.retrieve(
                key=f"ui_component_{component_id}",
                category="ui_components"
            )
            if ui_comp_data:
                component = UiComponentSpec(**ui_comp_data)
                is_ui_component = True
            else:
                # Try to find a regular component
                comp_data = self.shared_memory.retrieve(
                    key=f"component_{component_id}",
                    category="components"
                )
                if comp_data:
                    component = ArchitectureComponent(**comp_data)
        
        if not component:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"create_spec_{component_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Component with ID {component_id} not found"
            )
        
        # Create a task for creating the technical specification
        component_type = "UI component" if is_ui_component else "component"
        task = Task(
            task_id=f"create_spec_{component_id}",
            description=f"Create technical specification for {component_type} '{component.name}'",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "component_id": component_id,
                "component_name": component.name,
                "component_description": component.description,
                "component_type": component_type,
                "component_data": component.dict(),
                "architecture_context": architecture.dict() if architecture else None,
                "additional_requirements": additional_requirements or []
            },
            context=TaskContext(
                notes=(
                    f"Create a detailed technical specification for the {component_type} '{component.name}'. "
                    f"Include all necessary details for implementation, testing, and integration."
                )
            ),
            expected_output=(
                "A comprehensive technical specification including functionality, interfaces, "
                "data structures, algorithms, constraints, and implementation guidance."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the specification
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract specification from the result
                specification = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    specification = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            specification = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the raw text
                if not specification:
                    specification = {
                        "component_id": component_id,
                        "component_name": component.name,
                        "component_type": component_type,
                        "specification": result.result,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Ensure basic metadata is included
                    specification.update({
                        "component_id": component_id,
                        "component_name": component.name,
                        "component_type": component_type,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Store in shared memory if available
                if self.shared_memory:
                    spec_id = f"tech_spec_{component_id}"
                    self.shared_memory.store(
                        key=spec_id,
                        value=specification,
                        category="technical_specifications"
                    )
                
                logger.info(f"Created technical specification for {component_type} '{component.name}'")
                
                # Return the specification
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=specification,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "component_id": component_id,
                        "component_name": component.name,
                        "component_type": component_type
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing technical specification: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def analyze_design_feasibility(
        self, 
        design_url: str,
        design_description: str,
        constraints: List[Dict[str, Any]]
    ) -> TaskResult:
        """Analyze the technical feasibility of a UI/UX design.
        
        Args:
            design_url: URL to the design
            design_description: Description of the design
            constraints: Technical constraints to consider
            
        Returns:
            TaskResult containing the feasibility analysis
        """
        # Create a task for analyzing design feasibility
        task = Task(
            task_id=f"analyze_feasibility_{uuid.uuid4()}",
            description=f"Analyze technical feasibility of UI/UX design",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "design_url": design_url,
                "design_description": design_description,
                "constraints": constraints
            },
            context=TaskContext(
                notes=(
                    f"Analyze the technical feasibility of implementing the UI/UX design at {design_url}. "
                    f"Consider the provided constraints and identify any technical challenges or limitations."
                )
            ),
            expected_output=(
                "A comprehensive feasibility analysis including potential implementation challenges, "
                "technical limitations, recommended approaches, and alternatives if needed."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process the analysis
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract analysis from the result
                analysis = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    analysis = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            analysis = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the raw text
                if not analysis:
                    analysis = {
                        "design_url": design_url,
                        "design_description": design_description,
                        "analysis": result.result,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Ensure basic metadata is included
                    if "design_url" not in analysis:
                        analysis["design_url"] = design_url
                    if "design_description" not in analysis:
                        analysis["design_description"] = design_description
                    if "timestamp" not in analysis:
                        analysis["timestamp"] = datetime.now().isoformat()
                
                # Store in shared memory if available
                if self.shared_memory:
                    analysis_id = f"feasibility_analysis_{uuid.uuid4()}"
                    self.shared_memory.store(
                        key=analysis_id,
                        value=analysis,
                        category="feasibility_analyses"
                    )
                
                logger.info(f"Analyzed feasibility of UI/UX design")
                
                # Return the analysis
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=analysis,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "analysis_id": analysis_id if self.shared_memory else None
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing feasibility analysis: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def identify_tech_debt(
        self, 
        architecture_id: str
    ) -> TaskResult:
        """Identify potential technical debt in an existing architecture.
        
        Args:
            architecture_id: ID of the architecture to analyze
            
        Returns:
            TaskResult containing the technical debt analysis
        """
        # Get the architecture
        architecture = self.architectures.get(architecture_id)
        
        if not architecture and self.shared_memory:
            arch_data = self.shared_memory.retrieve(
                key=f"architecture_{architecture_id}",
                category="architectures"
            )
            if arch_data:
                architecture = SystemArchitecture(**arch_data)
        
        if not architecture:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"identify_tech_debt_{architecture_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Architecture with ID {architecture_id} not found"
            )
        
        # Create a task for identifying technical debt
        task = Task(
            task_id=f"identify_tech_debt_{architecture_id}",
            description=f"Identify technical debt in architecture '{architecture.name}'",
            agent_type=str(AgentRole.ARCHITECTURE_DESIGNER),
            requirements={
                "architecture_id": architecture_id,
                "architecture_name": architecture.name,
                "architecture_data": architecture.dict()
            },
            context=TaskContext(
                notes=(
                    f"Analyze the architecture '{architecture.name}' for potential technical debt. "
                    f"Identify areas where shortcuts might have been taken, where scalability might "
                    f"be limited, or where future maintenance might be challenging."
                )
            ),
            expected_output=(
                "A comprehensive analysis of potential technical debt, including specific areas "
                "of concern, impact assessment, and recommended remediation strategies."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process the analysis
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract analysis from the result
                analysis = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    analysis = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            analysis = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the raw text
                if not analysis:
                    analysis = {
                        "architecture_id": architecture_id,
                        "architecture_name": architecture.name,
                        "tech_debt_analysis": result.result,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Ensure basic metadata is included
                    if "architecture_id" not in analysis:
                        analysis["architecture_id"] = architecture_id
                    if "architecture_name" not in analysis:
                        analysis["architecture_name"] = architecture.name
                    if "timestamp" not in analysis:
                        analysis["timestamp"] = datetime.now().isoformat()
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"tech_debt_analysis_{architecture_id}",
                        value=analysis,
                        category="technical_debt"
                    )
                
                logger.info(f"Identified technical debt in architecture '{architecture.name}'")
                
                # Return the analysis
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=analysis,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "architecture_id": architecture_id,
                        "architecture_name": architecture.name
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing technical debt analysis: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def get_architecture(self, architecture_id: str) -> Optional[SystemArchitecture]:
        """Get a specific architecture.
        
        Args:
            architecture_id: ID of the architecture to retrieve
            
        Returns:
            SystemArchitecture if found, None otherwise
        """
        # Check local storage
        architecture = self.architectures.get(architecture_id)
        
        # If not found locally and shared memory is available, check there
        if not architecture and self.shared_memory:
            arch_data = self.shared_memory.retrieve(
                key=f"architecture_{architecture_id}",
                category="architectures"
            )
            if arch_data:
                architecture = SystemArchitecture(**arch_data)
                # Cache it locally
                self.architectures[architecture_id] = architecture
        
        return architecture
    
    def get_component(self, component_id: str) -> Optional[Union[ArchitectureComponent, UiComponentSpec]]:
        """Get a specific component.
        
        Args:
            component_id: ID of the component to retrieve
            
        Returns:
            ArchitectureComponent or UiComponentSpec if found, None otherwise
        """
        # Search through all architectures
        for architecture in self.architectures.values():
            # Check regular components
            for component in architecture.components:
                if component.id == component_id:
                    return component
            
            # Check UI components
            for ui_component in architecture.ui_components:
                if ui_component.id == component_id:
                    return ui_component
        
        # If not found locally and shared memory is available, check there
        if self.shared_memory:
            # Try as regular component
            comp_data = self.shared_memory.retrieve(
                key=f"component_{component_id}",
                category="components"
            )
            if comp_data:
                return ArchitectureComponent(**comp_data)
            
            # Try as UI component
            ui_comp_data = self.shared_memory.retrieve(
                key=f"ui_component_{component_id}",
                category="ui_components"
            )
            if ui_comp_data:
                return UiComponentSpec(**ui_comp_data)
        
        return None
    
    def set_ui_ux_designer_contact(self, contact_info: str) -> None:
        """Set the contact information for the UI/UX designer.
        
        Args:
            contact_info: Contact information for the UI/UX designer
        """
        self.ui_ux_designer_contact = contact_info
        logger.info(f"Updated UI/UX Designer contact to: {contact_info}")
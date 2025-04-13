"""
Frontend Agents for the multi-agent development system.

This module contains specialized agents for frontend development tasks, including
UI component development, frontend logic implementation, and frontend integration.
These agents work together to create high-quality, responsive, and accessible
user interfaces based on requirements and design specifications.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple, cast
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


class UIComponentProps(BaseModel):
    """Props definition for a UI component."""
    name: str
    type: str
    required: bool = False
    default_value: Optional[Any] = None
    description: str = ""
    validation: Optional[List[str]] = None


class UIComponentState(BaseModel):
    """State definition for a UI component."""
    name: str
    type: str
    initial_value: Optional[Any] = None
    description: str = ""


class UIComponentSpec(BaseModel):
    """Specification for a UI component."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    props: List[UIComponentProps] = Field(default_factory=list)
    state: List[UIComponentState] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    accessibility: Dict[str, Any] = Field(default_factory=dict)
    responsive_behavior: Dict[str, Any] = Field(default_factory=dict)
    design_url: Optional[str] = None
    design_source: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


class UIModule(BaseModel):
    """A module of related UI components."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    components: List[UIComponentSpec] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ApplicationState(BaseModel):
    """State management for a frontend application."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    state_structure: Dict[str, Any] = Field(default_factory=dict)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    selectors: List[Dict[str, Any]] = Field(default_factory=list)
    persistence: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class UIComponentDeveloper(BaseAgent):
    """Agent specialized in creating UI components."""
    
    def __init__(self, name, preferred_framework="React", design_system=None, model_provider=ModelProvider.ANTHROPIC, model_name="claude-3-sonnet-20240229", **kwargs):
        """Initialize the UI Component Developer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_framework: Preferred frontend framework
            design_system: Optional design system to follow
            model_provider: Model provider to use
            model_name: Model name to use
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        self.design_system = None
        super().__init__(
            name=name, 
            agent_type=AgentRole.UI_DEVELOPER,
            model_provider=model_provider,
            model_name=model_name, 
            **kwargs
        )
        self.preferred_framework = preferred_framework
        self.design_system = design_system or {}
        
        # Track UI components and modules
        self.components: Dict[str, UIComponentSpec] = {}
        self.modules: Dict[str, UIModule] = {}
        
        # Track implementation status
        self.implementation_status: Dict[str, Dict[str, Any]] = {}
        
        # Track UI library
        self.component_library: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"UI Component Developer Agent initialized with {preferred_framework} framework")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the UI Component Developer."""
        design_system_info = ""
        if self.design_system:
            design_system_info = (
                f"You should follow the provided design system, which includes specific "
                f"guidelines for colors, typography, spacing, and component patterns. "
            )
        
        return (
            f"You are {self.name}, a UI Component Developer specialized in creating "
            f"high-quality, reusable UI components using {self.preferred_framework}. "
            f"Your responsibilities include:\n"
            f"1. Creating responsive, accessible UI components\n"
            f"2. Following design systems and style guidelines\n"
            f"3. Optimizing components for performance\n"
            f"4. Creating clean, maintainable component code\n"
            f"5. Ensuring components work across different browsers and devices\n\n"
            f"{design_system_info}"
            f"When creating components, think about reusability, prop interfaces, state management, "
            f"accessibility, and responsive design. Document your components thoroughly, including "
            f"prop types, examples, and usage guidelines. Follow industry best practices for "
            f"{self.preferred_framework} development."
        )
    
    async def design_component(
        self, 
        component_name: str,
        description: str,
        requirements: List[Dict[str, Any]],
        design_url: Optional[str] = None,
        design_source: Optional[str] = None
    ) -> TaskResult:
        """Design a UI component specification based on requirements.
        
        Args:
            component_name: Name of the component
            description: Brief description of the component's purpose
            requirements: List of requirements for the component
            design_url: Optional URL to design mockups
            design_source: Optional base64 encoded design data
            
        Returns:
            TaskResult containing the component specification
        """
        # Create a task for component design
        task = Task(
            task_id=f"design_component_{component_name.lower().replace(' ', '_')}",
            description=f"Design UI component: {component_name}",
            agent_type=str(AgentRole.UI_DEVELOPER),
            requirements={
                "component_name": component_name,
                "description": description,
                "requirements": requirements,
                "has_design_url": design_url is not None,
                "has_design_source": design_source is not None,
                "framework": self.preferred_framework,
                "design_system": self.design_system
            },
            context=TaskContext(
                notes=(
                    f"Design a {self.preferred_framework} component called {component_name}. "
                    f"The component should meet all specified requirements and follow best practices. "
                    + (f"Reference the design at: {design_url}. " if design_url else "")
                    + (f"A design source is provided for reference. " if design_source else "")
                    + (f"Follow the established design system for consistency. " if self.design_system else "")
                )
            ),
            expected_output=(
                "A comprehensive component specification including props, state, events, "
                "accessibility considerations, and responsive behavior."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the component specification
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the component specification from the result
                component_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    component_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            component_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not component_data:
                    logger.warning(f"Could not parse component spec as JSON. Attempting to extract from text.")
                    component_data = self._extract_component_spec_from_text(result.result, component_name, description)
                
                # Process props
                props = []
                for prop_data in component_data.get("props", []):
                    prop = UIComponentProps(
                        name=prop_data.get("name", "prop"),
                        type=prop_data.get("type", "any"),
                        required=prop_data.get("required", False),
                        default_value=prop_data.get("default_value"),
                        description=prop_data.get("description", ""),
                        validation=prop_data.get("validation")
                    )
                    props.append(prop)
                
                # Process state
                state_items = []
                for state_data in component_data.get("state", []):
                    state = UIComponentState(
                        name=state_data.get("name", "state"),
                        type=state_data.get("type", "any"),
                        initial_value=state_data.get("initial_value"),
                        description=state_data.get("description", "")
                    )
                    state_items.append(state)
                
                # Create the component specification
                component = UIComponentSpec(
                    name=component_name,
                    description=description,
                    props=props,
                    state=state_items,
                    events=component_data.get("events", []),
                    accessibility=component_data.get("accessibility", {}),
                    responsive_behavior=component_data.get("responsive_behavior", {}),
                    design_url=design_url,
                    design_source=design_source,
                    dependencies=component_data.get("dependencies", [])
                )
                
                # Store the component
                self.components[component.id] = component
                
                # Initialize implementation status
                self.implementation_status[component.id] = {
                    "status": "designed",
                    "implemented": False,
                    "has_tests": False,
                    "has_documentation": False,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"ui_component_{component.id}",
                        value=component.dict(),
                        category="ui_components"
                    )
                
                logger.info(f"Created component specification for '{component_name}'")
                
                # Add to component library for reuse
                self.component_library[component_name.lower()] = {
                    "id": component.id,
                    "name": component.name,
                    "description": component.description,
                    "created_at": datetime.now().isoformat()
                }
                
                # Return the component specification as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=component.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"component_id": component.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing component specification: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_component_spec_from_text(
        self, 
        text: str, 
        component_name: str, 
        description: str
    ) -> Dict[str, Any]:
        """Extract structured component specification from unstructured text.
        
        Args:
            text: The text to extract from
            component_name: The name of the component
            description: The description of the component
            
        Returns:
            Structured component specification data
        """
        component_data = {
            "name": component_name,
            "description": description,
            "props": [],
            "state": [],
            "events": [],
            "accessibility": {},
            "responsive_behavior": {},
            "dependencies": []
        }
        
        # Extract props
        props_section = re.search(
            r'(?i)#+\s*(?:Props|Properties)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if props_section and props_section.group(1):
            props_text = props_section.group(1)
            
            # Check if there's a table format
            table_format = re.search(r'(?:\|\s*(.+?)\s*\|(?:\s*[-:]+\s*\|)+\n)((?:\|\s*.+?\s*\|\n)+)', props_text)
            
            if table_format:
                # Extract column headers and rows from Markdown table
                headers = [h.strip() for h in table_format.group(1).split('|') if h.strip()]
                rows = table_format.group(2).strip().split('\n')
                
                for row in rows:
                    row_values = [v.strip() for v in row.split('|') if v.strip()]
                    if not row_values:
                        continue
                        
                    prop = {}
                    for i, header in enumerate(headers):
                        if i < len(row_values):
                            value = row_values[i]
                            header_lower = header.lower()
                            
                            if header_lower in ["name", "prop"]:
                                prop["name"] = value
                            elif header_lower in ["type", "data type"]:
                                prop["type"] = value
                            elif header_lower in ["required"]:
                                prop["required"] = value.lower() in ["yes", "true", "y", "✓"]
                            elif header_lower in ["default", "default value"]:
                                if value.lower() not in ["none", "-", "n/a"]:
                                    prop["default_value"] = value
                            elif header_lower in ["description"]:
                                prop["description"] = value
                    
                    if "name" in prop:
                        component_data["props"].append(prop)
            else:
                # Try to extract from bullet points
                prop_items = re.findall(r'[-*]\s*([^\n]+)', props_text)
                
                for prop_text in prop_items:
                    # Try to match: name (type) [required/optional]: description
                    prop_match = re.search(r'`?([^`(]+)`?\s*(?:\(([^)]+)\))?\s*(?:\[([^\]]+)\])?(?::\s*(.+))?', prop_text)
                    
                    if prop_match:
                        prop_name = prop_match.group(1).strip()
                        prop_type = prop_match.group(2).strip() if prop_match.group(2) else "any"
                        prop_required = prop_match.group(3) and "required" in prop_match.group(3).lower()
                        prop_desc = prop_match.group(4).strip() if prop_match.group(4) else ""
                        
                        prop = {
                            "name": prop_name,
                            "type": prop_type,
                            "required": prop_required,
                            "description": prop_desc
                        }
                        
                        # Check for default value in description
                        default_match = re.search(r'default(?:\s*[:=]\s*|\s+is\s+)([^,.]+)', prop_desc, re.IGNORECASE)
                        if default_match:
                            default_value = default_match.group(1).strip()
                            if default_value.lower() not in ["none", "null", "undefined"]:
                                prop["default_value"] = default_value
                        
                        component_data["props"].append(prop)
                    else:
                        # Simplified format: just add the raw text
                        parts = prop_text.split(":")
                        if len(parts) >= 2:
                            prop_name = parts[0].strip()
                            prop_desc = parts[1].strip()
                            
                            component_data["props"].append({
                                "name": prop_name,
                                "description": prop_desc
                            })
        
        # Extract state
        state_section = re.search(
            r'(?i)#+\s*(?:State|Component State|Internal State)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if state_section and state_section.group(1):
            state_text = state_section.group(1)
            
            # Check if there's a table format
            table_format = re.search(r'(?:\|\s*(.+?)\s*\|(?:\s*[-:]+\s*\|)+\n)((?:\|\s*.+?\s*\|\n)+)', state_text)
            
            if table_format:
                # Extract column headers and rows from Markdown table
                headers = [h.strip() for h in table_format.group(1).split('|') if h.strip()]
                rows = table_format.group(2).strip().split('\n')
                
                for row in rows:
                    row_values = [v.strip() for v in row.split('|') if v.strip()]
                    if not row_values:
                        continue
                        
                    state_item = {}
                    for i, header in enumerate(headers):
                        if i < len(row_values):
                            value = row_values[i]
                            header_lower = header.lower()
                            
                            if header_lower in ["name", "state"]:
                                state_item["name"] = value
                            elif header_lower in ["type", "data type"]:
                                state_item["type"] = value
                            elif header_lower in ["initial", "initial value", "default"]:
                                if value.lower() not in ["none", "-", "n/a"]:
                                    state_item["initial_value"] = value
                            elif header_lower in ["description"]:
                                state_item["description"] = value
                    
                    if "name" in state_item:
                        component_data["state"].append(state_item)
            else:
                # Try to extract from bullet points
                state_items = re.findall(r'[-*]\s*([^\n]+)', state_text)
                
                for state_text in state_items:
                    # Try to match: name (type) [initial value]: description
                    state_match = re.search(r'`?([^`(]+)`?\s*(?:\(([^)]+)\))?\s*(?:\[([^\]]+)\])?(?::\s*(.+))?', state_text)
                    
                    if state_match:
                        state_name = state_match.group(1).strip()
                        state_type = state_match.group(2).strip() if state_match.group(2) else "any"
                        initial_value = state_match.group(3).strip() if state_match.group(3) else None
                        state_desc = state_match.group(4).strip() if state_match.group(4) else ""
                        
                        state_item = {
                            "name": state_name,
                            "type": state_type,
                            "description": state_desc
                        }
                        
                        if initial_value:
                            state_item["initial_value"] = initial_value
                        
                        component_data["state"].append(state_item)
                    else:
                        # Simplified format: just add the raw text
                        parts = state_text.split(":")
                        if len(parts) >= 2:
                            state_name = parts[0].strip()
                            state_desc = parts[1].strip()
                            
                            component_data["state"].append({
                                "name": state_name,
                                "description": state_desc
                            })
        
        # Extract events
        events_section = re.search(
            r'(?i)#+\s*(?:Events|Event Handlers|Callbacks)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if events_section and events_section.group(1):
            events_text = events_section.group(1)
            
            # Try to extract from bullet points
            event_items = re.findall(r'[-*]\s*([^\n]+)', events_text)
            
            for event_text in event_items:
                # Try to match: name(args): description
                event_match = re.search(r'`?([^`(]+)`?\s*(?:\(([^)]+)\))?\s*(?::\s*(.+))?', event_text)
                
                if event_match:
                    event_name = event_match.group(1).strip()
                    event_args = event_match.group(2).strip() if event_match.group(2) else ""
                    event_desc = event_match.group(3).strip() if event_match.group(3) else ""
                    
                    # Parse arguments
                    args = []
                    if event_args:
                        for arg in event_args.split(','):
                            arg = arg.strip()
                            if arg:
                                arg_match = re.search(r'([^:]+)(?::\s*(.+))?', arg)
                                if arg_match:
                                    arg_name = arg_match.group(1).strip()
                                    arg_type = arg_match.group(2).strip() if arg_match.group(2) else "any"
                                    args.append({
                                        "name": arg_name,
                                        "type": arg_type
                                    })
                                else:
                                    args.append({
                                        "name": arg,
                                        "type": "any"
                                    })
                    
                    component_data["events"].append({
                        "name": event_name,
                        "arguments": args,
                        "description": event_desc
                    })
                else:
                    # Just add a basic event entry
                    component_data["events"].append({
                        "description": event_text.strip()
                    })
        
        # Extract accessibility
        a11y_section = re.search(
            r'(?i)#+\s*(?:Accessibility|A11y)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if a11y_section and a11y_section.group(1):
            a11y_text = a11y_section.group(1).strip()
            
            # Extract ARIA roles
            aria_match = re.search(r'(?i)(?:ARIA|Roles?):\s*([^\n]+)', a11y_text)
            if aria_match:
                component_data["accessibility"]["aria_role"] = aria_match.group(1).strip()
            
            # Extract keyboard navigation
            keyboard_match = re.search(r'(?i)(?:Keyboard|Navigation):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', a11y_text)
            if keyboard_match:
                component_data["accessibility"]["keyboard_support"] = keyboard_match.group(1).strip()
            
            # Extract focus management
            focus_match = re.search(r'(?i)(?:Focus):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', a11y_text)
            if focus_match:
                component_data["accessibility"]["focus_management"] = focus_match.group(1).strip()
            
            # Extract other accessibility considerations
            component_data["accessibility"]["considerations"] = []
            consideration_items = re.findall(r'[-*]\s*([^\n]+)', a11y_text)
            component_data["accessibility"]["considerations"] = [c.strip() for c in consideration_items]
        
        # Extract responsive behavior
        responsive_section = re.search(
            r'(?i)#+\s*(?:Responsive|Responsive Design|Responsive Behavior)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if responsive_section and responsive_section.group(1):
            responsive_text = responsive_section.group(1).strip()
            
            # Extract breakpoints
            breakpoints_match = re.search(r'(?i)(?:Breakpoints):\s*([^\n]+)', responsive_text)
            if breakpoints_match:
                breakpoints_text = breakpoints_match.group(1).strip()
                component_data["responsive_behavior"]["breakpoints"] = breakpoints_text
            
            # Extract mobile behavior
            mobile_match = re.search(r'(?i)(?:Mobile|Small Screen|Phone):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', responsive_text)
            if mobile_match:
                component_data["responsive_behavior"]["mobile"] = mobile_match.group(1).strip()
            
            # Extract tablet behavior
            tablet_match = re.search(r'(?i)(?:Tablet|Medium Screen):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', responsive_text)
            if tablet_match:
                component_data["responsive_behavior"]["tablet"] = tablet_match.group(1).strip()
            
            # Extract desktop behavior
            desktop_match = re.search(r'(?i)(?:Desktop|Large Screen):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', responsive_text)
            if desktop_match:
                component_data["responsive_behavior"]["desktop"] = desktop_match.group(1).strip()
        
        # Extract dependencies
        deps_section = re.search(
            r'(?i)#+\s*(?:Dependencies|External Dependencies|Required Packages)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if deps_section and deps_section.group(1):
            deps_text = deps_section.group(1)
            
            # Try to extract from bullet points
            dep_items = re.findall(r'[-*]\s*([^\n]+)', deps_text)
            component_data["dependencies"] = [d.strip() for d in dep_items]
        
        return component_data
    
    async def implement_component(
        self, 
        component_id: str,
        styling_approach: str = "tailwind",
        implementation_details: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Implement a UI component based on its specification.
        
        Args:
            component_id: ID of the component to implement
            styling_approach: Styling approach (tailwind, css, emotion, etc.)
            implementation_details: Optional additional implementation details
            
        Returns:
            TaskResult containing the component implementation
        """
        # Check if component exists
        if component_id not in self.components:
            # Try to load from shared memory if available
            if self.shared_memory:
                component_data = self.shared_memory.retrieve(
                    key=f"ui_component_{component_id}",
                    category="ui_components"
                )
                if component_data:
                    self.components[component_id] = UIComponentSpec(**component_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_component_{component_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Component with ID {component_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_component_{component_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Component with ID {component_id} not found"
                )
        
        # Get the component specification
        component = self.components[component_id]
        
        # Create a task for implementing the component
        task = Task(
            task_id=f"implement_component_{component_id}",
            description=f"Implement UI component: {component.name}",
            agent_type=str(AgentRole.UI_DEVELOPER),
            requirements={
                "component_id": component_id,
                "component_name": component.name,
                "component_description": component.description,
                "props": [prop.dict() for prop in component.props],
                "state": [state.dict() for state in component.state],
                "events": component.events,
                "accessibility": component.accessibility,
                "responsive_behavior": component.responsive_behavior,
                "has_design_url": component.design_url is not None,
                "design_url": component.design_url,
                "framework": self.preferred_framework,
                "styling_approach": styling_approach,
                "design_system": self.design_system,
                "implementation_details": implementation_details or {}
            },
            context=TaskContext(
                notes=(
                    f"Implement the {component.name} component using {self.preferred_framework} "
                    f"and {styling_approach} for styling. The implementation should match the "
                    f"component specification and handle all props, state, and events appropriately. "
                    f"Ensure the component is accessible and responsive according to the specifications. "
                    + (f"Reference the design at: {component.design_url}. " if component.design_url else "")
                    + (f"Follow the established design system for consistency. " if self.design_system else "")
                )
            ),
            expected_output=(
                f"Complete {self.preferred_framework} implementation of the {component.name} component, "
                f"including all props, state management, event handlers, and styling."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Update implementation status
            if component_id in self.implementation_status:
                status = self.implementation_status[component_id]
                status["status"] = "implemented"
                status["implemented"] = True
                status["timestamp"] = datetime.now().isoformat()
                status["styling_approach"] = styling_approach
                
                # Update status
                self.implementation_status[component_id] = status
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"component_implementation_{component_id}",
                    value={
                        "component_id": component_id,
                        "component_name": component.name,
                        "framework": self.preferred_framework,
                        "styling_approach": styling_approach,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="component_implementations"
                )
                # Update implementation status in shared memory
                self.shared_memory.store(
                    key=f"component_implementation_status_{component_id}",
                    value=self.implementation_status[component_id],
                    category="component_implementation_status"
                )
            
            logger.info(f"Implemented {component.name} component using {self.preferred_framework} and {styling_approach}")
            
            # Return the implementation code
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,  # Original implementation code
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "component_id": component_id,
                    "component_name": component.name,
                    "framework": self.preferred_framework,
                    "styling_approach": styling_approach
                }
            )
            
            return updated_result
        
        return result
    
    async def create_component_tests(
        self, 
        component_id: str,
        test_framework: str = "jest"
    ) -> TaskResult:
        """Create tests for a UI component.
        
        Args:
            component_id: ID of the component to test
            test_framework: Testing framework to use
            
        Returns:
            TaskResult containing the component tests
        """
        # Check if component exists
        if component_id not in self.components:
            # Try to load from shared memory if available
            if self.shared_memory:
                component_data = self.shared_memory.retrieve(
                    key=f"ui_component_{component_id}",
                    category="ui_components"
                )
                if component_data:
                    self.components[component_id] = UIComponentSpec(**component_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"create_component_tests_{component_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Component with ID {component_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"create_component_tests_{component_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Component with ID {component_id} not found"
                )
        
        # Get the component specification
        component = self.components[component_id]
        
        # Get the component implementation if available
        implementation = None
        if self.shared_memory:
            impl_data = self.shared_memory.retrieve(
                key=f"component_implementation_{component_id}",
                category="component_implementations"
            )
            if impl_data:
                implementation = impl_data.get("code")
        
        # Create a task for creating component tests
        task = Task(
            task_id=f"create_component_tests_{component_id}",
            description=f"Create tests for {component.name} component",
            agent_type=str(AgentRole.UI_DEVELOPER),
            requirements={
                "component_id": component_id,
                "component_name": component.name,
                "component_description": component.description,
                "props": [prop.dict() for prop in component.props],
                "state": [state.dict() for state in component.state],
                "events": component.events,
                "framework": self.preferred_framework,
                "test_framework": test_framework,
                "has_implementation": implementation is not None,
                "implementation": implementation
            },
            context=TaskContext(
                notes=(
                    f"Create tests for the {component.name} component using {test_framework}. "
                    f"The tests should verify that the component renders correctly, props are "
                    f"handled properly, state changes work as expected, and events are triggered correctly. "
                    f"Include tests for accessibility and responsive behavior where applicable. "
                    + (f"Reference the implementation code when creating tests. " if implementation else "")
                )
            ),
            expected_output=(
                f"Complete test suite for the {component.name} component using {test_framework}, "
                f"covering rendering, props, state, events, and edge cases."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the tests
        if result.status == TaskStatus.COMPLETED and result.result:
            # Update implementation status
            if component_id in self.implementation_status:
                status = self.implementation_status[component_id]
                status["has_tests"] = True
                status["test_framework"] = test_framework
                status["timestamp"] = datetime.now().isoformat()
                
                # Update status
                self.implementation_status[component_id] = status
                
                # Store updated status in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"component_implementation_status_{component_id}",
                        value=status,
                        category="component_implementation_status"
                    )
            
            # Store tests in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"component_tests_{component_id}",
                    value={
                        "component_id": component_id,
                        "component_name": component.name,
                        "framework": self.preferred_framework,
                        "test_framework": test_framework,
                        "tests": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="component_tests"
                )
            
            logger.info(f"Created tests for {component.name} component using {test_framework}")
            
            # Return the tests
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "component_id": component_id,
                    "component_name": component.name,
                    "test_framework": test_framework
                }
            )
            
            return updated_result
        
        return result
    
    async def create_module(
        self, 
        module_name: str,
        description: str,
        component_ids: List[str]
    ) -> TaskResult:
        """Create a module grouping related UI components.
        
        Args:
            module_name: Name of the module
            description: Brief description of the module's purpose
            component_ids: List of component IDs to include in the module
            
        Returns:
            TaskResult containing the module specification
        """
        # Check if all components exist
        components = []
        missing_components = []
        
        for component_id in component_ids:
            if component_id in self.components:
                components.append(self.components[component_id])
            elif self.shared_memory:
                component_data = self.shared_memory.retrieve(
                    key=f"ui_component_{component_id}",
                    category="ui_components"
                )
                if component_data:
                    component = UIComponentSpec(**component_data)
                    self.components[component_id] = component
                    components.append(component)
                else:
                    missing_components.append(component_id)
            else:
                missing_components.append(component_id)
        
        if missing_components:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"create_module_{module_name.lower().replace(' ', '_')}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Components not found: {', '.join(missing_components)}"
            )
        
        # Create a task for organizing components into a module
        task = Task(
            task_id=f"create_module_{module_name.lower().replace(' ', '_')}",
            description=f"Create UI module: {module_name}",
            agent_type=str(AgentRole.UI_DEVELOPER),
            requirements={
                "module_name": module_name,
                "description": description,
                "components": [component.dict() for component in components],
                "framework": self.preferred_framework
            },
            context=TaskContext(
                notes=(
                    f"Create a {self.preferred_framework} module named {module_name} that organizes "
                    f"the specified components into a cohesive unit. Define how these components "
                    f"relate to each other and how they should be used together."
                )
            ),
            expected_output=(
                "A module specification including relationships between components, "
                "export structure, and usage guidelines."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the module
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Create the module
                module = UIModule(
                    name=module_name,
                    description=description,
                    components=components
                )
                
                # Store the module
                self.modules[module.id] = module
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"ui_module_{module.id}",
                        value=module.dict(),
                        category="ui_modules"
                    )
                
                logger.info(f"Created UI module '{module_name}' with {len(components)} components")
                
                # Return the module specification
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result={
                        "module": module.dict(),
                        "organization_notes": result.result
                    },
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"module_id": module.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing module creation: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def get_component(self, component_id: str) -> Optional[UIComponentSpec]:
        """Get a specific UI component.
        
        Args:
            component_id: ID of the component to retrieve
            
        Returns:
            UIComponentSpec if found, None otherwise
        """
        # Check local storage
        if component_id in self.components:
            return self.components[component_id]
        
        # Check shared memory if available
        if self.shared_memory:
            component_data = self.shared_memory.retrieve(
                key=f"ui_component_{component_id}",
                category="ui_components"
            )
            if component_data:
                component = UIComponentSpec(**component_data)
                # Cache locally
                self.components[component_id] = component
                return component
        
        return None
    
    def get_implementation_status(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get the implementation status for a UI component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            Implementation status if found, None otherwise
        """
        # Check local storage
        if component_id in self.implementation_status:
            return self.implementation_status[component_id]
        
        # Check shared memory if available
        if self.shared_memory:
            status_data = self.shared_memory.retrieve(
                key=f"component_implementation_status_{component_id}",
                category="component_implementation_status"
            )
            if status_data:
                # Cache locally
                self.implementation_status[component_id] = status_data
                return status_data
        
        return None


class FrontendLogicDeveloper(BaseAgent):
    """Agent specialized in implementing frontend application logic."""
    
    def __init__(
        self,
        name: str,
        preferred_framework: str = "React",
        preferred_state_management: str = "Redux",
        **kwargs
    ):
        # Initialize attributes before calling super().__init__
        self.preferred_framework = preferred_framework
        self.preferred_state_management = preferred_state_management
        self.design_system = None
        
        super().__init__(
            name=name,
            agent_type=AgentRole.FRONTEND_LOGIC,
            **kwargs
        )
        
        super().__init__(
            name=name,
            agent_type=AgentRole.FRONTEND_LOGIC,
            **kwargs
        )
        """Initialize the Frontend Logic Developer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_framework: Preferred frontend framework
            preferred_state_management: Preferred state management solution
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.FRONTEND_LOGIC, 
            **kwargs
        )
        self.preferred_framework = preferred_framework
        self.preferred_state_management = preferred_state_management
        
        # Track application state
        self.app_states: Dict[str, ApplicationState] = {}
        
        # Track logic implementations
        self.logic_implementations: Dict[str, Dict[str, Any]] = {}
        
        # Track API integration implementations
        self.api_integrations: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Frontend Logic Developer Agent initialized with {preferred_framework} framework and {preferred_state_management} state management")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Frontend Logic Developer."""
        return (
            f"You are {self.name}, a Frontend Logic Developer specialized in implementing "
            f"application logic and state management using {self.preferred_framework} and "
            f"{self.preferred_state_management}. "
            f"Your responsibilities include:\n"
            f"1. Designing application state structure\n"
            f"2. Implementing state management logic\n"
            f"3. Creating API integration code\n"
            f"4. Building business logic for the frontend\n"
            f"5. Ensuring data consistency and performance\n\n"
            f"Think carefully about state management patterns, data flow, side effects, "
            f"and performance optimizations. Write clean, maintainable code with appropriate "
            f"error handling, logging, and documentation. Consider the user experience when "
            f"implementing loading states, error states, and transitions."
        )
    
    async def design_application_state(
        self, 
        app_name: str,
        description: str,
        requirements: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> TaskResult:
        """Design the application state structure.
        
        Args:
            app_name: Name of the application
            description: Brief description of the application
            requirements: List of requirements for the state management
            entities: List of data entities to be managed in state
            
        Returns:
            TaskResult containing the application state design
        """
        # Create a task for designing application state
        task = Task(
            task_id=f"design_state_{app_name.lower().replace(' ', '_')}",
            description=f"Design application state for {app_name}",
            agent_type=str(AgentRole.FRONTEND_LOGIC),
            requirements={
                "app_name": app_name,
                "description": description,
                "requirements": requirements,
                "entities": entities,
                "framework": self.preferred_framework,
                "state_management": self.preferred_state_management
            },
            context=TaskContext(
                notes=(
                    f"Design the application state structure for {app_name} using "
                    f"{self.preferred_state_management} for state management. Define all "
                    f"state slices, actions, selectors, and persistence strategies. "
                    f"Consider performance, maintainability, and developer experience."
                )
            ),
            expected_output=(
                "A comprehensive application state design including state structure, "
                "actions, selectors, and persistence strategy."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the application state design
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the state design from the result
                state_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    state_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            state_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not state_data:
                    logger.warning(f"Could not parse application state as JSON. Attempting to extract from text.")
                    state_data = self._extract_app_state_from_text(result.result, app_name, description)
                
                # Create the application state
                app_state = ApplicationState(
                    name=app_name,
                    description=description,
                    state_structure=state_data.get("state_structure", {}),
                    actions=state_data.get("actions", []),
                    selectors=state_data.get("selectors", []),
                    persistence=state_data.get("persistence")
                )
                
                # Store the application state
                self.app_states[app_state.id] = app_state
                
                # Initialize logic implementations tracking
                self.logic_implementations[app_state.id] = {
                    "status": "designed",
                    "implemented_actions": [],
                    "pending_actions": [action.get("name", f"action_{i}") for i, action in enumerate(app_state.actions)],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"app_state_{app_state.id}",
                        value=app_state.dict(),
                        category="application_states"
                    )
                
                logger.info(f"Created application state design for '{app_name}' with {len(app_state.actions)} actions")
                
                # Return the application state design as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=app_state.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"app_state_id": app_state.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing application state design: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_app_state_from_text(
        self, 
        text: str, 
        app_name: str, 
        description: str
    ) -> Dict[str, Any]:
        """Extract structured application state data from unstructured text.
        
        Args:
            text: The text to extract from
            app_name: The name of the application
            description: The description of the application
            
        Returns:
            Structured application state data
        """
        state_data = {
            "name": app_name,
            "description": description,
            "state_structure": {},
            "actions": [],
            "selectors": [],
            "persistence": None
        }
        
        # Extract state structure
        state_section = re.search(
            r'(?i)#+\s*(?:State Structure|State|Store Structure)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if state_section and state_section.group(1):
            state_text = state_section.group(1)
            
            # Try to extract JSON structure
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', state_text, re.DOTALL)
            if json_match:
                try:
                    state_structure = json.loads(json_match.group(1))
                    state_data["state_structure"] = state_structure
                except:
                    # Failed to parse as JSON, try to build from sections
                    pass
            
            # If we couldn't parse JSON, try to build structure from sections
            if not state_data["state_structure"]:
                # Look for state slices
                slice_sections = re.findall(
                    r'(?i)#+\s*(?:Slice|State Slice|Store Slice):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+\s*(?:Slice|State Slice|Store Slice)|\n#+|\Z)',
                    state_text,
                    re.DOTALL
                )
                
                for title, content in slice_sections:
                    slice_name = title.strip()
                    slice_structure = {}
                    
                    # Try to extract JSON structure
                    json_match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        try:
                            slice_structure = json.loads(json_match.group(1))
                        except:
                            # If we can't parse as JSON, create a basic structure
                            structure_items = re.findall(r'[-*]\s*`?([^`:\n]+)`?(?:\s*:\s*([^\n]+))?', content)
                            for item in structure_items:
                                key = item[0].strip()
                                value_type = item[1].strip() if item[1] else "any"
                                slice_structure[key] = {"type": value_type}
                    else:
                        # Try to build from bullet points
                        structure_items = re.findall(r'[-*]\s*`?([^`:\n]+)`?(?:\s*:\s*([^\n]+))?', content)
                        for item in structure_items:
                            key = item[0].strip()
                            value_type = item[1].strip() if item[1] else "any"
                            slice_structure[key] = {"type": value_type}
                    
                    if slice_structure:
                        state_data["state_structure"][slice_name] = slice_structure
        
        # Extract actions
        actions_section = re.search(
            r'(?i)#+\s*(?:Actions|Action Creators|Action Types)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if actions_section and actions_section.group(1):
            actions_text = actions_section.group(1)
            
            # Look for individual actions
            action_sections = re.findall(
                r'(?i)(?:#{1,3}|[-*])\s*([^:\n]+)(?::\s*([^\n]+))?(?:\n+(.+?))?(?=\n+(?:#{1,3}|[-*])\s*[^:\n]+(?::\s*[^\n]+)?|\n#+|\Z)',
                actions_text,
                re.DOTALL
            )
            
            for match in action_sections:
                action_name = match[0].strip()
                action_type = match[1].strip() if len(match) > 1 and match[1] else ""
                action_content = match[2] if len(match) > 2 else ""
                
                action = {
                    "name": action_name,
                    "type": action_type or f"{action_name.upper().replace(' ', '_')}"
                }
                
                # Extract payload structure
                payload_match = re.search(r'(?i)(?:Payload|Parameters):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', action_content)
                if payload_match:
                    payload_text = payload_match.group(1).strip()
                    
                    # Try to parse as structure
                    structure_items = re.findall(r'[-*]\s*`?([^`:\n]+)`?(?:\s*:\s*([^\n]+))?', payload_text)
                    if structure_items:
                        payload = {}
                        for item in structure_items:
                            key = item[0].strip()
                            value_type = item[1].strip() if len(item) > 1 and item[1] else "any"
                            payload[key] = {"type": value_type}
                        
                        action["payload"] = payload
                    else:
                        action["payload"] = payload_text
                
                # Extract description
                desc_match = re.search(r'(?i)(?:Description|Purpose):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', action_content)
                if desc_match:
                    action["description"] = desc_match.group(1).strip()
                
                state_data["actions"].append(action)
        
        # Extract selectors
        selectors_section = re.search(
            r'(?i)#+\s*(?:Selectors|Selector Functions|State Selectors)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if selectors_section and selectors_section.group(1):
            selectors_text = selectors_section.group(1)
            
            # Look for individual selectors
            selector_items = re.findall(r'[-*]\s*([^\n]+)', selectors_text)
            
            for selector_text in selector_items:
                # Try to match: name(args): description
                selector_match = re.search(r'`?([^`(]+)`?\s*(?:\(([^)]+)\))?(?::\s*(.+))?', selector_text)
                
                if selector_match:
                    selector_name = selector_match.group(1).strip()
                    selector_args = selector_match.group(2).strip() if selector_match.group(2) else "state"
                    selector_desc = selector_match.group(3).strip() if selector_match.group(3) else ""
                    
                    state_data["selectors"].append({
                        "name": selector_name,
                        "arguments": selector_args,
                        "description": selector_desc
                    })
                else:
                    # Just add a basic selector entry
                    state_data["selectors"].append({
                        "name": selector_text.strip()
                    })
        
        # Extract persistence
        persistence_section = re.search(
            r'(?i)#+\s*(?:Persistence|State Persistence|Storage)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if persistence_section and persistence_section.group(1):
            persistence_text = persistence_section.group(1).strip()
            
            persistence = {
                "strategy": "none"
            }
            
            # Try to determine persistence strategy
            if re.search(r'(?i)local\s*storage', persistence_text):
                persistence["strategy"] = "localStorage"
            elif re.search(r'(?i)session\s*storage', persistence_text):
                persistence["strategy"] = "sessionStorage"
            elif re.search(r'(?i)cookie', persistence_text):
                persistence["strategy"] = "cookies"
            elif re.search(r'(?i)indexed\s*db', persistence_text):
                persistence["strategy"] = "indexedDB"
            
            # Try to extract persisted keys
            persisted_keys = []
            keys_match = re.search(r'(?i)(?:Persisted Keys|Keys to Persist|Persist):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', persistence_text)
            if keys_match:
                keys_text = keys_match.group(1).strip()
                
                # Try to extract from bullet points
                key_items = re.findall(r'[-*]\s*([^\n]+)', keys_text)
                if key_items:
                    persisted_keys = [k.strip() for k in key_items]
                else:
                    # Try to extract as comma-separated list
                    persisted_keys = [k.strip() for k in re.split(r',\s*', keys_text)]
            
            if persisted_keys:
                persistence["keys"] = persisted_keys
            
            state_data["persistence"] = persistence
        
        return state_data
    
    async def implement_state_management(
        self, 
        app_state_id: str,
        implementation_details: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Implement state management code based on application state design.
        
        Args:
            app_state_id: ID of the application state
            implementation_details: Optional additional implementation details
            
        Returns:
            TaskResult containing the state management implementation
        """
        # Check if application state exists
        if app_state_id not in self.app_states:
            # Try to load from shared memory if available
            if self.shared_memory:
                state_data = self.shared_memory.retrieve(
                    key=f"app_state_{app_state_id}",
                    category="application_states"
                )
                if state_data:
                    self.app_states[app_state_id] = ApplicationState(**state_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_state_{app_state_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Application state with ID {app_state_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_state_{app_state_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Application state with ID {app_state_id} not found"
                )
        
        # Get the application state
        app_state = self.app_states[app_state_id]
        
        # Create a task for implementing state management
        task = Task(
            task_id=f"implement_state_{app_state_id}",
            description=f"Implement state management for {app_state.name}",
            agent_type=str(AgentRole.FRONTEND_LOGIC),
            requirements={
                "app_state_id": app_state_id,
                "app_name": app_state.name,
                "state_structure": app_state.state_structure,
                "actions": app_state.actions,
                "selectors": app_state.selectors,
                "persistence": app_state.persistence,
               "framework": self.preferred_framework,
                "state_management": self.preferred_state_management,
                "implementation_details": implementation_details or {}
            },
            context=TaskContext(
                notes=(
                    f"Implement state management for {app_state.name} using "
                    f"{self.preferred_state_management} with {self.preferred_framework}. "
                    f"The implementation should include all state slices, actions, "
                    f"selectors, and persistence logic defined in the application state design."
                )
            ),
            expected_output=(
                f"Complete state management implementation using {self.preferred_state_management}, "
                f"including store configuration, actions, reducers, selectors, and middleware."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Update implementation status
            if app_state_id in self.logic_implementations:
                status = self.logic_implementations[app_state_id]
                
                # Mark all actions as implemented
                status["pending_actions"] = []
                status["implemented_actions"] = [action.get("name", f"action_{i}") for i, action in enumerate(app_state.actions)]
                status["status"] = "implemented"
                status["timestamp"] = datetime.now().isoformat()
                
                # Update status
                self.logic_implementations[app_state_id] = status
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"state_implementation_{app_state_id}",
                    value={
                        "app_state_id": app_state_id,
                        "app_name": app_state.name,
                        "framework": self.preferred_framework,
                        "state_management": self.preferred_state_management,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="state_implementations"
                )
                
                # Update implementation status
                self.shared_memory.store(
                    key=f"state_implementation_status_{app_state_id}",
                    value=self.logic_implementations[app_state_id],
                    category="state_implementation_status"
                )
            
            logger.info(f"Implemented state management for {app_state.name} using {self.preferred_state_management}")
            
            # Return the implementation code
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,  # Original implementation code
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "app_state_id": app_state_id,
                    "app_name": app_state.name,
                    "state_management": self.preferred_state_management
                }
            )
            
            return updated_result
        
        return result
    
    async def implement_api_integration(
        self, 
        api_name: str,
        endpoints: List[Dict[str, Any]],
        app_state_id: Optional[str] = None,
        error_handling_strategy: str = "standard"
    ) -> TaskResult:
        """Implement API integration code.
        
        Args:
            api_name: Name of the API
            endpoints: List of API endpoints to integrate with
            app_state_id: Optional ID of the application state to integrate with
            error_handling_strategy: Error handling strategy (standard, retry, circuit-breaker, etc.)
            
        Returns:
            TaskResult containing the API integration code
        """
        # Get application state if specified
        app_state = None
        if app_state_id:
            if app_state_id in self.app_states:
                app_state = self.app_states[app_state_id]
            elif self.shared_memory:
                state_data = self.shared_memory.retrieve(
                    key=f"app_state_{app_state_id}",
                    category="application_states"
                )
                if state_data:
                    app_state = ApplicationState(**state_data)
        
        # Create a task for implementing API integration
        task = Task(
            task_id=f"implement_api_{api_name.lower().replace(' ', '_')}",
            description=f"Implement API integration for {api_name}",
            agent_type=str(AgentRole.FRONTEND_LOGIC),
            requirements={
                "api_name": api_name,
                "endpoints": endpoints,
                "has_app_state": app_state is not None,
                "app_state": app_state.dict() if app_state else None,
                "framework": self.preferred_framework,
                "state_management": self.preferred_state_management if app_state else "none",
                "error_handling": error_handling_strategy
            },
            context=TaskContext(
                notes=(
                    f"Implement API integration code for {api_name} using {self.preferred_framework}. "
                    f"The implementation should include functions for calling each endpoint, "
                    f"handling responses, and proper error handling using the {error_handling_strategy} strategy. "
                    + (f"Integrate with the application state management using {self.preferred_state_management}. " 
                       if app_state else "")
                )
            ),
            expected_output=(
                f"Complete API integration code, including functions for each endpoint, "
                f"error handling, and state management integration if applicable."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create an ID for this integration
            integration_id = str(uuid.uuid4())
            
            # Store in our tracking dictionary
            self.api_integrations[integration_id] = {
                "api_name": api_name,
                "endpoints": endpoints,
                "app_state_id": app_state_id,
                "error_handling": error_handling_strategy,
                "implemented_at": datetime.now().isoformat()
            }
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"api_integration_{integration_id}",
                    value={
                        "api_name": api_name,
                        "endpoints": endpoints,
                        "app_state_id": app_state_id,
                        "error_handling": error_handling_strategy,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="api_integrations"
                )
            
            logger.info(f"Implemented API integration for {api_name} with {len(endpoints)} endpoints")
            
            # Return the implementation code
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "integration_id": integration_id,
                    "api_name": api_name,
                    "app_state_id": app_state_id,
                    "error_handling": error_handling_strategy
                }
            )
            
            return updated_result
        
        return result
    
    async def create_form_logic(
        self, 
        form_name: str,
        fields: List[Dict[str, Any]],
        validation_rules: List[Dict[str, Any]],
        submission_handler: Dict[str, Any]
    ) -> TaskResult:
        """Create form handling logic.
        
        Args:
            form_name: Name of the form
            fields: List of form fields
            validation_rules: List of validation rules for the fields
            submission_handler: Handler for form submission
            
        Returns:
            TaskResult containing the form handling logic implementation
        """
        # Create a task for implementing form logic
        task = Task(
            task_id=f"create_form_logic_{form_name.lower().replace(' ', '_')}",
            description=f"Create form handling logic for {form_name}",
            agent_type=str(AgentRole.FRONTEND_LOGIC),
            requirements={
                "form_name": form_name,
                "fields": fields,
                "validation_rules": validation_rules,
                "submission_handler": submission_handler,
                "framework": self.preferred_framework
            },
            context=TaskContext(
                notes=(
                    f"Implement form handling logic for {form_name} using {self.preferred_framework}. "
                    f"The implementation should handle form state management, validation according "
                    f"to the specified rules, and form submission. Include appropriate error handling, "
                    f"loading states, and success states."
                )
            ),
            expected_output=(
                f"Complete form handling implementation including state management, "
                f"validation, error handling, and submission handling."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                form_id = f"form_logic_{form_name.lower().replace(' ', '_')}"
                self.shared_memory.store(
                    key=form_id,
                    value={
                        "form_name": form_name,
                        "fields": fields,
                        "validation_rules": validation_rules,
                        "submission_handler": submission_handler,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="form_logic"
                )
            
            logger.info(f"Created form handling logic for {form_name}")
            
            # Return the implementation code
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "form_name": form_name,
                    "framework": self.preferred_framework
                }
            )
            
            return updated_result
        
        return result
    
    def get_application_state(self, app_state_id: str) -> Optional[ApplicationState]:
        """Get a specific application state.
        
        Args:
            app_state_id: ID of the application state to retrieve
            
        Returns:
            ApplicationState if found, None otherwise
        """
        # Check local storage
        if app_state_id in self.app_states:
            return self.app_states[app_state_id]
        
        # Check shared memory if available
        if self.shared_memory:
            state_data = self.shared_memory.retrieve(
                key=f"app_state_{app_state_id}",
                category="application_states"
            )
            if state_data:
                app_state = ApplicationState(**state_data)
                # Cache locally
                self.app_states[app_state_id] = app_state
                return app_state
        
        return None
    
    def get_implementation_status(self, app_state_id: str) -> Optional[Dict[str, Any]]:
        """Get the implementation status for an application state.
        
        Args:
            app_state_id: ID of the application state
            
        Returns:
            Implementation status if found, None otherwise
        """
        # Check local storage
        if app_state_id in self.logic_implementations:
            return self.logic_implementations[app_state_id]
        
        # Check shared memory if available
        if self.shared_memory:
            status_data = self.shared_memory.retrieve(
                key=f"state_implementation_status_{app_state_id}",
                category="state_implementation_status"
            )
            if status_data:
                # Cache locally
                self.logic_implementations[app_state_id] = status_data
                return status_data
        
        return None


class FrontendIntegrationDeveloper(BaseAgent):
    """Agent specialized in integrating frontend components and logic."""
    
    def __init__(
        self,
        name: str,
        preferred_framework: str = "React",
        **kwargs
    ):
        # Initialize attributes before calling super().__init__
        self.preferred_framework = preferred_framework
        
        super().__init__(
            name=name,
            agent_type=AgentRole.FRONTEND_INTEGRATION,
            **kwargs
        )
        """Initialize the Frontend Integration Developer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_framework: Preferred frontend framework
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.FRONTEND_INTEGRATION, 
            **kwargs
        )
        self.preferred_framework = preferred_framework
        
        # Track page implementations
        self.page_implementations: Dict[str, Dict[str, Any]] = {}
        
        # Track feature implementations
        self.feature_implementations: Dict[str, Dict[str, Any]] = {}
        
        # Track integration tests
        self.integration_tests: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Frontend Integration Developer Agent initialized with {preferred_framework} framework")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Frontend Integration Developer."""
        return (
            f"You are {self.name}, a Frontend Integration Developer specialized in connecting "
            f"UI components with business logic and backend services using {self.preferred_framework}. "
            f"Your responsibilities include:\n"
            f"1. Integrating UI components with state management\n"
            f"2. Connecting frontend to backend APIs\n"
            f"3. Implementing routing and navigation\n"
            f"4. Ensuring consistent data flow throughout the application\n"
            f"5. Implementing error handling and loading states\n\n"
            f"Focus on creating robust integrations that handle edge cases and provide good user experience. "
            f"Consider performance, error states, loading states, and overall user flow. Ensure that "
            f"components receive the data they need and that user actions are properly handled. "
            f"Pay special attention to providing consistent error handling and loading indicators."
        )
    
    async def implement_page(
        self, 
        page_name: str,
        description: str,
        components: List[Dict[str, Any]],
        state_dependencies: List[Dict[str, Any]],
        api_dependencies: List[Dict[str, Any]]
    ) -> TaskResult:
        """Implement a page that integrates components, state, and APIs.
        
        Args:
            page_name: Name of the page
            description: Brief description of the page's purpose
            components: List of UI components to use
            state_dependencies: List of state dependencies
            api_dependencies: List of API dependencies
            
        Returns:
            TaskResult containing the page implementation
        """
        # Create a task for implementing the page
        task = Task(
            task_id=f"implement_page_{page_name.lower().replace(' ', '_')}",
            description=f"Implement page: {page_name}",
            agent_type=str(AgentRole.FRONTEND_INTEGRATION),
            requirements={
                "page_name": page_name,
                "description": description,
                "components": components,
                "state_dependencies": state_dependencies,
                "api_dependencies": api_dependencies,
                "framework": self.preferred_framework
            },
            context=TaskContext(
                notes=(
                    f"Implement the {page_name} page using {self.preferred_framework}. "
                    f"The implementation should integrate the specified components with "
                    f"state management and API calls. Handle loading states, error states, "
                    f"and ensure proper data flow throughout the page."
                )
            ),
            expected_output=(
                f"Complete {self.preferred_framework} implementation of the {page_name} page, "
                f"including component integration, state management, API calls, routing, "
                f"and error handling."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create an ID for this page
            page_id = str(uuid.uuid4())
            
            # Store in our tracking dictionary
            self.page_implementations[page_id] = {
                "page_name": page_name,
                "description": description,
                "components": components,
                "state_dependencies": state_dependencies,
                "api_dependencies": api_dependencies,
                "implemented_at": datetime.now().isoformat()
            }
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"page_implementation_{page_id}",
                    value={
                        "page_name": page_name,
                        "description": description,
                        "components": components,
                        "state_dependencies": state_dependencies,
                        "api_dependencies": api_dependencies,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="page_implementations"
                )
            
            logger.info(f"Implemented {page_name} page with {len(components)} components")
            
            # Return the implementation code
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "page_id": page_id,
                    "page_name": page_name
                }
            )
            
            return updated_result
        
        return result
    
    async def implement_feature(
        self, 
        feature_name: str,
        description: str,
        components: List[Dict[str, Any]],
        logic: Dict[str, Any],
        api_integrations: List[Dict[str, Any]]
    ) -> TaskResult:
        """Implement a feature that integrates components, logic, and APIs.
        
        Args:
            feature_name: Name of the feature
            description: Brief description of the feature's purpose
            components: List of UI components to use
            logic: Frontend logic to integrate
            api_integrations: List of API integrations to use
            
        Returns:
            TaskResult containing the feature implementation
        """
        # Create a task for implementing the feature
        task = Task(
            task_id=f"implement_feature_{feature_name.lower().replace(' ', '_')}",
            description=f"Implement feature: {feature_name}",
            agent_type=str(AgentRole.FRONTEND_INTEGRATION),
            requirements={
                "feature_name": feature_name,
                "description": description,
                "components": components,
                "logic": logic,
                "api_integrations": api_integrations,
                "framework": self.preferred_framework
            },
            context=TaskContext(
                notes=(
                    f"Implement the {feature_name} feature using {self.preferred_framework}. "
                    f"The implementation should integrate the specified components with "
                    f"frontend logic and API integrations. Ensure proper data flow, "
                    f"error handling, and user experience."
                )
            ),
            expected_output=(
                f"Complete {self.preferred_framework} implementation of the {feature_name} feature, "
                f"including component integration, logic implementation, API calls, "
                f"and error handling."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create an ID for this feature
            feature_id = str(uuid.uuid4())
            
            # Store in our tracking dictionary
            self.feature_implementations[feature_id] = {
                "feature_name": feature_name,
                "description": description,
                "components": components,
                "logic": logic,
                "api_integrations": api_integrations,
                "implemented_at": datetime.now().isoformat()
            }
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"feature_implementation_{feature_id}",
                    value={
                        "feature_name": feature_name,
                        "description": description,
                        "components": components,
                        "logic": logic,
                        "api_integrations": api_integrations,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="feature_implementations"
                )
            
            logger.info(f"Implemented {feature_name} feature with {len(components)} components")
            
            # Return the implementation code
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "feature_id": feature_id,
                    "feature_name": feature_name
                }
            )
            
            return updated_result
        
        return result
    
    async def implement_routing(
        self, 
        app_name: str,
        pages: List[Dict[str, Any]],
        auth_requirements: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Implement application routing configuration.
        
        Args:
            app_name: Name of the application
            pages: List of pages to include in routing
            auth_requirements: Optional authentication requirements for routes
            
        Returns:
            TaskResult containing the routing implementation
        """
        # Create a task for implementing routing
        task = Task(
            task_id=f"implement_routing_{app_name.lower().replace(' ', '_')}",
            description=f"Implement routing for {app_name}",
            agent_type=str(AgentRole.FRONTEND_INTEGRATION),
            requirements={
                "app_name": app_name,
                "pages": pages,
                "has_auth": auth_requirements is not None,
                "auth_requirements": auth_requirements,
                "framework": self.preferred_framework
            },
            context=TaskContext(
                notes=(
                    f"Implement routing configuration for {app_name} using {self.preferred_framework}. "
                    f"The implementation should define routes for all specified pages with "
                    f"appropriate path patterns, parameters, and nesting. "
                    + (f"Include authentication protection for routes based on the specified requirements. " 
                       if auth_requirements else "")
                )
            ),
            expected_output=(
                f"Complete routing implementation for {app_name}, including route definitions, "
                f"nested routes, route parameters, and route protection if applicable."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"routing_{app_name.lower().replace(' ', '_')}",
                    value={
                        "app_name": app_name,
                        "pages": pages,
                        "auth_requirements": auth_requirements,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="routing_implementations"
                )
            
            logger.info(f"Implemented routing for {app_name} with {len(pages)} pages")
            
            # Return the implementation code
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "app_name": app_name,
                    "has_auth": auth_requirements is not None
                }
            )
            
            return updated_result
        
        return result
    
    async def create_integration_tests(
        self, 
        target_type: str,
        target_id: str,
        target_name: str,
        test_framework: str = "cypress"
    ) -> TaskResult:
        """Create integration tests for a page or feature.
        
        Args:
            target_type: Type of target to test ("page" or "feature")
            target_id: ID of the target to test
            target_name: Name of the target to test
            test_framework: Testing framework to use
            
        Returns:
            TaskResult containing the integration tests
        """
        # Try to get the implementation
        implementation = None
        
        if self.shared_memory:
            if target_type == "page":
                impl_data = self.shared_memory.retrieve(
                    key=f"page_implementation_{target_id}",
                    category="page_implementations"
                )
                if impl_data:
                    implementation = impl_data
            elif target_type == "feature":
                impl_data = self.shared_memory.retrieve(
                    key=f"feature_implementation_{target_id}",
                    category="feature_implementations"
                )
                if impl_data:
                    implementation = impl_data
        
        # Create a task for creating integration tests
        task = Task(
            task_id=f"create_integration_tests_{target_type}_{target_id}",
            description=f"Create integration tests for {target_name} {target_type}",
            agent_type=str(AgentRole.FRONTEND_INTEGRATION),
            requirements={
                "target_type": target_type,
                "target_id": target_id,
                "target_name": target_name,
                "has_implementation": implementation is not None,
                "implementation": implementation,
                "framework": self.preferred_framework,
                "test_framework": test_framework
            },
            context=TaskContext(
                notes=(
                    f"Create integration tests for the {target_name} {target_type} using {test_framework}. "
                    f"The tests should verify that components are properly integrated, data flows correctly, "
                    f"API calls are made and handled properly, and user interactions work as expected. "
                    f"Include tests for happy paths, error cases, and edge cases."
                    + (f"\n\nThe implementation code is available for reference." if implementation else "")
                )
            ),
            expected_output=(
                f"Complete integration test suite for the {target_name} {target_type} using {test_framework}, "
                f"covering component interaction, data flow, API integration, and user scenarios."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the tests
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create an ID for these tests
            test_id = str(uuid.uuid4())
            
            # Store in our tracking dictionary
            self.integration_tests[test_id] = {
                "target_type": target_type,
                "target_id": target_id,
                "target_name": target_name,
                "test_framework": test_framework,
                "created_at": datetime.now().isoformat()
            }
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"integration_tests_{target_type}_{target_id}",
                    value={
                        "target_type": target_type,
                        "target_id": target_id,
                        "target_name": target_name,
                        "test_framework": test_framework,
                        "tests": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="integration_tests"
                )
            
            logger.info(f"Created integration tests for {target_name} {target_type} using {test_framework}")
            
            # Return the tests
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "test_id": test_id,
                    "target_type": target_type,
                    "target_name": target_name,
                    "test_framework": test_framework
                }
            )
            
            return updated_result
        
        return result
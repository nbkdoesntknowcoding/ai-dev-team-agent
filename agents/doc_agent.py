"""
Documentation Writer Agent for the multi-agent development system.

This module contains the specialized agent for documentation tasks, including
generating user guides, API documentation, technical specifications, and other
documentation artifacts. The agent ensures clear, comprehensive, and accurate
documentation for various audiences.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple, cast
import uuid

from pydantic import BaseModel, Field, validator

from agents.base_agent import (
    BaseAgent, 
    Task, 
    TaskResult, 
    TaskStatus, 
    TaskPriority,
    TaskContext,
    AgentRole
)

# Set up logging
logger = logging.getLogger(__name__)


class DocumentationTemplate(BaseModel):
    """Template for a documentation artifact."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    audience: str  # "developer", "user", "admin", etc.
    format: str  # "markdown", "html", "rst", etc.


class DocumentationArtifact(BaseModel):
    """A complete documentation artifact."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    content: str
    template_id: Optional[str] = None
    audience: str
    format: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    related_artifacts: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DocumentationFeedback(BaseModel):
    """Feedback on a documentation artifact."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_id: str
    feedback_source: str
    feedback_text: str
    rating: Optional[int] = None  # 1-5 rating
    resolved: bool = False
    response: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DocumentationWriterAgent(BaseAgent):
    """Agent specialized in creating and managing documentation."""
    
    def __init__(
        self, 
        name: str = "Documentation Writer",
        preferred_formats: List[str] = ["markdown", "html"],
        documentation_standards: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize the Documentation Writer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_formats: List of preferred documentation formats
            documentation_standards: Optional documentation standards to follow
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.DOCUMENTATION, 
            **kwargs
        )
        self.preferred_formats = preferred_formats
        self.documentation_standards = documentation_standards or self._default_documentation_standards()
        
        # Track documentation templates
        self.templates: Dict[str, DocumentationTemplate] = {}
        
        # Track documentation artifacts
        self.artifacts: Dict[str, DocumentationArtifact] = {}
        
        # Track documentation feedback
        self.feedback: Dict[str, List[DocumentationFeedback]] = {}
        
        # Load default templates
        self._load_default_templates()
        
        logger.info(f"Documentation Writer Agent initialized with {', '.join(preferred_formats)} formats")
    
    def _default_documentation_standards(self) -> Dict[str, Any]:
        """Generate default documentation standards.
        
        Returns:
            Default documentation standards
        """
        return {
            "general": [
                "Use clear, concise language",
                "Avoid jargon unless necessary for the audience",
                "Include examples for complex concepts",
                "Use consistent terminology throughout",
                "Follow the inverted pyramid style (most important information first)",
                "Include a table of contents for longer documents",
                "Use appropriate headings and subheadings for organization"
            ],
            "api_documentation": [
                "Document all endpoints, parameters, and return values",
                "Include request and response examples",
                "Describe error codes and handling",
                "Specify authentication requirements",
                "Include rate limiting information",
                "Document versioning strategy"
            ],
            "user_guides": [
                "Start with an introduction and purpose",
                "Include step-by-step instructions with screenshots",
                "Provide troubleshooting sections for common issues",
                "Include a glossary of terms if needed",
                "Write from the user's perspective",
                "Include navigation aids (contents, index)"
            ],
            "code_documentation": [
                "Document function purpose, parameters, and return values",
                "Include usage examples",
                "Document exceptions and edge cases",
                "Explain complex algorithms or logic",
                "Include performance considerations",
                "Reference related functions or classes"
            ],
            "technical_specifications": [
                "Start with an executive summary",
                "Include system architecture diagrams",
                "Document components and their interactions",
                "Specify performance requirements",
                "Include security considerations",
                "Document data models and flows"
            ]
        }
    
    def _load_default_templates(self):
        """Load default documentation templates."""
        # API Documentation Template
        api_template = DocumentationTemplate(
            name="API Documentation",
            description="Template for documenting APIs",
            audience="developer",
            format="markdown",
            sections=[
                {
                    "name": "Introduction",
                    "description": "Overview of the API and its purpose",
                    "required": True
                },
                {
                    "name": "Authentication",
                    "description": "Authentication methods and requirements",
                    "required": True
                },
                {
                    "name": "Endpoints",
                    "description": "Detailed documentation of all API endpoints",
                    "required": True,
                    "subsections": [
                        {
                            "name": "Endpoint Information",
                            "items": ["Method", "Path", "Description", "Parameters", "Request Example", "Response Example", "Error Codes"]
                        }
                    ]
                },
                {
                    "name": "Rate Limiting",
                    "description": "Information about rate limits",
                    "required": False
                },
                {
                    "name": "Webhooks",
                    "description": "Webhook information if applicable",
                    "required": False
                },
                {
                    "name": "SDKs and Libraries",
                    "description": "Information about available SDKs and libraries",
                    "required": False
                },
                {
                    "name": "Versioning",
                    "description": "API versioning information",
                    "required": True
                }
            ]
        )
        self.templates[api_template.id] = api_template
        
        # User Guide Template
        user_guide_template = DocumentationTemplate(
            name="User Guide",
            description="Template for user guides and manuals",
            audience="user",
            format="markdown",
            sections=[
                {
                    "name": "Introduction",
                    "description": "Introduction to the product and the guide",
                    "required": True
                },
                {
                    "name": "Getting Started",
                    "description": "How to get started with the product",
                    "required": True
                },
                {
                    "name": "Features",
                    "description": "Detailed explanation of product features",
                    "required": True
                },
                {
                    "name": "Usage Instructions",
                    "description": "Step-by-step instructions for using the product",
                    "required": True
                },
                {
                    "name": "Troubleshooting",
                    "description": "Common issues and their solutions",
                    "required": True
                },
                {
                    "name": "FAQ",
                    "description": "Frequently asked questions",
                    "required": False
                },
                {
                    "name": "Glossary",
                    "description": "Definitions of terms used in the guide",
                    "required": False
                }
            ]
        )
        self.templates[user_guide_template.id] = user_guide_template
        
        # Technical Specification Template
        tech_spec_template = DocumentationTemplate(
            name="Technical Specification",
            description="Template for technical specifications",
            audience="developer",
            format="markdown",
            sections=[
                {
                    "name": "Executive Summary",
                    "description": "Brief summary of the specification",
                    "required": True
                },
                {
                    "name": "Introduction",
                    "description": "Introduction and purpose",
                    "required": True
                },
                {
                    "name": "System Architecture",
                    "description": "Overall system architecture",
                    "required": True
                },
                {
                    "name": "Components",
                    "description": "Detailed description of system components",
                    "required": True
                },
                {
                    "name": "Interfaces",
                    "description": "Description of interfaces between components",
                    "required": True
                },
                {
                    "name": "Data Model",
                    "description": "Data models and database schema",
                    "required": True
                },
                {
                    "name": "Security",
                    "description": "Security considerations and implementation",
                    "required": True
                },
                {
                    "name": "Performance Requirements",
                    "description": "Performance requirements and benchmarks",
                    "required": True
                },
                {
                    "name": "Implementation Considerations",
                    "description": "Notes about implementation",
                    "required": False
                }
            ]
        )
        self.templates[tech_spec_template.id] = tech_spec_template
        
        # Code Documentation Template
        code_doc_template = DocumentationTemplate(
            name="Code Documentation",
            description="Template for code documentation",
            audience="developer",
            format="markdown",
            sections=[
                {
                    "name": "Overview",
                    "description": "Overview of the code and its purpose",
                    "required": True
                },
                {
                    "name": "Installation",
                    "description": "Installation instructions",
                    "required": True
                },
                {
                    "name": "Usage",
                    "description": "How to use the code",
                    "required": True
                },
                {
                    "name": "API Reference",
                    "description": "Reference for all public APIs",
                    "required": True
                },
                {
                    "name": "Examples",
                    "description": "Usage examples",
                    "required": True
                },
                {
                    "name": "Contributing",
                    "description": "How to contribute to the codebase",
                    "required": False
                },
                {
                    "name": "License",
                    "description": "License information",
                    "required": True
                }
            ]
        )
        self.templates[code_doc_template.id] = code_doc_template
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Documentation Writer."""
        return (
            f"You are {self.name}, a Documentation Writer specialized in creating clear, "
            f"comprehensive, and accurate documentation for various audiences. "
            f"Your responsibilities include:\n"
            f"1. Creating user guides, API documentation, and technical specifications\n"
            f"2. Ensuring documentation is clear, accurate, and appropriate for the target audience\n"
            f"3. Maintaining consistent style and terminology across documentation\n"
            f"4. Incorporating feedback to improve documentation quality\n"
            f"5. Documenting complex technical concepts in an accessible way\n\n"
            f"When writing documentation, focus on clarity, completeness, and accuracy. "
            f"Use appropriate language for the target audience, include examples where helpful, "
            f"and organize information logically. Format documentation consistently "
            f"and follow established documentation standards and best practices."
        )
    
    async def create_documentation(
        self, 
        title: str,
        description: str,
        doc_type: str,
        source_material: Dict[str, Any],
        audience: str,
        format: Optional[str] = None,
        template_id: Optional[str] = None
    ) -> TaskResult:
        """Create a documentation artifact based on source material.
        
        Args:
            title: Title of the documentation
            description: Brief description of the documentation purpose
            doc_type: Type of documentation (api, user_guide, tech_spec, etc.)
            source_material: Source material to base the documentation on
            audience: Target audience (developer, user, admin, etc.)
            format: Optional format override (defaults to preferred format)
            template_id: Optional template ID to use
            
        Returns:
            TaskResult containing the documentation artifact
        """
        # Determine format
        if not format:
            format = self.preferred_formats[0] if self.preferred_formats else "markdown"
        
        # Find appropriate template if not specified
        template = None
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
        else:
            # Try to find a matching template by name/type
            for tmpl in self.templates.values():
                if (tmpl.name.lower().replace(' ', '_') == doc_type.lower().replace(' ', '_') or
                    doc_type.lower() in tmpl.name.lower()):
                    template = tmpl
                    template_id = tmpl.id
                    break
        
        # Get the relevant standards
        doc_standards = self.documentation_standards.get(doc_type, self.documentation_standards.get("general", []))
        
        # Create a task for documentation creation
        task = Task(
            task_id=f"create_doc_{doc_type}_{title.lower().replace(' ', '_')}",
            description=f"Create {doc_type} documentation: {title}",
            agent_type=str(AgentRole.DOCUMENTATION),
            requirements={
                "title": title,
                "description": description,
                "doc_type": doc_type,
                "source_material": source_material,
                "audience": audience,
                "format": format,
                "template": template.dict() if template else None,
                "standards": doc_standards
            },
            context=TaskContext(
                notes=(
                    f"Create {doc_type} documentation titled '{title}' for {audience} audience in {format} format. "
                    f"Use the provided source material and follow documentation standards. "
                    + (f"Use the provided template as a structure guide. " if template else "")
                )
            ),
            expected_output=(
                f"Complete {doc_type} documentation in {format} format that is clear, "
                f"comprehensive, accurate, and appropriate for {audience} audience."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the documentation artifact
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create the documentation artifact
            artifact = DocumentationArtifact(
                title=title,
                description=description,
                content=result.result,
                template_id=template_id,
                audience=audience,
                format=format,
                metadata={
                    "doc_type": doc_type,
                    "source_material_type": source_material.get("type", "unknown"),
                    "source_material_id": source_material.get("id", "unknown")
                },
                tags=[doc_type, audience, format]
            )
            
            # Store the artifact
            self.artifacts[artifact.id] = artifact
            
            # Initialize feedback tracking
            self.feedback[artifact.id] = []
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"documentation_{artifact.id}",
                    value=artifact.dict(),
                    category="documentation"
                )
            
            logger.info(f"Created {doc_type} documentation '{title}' for {audience} audience")
            
            # Return the documentation artifact
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=artifact.dict(),
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={"artifact_id": artifact.id}
            )
            
            return updated_result
        
        return result
    
    async def create_api_documentation(
        self, 
        api_name: str,
        api_spec: Dict[str, Any],
        include_examples: bool = True,
        format: Optional[str] = None
    ) -> TaskResult:
        """Create API documentation based on an API specification.
        
        Args:
            api_name: Name of the API
            api_spec: API specification (endpoints, parameters, etc.)
            include_examples: Whether to include examples
            format: Optional format override
            
        Returns:
            TaskResult containing the API documentation
        """
        # Find the API documentation template
        template_id = None
        for tmpl_id, tmpl in self.templates.items():
            if tmpl.name == "API Documentation":
                template_id = tmpl_id
                break
        
        # Get API documentation standards
        api_standards = self.documentation_standards.get("api_documentation", self.documentation_standards.get("general", []))
        
        # Create source material
        source_material = {
            "type": "api_specification",
            "id": api_spec.get("id", str(uuid.uuid4())),
            "api_name": api_name,
            "api_spec": api_spec,
            "include_examples": include_examples
        }
        
        # Create a task for API documentation
        task = Task(
            task_id=f"create_api_doc_{api_name.lower().replace(' ', '_')}",
            description=f"Create API documentation for {api_name}",
            agent_type=str(AgentRole.DOCUMENTATION),
            requirements={
                "api_name": api_name,
                "api_spec": api_spec,
                "include_examples": include_examples,
                "format": format or self.preferred_formats[0],
                "standards": api_standards,
                "template_id": template_id
            },
            context=TaskContext(
                notes=(
                    f"Create comprehensive API documentation for {api_name}. "
                    f"Include all endpoints, parameters, and return values. "
                    + ("Include request and response examples. " if include_examples else "")
                    + f"Follow API documentation standards and best practices."
                )
            ),
            expected_output=(
                f"Complete API documentation in {format or self.preferred_formats[0]} format "
                f"that thoroughly documents all aspects of the API."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the API documentation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create the documentation artifact
            artifact = DocumentationArtifact(
                title=f"{api_name} API Documentation",
                description=f"API documentation for {api_name}",
                content=result.result,
                template_id=template_id,
                audience="developer",
                format=format or self.preferred_formats[0],
                metadata={
                    "doc_type": "api_documentation",
                    "api_name": api_name,
                    "api_spec_id": api_spec.get("id", "unknown"),
                    "include_examples": include_examples
                },
                tags=["api_documentation", "developer", format or self.preferred_formats[0]]
            )
            
            # Store the artifact
            self.artifacts[artifact.id] = artifact
            
            # Initialize feedback tracking
            self.feedback[artifact.id] = []
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"api_documentation_{artifact.id}",
                    value=artifact.dict(),
                    category="documentation"
                )
            
            logger.info(f"Created API documentation for {api_name}")
            
            # Return the API documentation
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=artifact.dict(),
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={"artifact_id": artifact.id}
            )
            
            return updated_result
        
        return result
    
    async def create_user_guide(
        self, 
        product_name: str,
        features: List[Dict[str, Any]],
        workflows: List[Dict[str, Any]],
        screenshots: Optional[List[Dict[str, Any]]] = None,
        format: Optional[str] = None
    ) -> TaskResult:
        """Create a user guide for a product.
        
        Args:
            product_name: Name of the product
            features: List of product features
            workflows: List of common workflows
            screenshots: Optional screenshots to include
            format: Optional format override
            
        Returns:
            TaskResult containing the user guide
        """
        # Find the user guide template
        template_id = None
        for tmpl_id, tmpl in self.templates.items():
            if tmpl.name == "User Guide":
                template_id = tmpl_id
                break
        
        # Get user guide standards
        user_guide_standards = self.documentation_standards.get("user_guides", self.documentation_standards.get("general", []))
        
        # Create source material
        source_material = {
            "type": "product_information",
            "id": str(uuid.uuid4()),
            "product_name": product_name,
            "features": features,
            "workflows": workflows,
            "has_screenshots": screenshots is not None,
            "screenshots": screenshots or []
        }
        
        # Create a task for user guide creation
        task = Task(
            task_id=f"create_user_guide_{product_name.lower().replace(' ', '_')}",
            description=f"Create user guide for {product_name}",
            agent_type=str(AgentRole.DOCUMENTATION),
            requirements={
                "product_name": product_name,
                "features": features,
                "workflows": workflows,
                "screenshots": screenshots or [],
                "format": format or self.preferred_formats[0],
                "standards": user_guide_standards,
                "template_id": template_id
            },
            context=TaskContext(
                notes=(
                    f"Create a comprehensive user guide for {product_name}. "
                    f"Include sections on getting started, features, and common workflows. "
                    f"Write in a clear, user-friendly style appropriate for the end user. "
                    + (f"Incorporate the provided screenshots where relevant. " if screenshots else "")
                    + f"Follow user guide documentation standards and best practices."
                )
            ),
            expected_output=(
                f"Complete user guide in {format or self.preferred_formats[0]} format "
                f"that helps users understand and use the product effectively."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the user guide
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create the documentation artifact
            artifact = DocumentationArtifact(
                title=f"{product_name} User Guide",
                description=f"User guide for {product_name}",
                content=result.result,
                template_id=template_id,
                audience="user",
                format=format or self.preferred_formats[0],
                metadata={
                    "doc_type": "user_guide",
                    "product_name": product_name,
                    "has_screenshots": screenshots is not None
                },
                tags=["user_guide", "user", format or self.preferred_formats[0]]
            )
            
            # Store the artifact
            self.artifacts[artifact.id] = artifact
            
            # Initialize feedback tracking
            self.feedback[artifact.id] = []
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"user_guide_{artifact.id}",
                    value=artifact.dict(),
                    category="documentation"
                )
            
            logger.info(f"Created user guide for {product_name}")
            
            # Return the user guide
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=artifact.dict(),
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={"artifact_id": artifact.id}
            )
            
            return updated_result
        
        return result
    
    async def create_technical_specification(
        self, 
        component_name: str,
        architecture: Dict[str, Any],
        requirements: List[Dict[str, Any]],
        interfaces: List[Dict[str, Any]],
        format: Optional[str] = None
    ) -> TaskResult:
        """Create a technical specification for a component.
        
        Args:
            component_name: Name of the component
            architecture: Architecture information
            requirements: List of requirements
            interfaces: List of interfaces
            format: Optional format override
            
        Returns:
            TaskResult containing the technical specification
        """
        # Find the technical specification template
        template_id = None
        for tmpl_id, tmpl in self.templates.items():
            if tmpl.name == "Technical Specification":
                template_id = tmpl_id
                break
        
        # Get technical specification standards
        tech_spec_standards = self.documentation_standards.get("technical_specifications", self.documentation_standards.get("general", []))
        
        # Create source material
        source_material = {
            "type": "component_information",
            "id": str(uuid.uuid4()),
            "component_name": component_name,
            "architecture": architecture,
            "requirements": requirements,
            "interfaces": interfaces
        }
        
        # Create a task for technical specification creation
        task = Task(
            task_id=f"create_tech_spec_{component_name.lower().replace(' ', '_')}",
            description=f"Create technical specification for {component_name}",
            agent_type=str(AgentRole.DOCUMENTATION),
            requirements={
                "component_name": component_name,
                "architecture": architecture,
                "requirements": requirements,
                "interfaces": interfaces,
                "format": format or self.preferred_formats[0],
                "standards": tech_spec_standards,
                "template_id": template_id
            },
            context=TaskContext(
                notes=(
                    f"Create a comprehensive technical specification for {component_name}. "
                    f"Include architecture details, requirements, interfaces, and implementation considerations. "
                    f"Write in a clear, precise style appropriate for technical stakeholders. "
                    f"Follow technical specification documentation standards and best practices."
                )
            ),
            expected_output=(
                f"Complete technical specification in {format or self.preferred_formats[0]} format "
                f"that provides all necessary details for implementing and integrating the component."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the technical specification
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create the documentation artifact
            artifact = DocumentationArtifact(
                title=f"{component_name} Technical Specification",
                description=f"Technical specification for {component_name}",
                content=result.result,
                template_id=template_id,
                audience="developer",
                format=format or self.preferred_formats[0],
                metadata={
                    "doc_type": "technical_specification",
                    "component_name": component_name
                },
                tags=["technical_specification", "developer", format or self.preferred_formats[0]]
            )
            
            # Store the artifact
            self.artifacts[artifact.id] = artifact
            
            # Initialize feedback tracking
            self.feedback[artifact.id] = []
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"technical_specification_{artifact.id}",
                    value=artifact.dict(),
                    category="documentation"
                )
            
            logger.info(f"Created technical specification for {component_name}")
            
            # Return the technical specification
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=artifact.dict(),
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={"artifact_id": artifact.id}
            )
            
            return updated_result
        
        return result
    
    async def create_code_documentation(
        self, 
        module_name: str,
        code_files: List[Dict[str, Any]],
        include_examples: bool = True,
        format: Optional[str] = None
    ) -> TaskResult:
        """Create code documentation for a module.
        
        Args:
            module_name: Name of the module
            code_files: List of code files with content
            include_examples: Whether to include examples
            format: Optional format override
            
        Returns:
            TaskResult containing the code documentation
        """
        # Find the code documentation template
        template_id = None
        for tmpl_id, tmpl in self.templates.items():
            if tmpl.name == "Code Documentation":
                template_id = tmpl_id
                break
        
        # Get code documentation standards
        code_doc_standards = self.documentation_standards.get("code_documentation", self.documentation_standards.get("general", []))
        
        # Create source material
        source_material = {
            "type": "code_module",
            "id": str(uuid.uuid4()),
            "module_name": module_name,
            "code_files": code_files,
            "include_examples": include_examples
        }
        
        # Create a task for code documentation creation
        task = Task(
            task_id=f"create_code_doc_{module_name.lower().replace(' ', '_')}",
            description=f"Create code documentation for {module_name}",
            agent_type=str(AgentRole.DOCUMENTATION),
            requirements={
                "module_name": module_name,
                "code_files": code_files,
                "include_examples": include_examples,
                "format": format or self.preferred_formats[0],
                "standards": code_doc_standards,
                "template_id": template_id
            },
            context=TaskContext(
                notes=(
                    f"Create comprehensive code documentation for {module_name}. "
                    f"Document the purpose, usage, and API of the code. "
                    + (f"Include usage examples. " if include_examples else "")
                    + f"Write in a clear, precise style appropriate for developers. "
                    f"Follow code documentation standards and best practices."
                )
            ),
            expected_output=(
                f"Complete code documentation in {format or self.preferred_formats[0]} format "
                f"that helps developers understand and use the code effectively."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the code documentation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create the documentation artifact
            artifact = DocumentationArtifact(
                title=f"{module_name} Code Documentation",
                description=f"Code documentation for {module_name}",
                content=result.result,
                template_id=template_id,
                audience="developer",
                format=format or self.preferred_formats[0],
                metadata={
                    "doc_type": "code_documentation",
                    "module_name": module_name,
                    "include_examples": include_examples
                },
                tags=["code_documentation", "developer", format or self.preferred_formats[0]]
            )
            
            # Store the artifact
            self.artifacts[artifact.id] = artifact
            
            # Initialize feedback tracking
            self.feedback[artifact.id] = []
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"code_documentation_{artifact.id}",
                    value=artifact.dict(),
                    category="documentation"
                )
            
            logger.info(f"Created code documentation for {module_name}")
            
            # Return the code documentation
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=artifact.dict(),
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={"artifact_id": artifact.id}
            )
            
            return updated_result
        
        return result
    
    async def update_documentation(
        self, 
        artifact_id: str,
        changes: Dict[str, Any],
        reason: str
    ) -> TaskResult:
        """Update an existing documentation artifact.
        
        Args:
            artifact_id: ID of the artifact to update
            changes: Dictionary of changes (content, metadata, etc.)
            reason: Reason for the update
            
        Returns:
            TaskResult containing the updated artifact
        """
        # Check if artifact exists
        if artifact_id not in self.artifacts:
            # Try to load from shared memory if available
            if self.shared_memory:
                artifact_data = self.shared_memory.retrieve(
                    key=f"documentation_{artifact_id}",
                    category="documentation"
                )
                if artifact_data:
                    self.artifacts[artifact_id] = DocumentationArtifact(**artifact_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"update_documentation_{artifact_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Documentation artifact with ID {artifact_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"update_documentation_{artifact_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Documentation artifact with ID {artifact_id} not found"
                )
        
        # Get the artifact
        artifact = self.artifacts[artifact_id]
        
        # Get relevant standards
        doc_type = artifact.metadata.get("doc_type", "general")
        doc_standards = self.documentation_standards.get(doc_type, self.documentation_standards.get("general", []))
        
        # If we're updating content, create a task for it
        if "content" in changes:
            # Create a task for updating documentation
            task = Task(
                task_id=f"update_documentation_{artifact_id}",
                description=f"Update {artifact.title} documentation",
                agent_type=str(AgentRole.DOCUMENTATION),
                requirements={
                    "artifact_id": artifact_id,
                    "current_content": artifact.content,
                    "update_reason": reason,
                    "changes": changes,
                    "audience": artifact.audience,
                    "format": artifact.format,
                    "standards": doc_standards
                },
                context=TaskContext(
                    notes=(
                        f"Update the {artifact.title} documentation based on the specified changes. "
                        f"The reason for the update is: {reason}. "
                        f"Ensure the updated documentation maintains quality and follows documentation standards."
                    )
                ),
                expected_output=(
                    f"Updated documentation content that incorporates the requested changes "
                    f"while maintaining quality and consistency."
                ),
                priority=TaskPriority.HIGH
            )
            
            # Execute the task
            result = await self.execute_task(task)
            
            # If successful, update the artifact
            if result.status == TaskStatus.COMPLETED and result.result:
                # Update the content
                changes["content"] = result.result
            else:
                # Return the task result if not successful
                return result
        
        # Update the artifact with the changes
        artifact_dict = artifact.dict()
        for key, value in changes.items():
            if key in artifact_dict:
                artifact_dict[key] = value
        
        # Increment version
        version_parts = artifact_dict["version"].split('.')
        if len(version_parts) >= 3:
            version_parts[2] = str(int(version_parts[2]) + 1)
            artifact_dict["version"] = '.'.join(version_parts)
        
        # Update timestamp
        artifact_dict["updated_at"] = datetime.now().isoformat()
        
        # Create updated artifact
        updated_artifact = DocumentationArtifact(**artifact_dict)
        
        # Store the updated artifact
        self.artifacts[artifact_id] = updated_artifact
        
        # Store in shared memory if available
        if self.shared_memory:
            self.shared_memory.store(
                key=f"documentation_{artifact_id}",
                value=updated_artifact.dict(),
                category="documentation"
            )
            
            # Also store update history
            self.shared_memory.store(
                key=f"documentation_update_{artifact_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                value={
                    "artifact_id": artifact_id,
                    "reason": reason,
                    "changes": {k: "..." if k == "content" else v for k, v in changes.items()},
                    "version": updated_artifact.version,
                    "timestamp": updated_artifact.updated_at
                },
                category="documentation_updates"
            )
        
        logger.info(f"Updated documentation artifact {artifact.title} to version {updated_artifact.version}")
        
        # Return the updated artifact
        return TaskResult(
            agent_id=self.state.agent_id,
            agent_name=self.name,
            task_id=f"update_documentation_{artifact_id}",
            result=updated_artifact.dict(),
            status=TaskStatus.COMPLETED,
            timestamp=datetime.now().isoformat(),
            execution_time=0.0,
            metadata={
                "artifact_id": artifact_id,
                "version": updated_artifact.version
            }
        )
    
    async def process_feedback(
        self, 
        artifact_id: str,
        feedback_source: str,
        feedback_text: str,
        rating: Optional[int] = None
    ) -> TaskResult:
        """Process feedback on a documentation artifact.
        
        Args:
            artifact_id: ID of the artifact
            feedback_source: Source of the feedback
            feedback_text: Feedback text
            rating: Optional numerical rating (1-5)
            
        Returns:
            TaskResult containing the response to the feedback
        """
        # Check if artifact exists
        if artifact_id not in self.artifacts:
            # Try to load from shared memory if available
            if self.shared_memory:
                artifact_data = self.shared_memory.retrieve(
                    key=f"documentation_{artifact_id}",
                    category="documentation"
                )
                if artifact_data:
                    self.artifacts[artifact_id] = DocumentationArtifact(**artifact_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"process_feedback_{artifact_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Documentation artifact with ID {artifact_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"process_feedback_{artifact_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Documentation artifact with ID {artifact_id} not found"
                )
        
        # Get the artifact
        artifact = self.artifacts[artifact_id]
        
        # Create a task for processing feedback
        task = Task(
            task_id=f"process_feedback_{artifact_id}",
            description=f"Process feedback on {artifact.title}",
            agent_type=str(AgentRole.DOCUMENTATION),
            requirements={
                "artifact_id": artifact_id,
                "artifact_title": artifact.title,
                "artifact_type": artifact.metadata.get("doc_type", "documentation"),
                "feedback_source": feedback_source,
                "feedback_text": feedback_text,
                "rating": rating
            },
            context=TaskContext(
                notes=(
                    f"Process feedback on {artifact.title} documentation from {feedback_source}. "
                    f"Analyze the feedback, determine if it requires documentation updates, "
                    f"and provide a thoughtful response."
                )
            ),
            expected_output=(
                "A response to the feedback that acknowledges the feedback, "
                "explains whether changes will be made, and provides next steps."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the feedback and response
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create the feedback
            feedback_obj = DocumentationFeedback(
                artifact_id=artifact_id,
                feedback_source=feedback_source,
                feedback_text=feedback_text,
                rating=rating,
                response=result.result
            )
            
            # Store the feedback
            if artifact_id in self.feedback:
                self.feedback[artifact_id].append(feedback_obj)
            else:
                self.feedback[artifact_id] = [feedback_obj]
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"documentation_feedback_{feedback_obj.id}",
                    value=feedback_obj.dict(),
                    category="documentation_feedback"
                )
            
            logger.info(f"Processed feedback on {artifact.title} from {feedback_source}")
            
            # Return the response
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result={
                    "feedback_id": feedback_obj.id,
                    "response": result.result,
                    "needs_update": "update" in result.result.lower() or "change" in result.result.lower()
                },
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "feedback_id": feedback_obj.id,
                    "artifact_id": artifact_id
                }
            )
            
            return updated_result
        
        return result
    
    async def create_documentation_template(
        self, 
        name: str,
        description: str,
        audience: str,
        format: str,
        sections: List[Dict[str, Any]]
    ) -> TaskResult:
        """Create a new documentation template.
        
        Args:
            name: Template name
            description: Template description
            audience: Target audience
            format: Documentation format
            sections: List of sections
            
        Returns:
            TaskResult containing the template
        """
        # Create the template
        template = DocumentationTemplate(
            name=name,
            description=description,
            audience=audience,
            format=format,
            sections=sections
        )
        
        # Store the template
        self.templates[template.id] = template
        
        # Store in shared memory if available
        if self.shared_memory:
            self.shared_memory.store(
                key=f"documentation_template_{template.id}",
                value=template.dict(),
                category="documentation_templates"
            )
        
        logger.info(f"Created documentation template {name} with {len(sections)} sections")
        
        # Return the template
        return TaskResult(
            agent_id=self.state.agent_id,
            agent_name=self.name,
            task_id=f"create_template_{name.lower().replace(' ', '_')}",
            result=template.dict(),
            status=TaskStatus.COMPLETED,
            timestamp=datetime.now().isoformat(),
            execution_time=0.0,
            metadata={"template_id": template.id}
        )
    
    def get_documentation_artifact(self, artifact_id: str) -> Optional[DocumentationArtifact]:
        """Get a specific documentation artifact.
        
        Args:
            artifact_id: ID of the artifact to retrieve
            
        Returns:
            DocumentationArtifact if found, None otherwise
        """
        # Check local storage
        if artifact_id in self.artifacts:
            return self.artifacts[artifact_id]
        
        # Check shared memory if available
        if self.shared_memory:
            artifact_data = self.shared_memory.retrieve(
                key=f"documentation_{artifact_id}",
                category="documentation"
            )
            if artifact_data:
                artifact = DocumentationArtifact(**artifact_data)
                # Cache locally
                self.artifacts[artifact_id] = artifact
                return artifact
        
        return None
    
    def get_documentation_template(self, template_id: str) -> Optional[DocumentationTemplate]:
        """Get a specific documentation template.
        
        Args:
            template_id: ID of the template to retrieve
            
        Returns:
            DocumentationTemplate if found, None otherwise
        """
        # Check local storage
        if template_id in self.templates:
            return self.templates[template_id]
        
        # Check shared memory if available
        if self.shared_memory:
            template_data = self.shared_memory.retrieve(
                key=f"documentation_template_{template_id}",
                category="documentation_templates"
            )
            if template_data:
                template = DocumentationTemplate(**template_data)
                # Cache locally
                self.templates[template_id] = template
                return template
        
        return None
    
    def get_documentation_feedback(self, artifact_id: str) -> List[DocumentationFeedback]:
        """Get feedback for a specific documentation artifact.
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            List of DocumentationFeedback for the artifact
        """
        # Check local storage
        if artifact_id in self.feedback:
            return self.feedback[artifact_id]
        
        # Check shared memory if available
        if self.shared_memory:
            # Try to get all feedback for this artifact
            feedback_list = []
            
            # Get all keys in the documentation_feedback category
            feedback_keys = self.shared_memory.list_keys("documentation_feedback")
            
            # Filter for this artifact
            for key in feedback_keys:
                feedback_data = self.shared_memory.retrieve(key, "documentation_feedback")
                if feedback_data and feedback_data.get("artifact_id") == artifact_id:
                    feedback_list.append(DocumentationFeedback(**feedback_data))
            
            if feedback_list:
                # Cache locally
                self.feedback[artifact_id] = feedback_list
                return feedback_list
        
        return []
    
    def search_documentation(
        self, 
        query: str,
        doc_type: Optional[str] = None,
        audience: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for documentation artifacts matching the query.
        
        Args:
            query: Search query
            doc_type: Optional documentation type filter
            audience: Optional audience filter
            
        Returns:
            List of matching documentation artifacts
        """
        # Simple in-memory search implementation
        query_terms = query.lower().split()
        results = []
        
        for artifact_id, artifact in self.artifacts.items():
            # Apply filters if specified
            if doc_type and artifact.metadata.get("doc_type") != doc_type:
                continue
            
            if audience and artifact.audience != audience:
                continue
            
            # Check if query terms match
            score = 0
            for term in query_terms:
                if term in artifact.title.lower():
                    score += 3
                if term in artifact.description.lower():
                    score += 2
                if term in artifact.content.lower():
                    score += 1
                for tag in artifact.tags:
                    if term in tag.lower():
                        score += 2
            
            if score > 0:
                results.append({
                    "artifact_id": artifact_id,
                    "title": artifact.title,
                    "description": artifact.description,
                    "doc_type": artifact.metadata.get("doc_type", "documentation"),
                    "audience": artifact.audience,
                    "version": artifact.version,
                    "updated_at": artifact.updated_at,
                    "score": score
                })
        
        # Also check shared memory if available
        if self.shared_memory:
            # Get all keys in the documentation category
            doc_keys = self.shared_memory.list_keys("documentation")
            
            for key in doc_keys:
                # Skip if we already have this artifact in the results
                artifact_id = key.replace("documentation_", "")
                if any(result["artifact_id"] == artifact_id for result in results):
                    continue
                
                # Get the artifact
                artifact_data = self.shared_memory.retrieve(key, "documentation")
                if not artifact_data:
                    continue
                
                # Create a temporary artifact
                try:
                    artifact = DocumentationArtifact(**artifact_data)
                except:
                    continue
                
                # Apply filters if specified
                if doc_type and artifact.metadata.get("doc_type") != doc_type:
                    continue
                
                if audience and artifact.audience != audience:
                    continue
                
                # Check if query terms match
                score = 0
                for term in query_terms:
                    if term in artifact.title.lower():
                        score += 3
                    if term in artifact.description.lower():
                        score += 2
                    if term in artifact.content.lower():
                        score += 1
                    for tag in artifact.tags:
                        if term in tag.lower():
                            score += 2
                
                if score > 0:
                    results.append({
                        "artifact_id": artifact_id,
                        "title": artifact.title,
                        "description": artifact.description,
                        "doc_type": artifact.metadata.get("doc_type", "documentation"),
                        "audience": artifact.audience,
                        "version": artifact.version,
                        "updated_at": artifact.updated_at,
                        "score": score
                    })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
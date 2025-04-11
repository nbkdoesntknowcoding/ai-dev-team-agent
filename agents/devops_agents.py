"""
DevOps Agents for the multi-agent development system.

This module contains specialized agents for DevOps tasks, including infrastructure
development, deployment, and security analysis. These agents work together to
ensure reliable, secure, and scalable deployment of applications.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple, cast
import uuid
import os

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


class InfrastructureComponent(BaseModel):
    """Infrastructure component definition."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # e.g., "compute", "storage", "network", "database"
    description: str
    provider: str  # e.g., "aws", "azure", "gcp", "kubernetes"
    specifications: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    security_groups: List[Dict[str, Any]] = Field(default_factory=list)


class InfrastructureDesign(BaseModel):
    """Complete infrastructure design."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    components: List[InfrastructureComponent] = Field(default_factory=list)
    networking: Dict[str, Any] = Field(default_factory=dict)
    security: Dict[str, Any] = Field(default_factory=dict)
    environments: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DeploymentPipeline(BaseModel):
    """Deployment pipeline configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    repository: str
    branches: Dict[str, Any] = Field(default_factory=dict)
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    environments: List[Dict[str, Any]] = Field(default_factory=list)
    triggers: List[Dict[str, Any]] = Field(default_factory=list)
    notifications: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SecurityAssessment(BaseModel):
    """Security assessment for an application or infrastructure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    target_type: str  # "application", "infrastructure", "code"
    target_id: str
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    risk_score: float
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class InfrastructureDeveloper(BaseAgent):
    """Agent specialized in designing and implementing cloud infrastructure."""
    
    def __init__(self, name: str, preferred_cloud: str = "AWS", preferred_iac_tool: str = "terraform", **kwargs):
        # Initialize attributes before calling super().__init__
        self.preferred_cloud = preferred_cloud
        self.preferred_iac_tool = preferred_iac_tool
        
        super().__init__(
            name=name,
            agent_type=AgentRole.INFRASTRUCTURE,
            **kwargs
        )
        """Initialize the Infrastructure Developer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_cloud: Preferred cloud provider (aws, azure, gcp)
            preferred_iac_tool: Preferred IaC tool (terraform, cloudformation, etc.)
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.INFRASTRUCTURE, 
            **kwargs
        )
        self.preferred_cloud = preferred_cloud
        self.preferred_iac_tool = preferred_iac_tool
        
        # Track infrastructure designs
        self.infrastructure_designs: Dict[str, InfrastructureDesign] = {}
        
        # Track implementation status
        self.implementation_status: Dict[str, Dict[str, Any]] = {}
        
        # Track cost estimates
        self.cost_estimates: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Infrastructure Developer Agent initialized with {preferred_cloud} cloud and {preferred_iac_tool} IaC tool")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Infrastructure Developer."""
        return (
            f"You are {self.name}, an Infrastructure Developer specialized in designing "
            f"and implementing cloud infrastructure using {self.preferred_cloud} and "
            f"{self.preferred_iac_tool}. "
            f"Your responsibilities include:\n"
            f"1. Designing scalable and secure infrastructure\n"
            f"2. Implementing infrastructure as code (IaC) using {self.preferred_iac_tool}\n"
            f"3. Optimizing infrastructure for cost and performance\n"
            f"4. Ensuring high availability and fault tolerance\n"
            f"5. Implementing monitoring and logging\n\n"
            f"Always follow best practices for cloud architecture, including security, "
            f"scalability, reliability, performance, and cost optimization. Think carefully "
            f"about resource sizing, network design, security groups, and access controls. "
            f"When writing infrastructure as code, ensure it follows best practices, is well-documented, "
            f"and includes proper error handling and validation."
        )
    
    async def design_infrastructure(
        self, 
        project_name: str,
        description: str,
        requirements: List[Dict[str, Any]],
        environments: List[str] = ["dev", "staging", "prod"]
    ) -> TaskResult:
        """Design a complete cloud infrastructure based on requirements.
        
        Args:
            project_name: Name of the project
            description: Brief description of the project's purpose
            requirements: List of infrastructure requirements
            environments: List of deployment environments
            
        Returns:
            TaskResult containing the infrastructure design
        """
        # Create a task for infrastructure design
        task = Task(
            task_id=f"design_infrastructure_{project_name.lower().replace(' ', '_')}",
            description=f"Design infrastructure for {project_name}",
            agent_type=str(AgentRole.INFRASTRUCTURE),
            requirements={
                "project_name": project_name,
                "description": description,
                "requirements": requirements,
                "environments": environments,
                "cloud_provider": self.preferred_cloud
            },
            context=TaskContext(
                notes=(
                    f"Design a complete cloud infrastructure for {project_name} using {self.preferred_cloud}. "
                    f"The infrastructure should meet all requirements, follow best practices, "
                    f"and be deployable to {', '.join(environments)} environments. "
                    f"Consider security, scalability, reliability, and cost optimization."
                )
            ),
            expected_output=(
                "A comprehensive infrastructure design including all necessary components, "
                "networking, security, and environment configurations."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the infrastructure design
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the infrastructure design from the result
                infra_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    infra_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            infra_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not infra_data:
                    logger.warning(f"Could not parse infrastructure design as JSON. Attempting to extract from text.")
                    infra_data = self._extract_infrastructure_from_text(result.result, project_name, description)
                
                # Create infrastructure components
                components = []
                for component_data in infra_data.get("components", []):
                    component = InfrastructureComponent(
                        name=component_data.get("name", "component"),
                        type=component_data.get("type", "compute"),
                        description=component_data.get("description", ""),
                        provider=component_data.get("provider", self.preferred_cloud),
                        specifications=component_data.get("specifications", {}),
                        dependencies=component_data.get("dependencies", []),
                        configuration=component_data.get("configuration", {}),
                        security_groups=component_data.get("security_groups", [])
                    )
                    components.append(component)
                
                # Create the infrastructure design
                infra_design = InfrastructureDesign(
                    name=project_name,
                    description=description,
                    components=components,
                    networking=infra_data.get("networking", {}),
                    security=infra_data.get("security", {}),
                    environments=[
                        {
                            "name": env,
                            "configuration": infra_data.get("environments", {}).get(env, {})
                        }
                        for env in environments
                    ]
                )
                
                # Store the infrastructure design
                self.infrastructure_designs[infra_design.id] = infra_design
                
                # Initialize implementation status
                self.implementation_status[infra_design.id] = {
                    "status": "designed",
                    "implemented_components": [],
                    "pending_components": [component.id for component in components],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"infrastructure_{infra_design.id}",
                        value=infra_design.dict(),
                        category="infrastructure_designs"
                    )
                
                logger.info(f"Created infrastructure design for '{project_name}' with {len(components)} components")
                
                # Return the infrastructure design as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=infra_design.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"infrastructure_id": infra_design.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing infrastructure design: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_infrastructure_from_text(
        self, 
        text: str, 
        project_name: str, 
        description: str
    ) -> Dict[str, Any]:
        """Extract structured infrastructure data from unstructured text.
        
        Args:
            text: The text to extract from
            project_name: The name of the project
            description: The description of the project
            
        Returns:
            Structured infrastructure data
        """
        infra_data = {
            "name": project_name,
            "description": description,
            "components": [],
            "networking": {},
            "security": {},
            "environments": {}
        }
        
        # Extract components
        component_sections = re.findall(
            r'(?i)#+\s*(?:Component|Resource|Service):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+\s*(?:Component|Resource|Service|Networking|Security|Environment)|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in component_sections:
            component = {
                "name": title.strip(),
                "type": "compute",  # Default type
                "description": "",
                "provider": self.preferred_cloud,
                "specifications": {},
                "dependencies": [],
                "configuration": {},
                "security_groups": []
            }
            
            # Extract component type
            type_match = re.search(r'(?i)(?:Type|Service Type|Resource Type):\s*([^\n]+)', content)
            if type_match:
                component["type"] = type_match.group(1).strip().lower()
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Purpose):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if desc_match:
                component["description"] = desc_match.group(1).strip()
            
            # Extract provider
            provider_match = re.search(r'(?i)(?:Provider|Cloud Provider|Platform):\s*([^\n]+)', content)
            if provider_match:
                component["provider"] = provider_match.group(1).strip().lower()
            
            # Extract specifications
            specs_match = re.search(
                r'(?i)(?:Specifications|Specs|Configuration):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if specs_match:
                specs_text = specs_match.group(1)
                
                # Extract key-value pairs from bullet points
                specs_items = re.findall(r'[-*]\s*([^:]+):\s*([^\n]+)', specs_text)
                for key, value in specs_items:
                    component["specifications"][key.strip()] = value.strip()
            
            # Extract dependencies
            deps_match = re.search(
                r'(?i)(?:Dependencies|Depends On):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if deps_match:
                deps_text = deps_match.group(1)
                
                # Extract dependencies from bullet points
                deps_items = re.findall(r'[-*]\s*([^\n]+)', deps_text)
                component["dependencies"] = [dep.strip() for dep in deps_items]
            
            # Extract configuration
            config_match = re.search(
                r'(?i)(?:Detailed Configuration|Settings):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if config_match:
                config_text = config_match.group(1)
                
                # Extract key-value pairs from bullet points
                config_items = re.findall(r'[-*]\s*([^:]+):\s*([^\n]+)', config_text)
                for key, value in config_items:
                    component["configuration"][key.strip()] = value.strip()
            
            # Extract security groups
            security_match = re.search(
                r'(?i)(?:Security Groups|Access Control):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if security_match:
                security_text = security_match.group(1)
                
                # Extract security groups from bullet points
                security_items = re.findall(r'[-*]\s*([^\n]+)', security_text)
                for security_item in security_items:
                    # Try to parse into a structured format
                    sec_match = re.search(r'([^:]+)(?::\s*(.+))?', security_item)
                    if sec_match:
                        name = sec_match.group(1).strip()
                        rules = sec_match.group(2).strip() if sec_match.group(2) else ""
                        
                        component["security_groups"].append({
                            "name": name,
                            "rules": rules
                        })
                    else:
                        # Just add the raw text
                        component["security_groups"].append({
                            "description": security_item.strip()
                        })
            
            infra_data["components"].append(component)
        
        # Extract networking
        networking_section = re.search(
            r'(?i)#+\s*(?:Networking|Network Configuration)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if networking_section and networking_section.group(1):
            networking_text = networking_section.group(1)
            
            # Extract VPC configuration
            vpc_match = re.search(r'(?i)(?:VPC|Virtual Network):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', networking_text)
            if vpc_match:
                infra_data["networking"]["vpc"] = vpc_match.group(1).strip()
            
            # Extract subnets
            subnets_match = re.search(
                r'(?i)(?:Subnets|Subnet Configuration):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                networking_text
            )
            if subnets_match:
                subnets_text = subnets_match.group(1)
                
                # Extract subnets from bullet points
                subnet_items = re.findall(r'[-*]\s*([^\n]+)', subnets_text)
                infra_data["networking"]["subnets"] = []
                
                for subnet_item in subnet_items:
                    # Try to parse into a structured format
                    subnet_match = re.search(r'([^:]+)(?::\s*(.+))?', subnet_item)
                    if subnet_match:
                        name = subnet_match.group(1).strip()
                        cidr = subnet_match.group(2).strip() if subnet_match.group(2) else ""
                        
                        infra_data["networking"]["subnets"].append({
                            "name": name,
                            "cidr": cidr
                        })
                    else:
                        # Just add the raw text
                        infra_data["networking"]["subnets"].append({
                            "description": subnet_item.strip()
                        })
            
            # Extract routing configuration
            routing_match = re.search(
                r'(?i)(?:Routing|Routes|Route Tables):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                networking_text
            )
            if routing_match:
                routing_text = routing_match.group(1)
                
                # Extract routes from bullet points
                route_items = re.findall(r'[-*]\s*([^\n]+)', routing_text)
                infra_data["networking"]["routing"] = [route.strip() for route in route_items]
            
            # Extract load balancing configuration
            lb_match = re.search(
                r'(?i)(?:Load Balancing|Load Balancers):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                networking_text
            )
            if lb_match:
                lb_text = lb_match.group(1)
                
                # Extract load balancers from bullet points
                lb_items = re.findall(r'[-*]\s*([^\n]+)', lb_text)
                infra_data["networking"]["load_balancers"] = []
                
                for lb_item in lb_items:
                    # Try to parse into a structured format
                    lb_match = re.search(r'([^:]+)(?::\s*(.+))?', lb_item)
                    if lb_match:
                        name = lb_match.group(1).strip()
                        config = lb_match.group(2).strip() if lb_match.group(2) else ""
                        
                        infra_data["networking"]["load_balancers"].append({
                            "name": name,
                            "configuration": config
                        })
                    else:
                        # Just add the raw text
                        infra_data["networking"]["load_balancers"].append({
                            "description": lb_item.strip()
                        })
        
        # Extract security
        security_section = re.search(
            r'(?i)#+\s*(?:Security|Security Configuration)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if security_section and security_section.group(1):
            security_text = security_section.group(1)
            
            # Extract IAM configuration
            iam_match = re.search(r'(?i)(?:IAM|Identity and Access Management):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', security_text)
            if iam_match:
                infra_data["security"]["iam"] = iam_match.group(1).strip()
            
            # Extract security groups
            sg_match = re.search(
                r'(?i)(?:Security Groups|Firewall Rules):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                security_text
            )
            if sg_match:
                sg_text = sg_match.group(1)
                
                # Extract security groups from bullet points
                sg_items = re.findall(r'[-*]\s*([^\n]+)', sg_text)
                infra_data["security"]["security_groups"] = []
                
                for sg_item in sg_items:
                    # Try to parse into a structured format
                    sg_match = re.search(r'([^:]+)(?::\s*(.+))?', sg_item)
                    if sg_match:
                        name = sg_match.group(1).strip()
                        rules = sg_match.group(2).strip() if sg_match.group(2) else ""
                        
                        infra_data["security"]["security_groups"].append({
                            "name": name,
                            "rules": rules
                        })
                    else:
                        # Just add the raw text
                        infra_data["security"]["security_groups"].append({
                            "description": sg_item.strip()
                        })
            
            # Extract encryption configuration
            encryption_match = re.search(r'(?i)(?:Encryption|Data Protection):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', security_text)
            if encryption_match:
                infra_data["security"]["encryption"] = encryption_match.group(1).strip()
        
        # Extract environments
        environment_sections = re.findall(
            r'(?i)#+\s*(?:Environment|Deployment Environment):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+\s*(?:Environment|Deployment Environment)|\n#+|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in environment_sections:
            env_name = title.strip().lower()
            
            if env_name not in infra_data["environments"]:
                infra_data["environments"][env_name] = {}
            
            # Extract environment configuration
            config_match = re.search(r'(?i)(?:Configuration|Settings):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if config_match:
                infra_data["environments"][env_name]["configuration"] = config_match.group(1).strip()
            
            # Extract instance counts or sizes
            sizing_match = re.search(
                r'(?i)(?:Sizing|Scaling|Instance Count):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if sizing_match:
                sizing_text = sizing_match.group(1)
                
                # Extract sizing configuration from bullet points
                sizing_items = re.findall(r'[-*]\s*([^:]+):\s*([^\n]+)', sizing_text)
                
                if "sizing" not in infra_data["environments"][env_name]:
                    infra_data["environments"][env_name]["sizing"] = {}
                
                for key, value in sizing_items:
                    infra_data["environments"][env_name]["sizing"][key.strip()] = value.strip()
        
        return infra_data
    
    async def implement_infrastructure(
        self, 
        infrastructure_id: str,
        target_environment: str = "dev",
        implementation_details: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Implement infrastructure as code using the preferred IaC tool.
        
        Args:
            infrastructure_id: ID of the infrastructure design
            target_environment: Target environment to implement
            implementation_details: Optional additional implementation details
            
        Returns:
            TaskResult containing the infrastructure code
        """
        # Check if infrastructure design exists
        if infrastructure_id not in self.infrastructure_designs:
            # Try to load from shared memory if available
            if self.shared_memory:
                infra_data = self.shared_memory.retrieve(
                    key=f"infrastructure_{infrastructure_id}",
                    category="infrastructure_designs"
                )
                if infra_data:
                    self.infrastructure_designs[infrastructure_id] = InfrastructureDesign(**infra_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_infrastructure_{infrastructure_id}_{target_environment}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Infrastructure design with ID {infrastructure_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_infrastructure_{infrastructure_id}_{target_environment}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Infrastructure design with ID {infrastructure_id} not found"
                )
        
        # Get the infrastructure design
        infra_design = self.infrastructure_designs[infrastructure_id]
        
        # Find the target environment configuration
        target_env_config = None
        for env in infra_design.environments:
            if env["name"] == target_environment:
                target_env_config = env
                break
        
        if not target_env_config:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"implement_infrastructure_{infrastructure_id}_{target_environment}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Target environment '{target_environment}' not found in infrastructure design"
            )
        
        # Create a task for implementing infrastructure as code
        task = Task(
            task_id=f"implement_infrastructure_{infrastructure_id}_{target_environment}",
            description=f"Implement infrastructure as code for {infra_design.name} ({target_environment})",
            agent_type=str(AgentRole.INFRASTRUCTURE),
            requirements={
                "infrastructure_id": infrastructure_id,
                "infrastructure_name": infra_design.name,
                "components": [component.dict() for component in infra_design.components],
                "networking": infra_design.networking,
                "security": infra_design.security,
                "target_environment": target_environment,
                "environment_config": target_env_config,
                "cloud_provider": self.preferred_cloud,
                "iac_tool": self.preferred_iac_tool,
                "implementation_details": implementation_details or {}
            },
            context=TaskContext(
                notes=(
                    f"Implement the infrastructure for {infra_design.name} using {self.preferred_iac_tool} "
                    f"targeting the {target_environment} environment on {self.preferred_cloud}. "
                    f"The implementation should include all components, networking, and security "
                    f"configurations defined in the infrastructure design."
                )
            ),
            expected_output=(
                f"Complete infrastructure as code using {self.preferred_iac_tool}, "
                f"organized into logical modules with proper variables, outputs, and documentation."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Update implementation status
            if infrastructure_id in self.implementation_status:
                status = self.implementation_status[infrastructure_id]
                
                # Mark all components for this environment as implemented
                for component_id in status["pending_components"]:
                    # Find the component
                    for component in infra_design.components:
                        if component.id == component_id:
                            # Add to implemented list if not already there
                            if component_id not in status["implemented_components"]:
                                status["implemented_components"].append(component_id)
                            # Remove from pending
                            if component_id in status["pending_components"]:
                                status["pending_components"].remove(component_id)
                            break
                
                # Update status
                status["status"] = "partial" if status["pending_components"] else "completed"
                status[f"environment_{target_environment}"] = "implemented"
                status["timestamp"] = datetime.now().isoformat()
                
                # Update status
                self.implementation_status[infrastructure_id] = status
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"infrastructure_code_{infrastructure_id}_{target_environment}",
                    value={
                        "infrastructure_id": infrastructure_id,
                        "infrastructure_name": infra_design.name,
                        "target_environment": target_environment,
                        "cloud_provider": self.preferred_cloud,
                        "iac_tool": self.preferred_iac_tool,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="infrastructure_code"
                )
                
                # Update implementation status
                self.shared_memory.store(
                    key=f"infrastructure_implementation_status_{infrastructure_id}",
                    value=self.implementation_status[infrastructure_id],
                    category="infrastructure_implementation_status"
                )
            
            logger.info(f"Implemented infrastructure for {infra_design.name} ({target_environment}) using {self.preferred_iac_tool}")
            
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
                   "infrastructure_id": infrastructure_id,
                    "infrastructure_name": infra_design.name,
                    "target_environment": target_environment,
                    "iac_tool": self.preferred_iac_tool
                }
            )
            
            return updated_result
        
        return result
    
    async def estimate_infrastructure_cost(
        self, 
        infrastructure_id: str,
        target_environment: str = "prod",
        usage_pattern: str = "moderate"
    ) -> TaskResult:
        """Estimate the cost of deploying and running the infrastructure.
        
        Args:
            infrastructure_id: ID of the infrastructure design
            target_environment: Target environment for cost estimation
            usage_pattern: Expected usage pattern (low, moderate, high)
            
        Returns:
            TaskResult containing the cost estimate
        """
        # Check if infrastructure design exists
        if infrastructure_id not in self.infrastructure_designs:
            # Try to load from shared memory if available
            if self.shared_memory:
                infra_data = self.shared_memory.retrieve(
                    key=f"infrastructure_{infrastructure_id}",
                    category="infrastructure_designs"
                )
                if infra_data:
                    self.infrastructure_designs[infrastructure_id] = InfrastructureDesign(**infra_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"estimate_cost_{infrastructure_id}_{target_environment}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Infrastructure design with ID {infrastructure_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"estimate_cost_{infrastructure_id}_{target_environment}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Infrastructure design with ID {infrastructure_id} not found"
                )
        
        # Get the infrastructure design
        infra_design = self.infrastructure_designs[infrastructure_id]
        
        # Find the target environment configuration
        target_env_config = None
        for env in infra_design.environments:
            if env["name"] == target_environment:
                target_env_config = env
                break
        
        if not target_env_config:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"estimate_cost_{infrastructure_id}_{target_environment}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Target environment '{target_environment}' not found in infrastructure design"
            )
        
        # Create a task for estimating infrastructure cost
        task = Task(
            task_id=f"estimate_cost_{infrastructure_id}_{target_environment}",
            description=f"Estimate cost for {infra_design.name} infrastructure ({target_environment})",
            agent_type=str(AgentRole.INFRASTRUCTURE),
            requirements={
                "infrastructure_id": infrastructure_id,
                "infrastructure_name": infra_design.name,
                "components": [component.dict() for component in infra_design.components],
                "target_environment": target_environment,
                "environment_config": target_env_config,
                "usage_pattern": usage_pattern,
                "cloud_provider": self.preferred_cloud
            },
            context=TaskContext(
                notes=(
                    f"Estimate the monthly cost of deploying and running the {infra_design.name} "
                    f"infrastructure in the {target_environment} environment on {self.preferred_cloud} "
                    f"with a {usage_pattern} usage pattern. Provide a breakdown of costs by component "
                    f"and service type."
                )
            ),
            expected_output=(
                "A comprehensive cost estimate including monthly costs, cost by component, "
                "cost by service type, and cost optimization recommendations."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the cost estimate
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the cost estimate from the result
                cost_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    cost_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            cost_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not cost_data:
                    cost_data = self._extract_cost_estimate_from_text(result.result, infra_design.name, target_environment)
                
                # Store the cost estimate
                self.cost_estimates[f"{infrastructure_id}_{target_environment}"] = {
                    "infrastructure_id": infrastructure_id,
                    "infrastructure_name": infra_design.name,
                    "target_environment": target_environment,
                    "usage_pattern": usage_pattern,
                    "estimate": cost_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"cost_estimate_{infrastructure_id}_{target_environment}",
                        value=self.cost_estimates[f"{infrastructure_id}_{target_environment}"],
                        category="cost_estimates"
                    )
                
                logger.info(f"Created cost estimate for {infra_design.name} ({target_environment}) with {usage_pattern} usage")
                
                # Return the cost estimate
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=cost_data,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "infrastructure_id": infrastructure_id,
                        "infrastructure_name": infra_design.name,
                        "target_environment": target_environment,
                        "usage_pattern": usage_pattern
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing cost estimate: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_cost_estimate_from_text(
        self, 
        text: str, 
        infrastructure_name: str, 
        target_environment: str
    ) -> Dict[str, Any]:
        """Extract structured cost estimate data from unstructured text.
        
        Args:
            text: The text to extract from
            infrastructure_name: The name of the infrastructure
            target_environment: The target environment
            
        Returns:
            Structured cost estimate data
        """
        cost_data = {
            "infrastructure_name": infrastructure_name,
            "environment": target_environment,
            "total_monthly_cost": None,
            "cost_by_component": {},
            "cost_by_service": {},
            "optimization_recommendations": []
        }
        
        # Extract total monthly cost
        total_cost_match = re.search(r'(?i)(?:Total Monthly Cost|Total Cost|Monthly Cost):\s*\$?([\d,.]+)', text)
        if total_cost_match:
            try:
                cost_data["total_monthly_cost"] = float(total_cost_match.group(1).replace(',', ''))
            except ValueError:
                # If we can't parse as float, just use the string
                cost_data["total_monthly_cost"] = total_cost_match.group(1)
        
        # Extract cost by component
        component_cost_section = re.search(
            r'(?i)(?:Cost by Component|Component Costs|Costs by Component)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if component_cost_section and component_cost_section.group(1):
            component_costs_text = component_cost_section.group(1)
            
            # Try to extract from a table format first
            table_matches = re.findall(r'\|\s*([^|]+)\s*\|\s*\$?([\d,.]+)\s*\|', component_costs_text)
            if table_matches:
                for name, cost in table_matches:
                    try:
                        cost_data["cost_by_component"][name.strip()] = float(cost.replace(',', ''))
                    except ValueError:
                        cost_data["cost_by_component"][name.strip()] = cost
            else:
                # Try to extract from bullet points
                bullet_matches = re.findall(r'[-*]\s*([^:]+):\s*\$?([\d,.]+)', component_costs_text)
                for name, cost in bullet_matches:
                    try:
                        cost_data["cost_by_component"][name.strip()] = float(cost.replace(',', ''))
                    except ValueError:
                        cost_data["cost_by_component"][name.strip()] = cost
        
        # Extract cost by service
        service_cost_section = re.search(
            r'(?i)(?:Cost by Service|Service Costs|Costs by Service Type)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if service_cost_section and service_cost_section.group(1):
            service_costs_text = service_cost_section.group(1)
            
            # Try to extract from a table format first
            table_matches = re.findall(r'\|\s*([^|]+)\s*\|\s*\$?([\d,.]+)\s*\|', service_costs_text)
            if table_matches:
                for name, cost in table_matches:
                    try:
                        cost_data["cost_by_service"][name.strip()] = float(cost.replace(',', ''))
                    except ValueError:
                        cost_data["cost_by_service"][name.strip()] = cost
            else:
                # Try to extract from bullet points
                bullet_matches = re.findall(r'[-*]\s*([^:]+):\s*\$?([\d,.]+)', service_costs_text)
                for name, cost in bullet_matches:
                    try:
                        cost_data["cost_by_service"][name.strip()] = float(cost.replace(',', ''))
                    except ValueError:
                        cost_data["cost_by_service"][name.strip()] = cost
        
        # Extract optimization recommendations
        optimization_section = re.search(
            r'(?i)(?:Cost Optimization|Optimization Recommendations|Recommendations)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if optimization_section and optimization_section.group(1):
            optimization_text = optimization_section.group(1)
            
            # Extract recommendations from bullet points
            bullet_matches = re.findall(r'[-*]\s*([^\n]+)', optimization_text)
            cost_data["optimization_recommendations"] = [rec.strip() for rec in bullet_matches]
        
        return cost_data
    
    def get_infrastructure_design(self, infrastructure_id: str) -> Optional[InfrastructureDesign]:
        """Get a specific infrastructure design.
        
        Args:
            infrastructure_id: ID of the infrastructure design to retrieve
            
        Returns:
            InfrastructureDesign if found, None otherwise
        """
        # Check local storage
        if infrastructure_id in self.infrastructure_designs:
            return self.infrastructure_designs[infrastructure_id]
        
        # Check shared memory if available
        if self.shared_memory:
            infra_data = self.shared_memory.retrieve(
                key=f"infrastructure_{infrastructure_id}",
                category="infrastructure_designs"
            )
            if infra_data:
                infrastructure = InfrastructureDesign(**infra_data)
                # Cache locally
                self.infrastructure_designs[infrastructure_id] = infrastructure
                return infrastructure
        
        return None
    
    def get_implementation_status(self, infrastructure_id: str) -> Optional[Dict[str, Any]]:
        """Get the implementation status for an infrastructure design.
        
        Args:
            infrastructure_id: ID of the infrastructure design
            
        Returns:
            Implementation status if found, None otherwise
        """
        # Check local storage
        if infrastructure_id in self.implementation_status:
            return self.implementation_status[infrastructure_id]
        
        # Check shared memory if available
        if self.shared_memory:
            status_data = self.shared_memory.retrieve(
                key=f"infrastructure_implementation_status_{infrastructure_id}",
                category="infrastructure_implementation_status"
            )
            if status_data:
                # Cache locally
                self.implementation_status[infrastructure_id] = status_data
                return status_data
        
        return None


class DeploymentSpecialist(BaseAgent):
    """Agent specialized in deployment processes and CI/CD pipelines."""
    
    def __init__(
        self, 
        name: str,
        preferred_ci_tool: str = "GitHub Actions",
        preferred_deployment_strategy: str = "Blue-Green",
        **kwargs
    ):
        """Initialize the Deployment Specialist agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_ci_tool: Preferred CI/CD tool (github_actions, jenkins, etc.)
            preferred_deployment_strategy: Preferred deployment strategy
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.DEPLOYMENT, 
            **kwargs
        )
        self.preferred_ci_tool = preferred_ci_tool
        self.preferred_deployment_strategy = preferred_deployment_strategy
        
        # Track deployment pipelines
        self.deployment_pipelines: Dict[str, DeploymentPipeline] = {}
        
        # Track deployment plans
        self.deployment_plans: Dict[str, Dict[str, Any]] = {}
        
        # Track deployment history
        self.deployment_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"Deployment Specialist Agent initialized with {self.preferred_ci_tool} CI tool and {self.preferred_deployment_strategy} strategy")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Deployment Specialist."""
        return (
            f"You are {self.name}, a Deployment Specialist focused on automating "
            f"deployment processes and implementing CI/CD pipelines, primarily using "
            f"{self.preferred_ci_tool} and {self.preferred_deployment_strategy} deployments. "
            f"Your responsibilities include:\n"
            f"1. Designing and implementing CI/CD pipelines\n"
            f"2. Creating deployment strategies and plans\n"
            f"3. Automating build, test, and deployment processes\n"
            f"4. Ensuring reliable and consistent deployments\n"
            f"5. Managing environment configurations\n\n"
            f"Always follow best practices for CI/CD, including automation, early feedback, "
            f"consistency, rollback mechanisms, and proper environment separation. Think carefully "
            f"about pipeline stages, testing strategies, and deployment verification. When writing "
            f"pipeline configurations, ensure they are well-documented, secure, and maintainable."
        )
    
    async def design_ci_cd_pipeline(
        self, 
        project_name: str,
        repository_url: str,
        project_type: str,
        environments: List[str] = ["dev", "staging", "prod"],
        testing_requirements: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Design a complete CI/CD pipeline for a project.
        
        Args:
            project_name: Name of the project
            repository_url: URL of the source code repository
            project_type: Type of project (frontend, backend, fullstack, etc.)
            environments: List of deployment environments
            testing_requirements: Optional testing requirements
            
        Returns:
            TaskResult containing the CI/CD pipeline design
        """
        # Create a task for designing the CI/CD pipeline
        task = Task(
            task_id=f"design_pipeline_{project_name.lower().replace(' ', '_')}",
            description=f"Design CI/CD pipeline for {project_name}",
            agent_type=str(AgentRole.DEPLOYMENT),
            requirements={
                "project_name": project_name,
                "repository_url": repository_url,
                "project_type": project_type,
                "environments": environments,
                "testing_requirements": testing_requirements or {},
                "ci_tool": self.preferred_ci_tool,
                "deployment_strategy": self.preferred_deployment_strategy
            },
            context=TaskContext(
                notes=(
                    f"Design a complete CI/CD pipeline for {project_name} using {self.preferred_ci_tool}. "
                    f"The pipeline should include stages for building, testing, and deploying the application "
                    f"to {', '.join(environments)} environments using a {self.preferred_deployment_strategy} "
                    f"deployment strategy. Consider the specific requirements of a {project_type} project."
                )
            ),
            expected_output=(
                "A comprehensive CI/CD pipeline design including stages, triggers, "
                "environments, testing strategies, and deployment approaches."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the pipeline design
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the pipeline design from the result
                pipeline_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    pipeline_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            pipeline_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not pipeline_data:
                    logger.warning(f"Could not parse pipeline design as JSON. Attempting to extract from text.")
                    pipeline_data = self._extract_pipeline_from_text(result.result, project_name, repository_url, project_type)
                
                # Create the deployment pipeline
                pipeline = DeploymentPipeline(
                    name=project_name,
                    description=f"CI/CD pipeline for {project_name} ({project_type})",
                    repository=repository_url,
                    branches=pipeline_data.get("branches", {}),
                    stages=pipeline_data.get("stages", []),
                    environments=[{"name": env, "configuration": pipeline_data.get("environments", {}).get(env, {})} for env in environments],
                    triggers=pipeline_data.get("triggers", []),
                    notifications=pipeline_data.get("notifications", [])
                )
                
                # Store the pipeline
                self.deployment_pipelines[pipeline.id] = pipeline
                
                # Initialize deployment history
                self.deployment_history[pipeline.id] = []
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"pipeline_{pipeline.id}",
                        value=pipeline.dict(),
                        category="deployment_pipelines"
                    )
                
                logger.info(f"Created CI/CD pipeline design for '{project_name}' with {len(pipeline.stages)} stages")
                
                # Return the pipeline design as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=pipeline.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"pipeline_id": pipeline.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing CI/CD pipeline design: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_pipeline_from_text(
        self, 
        text: str, 
        project_name: str, 
        repository_url: str,
        project_type: str
    ) -> Dict[str, Any]:
        """Extract structured CI/CD pipeline data from unstructured text.
        
        Args:
            text: The text to extract from
            project_name: The name of the project
            repository_url: The URL of the source code repository
            project_type: The type of project
            
        Returns:
            Structured pipeline data
        """
        pipeline_data = {
            "name": project_name,
            "repository": repository_url,
            "project_type": project_type,
            "branches": {},
            "stages": [],
            "environments": {},
            "triggers": [],
            "notifications": []
        }
        
        # Extract branches configuration
        branches_section = re.search(
            r'(?i)#+\s*(?:Branches|Branch Configuration|Git Branches)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if branches_section and branches_section.group(1):
            branches_text = branches_section.group(1)
            
            # Extract main branch
            main_branch_match = re.search(r'(?i)(?:Main Branch|Primary Branch|Production Branch):\s*([^\n]+)', branches_text)
            if main_branch_match:
                pipeline_data["branches"]["main"] = main_branch_match.group(1).strip()
            
            # Extract development branch
            dev_branch_match = re.search(r'(?i)(?:Development Branch|Dev Branch):\s*([^\n]+)', branches_text)
            if dev_branch_match:
                pipeline_data["branches"]["development"] = dev_branch_match.group(1).strip()
            
            # Extract feature branches pattern
            feature_branch_match = re.search(r'(?i)(?:Feature Branches|Feature Branch Pattern):\s*([^\n]+)', branches_text)
            if feature_branch_match:
                pipeline_data["branches"]["feature"] = feature_branch_match.group(1).strip()
            
            # Extract release branches pattern
            release_branch_match = re.search(r'(?i)(?:Release Branches|Release Branch Pattern):\s*([^\n]+)', branches_text)
            if release_branch_match:
                pipeline_data["branches"]["release"] = release_branch_match.group(1).strip()
        
        # Extract pipeline stages
        stages_section = re.search(
            r'(?i)#+\s*(?:Stages|Pipeline Stages|CI/CD Stages)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if stages_section and stages_section.group(1):
            stages_text = stages_section.group(1)
            
            # Look for individual stages
            stage_sections = re.findall(
                r'(?i)(?:#{1,3}|[-*])\s*([^:\n]+)(?::|Stage)(?:\n+(.+?))?(?=\n+(?:#{1,3}|[-*])\s*[^:\n]+(?::|Stage)|\n#+|\Z)',
                stages_text,
                re.DOTALL
            )
            
            for title, content in stage_sections:
                stage = {
                    "name": title.strip(),
                    "steps": [],
                    "environment": None,
                    "conditions": []
                }
                
                # Extract steps
                steps_match = re.search(
                    r'(?i)(?:Steps|Commands|Actions):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                    content
                )
                if steps_match:
                    steps_text = steps_match.group(1)
                    
                    # Extract steps from bullet points
                    step_items = re.findall(r'[-*]\s*([^\n]+)', steps_text)
                    stage["steps"] = [step.strip() for step in step_items]
                
                # Extract environment
                env_match = re.search(r'(?i)(?:Environment|Runs On):\s*([^\n]+)', content)
                if env_match:
                    stage["environment"] = env_match.group(1).strip()
                
                # Extract conditions
                conditions_match = re.search(
                    r'(?i)(?:Conditions|Criteria|When):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                    content
                )
                if conditions_match:
                    conditions_text = conditions_match.group(1)
                    
                    # Extract conditions from bullet points
                    condition_items = re.findall(r'[-*]\s*([^\n]+)', conditions_text)
                    stage["conditions"] = [condition.strip() for condition in condition_items]
                
                pipeline_data["stages"].append(stage)
        
        # Extract environments
        environment_sections = re.findall(
            r'(?i)#+\s*(?:Environment|Deployment Environment):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+\s*(?:Environment|Deployment Environment)|\n#+|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in environment_sections:
            env_name = title.strip().lower()
            
            if env_name not in pipeline_data["environments"]:
                pipeline_data["environments"][env_name] = {}
            
            # Extract environment configuration
            config_match = re.search(r'(?i)(?:Configuration|Settings):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if config_match:
                pipeline_data["environments"][env_name]["configuration"] = config_match.group(1).strip()
            
            # Extract deployment targets
            targets_match = re.search(
                r'(?i)(?:Deployment Targets|Targets):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if targets_match:
                targets_text = targets_match.group(1)
                
                # Extract targets from bullet points
                target_items = re.findall(r'[-*]\s*([^\n]+)', targets_text)
                pipeline_data["environments"][env_name]["targets"] = [target.strip() for target in target_items]
            
            # Extract approval requirements
            approval_match = re.search(r'(?i)(?:Approval|Approvals Required):\s*([^\n]+)', content)
            if approval_match:
                pipeline_data["environments"][env_name]["approval_required"] = "yes" in approval_match.group(1).lower() or "true" in approval_match.group(1).lower()
        
        # Extract triggers
        triggers_section = re.search(
            r'(?i)#+\s*(?:Triggers|Pipeline Triggers)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if triggers_section and triggers_section.group(1):
            triggers_text = triggers_section.group(1)
            
            # Extract triggers from bullet points
            trigger_items = re.findall(r'[-*]\s*([^\n]+)', triggers_text)
            
            for trigger_text in trigger_items:
                # Try to parse into a structured format
                trigger_match = re.search(r'([^:]+)(?::\s*(.+))?', trigger_text)
                if trigger_match:
                    trigger_type = trigger_match.group(1).strip()
                    trigger_config = trigger_match.group(2).strip() if trigger_match.group(2) else ""
                    
                    pipeline_data["triggers"].append({
                        "type": trigger_type,
                        "configuration": trigger_config
                    })
                else:
                    # Just add the raw text
                    pipeline_data["triggers"].append({
                        "description": trigger_text.strip()
                    })
        
        # Extract notifications
        notifications_section = re.search(
            r'(?i)#+\s*(?:Notifications|Alerts)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if notifications_section and notifications_section.group(1):
            notifications_text = notifications_section.group(1)
            
            # Extract notifications from bullet points
            notification_items = re.findall(r'[-*]\s*([^\n]+)', notifications_text)
            
            for notification_text in notification_items:
                # Try to parse into a structured format
                notification_match = re.search(r'([^:]+)(?::\s*(.+))?', notification_text)
                if notification_match:
                    notification_type = notification_match.group(1).strip()
                    notification_config = notification_match.group(2).strip() if notification_match.group(2) else ""
                    
                    pipeline_data["notifications"].append({
                        "type": notification_type,
                        "configuration": notification_config
                    })
                else:
                    # Just add the raw text
                    pipeline_data["notifications"].append({
                        "description": notification_text.strip()
                    })
        
        return pipeline_data
    
    async def implement_ci_cd_pipeline(
        self, 
        pipeline_id: str,
        implementation_details: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Implement a CI/CD pipeline using the preferred CI tool.
        
        Args:
            pipeline_id: ID of the deployment pipeline
            implementation_details: Optional additional implementation details
            
        Returns:
            TaskResult containing the CI/CD pipeline implementation
        """
        # Check if pipeline exists
        if pipeline_id not in self.deployment_pipelines:
            # Try to load from shared memory if available
            if self.shared_memory:
                pipeline_data = self.shared_memory.retrieve(
                    key=f"pipeline_{pipeline_id}",
                    category="deployment_pipelines"
                )
                if pipeline_data:
                    self.deployment_pipelines[pipeline_id] = DeploymentPipeline(**pipeline_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_pipeline_{pipeline_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Deployment pipeline with ID {pipeline_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_pipeline_{pipeline_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Deployment pipeline with ID {pipeline_id} not found"
                )
        
        # Get the pipeline
        pipeline = self.deployment_pipelines[pipeline_id]
        
        # Create a task for implementing the CI/CD pipeline
        task = Task(
            task_id=f"implement_pipeline_{pipeline_id}",
            description=f"Implement CI/CD pipeline for {pipeline.name}",
            agent_type=str(AgentRole.DEPLOYMENT),
            requirements={
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline.name,
                "repository": pipeline.repository,
                "branches": pipeline.branches,
                "stages": pipeline.stages,
                "environments": pipeline.environments,
                "triggers": pipeline.triggers,
                "ci_tool": self.preferred_ci_tool,
                "implementation_details": implementation_details or {}
            },
            context=TaskContext(
                notes=(
                    f"Implement a CI/CD pipeline for {pipeline.name} using {self.preferred_ci_tool}. "
                    f"The implementation should include all stages, triggers, and environment configurations "
                    f"defined in the pipeline design. Create the necessary configuration files "
                    f"and scripts for the CI/CD tool."
                )
            ),
            expected_output=(
                f"Complete {self.preferred_ci_tool} configuration files and scripts "
                f"that implement the CI/CD pipeline according to the design."
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
                    key=f"pipeline_implementation_{pipeline_id}",
                    value={
                        "pipeline_id": pipeline_id,
                        "pipeline_name": pipeline.name,
                        "ci_tool": self.preferred_ci_tool,
                        "implementation": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="pipeline_implementations"
                )
            
            logger.info(f"Implemented CI/CD pipeline for {pipeline.name} using {self.preferred_ci_tool}")
            
            # Return the implementation
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
                    "pipeline_id": pipeline_id,
                    "pipeline_name": pipeline.name,
                    "ci_tool": self.preferred_ci_tool
                }
            )
            
            return updated_result
        
        return result
    
    async def create_deployment_plan(
        self, 
        application_name: str,
        version: str,
        source_environment: str,
        target_environment: str,
        components: List[Dict[str, Any]],
        rollback_strategy: Optional[str] = None
    ) -> TaskResult:
        """Create a deployment plan for a specific application version.
        
        Args:
            application_name: Name of the application
            version: Version to deploy
            source_environment: Source environment
            target_environment: Target environment
            components: List of components to deploy
            rollback_strategy: Optional rollback strategy
            
        Returns:
            TaskResult containing the deployment plan
        """
        # Create a task for creating a deployment plan
        task = Task(
            task_id=f"deployment_plan_{application_name}_{version}_{target_environment}",
            description=f"Create deployment plan for {application_name} v{version} to {target_environment}",
            agent_type=str(AgentRole.DEPLOYMENT),
            requirements={
                "application_name": application_name,
                "version": version,
                "source_environment": source_environment,
                "target_environment": target_environment,
                "components": components,
                "rollback_strategy": rollback_strategy or self.preferred_deployment_strategy,
                "deployment_strategy": self.preferred_deployment_strategy
            },
            context=TaskContext(
                notes=(
                    f"Create a detailed deployment plan for {application_name} version {version} "
                    f"from {source_environment} to {target_environment} using a {self.preferred_deployment_strategy} "
                    f"deployment strategy. The plan should include pre-deployment tasks, deployment steps, "
                    f"testing, verification, and rollback procedures."
                )
            ),
            expected_output=(
                "A comprehensive deployment plan including preparation steps, deployment procedures, "
                "verification methods, and rollback instructions."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the deployment plan
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the deployment plan from the result
                plan_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    plan_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            plan_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the text as is with minimal structure
                if not plan_data:
                    plan_data = {
                        "application_name": application_name,
                        "version": version,
                        "source_environment": source_environment,
                        "target_environment": target_environment,
                        "deployment_strategy": self.preferred_deployment_strategy,
                        "plan": result.result
                    }
                
                # Generate plan ID
                plan_id = f"{application_name}_{version}_{target_environment}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Store the deployment plan
                self.deployment_plans[plan_id] = {
                    "id": plan_id,
                    "application_name": application_name,
                    "version": version,
                    "source_environment": source_environment,
                    "target_environment": target_environment,
                    "components": components,
                    "deployment_strategy": self.preferred_deployment_strategy,
                    "rollback_strategy": rollback_strategy or self.preferred_deployment_strategy,
                    "plan": plan_data,
                    "created_at": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"deployment_plan_{plan_id}",
                        value=self.deployment_plans[plan_id],
                        category="deployment_plans"
                    )
                
                logger.info(f"Created deployment plan for {application_name} v{version} to {target_environment}")
                
                # Return the deployment plan
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=self.deployment_plans[plan_id],
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "plan_id": plan_id,
                        "application_name": application_name,
                        "version": version,
                        "target_environment": target_environment
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing deployment plan: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def create_rollback_plan(
        self, 
        deployment_plan_id: str
    ) -> TaskResult:
        """Create a rollback plan for a specific deployment.
        
        Args:
            deployment_plan_id: ID of the deployment plan
            
        Returns:
            TaskResult containing the rollback plan
        """
        # Check if deployment plan exists
        if deployment_plan_id not in self.deployment_plans:
            # Try to load from shared memory if available
            if self.shared_memory:
                plan_data = self.shared_memory.retrieve(
                    key=f"deployment_plan_{deployment_plan_id}",
                    category="deployment_plans"
                )
                if plan_data:
                    self.deployment_plans[deployment_plan_id] = plan_data
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"rollback_plan_{deployment_plan_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Deployment plan with ID {deployment_plan_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"rollback_plan_{deployment_plan_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Deployment plan with ID {deployment_plan_id} not found"
                )
        
        # Get the deployment plan
        deployment_plan = self.deployment_plans[deployment_plan_id]
        
        # Create a task for creating a rollback plan
        task = Task(
            task_id=f"rollback_plan_{deployment_plan_id}",
            description=f"Create rollback plan for {deployment_plan['application_name']} v{deployment_plan['version']}",
            agent_type=str(AgentRole.DEPLOYMENT),
            requirements={
                "deployment_plan_id": deployment_plan_id,
                "application_name": deployment_plan["application_name"],
                "version": deployment_plan["version"],
                "source_environment": deployment_plan["source_environment"],
                "target_environment": deployment_plan["target_environment"],
                "components": deployment_plan["components"],
                "deployment_strategy": deployment_plan["deployment_strategy"],
                "rollback_strategy": deployment_plan["rollback_strategy"],
                "deployment_plan": deployment_plan["plan"]
            },
            context=TaskContext(
                notes=(
                    f"Create a detailed rollback plan for {deployment_plan['application_name']} "
                    f"version {deployment_plan['version']} deployment to {deployment_plan['target_environment']}. "
                    f"The rollback plan should define conditions for triggering a rollback, "
                    f"steps to revert the deployment, and verification procedures."
                )
            ),
            expected_output=(
                "A comprehensive rollback plan including trigger conditions, rollback procedures, "
                "verification methods, and post-rollback actions."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the rollback plan
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                rollback_id = f"rollback_plan_{deployment_plan_id}"
                self.shared_memory.store(
                    key=rollback_id,
                    value={
                        "deployment_plan_id": deployment_plan_id,
                        "application_name": deployment_plan["application_name"],
                        "version": deployment_plan["version"],
                        "target_environment": deployment_plan["target_environment"],
                        "rollback_strategy": deployment_plan["rollback_strategy"],
                        "plan": result.result,
                        "created_at": datetime.now().isoformat()
                    },
                    category="rollback_plans"
                )
            
            logger.info(f"Created rollback plan for {deployment_plan['application_name']} v{deployment_plan['version']}")
            
            # Return the rollback plan
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
                    "deployment_plan_id": deployment_plan_id,
                    "application_name": deployment_plan["application_name"],
                    "version": deployment_plan["version"],
                    "target_environment": deployment_plan["target_environment"]
                }
            )
            
            return updated_result
        
        return result
    
    async def create_monitoring_configuration(
        self, 
        application_name: str,
        components: List[Dict[str, Any]],
        environment: str,
        alert_thresholds: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Create monitoring configuration for an application.
        
        Args:
            application_name: Name of the application
            components: List of components to monitor
            environment: Target environment
            alert_thresholds: Optional alert thresholds
            
        Returns:
            TaskResult containing the monitoring configuration
        """
        # Create a task for creating monitoring configuration
        task = Task(
            task_id=f"monitoring_config_{application_name}_{environment}",
            description=f"Create monitoring configuration for {application_name} in {environment}",
            agent_type=str(AgentRole.DEPLOYMENT),
            requirements={
                "application_name": application_name,
                "components": components,
                "environment": environment,
                "alert_thresholds": alert_thresholds or {}
            },
            context=TaskContext(
                notes=(
                    f"Create a comprehensive monitoring configuration for {application_name} "
                    f"in the {environment} environment. The configuration should include "
                    f"metrics to collect, logging configuration, dashboards, and alerts."
                )
            ),
            expected_output=(
                "A complete monitoring configuration including metrics, logs, dashboards, "
                "and alerts for the application and its components."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the monitoring configuration
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                config_id = f"monitoring_config_{application_name}_{environment}"
                self.shared_memory.store(
                    key=config_id,
                    value={
                        "application_name": application_name,
                        "components": components,
                        "environment": environment,
                        "alert_thresholds": alert_thresholds or {},
                        "configuration": result.result,
                        "created_at": datetime.now().isoformat()
                    },
                    category="monitoring_configurations"
                )
            
            logger.info(f"Created monitoring configuration for {application_name} in {environment}")
            
            # Return the monitoring configuration
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
                    "application_name": application_name,
                    "environment": environment
                }
            )
            
            return updated_result
        
        return result
    
    def get_deployment_pipeline(self, pipeline_id: str) -> Optional[DeploymentPipeline]:
        """Get a specific deployment pipeline.
        
        Args:
            pipeline_id: ID of the deployment pipeline to retrieve
            
        Returns:
            DeploymentPipeline if found, None otherwise
        """
        # Check local storage
        if pipeline_id in self.deployment_pipelines:
            return self.deployment_pipelines[pipeline_id]
        
        # Check shared memory if available
        if self.shared_memory:
            pipeline_data = self.shared_memory.retrieve(
                key=f"pipeline_{pipeline_id}",
                category="deployment_pipelines"
            )
            if pipeline_data:
                pipeline = DeploymentPipeline(**pipeline_data)
                # Cache locally
                self.deployment_pipelines[pipeline_id] = pipeline
                return pipeline
        
        return None
    
    def get_deployment_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific deployment plan.
        
        Args:
            plan_id: ID of the deployment plan to retrieve
            
        Returns:
            Deployment plan if found, None otherwise
        """
        # Check local storage
        if plan_id in self.deployment_plans:
            return self.deployment_plans[plan_id]
        
        # Check shared memory if available
        if self.shared_memory:
            plan_data = self.shared_memory.retrieve(
                key=f"deployment_plan_{plan_id}",
                category="deployment_plans"
            )
            if plan_data:
                # Cache locally
                self.deployment_plans[plan_id] = plan_data
                return plan_data
        
        return None


class SecurityAnalyst(BaseAgent):
    """Agent specialized in security analysis and implementation."""
    
    def __init__(self, name, security_frameworks=None, **kwargs):
        """Initialize the Security Analyst agent.
        
        Args:
            name: Human-readable name for this agent
            security_frameworks: List of security frameworks to reference
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.SECURITY, 
            **kwargs
        )
        self.security_frameworks = security_frameworks or ["OWASP", "NIST"]
        self.security_frameworks = security_frameworks
        
        # Track security assessments
        self.security_assessments: Dict[str, SecurityAssessment] = {}
        
        # Track security implementations
        self.security_implementations: Dict[str, Dict[str, Any]] = {}
        
        # Track vulnerability database
        self.vulnerability_database: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Security Analyst Agent initialized with {', '.join(security_frameworks)} frameworks")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Security Analyst."""
        return (
            f"You are {self.name}, a Security Analyst specialized in identifying "
            f"and mitigating security risks in software and infrastructure, "
            f"with expertise in {', '.join(self.security_frameworks)}. "
            f"Your responsibilities include:\n"
            f"1. Performing security reviews of code and architecture\n"
            f"2. Identifying potential security vulnerabilities\n"
            f"3. Recommending security controls and mitigations\n"
            f"4. Implementing secure coding practices\n"
            f"5. Ensuring compliance with security standards\n\n"
            f"Always approach security from a holistic perspective, considering not just code but "
            f"also infrastructure, data handling, and operational aspects. Think about attack vectors, "
            f"threat models, and defense in depth. When making recommendations, prioritize based on "
            f"risk and impact, and provide practical implementation guidance."
        )
    
    async def perform_security_assessment(
        self, 
        target_name: str,
        target_type: str,
        target_details: Dict[str, Any],
        compliance_requirements: Optional[List[str]] = None
    ) -> TaskResult:
        """Perform a security assessment of an application, infrastructure, or codebase.
        
        Args:
            target_name: Name of the assessment target
            target_type: Type of target (application, infrastructure, code)
            target_details: Details of the assessment target
            compliance_requirements: Optional compliance requirements
            
        Returns:
            TaskResult containing the security assessment
        """
        # Create a task for the security assessment
        task = Task(
            task_id=f"security_assessment_{target_name.lower().replace(' ', '_')}",
            description=f"Perform security assessment of {target_name} ({target_type})",
            agent_type=str(AgentRole.SECURITY),
            requirements={
                "target_name": target_name,
                "target_type": target_type,
                "target_details": target_details,
                "compliance_requirements": compliance_requirements or [],
                "security_frameworks": self.security_frameworks
            },
            context=TaskContext(
                notes=(
                    f"Perform a comprehensive security assessment of {target_name} ({target_type}). "
                    f"Identify potential vulnerabilities, assess risks, and provide detailed "
                    f"recommendations for security improvements. Reference {', '.join(self.security_frameworks)} "
                    f"as appropriate."
                    + (f" Ensure compliance with {', '.join(compliance_requirements)}." 
                       if compliance_requirements else "")
                )
            ),
            expected_output=(
                "A detailed security assessment including findings, risk ratings, "
                "vulnerability details, and prioritized recommendations."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the security assessment
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the security assessment from the result
                assessment_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    assessment_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            assessment_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not assessment_data:
                    logger.warning(f"Could not parse security assessment as JSON. Attempting to extract from text.")
                    assessment_data = self._extract_assessment_from_text(result.result, target_name, target_type)
                
                # Calculate risk score
                risk_score = 0.0
                if "findings" in assessment_data and assessment_data["findings"]:
                    total_risk = sum(finding.get("risk_score", 0) for finding in assessment_data["findings"])
                    risk_score = total_risk / len(assessment_data["findings"])
                
                # Create the security assessment
                assessment = SecurityAssessment(
                    name=f"{target_name} Security Assessment",
                    target_type=target_type,
                    target_id=str(uuid.uuid4()),  # Generate a target ID
                    findings=assessment_data.get("findings", []),
                    risk_score=risk_score,
                    recommendations=assessment_data.get("recommendations", [])
                )
                
                # Store the security assessment
                self.security_assessments[assessment.id] = assessment
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"security_assessment_{assessment.id}",
                        value=assessment.dict(),
                        category="security_assessments"
                    )
                
                logger.info(f"Created security assessment for '{target_name}' with {len(assessment.findings)} findings")
                
                # Return the security assessment as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=assessment.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"assessment_id": assessment.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing security assessment: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_assessment_from_text(
        self, 
        text: str, 
        target_name: str, 
        target_type: str
    ) -> Dict[str, Any]:
        """Extract structured security assessment data from unstructured text.
        
        Args:
            text: The text to extract from
            target_name: The name of the assessment target
            target_type: The type of target
            
        Returns:
            Structured security assessment data
        """
        assessment_data = {
            "name": f"{target_name} Security Assessment",
            "target_type": target_type,
            "findings": [],
            "recommendations": []
        }
        
        # Extract findings
        findings_section = re.search(
            r'(?i)#+\s*(?:Findings|Vulnerabilities|Security Issues)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if findings_section and findings_section.group(1):
            findings_text = findings_section.group(1)
            
            # Look for individual findings
            finding_sections = re.findall(
                r'(?i)(?:#{1,3}|[-*])\s*([^:\n]+)(?::|Vulnerability|Issue)(?:\n+(.+?))?(?=\n+(?:#{1,3}|[-*])\s*[^:\n]+(?::|Vulnerability|Issue)|\n#+|\Z)',
                findings_text,
                re.DOTALL
            )
            
            for title, content in finding_sections:
                finding = {
                    "title": title.strip(),
                    "description": "",
                    "severity": "medium",  # Default severity
                    "risk_score": 5.0,  # Default risk score (scale of 1-10)
                    "affected_components": [],
                    "mitigation": ""
                }
                
                # Extract description
                desc_match = re.search(r'(?i)(?:Description|Details):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
                if desc_match:
                    finding["description"] = desc_match.group(1).strip()
                
                # Extract severity
                severity_match = re.search(r'(?i)(?:Severity|Impact):\s*([^\n]+)', content)
                if severity_match:
                    severity_text = severity_match.group(1).strip().lower()
                    if "critical" in severity_text:
                        finding["severity"] = "critical"
                        finding["risk_score"] = 9.0
                    elif "high" in severity_text:
                        finding["severity"] = "high"
                        finding["risk_score"] = 7.0
                    elif "medium" in severity_text:
                        finding["severity"] = "medium"
                        finding["risk_score"] = 5.0
                    elif "low" in severity_text:
                        finding["severity"] = "low"
                        finding["risk_score"] = 3.0
                    elif "info" in severity_text or "informational" in severity_text:
                        finding["severity"] = "info"
                        finding["risk_score"] = 1.0
                
                # Extract risk score if explicitly provided
                risk_match = re.search(r'(?i)(?:Risk Score|CVSS|Risk):\s*([0-9.]+)', content)
                if risk_match:
                    try:
                        risk_score = float(risk_match.group(1))
                        if 0 <= risk_score <= 10:
                            finding["risk_score"] = risk_score
                    except ValueError:
                        pass
                
                # Extract affected components
                affected_match = re.search(
                    r'(?i)(?:Affected Components|Affected Areas|Impact):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                    content
                )
                if affected_match:
                    affected_text = affected_match.group(1)
                    
                    # Extract components from bullet points
                    affected_items = re.findall(r'[-*]\s*([^\n]+)', affected_text)
                    finding["affected_components"] = [item.strip() for item in affected_items]
                
                # Extract mitigation
                mitigation_match = re.search(r'(?i)(?:Mitigation|Remediation|Fix):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
                if mitigation_match:
                    finding["mitigation"] = mitigation_match.group(1).strip()
                
                assessment_data["findings"].append(finding)
        
        # Extract recommendations
        recommendations_section = re.search(
            r'(?i)#+\s*(?:Recommendations|Remediation|Security Controls)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if recommendations_section and recommendations_section.group(1):
            recommendations_text = recommendations_section.group(1)
            
            # Look for individual recommendations
            recommendation_sections = re.findall(
                r'(?i)(?:#{1,3}|[-*])\s*([^:\n]+)(?::|Recommendation|Control)(?:\n+(.+?))?(?=\n+(?:#{1,3}|[-*])\s*[^:\n]+(?::|Recommendation|Control)|\n#+|\Z)',
                recommendations_text,
                re.DOTALL
            )
            
            for title, content in recommendation_sections:
                recommendation = {
                    "title": title.strip(),
                    "description": "",
                    "priority": "medium",  # Default priority
                    "implementation_complexity": "medium",  # Default complexity
                    "addressed_findings": []
                }
                
                # Extract description
                desc_match = re.search(r'(?i)(?:Description|Details):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
                if desc_match:
                    recommendation["description"] = desc_match.group(1).strip()
                else:
                    # If no explicit description section, use the first paragraph
                    paragraphs = re.split(r'\n\n+', content)
                    if paragraphs:
                        recommendation["description"] = paragraphs[0].strip()
                
                # Extract priority
                priority_match = re.search(r'(?i)(?:Priority|Importance):\s*([^\n]+)', content)
                if priority_match:
                    priority_text = priority_match.group(1).strip().lower()
                    if "critical" in priority_text or "high" in priority_text:
                        recommendation["priority"] = "high"
                    elif "medium" in priority_text:
                        recommendation["priority"] = "medium"
                    elif "low" in priority_text:
                        recommendation["priority"] = "low"
                
                # Extract implementation complexity
                complexity_match = re.search(r'(?i)(?:Complexity|Implementation Complexity|Effort):\s*([^\n]+)', content)
                if complexity_match:
                    complexity_text = complexity_match.group(1).strip().lower()
                    if "high" in complexity_text or "complex" in complexity_text:
                        recommendation["implementation_complexity"] = "high"
                    elif "medium" in complexity_text:
                        recommendation["implementation_complexity"] = "medium"
                    elif "low" in complexity_text or "simple" in complexity_text:
                        recommendation["implementation_complexity"] = "low"
                
                # Extract addressed findings
                addressed_match = re.search(
                    r'(?i)(?:Addresses|Mitigates|Related To):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                    content
                )
                if addressed_match:
                    addressed_text = addressed_match.group(1)
                    
                    # Extract findings from bullet points
                    addressed_items = re.findall(r'[-*]\s*([^\n]+)', addressed_text)
                    recommendation["addressed_findings"] = [item.strip() for item in addressed_items]
                
                assessment_data["recommendations"].append(recommendation)
        
        return assessment_data
    
    async def implement_security_controls(
        self, 
        assessment_id: str,
        selected_recommendations: List[str],
        implementation_context: Dict[str, Any]
    ) -> TaskResult:
        """Implement security controls based on assessment recommendations.
        
        Args:
            assessment_id: ID of the security assessment
            selected_recommendations: List of recommendation IDs or titles to implement
            implementation_context: Context information for implementation
            
        Returns:
            TaskResult containing the security implementation
        """
        # Check if assessment exists
        if assessment_id not in self.security_assessments:
            # Try to load from shared memory if available
            if self.shared_memory:
                assessment_data = self.shared_memory.retrieve(
                    key=f"security_assessment_{assessment_id}",
                    category="security_assessments"
                )
                if assessment_data:
                    self.security_assessments[assessment_id] = SecurityAssessment(**assessment_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_security_{assessment_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Security assessment with ID {assessment_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_security_{assessment_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Security assessment with ID {assessment_id} not found"
                )
        
        # Get the security assessment
        assessment = self.security_assessments[assessment_id]
        
        # Filter recommendations to implement
        recommendations_to_implement = []
        for rec in assessment.recommendations:
            # Match by ID or title
            if rec.get("id", "") in selected_recommendations or rec.get("title", "") in selected_recommendations:
                recommendations_to_implement.append(rec)
        
        if not recommendations_to_implement:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"implement_security_{assessment_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error="No matching recommendations found to implement"
            )
        
        # Create a task for implementing security controls
        task = Task(
            task_id=f"implement_security_{assessment_id}",
            description=f"Implement security controls for {assessment.name}",
            agent_type=str(AgentRole.SECURITY),
            requirements={
                "assessment_id": assessment_id,
                "assessment_name": assessment.name,
                "target_type": assessment.target_type,
                "recommendations": recommendations_to_implement,
                "implementation_context": implementation_context
            },
            context=TaskContext(
                notes=(
                    f"Implement the selected security controls for {assessment.name}. "
                    f"The implementation should include detailed code, configuration, "
                    f"or process changes to address the security findings."
                )
            ),
            expected_output=(
                "Detailed implementation of security controls, including code, "
                "configuration, or process changes with explanations."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Create an implementation ID
            implementation_id = f"security_implementation_{assessment_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store the implementation
            self.security_implementations[implementation_id] = {
                "assessment_id": assessment_id,
                "assessment_name": assessment.name,
                "target_type": assessment.target_type,
                "implemented_recommendations": [r.get("title", "") for r in recommendations_to_implement],
                "implementation": result.result,
                "implementation_context": implementation_context,
                "created_at": datetime.now().isoformat()
            }
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=implementation_id,
                    value=self.security_implementations[implementation_id],
                    category="security_implementations"
                )
            
            logger.info(f"Implemented security controls for {assessment.name}")
            
            # Return the implementation
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
                    "implementation_id": implementation_id,
                    "assessment_id": assessment_id,
                    "assessment_name": assessment.name
                }
            )
            
            return updated_result
        
        return result
    
    async def create_security_policy(
        self, 
        policy_name: str,
        policy_type: str,
        compliance_frameworks: List[str],
        target_audience: List[str]
    ) -> TaskResult:
        """Create a security policy document.
        
        Args:
            policy_name: Name of the policy
            policy_type: Type of policy (data protection, access control, etc.)
            compliance_frameworks: List of compliance frameworks to address
            target_audience: List of target audience roles
            
        Returns:
            TaskResult containing the security policy
        """
        # Create a task for creating a security policy
        task = Task(
            task_id=f"security_policy_{policy_name.lower().replace(' ', '_')}",
            description=f"Create {policy_name} security policy",
            agent_type=str(AgentRole.SECURITY),
            requirements={
                "policy_name": policy_name,
                "policy_type": policy_type,
                "compliance_frameworks": compliance_frameworks,
                "target_audience": target_audience,
                "security_frameworks": self.security_frameworks
            },
            context=TaskContext(
                notes=(
                    f"Create a comprehensive {policy_name} security policy of type {policy_type}. "
                    f"The policy should address requirements from {', '.join(compliance_frameworks)} "
                    f"and be suitable for {', '.join(target_audience)}. Reference "
                    f"{', '.join(self.security_frameworks)} as appropriate."
                )
            ),
            expected_output=(
                "A complete security policy document including purpose, scope, policy statements, "
                "roles and responsibilities, compliance requirements, and implementation guidance."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the security policy
        if result.status == TaskStatus.COMPLETED and result.result:
            # Generate policy ID
            policy_id = f"security_policy_{policy_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=policy_id,
                    value={
                        "policy_name": policy_name,
                        "policy_type": policy_type,
                        "compliance_frameworks": compliance_frameworks,
                        "target_audience": target_audience,
                        "policy_document": result.result,
                        "created_at": datetime.now().isoformat()
                    },
                    category="security_policies"
                )
            
            logger.info(f"Created {policy_name} security policy")
            
            # Return the policy
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
                    "policy_id": policy_id,
                    "policy_name": policy_name,
                    "policy_type": policy_type
                }
            )
            
            return updated_result
        
        return result
    
    async def analyze_dependencies_for_vulnerabilities(
        self, 
        dependencies: List[Dict[str, str]],
        project_type: str
    ) -> TaskResult:
        """Analyze dependencies for security vulnerabilities.
        
        Args:
            dependencies: List of dependencies with name and version
            project_type: Type of project (frontend, backend, etc.)
            
        Returns:
            TaskResult containing the vulnerability analysis
        """
        # Create a task for analyzing dependencies
        task = Task(
            task_id=f"dependency_analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            description=f"Analyze dependencies for vulnerabilities ({project_type} project)",
            agent_type=str(AgentRole.SECURITY),
            requirements={
                "dependencies": dependencies,
                "project_type": project_type
            },
            context=TaskContext(
                notes=(
                    f"Analyze the provided dependencies for a {project_type} project for known "
                    f"security vulnerabilities. Identify vulnerable dependencies, describe "
                    f"the vulnerabilities, and recommend upgrades or mitigations."
                )
            ),
            expected_output=(
                "A comprehensive dependency vulnerability analysis including vulnerable packages, "
                "vulnerability details, severity ratings, and recommended actions."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process the vulnerability analysis
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the vulnerability analysis from the result
                analysis_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    analysis_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            analysis_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the text as is with minimal structure
                if not analysis_data:
                    analysis_data = {
                        "project_type": project_type,
                        "dependencies_analyzed": len(dependencies),
                        "analysis": result.result
                    }
                
                # Store in shared memory if available
                if self.shared_memory:
                    analysis_id = f"dependency_analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    self.shared_memory.store(
                        key=analysis_id,
                        value={
                            "project_type": project_type,
                            "dependencies": dependencies,
                            "analysis": analysis_data,
                            "created_at": datetime.now().isoformat()
                        },
                        category="dependency_analyses"
                    )
                
                    # Update the result metadata
                    result.metadata = {
                        "analysis_id": analysis_id,
                        "project_type": project_type,
                        "dependencies_analyzed": len(dependencies)
                    }
                
                logger.info(f"Analyzed {len(dependencies)} dependencies for {project_type} project")
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing dependency analysis: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def setup_security_monitoring(
        self, 
        application_name: str,
        environment: str,
        components: List[Dict[str, Any]],
        threat_model: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Create security monitoring and alerting configuration.
        
        Args:
            application_name: Name of the application
            environment: Target environment
            components: List of components to monitor
            threat_model: Optional threat model information
            
        Returns:
            TaskResult containing the security monitoring configuration
        """
        # Create a task for security monitoring setup
        task = Task(
            task_id=f"security_monitoring_{application_name}_{environment}",
            description=f"Setup security monitoring for {application_name} in {environment}",
            agent_type=str(AgentRole.SECURITY),
            requirements={
                "application_name": application_name,
                "environment": environment,
                "components": components,
                "threat_model": threat_model or {}
            },
            context=TaskContext(
                notes=(
                    f"Create a comprehensive security monitoring and alerting configuration "
                    f"for {application_name} in the {environment} environment. The configuration "
                    f"should include logging, monitoring, alerting, and incident response procedures "
                    f"tailored to the security needs of the application."
                )
            ),
            expected_output=(
                "A complete security monitoring configuration including log sources, "
                "monitoring rules, alert thresholds, and incident response procedures."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the monitoring configuration
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                config_id = f"security_monitoring_{application_name}_{environment}"
                self.shared_memory.store(
                    key=config_id,
                    value={
                        "application_name": application_name,
                        "environment": environment,
                        "components": components,
                        "threat_model": threat_model,
                        "configuration": result.result,
                        "created_at": datetime.now().isoformat()
                    },
                    category="security_monitoring"
                )
            
            logger.info(f"Created security monitoring configuration for {application_name} in {environment}")
            
            # Return the monitoring configuration
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
                    "application_name": application_name,
                    "environment": environment
                }
            )
            
            return updated_result
        
        return result
    
    def get_security_assessment(self, assessment_id: str) -> Optional[SecurityAssessment]:
        """Get a specific security assessment.
        
        Args:
            assessment_id: ID of the security assessment to retrieve
            
        Returns:
            SecurityAssessment if found, None otherwise
        """
        # Check local storage
        if assessment_id in self.security_assessments:
            return self.security_assessments[assessment_id]
        
        # Check shared memory if available
        if self.shared_memory:
            assessment_data = self.shared_memory.retrieve(
                key=f"security_assessment_{assessment_id}",
                category="security_assessments"
            )
            if assessment_data:
                assessment = SecurityAssessment(**assessment_data)
                # Cache locally
                self.security_assessments[assessment_id] = assessment
                return assessment
        
        return None
    
    def get_security_implementation(self, implementation_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific security implementation.
        
        Args:
            implementation_id: ID of the security implementation to retrieve
            
        Returns:
            Security implementation if found, None otherwise
        """
        # Check local storage
        if implementation_id in self.security_implementations:
            return self.security_implementations[implementation_id]
        
        # Check shared memory if available
        if self.shared_memory:
            implementation_data = self.shared_memory.retrieve(
                key=implementation_id,
                category="security_implementations"
            )
            if implementation_data:
                # Cache locally
                self.security_implementations[implementation_id] = implementation_data
                return implementation_data
        
        return None
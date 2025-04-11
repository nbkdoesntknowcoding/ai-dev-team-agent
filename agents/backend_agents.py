"""
Backend Agents for the multi-agent development system.

This module contains specialized agents for backend development tasks, including
API development, database design, and backend logic implementation. These agents
work together and with other agents to implement the server-side components of
applications based on the system architecture.
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


class Endpoint(BaseModel):
    """API endpoint definition."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    path: str
    method: str  # GET, POST, PUT, DELETE, etc.
    description: str
    request_params: Optional[Dict[str, Any]] = None
    request_body: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    status_codes: Dict[str, str] = Field(default_factory=dict)
    auth_required: bool = False
    rate_limited: bool = False
    tags: List[str] = Field(default_factory=list)


class ApiSpecification(BaseModel):
    """API specification containing multiple endpoints."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    base_path: str
    version: str
    endpoints: List[Endpoint] = Field(default_factory=list)
    auth_schemes: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DatabaseColumn(BaseModel):
    """Definition of a database column."""
    name: str
    data_type: str
    nullable: bool = False
    primary_key: bool = False
    foreign_key: Optional[str] = None  # Format: "table.column"
    unique: bool = False
    default: Optional[Any] = None
    description: str = ""
    constraints: List[str] = Field(default_factory=list)


class DatabaseTable(BaseModel):
    """Definition of a database table."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    columns: List[DatabaseColumn] = Field(default_factory=list)
    indexes: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)


class DatabaseSchema(BaseModel):
    """Complete database schema."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    tables: List[DatabaseTable] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class BusinessLogic(BaseModel):
    """Business logic component."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    functions: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    integration_points: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class APIDeveloper(BaseAgent):
    """Agent specialized in designing and implementing API endpoints."""
    
    def __init__(self, name, preferred_api_style="REST", **kwargs):
        # Initialize attributes before calling super().__init__
        self.preferred_api_style = preferred_api_style
        
        super().__init__(
            name=name,
            agent_type=AgentRole.API_DEVELOPER,
            **kwargs
        )
        """Initialize the API Developer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_api_style: Preferred API style (REST, GraphQL, gRPC, etc.)
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.API_DEVELOPER, 
            **kwargs
        )
        self.preferred_api_style = preferred_api_style
        
        # Track API specifications
        self.api_specifications: Dict[str, ApiSpecification] = {}
        
        # Knowledge base of common API patterns and best practices
        self.api_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Track implementation status
        self.implementation_status: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"API Developer Agent initialized with {preferred_api_style} style preference")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the API Developer."""
        return (
            f"You are {self.name}, an API Developer specialized in designing and implementing "
            f"API endpoints using {self.preferred_api_style}. "
            f"Your responsibilities include:\n"
            f"1. Designing API endpoints based on system requirements\n"
            f"2. Creating API specifications with request/response schemas\n"
            f"3. Implementing API endpoints using appropriate frameworks\n"
            f"4. Ensuring API security, performance, and scalability\n"
            f"5. Following API best practices and standards\n\n"
            f"Think step-by-step when designing APIs. Consider RESTful principles, "
            f"consistency, versioning, error handling, and documentation. Be thorough "
            f"in your specifications, including all necessary endpoints, parameters, "
            f"request bodies, response schemas, and status codes.\n\n"
            f"When implementing APIs, focus on security (authentication, authorization, input validation), "
            f"performance optimization, and proper error handling. Document your APIs "
            f"thoroughly for other developers to understand and use."
        )
    
    async def design_api(
        self, 
        api_name: str,
        description: str,
        requirements: List[Dict[str, Any]],
        database_schema: Optional[Dict[str, Any]] = None,
        security_requirements: Optional[List[Dict[str, Any]]] = None
    ) -> TaskResult:
        """Design a complete API based on requirements.
        
        Args:
            api_name: Name of the API
            description: Brief description of the API's purpose
            requirements: List of requirements for the API
            database_schema: Optional database schema to integrate with
            security_requirements: Optional security requirements
            
        Returns:
            TaskResult containing the API specification
        """
        # Create a task for API design
        task = Task(
            task_id=f"design_api_{api_name.lower().replace(' ', '_')}",
            description=f"Design API for {api_name}",
            agent_type=str(AgentRole.API_DEVELOPER),
            requirements={
                "api_name": api_name,
                "description": description,
                "requirements": requirements,
                "database_schema": database_schema,
                "security_requirements": security_requirements,
                "preferred_style": self.preferred_api_style
            },
            context=TaskContext(
                notes=(
                    f"Design a complete API for {api_name} that meets all requirements. "
                    f"The API should follow {self.preferred_api_style} principles and best practices. "
                    + (f"It should integrate with the provided database schema. " if database_schema else "")
                    + (f"It must meet the specified security requirements. " if security_requirements else "")
                )
            ),
            expected_output=(
                "A comprehensive API specification including base path, endpoints, "
                "request/response schemas, authentication, and error handling."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the API specification
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the API specification from the result
                api_spec_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    api_spec_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            api_spec_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not api_spec_data:
                    logger.warning(f"Could not parse API specification as JSON. Attempting to extract from text.")
                    api_spec_data = self._extract_api_spec_from_text(result.result, api_name, description)
                
                # Create endpoints
                endpoints = []
                for endpoint_data in api_spec_data.get("endpoints", []):
                    endpoints.append(
                        Endpoint(
                            path=endpoint_data.get("path", "/"),
                            method=endpoint_data.get("method", "GET"),
                            description=endpoint_data.get("description", ""),
                            request_params=endpoint_data.get("request_params"),
                            request_body=endpoint_data.get("request_body"),
                            response_schema=endpoint_data.get("response_schema"),
                            status_codes=endpoint_data.get("status_codes", {}),
                            auth_required=endpoint_data.get("auth_required", False),
                            rate_limited=endpoint_data.get("rate_limited", False),
                            tags=endpoint_data.get("tags", [])
                        )
                    )
                
                # Create the API specification
                api_spec = ApiSpecification(
                    name=api_name,
                    description=description,
                    base_path=api_spec_data.get("base_path", f"/api/{api_name.lower().replace(' ', '-')}"),
                    version=api_spec_data.get("version", "v1"),
                    endpoints=endpoints,
                    auth_schemes=api_spec_data.get("auth_schemes")
                )
                
                # Store the API specification
                self.api_specifications[api_spec.id] = api_spec
                
                # Initialize implementation status
                self.implementation_status[api_spec.id] = {
                    "status": "designed",
                    "implemented_endpoints": [],
                    "pending_endpoints": [endpoint.id for endpoint in endpoints],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"api_spec_{api_spec.id}",
                        value=api_spec.dict(),
                        category="api_specifications"
                    )
                
                logger.info(f"Created API specification for '{api_name}' with {len(endpoints)} endpoints")
                
                # Return the API specification as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=api_spec.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"api_spec_id": api_spec.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing API specification: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_api_spec_from_text(
        self, 
        text: str, 
        api_name: str, 
        description: str
    ) -> Dict[str, Any]:
        """Extract structured API specification data from unstructured text.
        
        Args:
            text: The text to extract from
            api_name: The name of the API
            description: The description of the API
            
        Returns:
            Structured API specification data
        """
        api_spec_data = {
            "name": api_name,
            "description": description,
            "base_path": f"/api/{api_name.lower().replace(' ', '-')}",
            "version": "v1",
            "endpoints": [],
            "auth_schemes": {}
        }
        
        # Extract base path and version
        base_path_match = re.search(r'(?i)Base(?:\s+|-)Path:?\s+([^\n]+)', text)
        if base_path_match:
            api_spec_data["base_path"] = base_path_match.group(1).strip()
        
        version_match = re.search(r'(?i)Version:?\s+([^\n]+)', text)
        if version_match:
            api_spec_data["version"] = version_match.group(1).strip()
        
        # Extract auth schemes
        auth_section = re.search(
            r'(?i)(?:Authentication|Auth Schemes|Security):\s*\n+((?:.+\n)+?)(?:\n|$)',
            text
        )
        if auth_section:
            auth_text = auth_section.group(1)
            auth_schemes = {}
            
            # Look for common auth types
            if re.search(r'(?i)bearer|jwt|token', auth_text):
                auth_schemes["bearer"] = {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "JWT Authentication"
                }
            
            if re.search(r'(?i)basic', auth_text):
                auth_schemes["basic"] = {
                    "type": "http",
                    "scheme": "basic",
                    "description": "Basic Authentication"
                }
            
            if re.search(r'(?i)api[- ]?key', auth_text):
                auth_schemes["apiKey"] = {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API Key Authentication"
                }
            
            if re.search(r'(?i)oauth|oauth2', auth_text):
                auth_schemes["oauth2"] = {
                    "type": "oauth2",
                    "description": "OAuth2 Authentication"
                }
            
            if auth_schemes:
                api_spec_data["auth_schemes"] = auth_schemes
        
        # Extract endpoints
        endpoint_sections = re.findall(
            r'(?i)#+\s*(?:Endpoint|API|Route):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+\s*(?:Endpoint|API|Route)|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in endpoint_sections:
            endpoint = {
                "path": "/",
                "method": "GET",
                "description": "",
                "request_params": {},
                "request_body": None,
                "response_schema": None,
                "status_codes": {},
                "auth_required": False,
                "rate_limited": False,
                "tags": []
            }
            
            # Extract path and method from title
            path_method_match = re.search(r'(?i)(`)?([^`]+)(`)?\s+(?:[-–—])\s+([A-Z]+)', title)
            if path_method_match:
                endpoint["path"] = path_method_match.group(2).strip()
                endpoint["method"] = path_method_match.group(4).strip()
            else:
                # Try alternative formats
                path_match = re.search(r'(?i)(?:Path|Route|URL):\s*(`)?([^`\n]+)(`)?', title + "\n" + content)
                method_match = re.search(r'(?i)(?:Method|HTTP Method|Verb):\s*([A-Z]+)', title + "\n" + content)
                
                if path_match:
                    endpoint["path"] = path_match.group(2).strip()
                if method_match:
                    endpoint["method"] = method_match.group(1).strip()
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Purpose):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if desc_match:
                endpoint["description"] = desc_match.group(1).strip()
            
            # Extract request parameters
            params_match = re.search(
                r'(?i)(?:Request Parameters|Query Parameters|Path Parameters):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if params_match:
                params_text = params_match.group(1)
                param_items = re.findall(r'[-*]\s*([^\n]+)', params_text)
                
                for param in param_items:
                    param_parts = re.search(r'`?([^`:\s]+)`?\s*(?:\(([^)]+)\))?(?::\s*(.+))?', param)
                    if param_parts:
                        param_name = param_parts.group(1).strip()
                        param_type = param_parts.group(2).strip() if param_parts.group(2) else "string"
                        param_desc = param_parts.group(3).strip() if param_parts.group(3) else ""
                        
                        endpoint["request_params"][param_name] = {
                            "type": param_type,
                            "description": param_desc,
                            "required": "optional" not in param.lower()
                        }
                    else:
                        # Just add a simple entry if we can't parse the structure
                        endpoint["request_params"][param.strip()] = {
                            "type": "string",
                            "description": "",
                            "required": "optional" not in param.lower()
                        }
            
            # Extract request body
            body_match = re.search(
                r'(?i)(?:Request Body|Body|Payload):\s*\n+(```(?:json)?\n(?:.+\n)+?```|(?:[-*]\s*[^\n]+\n*)+)',
                content,
                re.DOTALL
            )
            if body_match:
                body_text = body_match.group(1)
                
                # Try to parse JSON
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', body_text, re.DOTALL)
                if json_match:
                    try:
                        body_json = json.loads(json_match.group(1))
                        endpoint["request_body"] = body_json
                    except:
                        # If we can't parse it, just use it as a string
                        endpoint["request_body"] = json_match.group(1).strip()
                else:
                    # Try to build a structure from bullet points
                    body_items = re.findall(r'[-*]\s*([^\n]+)', body_text)
                    body_struct = {}
                    
                    for item in body_items:
                        item_parts = re.search(r'`?([^`:\s]+)`?\s*(?:\(([^)]+)\))?(?::\s*(.+))?', item)
                        if item_parts:
                            item_name = item_parts.group(1).strip()
                            item_type = item_parts.group(2).strip() if item_parts.group(2) else "string"
                            item_desc = item_parts.group(3).strip() if item_parts.group(3) else ""
                            
                            body_struct[item_name] = {
                                "type": item_type,
                                "description": item_desc,
                                "required": "optional" not in item.lower()
                            }
                    
                    if body_struct:
                        endpoint["request_body"] = {
                            "type": "object",
                            "properties": body_struct
                        }
            
            # Extract response schema
            response_match = re.search(
                r'(?i)(?:Response|Response Body|Response Schema):\s*\n+(```(?:json)?\n(?:.+\n)+?```|(?:[-*]\s*[^\n]+\n*)+)',
                content,
                re.DOTALL
            )
            if response_match:
                response_text = response_match.group(1)
                
                # Try to parse JSON
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    try:
                        response_json = json.loads(json_match.group(1))
                        endpoint["response_schema"] = response_json
                    except:
                        # If we can't parse it, just use it as a string
                        endpoint["response_schema"] = json_match.group(1).strip()
                else:
                    # Try to build a structure from bullet points
                    response_items = re.findall(r'[-*]\s*([^\n]+)', response_text)
                    response_struct = {}
                    
                    for item in response_items:
                        item_parts = re.search(r'`?([^`:\s]+)`?\s*(?:\(([^)]+)\))?(?::\s*(.+))?', item)
                        if item_parts:
                            item_name = item_parts.group(1).strip()
                            item_type = item_parts.group(2).strip() if item_parts.group(2) else "string"
                            item_desc = item_parts.group(3).strip() if item_parts.group(3) else ""
                            
                            response_struct[item_name] = {
                                "type": item_type,
                                "description": item_desc
                            }
                    
                    if response_struct:
                        endpoint["response_schema"] = {
                            "type": "object",
                            "properties": response_struct
                        }
            
            # Extract status codes
            status_codes_match = re.search(
                r'(?i)(?:Status Codes|Response Codes|HTTP Status):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if status_codes_match:
                codes_text = status_codes_match.group(1)
                code_items = re.findall(r'[-*]\s*([^\n]+)', codes_text)
                
                for code_item in code_items:
                    code_parts = re.search(r'(?:(\d{3})(?:\s*[-:]\s*|\s+)([^\n]+))', code_item)
                    if code_parts:
                        code = code_parts.group(1).strip()
                        description = code_parts.group(2).strip()
                        endpoint["status_codes"][code] = description
            
            # Extract auth requirement
            if re.search(r'(?i)(?:requires auth|authentication required|auth required)', content):
                endpoint["auth_required"] = True
            
            # Extract rate limiting
            if re.search(r'(?i)(?:rate limited|rate limit)', content):
                endpoint["rate_limited"] = True
            
            # Extract tags
            tags_match = re.search(r'(?i)(?:Tags|Categories):\s*([^\n]+)', content)
            if tags_match:
                tags_text = tags_match.group(1)
                tags = re.findall(r'(?:`([^`]+)`|([^,\s]+))', tags_text)
                endpoint["tags"] = [t[0] or t[1] for t in tags if t[0] or t[1]]
            
            api_spec_data["endpoints"].append(endpoint)
        
        return api_spec_data
    
    async def implement_endpoint(
        self, 
        api_spec_id: str,
        endpoint_id: str,
        framework: str = "FastAPI",
        implementation_details: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Implement a specific API endpoint.
        
        Args:
            api_spec_id: ID of the API specification
            endpoint_id: ID of the endpoint to implement
            framework: Web framework to use (FastAPI, Flask, Express, etc.)
            implementation_details: Optional additional implementation details
            
        Returns:
            TaskResult containing the endpoint implementation
        """
        # Check if API specification exists
        if api_spec_id not in self.api_specifications:
            # Try to load from shared memory if available
            if self.shared_memory:
                spec_data = self.shared_memory.retrieve(
                    key=f"api_spec_{api_spec_id}",
                    category="api_specifications"
                )
                if spec_data:
                    self.api_specifications[api_spec_id] = ApiSpecification(**spec_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_endpoint_{endpoint_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"API specification with ID {api_spec_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_endpoint_{endpoint_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"API specification with ID {api_spec_id} not found"
                )
        
        # Get the API specification
        api_spec = self.api_specifications[api_spec_id]
        
        # Find the endpoint
        target_endpoint = None
        for endpoint in api_spec.endpoints:
            if endpoint.id == endpoint_id:
                target_endpoint = endpoint
                break
        
        if not target_endpoint:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"implement_endpoint_{endpoint_id}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Endpoint with ID {endpoint_id} not found in API specification {api_spec.name}"
            )
        
        # Create a task for implementing the endpoint
        task = Task(
            task_id=f"implement_endpoint_{endpoint_id}",
            description=f"Implement {target_endpoint.method} {target_endpoint.path} endpoint",
            agent_type=str(AgentRole.API_DEVELOPER),
            requirements={
                "api_spec_id": api_spec_id,
                "api_name": api_spec.name,
                "endpoint_id": endpoint_id,
                "endpoint_path": target_endpoint.path,
                "endpoint_method": target_endpoint.method,
                "endpoint_description": target_endpoint.description,
                "request_params": target_endpoint.request_params,
                "request_body": target_endpoint.request_body,
                "response_schema": target_endpoint.response_schema,
                "status_codes": target_endpoint.status_codes,
                "auth_required": target_endpoint.auth_required,
                "framework": framework,
                "implementation_details": implementation_details or {}
            },
            context=TaskContext(
                notes=(
                    f"Implement the {target_endpoint.method} {target_endpoint.path} endpoint "
                    f"for the {api_spec.name} API using {framework}. The implementation should "
                    f"handle request validation, appropriate error responses, and conform to "
                    f"the specified response schema."
                )
            ),
            expected_output=(
                f"Complete {framework} implementation code for the endpoint, including "
                f"request handling, validation, business logic integration, and error handling."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Update implementation status
            if api_spec_id in self.implementation_status:
                status = self.implementation_status[api_spec_id]
                
                # Remove from pending and add to implemented
                if endpoint_id in status["pending_endpoints"]:
                    status["pending_endpoints"].remove(endpoint_id)
                
                if endpoint_id not in status["implemented_endpoints"]:
                    status["implemented_endpoints"].append(endpoint_id)
                
                status["status"] = "partial" if status["pending_endpoints"] else "completed"
                status["timestamp"] = datetime.now().isoformat()
                
                # Update status
                self.implementation_status[api_spec_id] = status
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"endpoint_implementation_{endpoint_id}",
                    value={
                        "api_spec_id": api_spec_id,
                        "endpoint_id": endpoint_id,
                        "framework": framework,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="endpoint_implementations"
                )
                
                # Update implementation status
                self.shared_memory.store(
                    key=f"api_implementation_status_{api_spec_id}",
                    value=self.implementation_status[api_spec_id],
                    category="api_implementation_status"
                )
            
            logger.info(f"Implemented {target_endpoint.method} {target_endpoint.path} endpoint using {framework}")
            
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
                    "api_spec_id": api_spec_id,
                    "endpoint_id": endpoint_id,
                    "framework": framework
                }
            )
            
            return updated_result
        
        return result
    
    async def generate_api_documentation(
        self, 
        api_spec_id: str,
        format: str = "OpenAPI",
        include_examples: bool = True
    ) -> TaskResult:
        """Generate documentation for an API specification.
        
        Args:
            api_spec_id: ID of the API specification
            format: Documentation format (OpenAPI, API Blueprint, etc.)
            include_examples: Whether to include example requests and responses
            
        Returns:
            TaskResult containing the API documentation
        """
        # Check if API specification exists
        if api_spec_id not in self.api_specifications:
            # Try to load from shared memory if available
            if self.shared_memory:
                spec_data = self.shared_memory.retrieve(
                    key=f"api_spec_{api_spec_id}",
                    category="api_specifications"
                )
                if spec_data:
                    self.api_specifications[api_spec_id] = ApiSpecification(**spec_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"generate_api_docs_{api_spec_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"API specification with ID {api_spec_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"generate_api_docs_{api_spec_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"API specification with ID {api_spec_id} not found"
                )
        
        # Get the API specification
        api_spec = self.api_specifications[api_spec_id]
        
        # Create a task for generating API documentation
        task = Task(
            task_id=f"generate_api_docs_{api_spec_id}",
            description=f"Generate {format} documentation for {api_spec.name} API",
            agent_type=str(AgentRole.API_DEVELOPER),
            requirements={
                "api_spec_id": api_spec_id,
                "api_name": api_spec.name,
                "api_description": api_spec.description,
                "base_path": api_spec.base_path,
                "version": api_spec.version,
                "endpoints": [endpoint.dict() for endpoint in api_spec.endpoints],
                "auth_schemes": api_spec.auth_schemes,
                "format": format,
                "include_examples": include_examples
            },
            context=TaskContext(
                notes=(
                    f"Generate comprehensive {format} documentation for the {api_spec.name} API. "
                    f"The documentation should include all endpoints, request/response schemas, "
                    f"parameters, status codes, authentication, and error handling."
                    + (f" Include example requests and responses." if include_examples else "")
                )
            ),
            expected_output=(
                f"Complete {format} documentation for the API that can be used by developers "
                f"to understand and consume the API."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the documentation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"api_documentation_{api_spec_id}_{format.lower()}",
                    value={
                        "api_spec_id": api_spec_id,
                        "api_name": api_spec.name,
                        "format": format,
                        "documentation": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="api_documentation"
                )
            
            logger.info(f"Generated {format} documentation for {api_spec.name} API")
            
            # Return the documentation
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
                    "api_spec_id": api_spec_id,
                    "api_name": api_spec.name,
                    "format": format
                }
            )
            
            return updated_result
        
        return result
    
    def get_api_specification(self, api_spec_id: str) -> Optional[ApiSpecification]:
        """Get a specific API specification.
        
        Args:
            api_spec_id: ID of the API specification to retrieve
            
        Returns:
            ApiSpecification if found, None otherwise
        """
        # Check local storage
        if api_spec_id in self.api_specifications:
            return self.api_specifications[api_spec_id]
        
        # Check shared memory if available
        if self.shared_memory:
            spec_data = self.shared_memory.retrieve(
                key=f"api_spec_{api_spec_id}",
                category="api_specifications"
            )
            if spec_data:
                api_spec = ApiSpecification(**spec_data)
                # Cache locally
                self.api_specifications[api_spec_id] = api_spec
                return api_spec
        
        return None
    
    def get_implementation_status(self, api_spec_id: str) -> Optional[Dict[str, Any]]:
        """Get the implementation status for an API specification.
        
        Args:
            api_spec_id: ID of the API specification
            
        Returns:
            Implementation status if found, None otherwise
        """
        # Check local storage
        if api_spec_id in self.implementation_status:
            return self.implementation_status[api_spec_id]
        
        # Check shared memory if available
        if self.shared_memory:
            status_data = self.shared_memory.retrieve(
                key=f"api_implementation_status_{api_spec_id}",
                category="api_implementation_status"
            )
            if status_data:
                # Cache locally
                self.implementation_status[api_spec_id] = status_data
                return status_data
        
        return None


class DatabaseDesigner(BaseAgent):
    """Agent specialized in designing and implementing database schemas."""
    
    def __init__(self, name, preferred_db_type="SQL", **kwargs):
        # Initialize attributes before calling super().__init__
        self.preferred_db_type = preferred_db_type
        
        super().__init__(
            name=name,
            agent_type=AgentRole.DATABASE_DESIGNER,
            **kwargs
        )
        """Initialize the Database Designer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_db_type: Preferred database type (relational, document, etc.)
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.DATABASE_DESIGNER, 
            **kwargs
        )
        self.preferred_db_type = preferred_db_type
        
        # Track database schemas
        self.database_schemas: Dict[str, DatabaseSchema] = {}
        
        # Database patterns and best practices
        self.database_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Track implementation status
        self.implementation_status: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Database Designer Agent initialized with {preferred_db_type} database preference")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Database Designer."""
        return (
            f"You are {self.name}, a Database Designer specialized in {self.preferred_db_type} "
            f"database design and implementation. "
            f"Your responsibilities include:\n"
            f"1. Designing database schemas based on data requirements\n"
            f"2. Creating table structures with appropriate relationships\n"
            f"3. Implementing database schemas using appropriate tools\n"
            f"4. Optimizing database performance and reliability\n"
            f"5. Following database best practices and standards\n\n"
            f"Think step-by-step when designing databases. Consider data integrity, normalization, "
            f"performance, scalability, and security. Be thorough in your designs, including all "
            f"necessary tables, columns, relationships, indexes, and constraints.\n\n"
            f"When implementing databases, focus on data integrity, performance optimization, "
            f"and security. Document your database schema thoroughly for other developers "
            f"to understand and use."
        )
    
    async def design_database_schema(
        self, 
        schema_name: str,
        description: str,
        data_requirements: List[Dict[str, Any]],
        existing_systems: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Design a complete database schema based on requirements.
        
        Args:
            schema_name: Name of the database schema
            description: Brief description of the database purpose
            data_requirements: Data requirements to incorporate into the schema
            existing_systems: Optional existing systems to integrate with
            
        Returns:
            TaskResult containing the database schema
        """
        # Create a task for database schema design
        task = Task(
            task_id=f"design_db_schema_{schema_name.lower().replace(' ', '_')}",
            description=f"Design database schema for {schema_name}",
            agent_type=str(AgentRole.DATABASE_DESIGNER),
            requirements={
                "schema_name": schema_name,
                "description": description,
                "data_requirements": data_requirements,
                "existing_systems": existing_systems,
                "database_type": self.preferred_db_type
            },
            context=TaskContext(
                notes=(
                    f"Design a complete database schema for {schema_name} that meets all data requirements. "
                    f"The schema should follow {self.preferred_db_type} database principles and best practices. "
                    + (f"It should integrate with the existing systems provided. " if existing_systems else "")
                )
            ),
            expected_output=(
                "A comprehensive database schema including tables, columns, relationships, "
                "indexes, constraints, and any other relevant structures."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the database schema
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the database schema from the result
                schema_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    schema_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            schema_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not schema_data:
                    logger.warning(f"Could not parse database schema as JSON. Attempting to extract from text.")
                    schema_data = self._extract_db_schema_from_text(result.result, schema_name, description)
                
                # Create tables
                tables = []
                for table_data in schema_data.get("tables", []):
                    # Process columns
                    columns = []
                    for column_data in table_data.get("columns", []):
                        column = DatabaseColumn(
                            name=column_data.get("name", "column"),
                            data_type=column_data.get("data_type", "varchar"),
                            nullable=column_data.get("nullable", False),
                            primary_key=column_data.get("primary_key", False),
                            foreign_key=column_data.get("foreign_key"),
                            unique=column_data.get("unique", False),
                            default=column_data.get("default"),
                            description=column_data.get("description", ""),
                            constraints=column_data.get("constraints", [])
                        )
                        columns.append(column)
                    
                    # Create the table
                    table = DatabaseTable(
                        name=table_data.get("name", "table"),
                        description=table_data.get("description", ""),
                        columns=columns,
                        indexes=table_data.get("indexes", []),
                        constraints=table_data.get("constraints", [])
                    )
                    tables.append(table)
                
                # Create the database schema
                db_schema = DatabaseSchema(
                    name=schema_name,
                    description=description,
                    tables=tables,
                    relationships=schema_data.get("relationships", [])
                )
                
                # Store the database schema
                self.database_schemas[db_schema.id] = db_schema
                
                # Initialize implementation status
                self.implementation_status[db_schema.id] = {
                    "status": "designed",
                    "implemented_tables": [],
                    "pending_tables": [table.id for table in tables],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"db_schema_{db_schema.id}",
                        value=db_schema.dict(),
                        category="database_schemas"
                    )
                
                logger.info(f"Created database schema for '{schema_name}' with {len(tables)} tables")
                
                # Return the database schema as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=db_schema.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"db_schema_id": db_schema.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing database schema: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_db_schema_from_text(
        self, 
        text: str, 
        schema_name: str, 
        description: str
    ) -> Dict[str, Any]:
        """Extract structured database schema data from unstructured text.
        
        Args:
            text: The text to extract from
            schema_name: The name of the schema
            description: The description of the schema
            
        Returns:
            Structured database schema data
        """
        schema_data = {
            "name": schema_name,
            "description": description,
            "tables": [],
            "relationships": []
        }
        
        # Extract tables
        table_sections = re.findall(
            r'(?i)#+\s*(?:Table|Entity):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+\s*(?:Table|Entity|Relationship)|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in table_sections:
            table = {
                "name": title.strip(),
                "description": "",
                "columns": [],
                "indexes": [],
                "constraints": []
            }
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Purpose):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if desc_match:
                table["description"] = desc_match.group(1).strip()
            
            # Extract columns
            column_section = re.search(
                r'(?i)(?:Columns|Fields|Attributes):\s*\n+((?:.+\n)+?)(?:\n\n|\Z)',
                content
            )
            
            if column_section:
                # First, check if there's a table format
                table_format = re.search(r'(?:\|\s*(.+?)\s*\|(?:\s*[-:]+\s*\|)+\n)((?:\|\s*.+?\s*\|\n)+)', column_section.group(1))
                
                if table_format:
                    # Extract column headers and rows from Markdown table
                    headers = [h.strip() for h in table_format.group(1).split('|') if h.strip()]
                    rows = table_format.group(2).strip().split('\n')
                    
                    for row in rows:
                        row_values = [v.strip() for v in row.split('|') if v.strip()]
                        if not row_values:
                            continue
                            
                        column = {}
                        for i, header in enumerate(headers):
                            if i < len(row_values):
                                value = row_values[i]
                                header_lower = header.lower()
                                
                                if header_lower in ["name", "column"]:
                                    column["name"] = value
                                elif header_lower in ["type", "data type", "datatype"]:
                                    column["data_type"] = value
                                elif header_lower in ["nullable", "null"]:
                                    column["nullable"] = value.lower() in ["yes", "true", "y", "✓"]
                                elif header_lower in ["primary key", "pk"]:
                                    column["primary_key"] = value.lower() in ["yes", "true", "y", "✓", "pk"]
                                elif header_lower in ["foreign key", "fk", "references"]:
                                    if value and value.lower() not in ["no", "false", "n", "-"]:
                                        column["foreign_key"] = value
                                elif header_lower in ["unique"]:
                                    column["unique"] = value.lower() in ["yes", "true", "y", "✓"]
                                elif header_lower in ["default"]:
                                    column["default"] = value if value.lower() not in ["none", "-"] else None
                                elif header_lower in ["description"]:
                                    column["description"] = value
                        
                        if "name" in column:
                            table["columns"].append(column)
                else:
                    # Try to extract from bullet points
                    column_items = re.findall(r'[-*]\s*([^\n]+)', column_section.group(1))
                    
                    for column_text in column_items:
                        # Try to match: name (type) [constraints]: description
                        column_match = re.search(r'`?([^`(]+)`?\s*(?:\(([^)]+)\))?\s*(?:\[([^\]]+)\])?(?::\s*(.+))?', column_text)
                        
                        if column_match:
                            column_name = column_match.group(1).strip()
                            column_type = column_match.group(2).strip() if column_match.group(2) else "varchar"
                            constraints_text = column_match.group(3) if column_match.group(3) else ""
                            description = column_match.group(4).strip() if column_match.group(4) else ""
                            
                            # Parse constraints
                            constraints = []
                            is_primary = "primary key" in constraints_text.lower() or "pk" in constraints_text.lower()
                            is_nullable = not ("not null" in constraints_text.lower())
                            is_unique = "unique" in constraints_text.lower()
                            foreign_key = None
                            
                            fk_match = re.search(r'(?:references|fk)\s*(?:to)?\s*([^,\s]+)', constraints_text, re.IGNORECASE)
                            if fk_match:
                                foreign_key = fk_match.group(1)
                            
                            default_match = re.search(r'default\s*(?:=|:)?\s*([^,\s]+)', constraints_text, re.IGNORECASE)
                            default_value = default_match.group(1) if default_match else None
                            
                            table["columns"].append({
                                "name": column_name,
                                "data_type": column_type,
                                "nullable": is_nullable,
                                "primary_key": is_primary,
                                "foreign_key": foreign_key,
                                "unique": is_unique,
                                "default": default_value,
                                "description": description,
                                "constraints": [c.strip() for c in constraints_text.split(',') if c.strip()]
                            })
                        else:
                            # Simplified format: just add the raw text
                            parts = column_text.split(":")
                            if len(parts) >= 2:
                                name_type = parts[0].strip()
                                description = parts[1].strip()
                                
                                # Try to separate name and type
                                name_type_match = re.search(r'`?([^`(]+)`?\s*(?:\(([^)]+)\))?', name_type)
                                if name_type_match:
                                    name = name_type_match.group(1).strip()
                                    data_type = name_type_match.group(2).strip() if name_type_match.group(2) else "varchar"
                                    
                                    table["columns"].append({
                                        "name": name,
                                        "data_type": data_type,
                                        "description": description
                                    })
            
            # Extract indexes
            index_section = re.search(
                r'(?i)(?:Indexes|Index):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if index_section:
                index_items = re.findall(r'[-*]\s*([^\n]+)', index_section.group(1))
                
                for index_text in index_items:
                    index_match = re.search(r'(?:(\w+)(?:\s*:)?)?\s*(?:on)?\s*\(?([^)]+)\)?(?:\s*\(([^)]+)\))?', index_text, re.IGNORECASE)
                    
                    if index_match:
                        index_name = index_match.group(1).strip() if index_match.group(1) else f"idx_{table['name'].lower()}"
                        index_type = index_match.group(2).strip() if index_match.group(2) else "btree"
                        columns = index_match.group(3) if index_match.group(3) else ""
                        
                        if not columns:
                            # Try to find column names in the text
                            columns = re.search(r'(?:columns|fields)?\s*:?\s*([^:]+)$', index_text, re.IGNORECASE)
                            columns = columns.group(1).strip() if columns else ""
                        
                        # Parse columns
                        column_list = [c.strip() for c in re.split(r'[,\s]+', columns) if c.strip()]
                        
                        if column_list:
                            table["indexes"].append({
                                "name": index_name,
                                "type": index_type,
                                "columns": column_list
                            })
                    else:
                        # Just add a basic index entry
                        table["indexes"].append({
                            "description": index_text.strip()
                        })
            
            # Extract constraints
            constraint_section = re.search(
                r'(?i)(?:Constraints|Constraint):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if constraint_section:
                constraint_items = re.findall(r'[-*]\s*([^\n]+)', constraint_section.group(1))
                
                for constraint_text in constraint_items:
                    table["constraints"].append({
                        "description": constraint_text.strip()
                    })
            
            schema_data["tables"].append(table)
        
        # Extract relationships
        relationship_sections = re.findall(
            r'(?i)#+\s*(?:Relationship|Relationships)(?:\n+(.+?))?(?=\n#+\s*(?:Table|Entity|Relationship)|\Z)',
            text,
            re.DOTALL
        )
        
        for content in relationship_sections:
            if not content:
                continue
                
            # Try to extract from bullet points
            relationship_items = re.findall(r'[-*]\s*([^\n]+)', content)
            
            for rel_text in relationship_items:
                # Try to match various relationship formats
                rel_match = re.search(r'(?:(\w+)(?:\s*:)?\s*)?([^-<>]+)\s*(?:(-+|<-+|--+>|<--+>))\s*([^:(]+)(?:\s*\(([^)]+)\))?', rel_text)
                
                if rel_match:
                    rel_name = rel_match.group(1).strip() if rel_match.group(1) else ""
                    source_table = rel_match.group(2).strip()
                    rel_type = rel_match.group(3).strip()
                    target_table = rel_match.group(4).strip()
                    cardinality = rel_match.group(5).strip() if rel_match.group(5) else ""
                    
                    # Determine relationship type
                    if "->" in rel_type or rel_type.endswith("-"):
                        rel_type = "one-to-many"
                    elif "<-" in rel_type:
                        rel_type = "many-to-one"
                    elif "<->" in rel_type:
                        rel_type = "many-to-many"
                    else:
                        rel_type = "one-to-one"
                    
                    # Override with explicit cardinality if provided
                    if cardinality:
                        if cardinality.lower() in ["1:n", "1:many", "one-to-many"]:
                            rel_type = "one-to-many"
                        elif cardinality.lower() in ["n:1", "many:1", "many-to-one"]:
                            rel_type = "many-to-one"
                        elif cardinality.lower() in ["n:n", "many:many", "many-to-many"]:
                            rel_type = "many-to-many"
                        elif cardinality.lower() in ["1:1", "one:one", "one-to-one"]:
                            rel_type = "one-to-one"
                    
                    schema_data["relationships"].append({
                        "name": rel_name,
                        "source_table": source_table,
                        "target_table": target_table,
                        "type": rel_type,
                        "cardinality": cardinality
                    })
                else:
                    # Just add the raw text as a relationship description
                    schema_data["relationships"].append({
                        "description": rel_text.strip()
                    })
        
        return schema_data
    
    async def implement_database(
        self, 
        schema_id: str,
        db_platform: str = "PostgreSQL",
        implementation_details: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Generate database implementation scripts based on a schema.
        
        Args:
            schema_id: ID of the database schema
            db_platform: Database platform (PostgreSQL, MySQL, MongoDB, etc.)
            implementation_details: Optional additional implementation details
            
        Returns:
            TaskResult containing the database implementation scripts
        """
        # Check if database schema exists
        if schema_id not in self.database_schemas:
            # Try to load from shared memory if available
            if self.shared_memory:
                schema_data = self.shared_memory.retrieve(
                    key=f"db_schema_{schema_id}",
                    category="database_schemas"
                )
                if schema_data:
                    self.database_schemas[schema_id] = DatabaseSchema(**schema_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_database_{schema_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Database schema with ID {schema_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_database_{schema_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Database schema with ID {schema_id} not found"
                )
        
        # Get the database schema
        db_schema = self.database_schemas[schema_id]
        
        # Create a task for implementing the database
        task = Task(
            task_id=f"implement_database_{schema_id}",
            description=f"Implement {db_schema.name} database schema using {db_platform}",
            agent_type=str(AgentRole.DATABASE_DESIGNER),
            requirements={
                "schema_id": schema_id,
                "schema_name": db_schema.name,
                "db_platform": db_platform,
                "tables": [table.dict() for table in db_schema.tables],
                "relationships": db_schema.relationships,
                "implementation_details": implementation_details or {}
            },
            context=TaskContext(
                notes=(
                    f"Generate database implementation scripts for the {db_schema.name} schema "
                    f"using {db_platform}. The scripts should create all tables, columns, "
                    f"relationships, indexes, and constraints defined in the schema."
                )
            ),
            expected_output=(
                f"Complete {db_platform} implementation scripts that can be executed "
                f"to create the database schema with all necessary structures."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Update implementation status
            if schema_id in self.implementation_status:
                status = self.implementation_status[schema_id]
                
                # Mark all tables as implemented
                status["pending_tables"] = []
                status["implemented_tables"] = [table.id for table in db_schema.tables]
                status["status"] = "completed"
                status["timestamp"] = datetime.now().isoformat()
                
                # Update status
                self.implementation_status[schema_id] = status
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"db_implementation_{schema_id}_{db_platform.lower()}",
                    value={
                        "schema_id": schema_id,
                        "schema_name": db_schema.name,
                        "db_platform": db_platform,
                        "scripts": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="database_implementations"
                )
                
                # Update implementation status
                self.shared_memory.store(
                    key=f"db_implementation_status_{schema_id}",
                    value=self.implementation_status[schema_id],
                    category="database_implementation_status"
                )
            
            logger.info(f"Implemented {db_schema.name} database schema using {db_platform}")
            
            # Return the implementation scripts
            updated_result = TaskResult(
                agent_id=result.agent_id,
                agent_name=result.agent_name,
                task_id=result.task_id,
                result=result.result,  # Original implementation scripts
                status=result.status,
                timestamp=result.timestamp,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                metadata={
                    "schema_id": schema_id,
                    "schema_name": db_schema.name,
                    "db_platform": db_platform
                }
            )
            
            return updated_result
        
        return result
    
    async def generate_database_migrations(
        self, 
        schema_id: str,
        previous_schema_id: Optional[str] = None,
        db_platform: str = "PostgreSQL"
    ) -> TaskResult:
        """Generate database migration scripts between schema versions.
        
        Args:
            schema_id: ID of the target database schema
            previous_schema_id: Optional ID of the previous schema version
            db_platform: Database platform (PostgreSQL, MySQL, MongoDB, etc.)
            
        Returns:
            TaskResult containing the database migration scripts
        """
        # Check if target schema exists
        if schema_id not in self.database_schemas:
            # Try to load from shared memory if available
            if self.shared_memory:
                schema_data = self.shared_memory.retrieve(
                    key=f"db_schema_{schema_id}",
                    category="database_schemas"
                )
                if schema_data:
                    self.database_schemas[schema_id] = DatabaseSchema(**schema_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"generate_migrations_{schema_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Database schema with ID {schema_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"generate_migrations_{schema_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Database schema with ID {schema_id} not found"
                )
        
        # Get the target schema
        target_schema = self.database_schemas[schema_id]
        
        # Get the previous schema if specified
        previous_schema = None
        if previous_schema_id:
            if previous_schema_id in self.database_schemas:
                previous_schema = self.database_schemas[previous_schema_id]
            elif self.shared_memory:
                prev_schema_data = self.shared_memory.retrieve(
                    key=f"db_schema_{previous_schema_id}",
                    category="database_schemas"
                )
                if prev_schema_data:
                    previous_schema = DatabaseSchema(**prev_schema_data)
        
        # Create a task for generating migrations
        task = Task(
            task_id=f"generate_migrations_{schema_id}",
            description=f"Generate database migrations for {target_schema.name} schema using {db_platform}",
            agent_type=str(AgentRole.DATABASE_DESIGNER),
            requirements={
                "schema_id": schema_id,
                "schema_name": target_schema.name,
                "target_schema": target_schema.dict(),
                "has_previous_schema": previous_schema is not None,
                "previous_schema": previous_schema.dict() if previous_schema else None,
                "db_platform": db_platform
            },
            context=TaskContext(
                notes=(
                    f"Generate database migration scripts for the {target_schema.name} schema "
                    f"using {db_platform}. "
                    + (f"The migrations should transform the previous schema into the target schema. " 
                       if previous_schema else 
                       "Create a new migration script for the initial schema setup. ")
                    + f"Include both 'up' and 'down' migrations for rollback support."
                )
            ),
            expected_output=(
                f"Complete {db_platform} migration scripts that can be executed "
                f"to update the database schema with proper versioning."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the migration scripts
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                migration_id = str(uuid.uuid4())
                self.shared_memory.store(
                    key=f"db_migration_{migration_id}",
                    value={
                        "schema_id": schema_id,
                        "schema_name": target_schema.name,
                        "previous_schema_id": previous_schema_id,
                        "db_platform": db_platform,
                        "scripts": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="database_migrations"
                )
            
            logger.info(f"Generated database migrations for {target_schema.name} schema using {db_platform}")
            
            # Return the migration scripts
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
                    "schema_id": schema_id,
                    "schema_name": target_schema.name,
                    "previous_schema_id": previous_schema_id,
                    "db_platform": db_platform
                }
            )
            
            return updated_result
        
        return result
    
    async def optimize_database_schema(
        self, 
        schema_id: str,
        optimization_goals: List[str],
        current_issues: Optional[List[Dict[str, Any]]] = None
    ) -> TaskResult:
        """Optimize a database schema for specific goals.
        
        Args:
            schema_id: ID of the database schema to optimize
            optimization_goals: List of optimization goals (performance, storage, etc.)
            current_issues: Optional list of current issues with the schema
            
        Returns:
            TaskResult containing the optimized database schema
        """
        # Check if schema exists
        if schema_id not in self.database_schemas:
            # Try to load from shared memory if available
            if self.shared_memory:
                schema_data = self.shared_memory.retrieve(
                    key=f"db_schema_{schema_id}",
                    category="database_schemas"
                )
                if schema_data:
                    self.database_schemas[schema_id] = DatabaseSchema(**schema_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"optimize_schema_{schema_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Database schema with ID {schema_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"optimize_schema_{schema_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Database schema with ID {schema_id} not found"
                )
        
        # Get the database schema
        db_schema = self.database_schemas[schema_id]
        
        # Create a task for optimizing the schema
        task = Task(
            task_id=f"optimize_schema_{schema_id}",
            description=f"Optimize {db_schema.name} database schema",
            agent_type=str(AgentRole.DATABASE_DESIGNER),
            requirements={
                "schema_id": schema_id,
                "schema_name": db_schema.name,
                "current_schema": db_schema.dict(),
                "optimization_goals": optimization_goals,
                "current_issues": current_issues or []
            },
            context=TaskContext(
                notes=(
                    f"Optimize the {db_schema.name} database schema for the following goals: "
                    f"{', '.join(optimization_goals)}. "
                    + (f"Address the following issues: {json.dumps(current_issues)}. " if current_issues else "")
                    + f"Maintain data integrity and functional requirements while making optimizations."
                )
            ),
            expected_output=(
                "An optimized database schema with specific changes to improve the schema "
                "based on the optimization goals, including justification for each change."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the optimized schema
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the optimized schema from the result
                optimized_schema_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    optimized_schema_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            optimized_schema_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, use the original schema with optimization notes
                if not optimized_schema_data:
                    # Create a new schema based on the original
                    new_schema_id = str(uuid.uuid4())
                    new_schema = DatabaseSchema(
                        id=new_schema_id,
                        name=f"{db_schema.name} (Optimized)",
                        description=f"Optimized version of {db_schema.name} schema",
                        tables=db_schema.tables,
                        relationships=db_schema.relationships
                    )
                    
                    # Store the optimization results in shared memory
                    if self.shared_memory:
                        self.shared_memory.store(
                            key=f"schema_optimization_{schema_id}",
                            value={
                                "original_schema_id": schema_id,
                                "schema_name": db_schema.name,
                                "optimization_goals": optimization_goals,
                                "optimization_notes": result.result,
                                "timestamp": datetime.now().isoformat()
                            },
                            category="schema_optimizations"
                        )
                        
                        # Store the new schema as well
                        self.shared_memory.store(
                            key=f"db_schema_{new_schema_id}",
                            value=new_schema.dict(),
                            category="database_schemas"
                        )
                    
                    # Store the new schema locally
                    self.database_schemas[new_schema_id] = new_schema
                    
                    logger.info(f"Created optimized schema for {db_schema.name}")
                    
                    # Return the optimization notes and new schema ID
                    updated_result = TaskResult(
                        agent_id=result.agent_id,
                        agent_name=result.agent_name,
                        task_id=result.task_id,
                        result={
                            "original_schema_id": schema_id,
                            "optimized_schema_id": new_schema_id,
                            "optimization_notes": result.result
                        },
                        status=result.status,
                        timestamp=result.timestamp,
                        execution_time=result.execution_time,
                        token_usage=result.token_usage,
                        metadata={
                            "original_schema_id": schema_id,
                            "optimized_schema_id": new_schema_id
                        }
                    )
                    
                    return updated_result
                
                # Process the optimized schema data
                # Create tables
                tables = []
                for table_data in optimized_schema_data.get("tables", []):
                    # Process columns
                    columns = []
                    for column_data in table_data.get("columns", []):
                        column = DatabaseColumn(
                            name=column_data.get("name", "column"),
                            data_type=column_data.get("data_type", "varchar"),
                            nullable=column_data.get("nullable", False),
                            primary_key=column_data.get("primary_key", False),
                            foreign_key=column_data.get("foreign_key"),
                            unique=column_data.get("unique", False),
                            default=column_data.get("default"),
                            description=column_data.get("description", ""),
                            constraints=column_data.get("constraints", [])
                        )
                        columns.append(column)
                    
                    # Create the table
                    table = DatabaseTable(
                        name=table_data.get("name", "table"),
                        description=table_data.get("description", ""),
                        columns=columns,
                        indexes=table_data.get("indexes", []),
                        constraints=table_data.get("constraints", [])
                    )
                    tables.append(table)
                
                # Create the optimized database schema
                new_schema_id = str(uuid.uuid4())
                optimized_schema = DatabaseSchema(
                    id=new_schema_id,
                    name=f"{db_schema.name} (Optimized)",
                    description=f"Optimized version of {db_schema.name} schema",
                    tables=tables,
                    relationships=optimized_schema_data.get("relationships", [])
                )
                
                # Store the optimized schema
                self.database_schemas[optimized_schema.id] = optimized_schema
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"db_schema_{optimized_schema.id}",
                        value=optimized_schema.dict(),
                        category="database_schemas"
                    )
                    
                    # Store optimization metadata
                    self.shared_memory.store(
                        key=f"schema_optimization_{schema_id}_{optimized_schema.id}",
                        value={
                            "original_schema_id": schema_id,
                            "optimized_schema_id": optimized_schema.id,
                            "optimization_goals": optimization_goals,
                            "changes_made": optimized_schema_data.get("changes_made", []),
                            "timestamp": datetime.now().isoformat()
                        },
                        category="schema_optimizations"
                    )
                
                logger.info(f"Created optimized schema for {db_schema.name} with {len(tables)} tables")
                
                # Return the optimized schema
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=optimized_schema.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "original_schema_id": schema_id,
                        "optimized_schema_id": optimized_schema.id
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing optimized schema: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def get_database_schema(self, schema_id: str) -> Optional[DatabaseSchema]:
        """Get a specific database schema.
        
        Args:
            schema_id: ID of the database schema to retrieve
            
        Returns:
            DatabaseSchema if found, None otherwise
        """
        # Check local storage
        if schema_id in self.database_schemas:
            return self.database_schemas[schema_id]
        
        # Check shared memory if available
        if self.shared_memory:
            schema_data = self.shared_memory.retrieve(
                key=f"db_schema_{schema_id}",
                category="database_schemas"
            )
            if schema_data:
                db_schema = DatabaseSchema(**schema_data)
                # Cache locally
                self.database_schemas[schema_id] = db_schema
                return db_schema
        
        return None
    
    def get_implementation_status(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Get the implementation status for a database schema.
        
        Args:
            schema_id: ID of the database schema
            
        Returns:
            Implementation status if found, None otherwise
        """
        # Check local storage
        if schema_id in self.implementation_status:
            return self.implementation_status[schema_id]
        
        # Check shared memory if available
        if self.shared_memory:
            status_data = self.shared_memory.retrieve(
                key=f"db_implementation_status_{schema_id}",
                category="database_implementation_status"
            )
            if status_data:
                # Cache locally
                self.implementation_status[schema_id] = status_data
                return status_data
        
        return None


class BackendLogicDeveloper(BaseAgent):
    """Agent specialized in implementing backend business logic."""
    
    def __init__(self, name, preferred_language="Python", **kwargs):
        # Initialize attributes before calling super().__init__
        self.preferred_language = preferred_language
        
        super().__init__(
            name=name,
            agent_type=AgentRole.BACKEND_LOGIC,
            **kwargs
        )
        """Initialize the Backend Logic Developer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_language: Preferred programming language
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.BACKEND_LOGIC, 
            **kwargs
        )
        self.preferred_language = preferred_language
        
        # Track business logic components
        self.business_logic: Dict[str, BusinessLogic] = {}
        
        # Track implementations of logic
        self.implementations: Dict[str, Dict[str, Any]] = {}
        
        # Testing results
        self.test_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Backend Logic Developer Agent initialized with {preferred_language} preference")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Backend Logic Developer."""
        return (
            f"You are {self.name}, a Backend Logic Developer specialized in implementing "
            f"business logic using {self.preferred_language}. "
            f"Your responsibilities include:\n"
            f"1. Implementing business rules and processes\n"
            f"2. Creating efficient and maintainable backend code\n"
            f"3. Integrating with databases, APIs, and external systems\n"
            f"4. Ensuring code security, reliability, and performance\n"
            f"5. Writing unit tests for business logic\n\n"
            f"Think step-by-step when implementing business logic. Write clean, maintainable "
            f"code with appropriate error handling, logging, and documentation. Consider "
            f"edge cases, performance implications, and security best practices.\n\n"
            f"Your code should be well-structured, properly commented, and follow best "
            f"practices for {self.preferred_language}. Include comprehensive tests to validate "
            f"functionality and handle edge cases."
        )
    
    async def design_business_logic(
        self, 
        component_name: str,
        description: str,
        requirements: List[Dict[str, Any]],
        database_schema: Optional[Dict[str, Any]] = None,
        api_spec: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Design business logic components based on requirements.
        
        Args:
            component_name: Name of the business logic component
            description: Brief description of the component's purpose
            requirements: List of functional requirements
            database_schema: Optional database schema the logic will interact with
            api_spec: Optional API specification the logic will support
            
        Returns:
            TaskResult containing the business logic design
        """
        # Create a task for business logic design
        task = Task(
            task_id=f"design_logic_{component_name.lower().replace(' ', '_')}",
            description=f"Design business logic for {component_name}",
            agent_type=str(AgentRole.BACKEND_LOGIC),
            requirements={
                "component_name": component_name,
                "description": description,
                "requirements": requirements,
                "database_schema": database_schema,
                "api_spec": api_spec,
                "language": self.preferred_language
            },
            context=TaskContext(
                notes=(
                    f"Design business logic components for {component_name} that meet all requirements. "
                    f"The logic should be implemented in {self.preferred_language} and follow best practices. "
                    + (f"It should interact with the provided database schema. " if database_schema else "")
                    + (f"It should support the specified API. " if api_spec else "")
                )
            ),
            expected_output=(
                "A comprehensive business logic design including component structure, "
                "functions, data flow, and integration points."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the business logic design
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the business logic design from the result
                logic_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    logic_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            logic_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not logic_data:
                    logger.warning(f"Could not parse business logic as JSON. Attempting to extract from text.")
                    logic_data = self._extract_business_logic_from_text(result.result, component_name, description)
                
                # Create the business logic component
                business_logic = BusinessLogic(
                    name=component_name,
                    description=description,
                    functions=logic_data.get("functions", []),
                    dependencies=logic_data.get("dependencies", []),
                    integration_points=logic_data.get("integration_points", [])
                )
                
                # Store the business logic
                self.business_logic[business_logic.id] = business_logic
                
                # Initialize implementations tracking
                self.implementations[business_logic.id] = {
                    "status": "designed",
                    "implemented_functions": [],
                    "pending_functions": [f.get("name", f"function_{i}") for i, f in enumerate(business_logic.functions)],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"business_logic_{business_logic.id}",
                        value=business_logic.dict(),
                        category="business_logic"
                    )
                
                logger.info(f"Created business logic design for '{component_name}' with {len(business_logic.functions)} functions")
                
                # Return the business logic design as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=business_logic.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"business_logic_id": business_logic.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing business logic design: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_business_logic_from_text(
        self, 
        text: str, 
        component_name: str, 
        description: str
    ) -> Dict[str, Any]:
        """Extract structured business logic data from unstructured text.
        
        Args:
            text: The text to extract from
            component_name: The name of the component
            description: The description of the component
            
        Returns:
            Structured business logic data
        """
        logic_data = {
            "name": component_name,
            "description": description,
            "functions": [],
            "dependencies": [],
            "integration_points": []
        }
        
        # Extract functions
        function_sections = re.findall(
            r'(?i)#+\s*(?:Function|Method):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+\s*(?:Function|Method|Dependency|Integration)|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in function_sections:
            function = {
                "name": title.strip(),
                "description": "",
                "parameters": [],
                "returns": None,
                "pseudocode": "",
                "business_rules": []
            }
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Purpose):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if desc_match:
                function["description"] = desc_match.group(1).strip()
            
            # Extract parameters
            params_match = re.search(
                r'(?i)(?:Parameters|Inputs|Arguments):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if params_match:
                params_text = params_match.group(1)
                param_items = re.findall(r'[-*]\s*([^\n]+)', params_text)
                
                for param in param_items:
                    param_parts = re.search(r'`?([^`:\s]+)`?\s*(?:\(([^)]+)\))?(?::\s*(.+))?', param)
                    if param_parts:
                        param_name = param_parts.group(1).strip()
                        param_type = param_parts.group(2).strip() if param_parts.group(2) else "any"
                        param_desc = param_parts.group(3).strip() if param_parts.group(3) else ""
                        
                        function["parameters"].append({
                            "name": param_name,
                            "type": param_type,
                            "description": param_desc,
                            "required": "optional" not in param.lower()
                        })
                    else:
                        # Just add a simple entry if we can't parse the structure
                        function["parameters"].append({
                            "description": param.strip()
                        })
            
            # Extract return value
            returns_match = re.search(r'(?i)(?:Returns|Return Value|Output):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if returns_match:
                returns_text = returns_match.group(1).strip()
                returns_parts = re.search(r'(?:\(([^)]+)\))?\s*(.+)', returns_text)
                
                if returns_parts:
                    return_type = returns_parts.group(1).strip() if returns_parts.group(1) else "any"
                    return_desc = returns_parts.group(2).strip() if returns_parts.group(2) else returns_text
                    
                    function["returns"] = {
                        "type": return_type,
                        "description": return_desc
                    }
                else:
                    function["returns"] = {
                        "description": returns_text
                    }
            
            # Extract pseudocode or algorithm
            code_match = re.search(r'(?i)(?:Pseudocode|Algorithm|Implementation):\s*\n+```(?:[a-z]*\n)?(.*?)```', content, re.DOTALL)
            if code_match:
                function["pseudocode"] = code_match.group(1).strip()
            else:
                # Try to find an outlined algorithm
                algo_match = re.search(r'(?i)(?:Pseudocode|Algorithm|Implementation|Steps):\s*\n+((?:[-*0-9.]\s*[^\n]+\n*)+)', content)
                if algo_match:
                    function["pseudocode"] = algo_match.group(1).strip()
            
            # Extract business rules
            rules_match = re.search(
                r'(?i)(?:Business Rules|Rules|Constraints):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if rules_match:
                rules_text = rules_match.group(1)
                rule_items = re.findall(r'[-*]\s*([^\n]+)', rules_text)
                function["business_rules"] = [rule.strip() for rule in rule_items]
            
            logic_data["functions"].append(function)
        
        # Extract dependencies
        dependencies_section = re.search(
            r'(?i)#+\s*(?:Dependencies|External Libraries|Required Modules)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if dependencies_section and dependencies_section.group(1):
            dependencies_text = dependencies_section.group(1)
            dependency_items = re.findall(r'[-*]\s*([^\n]+)', dependencies_text)
            logic_data["dependencies"] = [dep.strip() for dep in dependency_items]
        
        # Extract integration points
        integration_sections = re.findall(
            r'(?i)#+\s*(?:Integration|Integration Point|External System):\s*([^\n]+)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        
        for title, content in integration_sections:
            integration = {
                "name": title.strip(),
                "type": "unknown",
                "description": ""
            }
            
            # Extract type
            type_match = re.search(r'(?i)(?:Type|Integration Type):\s*([^\n]+)', content)
            if type_match:
                integration["type"] = type_match.group(1).strip()
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Details):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if desc_match:
                integration["description"] = desc_match.group(1).strip()
            
            # Try to determine type from name or content if not specified
            if integration["type"] == "unknown":
                content_lower = content.lower()
                if re.search(r'database|db|data ?store|sql', content_lower):
                    integration["type"] = "database"
                elif re.search(r'api|endpoint|rest|graphql|service', content_lower):
                    integration["type"] = "api"
                elif re.search(r'event|message|queue|topic|pub[/\-]?sub', content_lower):
                    integration["type"] = "messaging"
                elif re.search(r'file|storage|s3|blob', content_lower):
                    integration["type"] = "file_storage"
                elif re.search(r'cache|redis|memcached', content_lower):
                    integration["type"] = "cache"
                elif re.search(r'auth|authentication|identity', content_lower):
                    integration["type"] = "authentication"
            
            logic_data["integration_points"].append(integration)
        
        return logic_data
    
    async def implement_function(
        self, 
        logic_id: str,
        function_name: str,
        implementation_details: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Implement a specific business logic function.
        
        Args:
            logic_id: ID of the business logic component
            function_name: Name of the function to implement
            implementation_details: Optional additional implementation details
            
        Returns:
            TaskResult containing the function implementation
        """
        # Check if business logic component exists
        if logic_id not in self.business_logic:
            # Try to load from shared memory if available
            if self.shared_memory:
                logic_data = self.shared_memory.retrieve(
                    key=f"business_logic_{logic_id}",
                    category="business_logic"
                )
                if logic_data:
                    self.business_logic[logic_id] = BusinessLogic(**logic_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_function_{function_name}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Business logic component with ID {logic_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_function_{function_name}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Business logic component with ID {logic_id} not found"
                )
        
        # Get the business logic component
        logic = self.business_logic[logic_id]
        
        # Find the function specification
        function_spec = None
        for func in logic.functions:
            if func.get("name") == function_name:
                function_spec = func
                break
        
        if not function_spec:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"implement_function_{function_name}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Function {function_name} not found in business logic component {logic.name}"
            )
        
        # Create a task for implementing the function
        task = Task(
            task_id=f"implement_function_{function_name}",
            description=f"Implement {function_name} function for {logic.name}",
            agent_type=str(AgentRole.BACKEND_LOGIC),
            requirements={
                "logic_id": logic_id,
                "component_name": logic.name,
                "function_name": function_name,
                "function_spec": function_spec,
                "language": self.preferred_language,
                "implementation_details": implementation_details or {}
            },
            context=TaskContext(
                notes=(
                    f"Implement the {function_name} function for {logic.name} in {self.preferred_language}. "
                    f"Follow the function specification including all parameters, return values, "
                    f"and business rules. Include proper error handling, logging, and documentation."
                )
            ),
            expected_output=(
                f"Complete {self.preferred_language} implementation of the function, including "
                f"documentation, error handling, and any necessary helper functions."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the implementation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Update implementation status
            if logic_id in self.implementations:
                status = self.implementations[logic_id]
                
                # Remove from pending and add to implemented
                if function_name in status["pending_functions"]:
                    status["pending_functions"].remove(function_name)
                
                if function_name not in status["implemented_functions"]:
                    status["implemented_functions"].append(function_name)
                
                status["status"] = "partial" if status["pending_functions"] else "completed"
                status["timestamp"] = datetime.now().isoformat()
                
                # Update status
                self.implementations[logic_id] = status
            
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"function_implementation_{logic_id}_{function_name}",
                    value={
                        "logic_id": logic_id,
                        "component_name": logic.name,
                        "function_name": function_name,
                        "language": self.preferred_language,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="function_implementations"
                )
                
                # Update implementation status
                self.shared_memory.store(
                    key=f"logic_implementation_status_{logic_id}",
                    value=self.implementations[logic_id],
                    category="logic_implementation_status"
                )
            
            logger.info(f"Implemented {function_name} function for {logic.name} in {self.preferred_language}")
            
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
                    "logic_id": logic_id,
                    "function_name": function_name,
                    "language": self.preferred_language
                }
            )
            
            return updated_result
        
        return result
    
    async def create_unit_tests(
        self, 
        logic_id: str,
        function_name: str,
        test_framework: str = "pytest"
    ) -> TaskResult:
        """Create unit tests for a business logic function.
        
        Args:
            logic_id: ID of the business logic component
            function_name: Name of the function to test
            test_framework: Testing framework to use
            
        Returns:
            TaskResult containing the unit tests
        """
        # Check if business logic component exists
        if logic_id not in self.business_logic:
            # Try to load from shared memory if available
            if self.shared_memory:
                logic_data = self.shared_memory.retrieve(
                    key=f"business_logic_{logic_id}",
                    category="business_logic"
                )
                if logic_data:
                    self.business_logic[logic_id] = BusinessLogic(**logic_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"create_tests_{function_name}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Business logic component with ID {logic_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"create_tests_{function_name}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Business logic component with ID {logic_id} not found"
                )
        
        # Get the business logic component
        logic = self.business_logic[logic_id]
        
        # Find the function specification
        function_spec = None
        for func in logic.functions:
            if func.get("name") == function_name:
                function_spec = func
                break
        
        if not function_spec:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"create_tests_{function_name}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Function {function_name} not found in business logic component {logic.name}"
            )
        
        # Get the function implementation if available
        function_impl = None
        if self.shared_memory:
            impl_data = self.shared_memory.retrieve(
                key=f"function_implementation_{logic_id}_{function_name}",
                category="function_implementations"
            )
            if impl_data:
                function_impl = impl_data.get("code")
        
        # Create a task for creating unit tests
        task = Task(
            task_id=f"create_tests_{function_name}",
            description=f"Create unit tests for {function_name} function",
            agent_type=str(AgentRole.BACKEND_LOGIC),
            requirements={
                "logic_id": logic_id,
                "component_name": logic.name,
                "function_name": function_name,
                "function_spec": function_spec,
                "function_implementation": function_impl,
                "language": self.preferred_language,
                "test_framework": test_framework
            },
            context=TaskContext(
                notes=(
                    f"Create comprehensive unit tests for the {function_name} function of {logic.name} "
                    f"using {test_framework}. Cover all normal operation cases, edge cases, and error cases. "
                    f"Follow testing best practices and ensure good test coverage."
                    + (f"\n\nThe function implementation is available for reference." if function_impl else "")
                )
            ),
            expected_output=(
                f"Complete unit tests for the function using {test_framework}, including test cases "
                f"for normal operation, edge cases, and error handling."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the unit tests
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"function_tests_{logic_id}_{function_name}",
                    value={
                        "logic_id": logic_id,
                        "component_name": logic.name,
                        "function_name": function_name,
                        "test_framework": test_framework,
                        "tests": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="function_tests"
                )
            
            logger.info(f"Created unit tests for {function_name} function using {test_framework}")
            
            # Return the unit tests
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
                    "logic_id": logic_id,
                    "function_name": function_name,
                    "test_framework": test_framework
                }
            )
            
            return updated_result
        
        return result
    
    async def create_integration_code(
        self, 
        logic_id: str,
        integration_name: str,
        target_system: Dict[str, Any]
    ) -> TaskResult:
        """Create integration code for connecting business logic with external systems.
        
        Args:
            logic_id: ID of the business logic component
            integration_name: Name of the integration point
            target_system: Details of the system to integrate with
            
        Returns:
            TaskResult containing the integration code
        """
        # Check if business logic component exists
        if logic_id not in self.business_logic:
            # Try to load from shared memory if available
            if self.shared_memory:
                logic_data = self.shared_memory.retrieve(
                    key=f"business_logic_{logic_id}",
                    category="business_logic"
                )
                if logic_data:
                    self.business_logic[logic_id] = BusinessLogic(**logic_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"create_integration_{integration_name}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Business logic component with ID {logic_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"create_integration_{integration_name}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Business logic component with ID {logic_id} not found"
                )
        
        # Get the business logic component
        logic = self.business_logic[logic_id]
        
        # Find the integration point
        integration_point = None
        for integration in logic.integration_points:
            if integration.get("name") == integration_name:
                integration_point = integration
                break
        
        if not integration_point:
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"create_integration_{integration_name}",
                result=None,
                status=TaskStatus.FAILED,
                execution_time=0.0,
                error=f"Integration point {integration_name} not found in business logic component {logic.name}"
            )
        
        # Create a task for creating integration code
        task = Task(
            task_id=f"create_integration_{integration_name}",
            description=f"Create integration code for {integration_name}",
            agent_type=str(AgentRole.BACKEND_LOGIC),
            requirements={
                "logic_id": logic_id,
                "component_name": logic.name,
                "integration_name": integration_name,
                "integration_point": integration_point,
                "target_system": target_system,
                "language": self.preferred_language
            },
            context=TaskContext(
                notes=(
                    f"Create integration code for connecting {logic.name} with {integration_name} "
                    f"using {self.preferred_language}. The integration should handle all necessary "
                    f"data transformations, error handling, and retries. Follow best practices "
                    f"for integrating with {target_system.get('type', 'external system')}."
                )
            ),
            expected_output=(
                f"Complete integration code for connecting {logic.name} with {integration_name}, "
                f"including connection management, error handling, and data transformation."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the integration code
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                self.shared_memory.store(
                    key=f"integration_code_{logic_id}_{integration_name}",
                    value={
                        "logic_id": logic_id,
                        "component_name": logic.name,
                        "integration_name": integration_name,
                        "target_system": target_system,
                        "code": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="integration_code"
                )
            
            logger.info(f"Created integration code for {integration_name} in {logic.name}")
            
            # Return the integration code
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
                    "logic_id": logic_id,
                    "integration_name": integration_name,
                    "target_system_type": target_system.get("type")
                }
            )
            
            return updated_result
        
        return result
    
    def get_business_logic(self, logic_id: str) -> Optional[BusinessLogic]:
        """Get a specific business logic component.
        
        Args:
            logic_id: ID of the business logic component to retrieve
            
        Returns:
            BusinessLogic if found, None otherwise
        """
        # Check local storage
        if logic_id in self.business_logic:
            return self.business_logic[logic_id]
        
        # Check shared memory if available
        if self.shared_memory:
            logic_data = self.shared_memory.retrieve(
                key=f"business_logic_{logic_id}",
                category="business_logic"
            )
            if logic_data:
                business_logic = BusinessLogic(**logic_data)
                # Cache locally
                self.business_logic[logic_id] = business_logic
                return business_logic
        
        return None
    
    def get_implementation_status(self, logic_id: str) -> Optional[Dict[str, Any]]:
        """Get the implementation status for a business logic component.
        
        Args:
            logic_id: ID of the business logic component
            
        Returns:
            Implementation status if found, None otherwise
        """
        # Check local storage
        if logic_id in self.implementations:
            return self.implementations[logic_id]
        
        # Check shared memory if available
        if self.shared_memory:
            status_data = self.shared_memory.retrieve(
                key=f"logic_implementation_status_{logic_id}",
                category="logic_implementation_status"
            )
            if status_data:
                # Cache locally
                self.implementations[logic_id] = status_data
                return status_data
        
        return None
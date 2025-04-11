"""
Configuration for different agent types in the multi-agent development system.

This module defines the configuration settings for each type of agent in the system,
including their default models, parameters, roles, permissions, interaction patterns,
and specialized tools.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import yaml
from pathlib import Path

# Configuration constants
DEFAULT_MODEL_PROVIDER = "anthropic"
DEFAULT_MODEL_NAME = "claude-3-haiku-20240307"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000

# Set up logging
logger = logging.getLogger(__name__)


class AgentTier(str, Enum):
    """Tiers of agents based on capabilities and resource usage."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AgentPermission(str, Enum):
    """Permission levels for agents."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    EXECUTE = "execute"
    ADMIN = "admin"


class AgentConfig:
    """Configuration manager for agent settings."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        env_prefix: str = "AGENT_",
        load_defaults: bool = True
    ):
        """Initialize the agent configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            env_prefix: Prefix for environment variables
            load_defaults: Whether to load default configurations
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.config = {}
        
        # Load configurations in order of precedence
        if load_defaults:
            self._load_default_config()
        
        if config_path:
            self._load_config_file(config_path)
        
        self._load_env_variables()
        
        logger.info(f"Agent configuration initialized with {len(self.config)} agent types")
    
    def _load_default_config(self) -> None:
        """Load default configuration for all agent types."""
        self.config = {
            # Project Management Agents
            "project_manager": {
                "description": "Coordinates all activities and interfaces with human stakeholders",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": "claude-3-sonnet-20240229",
                "temperature": 0.3,
                "max_tokens": 4000,
                "tier": AgentTier.EXPERT,
                "permissions": [AgentPermission.ADMIN],
                "tools": ["task_management", "scheduling", "documentation", "communication"],
                "can_delegate": True,
                "system_prompt": (
                    "You are the Project Manager agent, responsible for coordinating all activities "
                    "across the development team. You interface directly with human stakeholders, "
                    "gather requirements, set priorities, and ensure all tasks are properly assigned "
                    "and completed. You have a high-level understanding of the entire project and "
                    "make strategic decisions about resource allocation and workflow."
                )
            },
            
            "architecture_designer": {
                "description": "Designs system architecture and makes technology decisions",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": "claude-3-sonnet-20240229",
                "temperature": 0.2,
                "max_tokens": 4000,
                "tier": AgentTier.EXPERT,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["diagramming", "code_analysis", "documentation"],
                "can_delegate": False,
                "system_prompt": (
                    "You are the Architecture Designer agent, responsible for designing the overall "
                    "system architecture and making key technology decisions. You evaluate requirements, "
                    "consider scalability, security, and maintainability, and create architectural "
                    "diagrams and specifications. You make informed recommendations about frameworks, "
                    "libraries, and design patterns that best fit the project's needs."
                )
            },
            
            # Frontend Development Agents
            "ui_developer": {
                "description": "Creates UI components and implements designs",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.3,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["code_generation", "code_analysis", "ui_frameworks"],
                "specializations": ["react", "vue", "angular", "tailwind"],
                "system_prompt": (
                    "You are the UI Component Developer agent, responsible for creating UI components "
                    "and implementing visual designs. You translate design mockups and wireframes into "
                    "responsive, accessible, and cross-browser compatible UI components. You follow "
                    "best practices for reusability, performance, and design consistency."
                )
            },
            
            "frontend_logic": {
                "description": "Implements frontend business logic and state management",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["code_generation", "code_analysis", "debugging"],
                "specializations": ["state_management", "form_handling", "data_fetching"],
                "system_prompt": (
                    "You are the Frontend Logic Developer agent, responsible for implementing business "
                    "logic and state management on the frontend. You create maintainable and testable "
                    "code that handles user interactions, form validations, data transformations, and "
                    "application state. You implement efficient algorithms and follow best practices "
                    "for error handling and performance optimization."
                )
            },
            
            "frontend_integration": {
                "description": "Integrates frontend components with backend systems",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["api_integration", "authentication", "data_handling"],
                "specializations": ["rest_api", "graphql", "websockets"],
                "system_prompt": (
                    "You are the Frontend Integration Developer agent, responsible for connecting "
                    "frontend components to backend systems. You implement API calls, handle "
                    "authentication, manage data formats, and ensure proper error handling. You "
                    "optimize data fetching strategies and implement caching where appropriate. "
                    "You understand both frontend and backend constraints and create reliable "
                    "interfaces between them."
                )
            },
            
            # Backend Development Agents
            "api_developer": {
                "description": "Designs and implements API endpoints",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["code_generation", "api_design", "documentation"],
                "specializations": ["rest", "graphql", "websockets"],
                "system_prompt": (
                    "You are the API Developer agent, responsible for designing and implementing "
                    "API endpoints. You create clear, consistent, and well-documented interfaces "
                    "following REST or GraphQL principles. You implement proper authentication, "
                    "input validation, error handling, and performance optimizations. You design "
                    "endpoints that are easy to use, secure, and meet the needs of frontend clients."
                )
            },
            
            "database_designer": {
                "description": "Designs database schemas and optimizes queries",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.ADVANCED,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["database_design", "query_optimization", "data_modeling"],
                "specializations": ["sql", "nosql", "graph_databases"],
                "system_prompt": (
                    "You are the Database Designer agent, responsible for designing database schemas "
                    "and optimizing queries. You create efficient data models that balance performance, "
                    "flexibility, and data integrity. You optimize complex queries, implement proper "
                    "indexing strategies, and ensure data consistency and security. You understand "
                    "both relational and NoSQL database paradigms and choose appropriate solutions "
                    "for different data requirements."
                )
            },
            
            "backend_logic": {
                "description": "Implements business logic and application workflows",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["code_generation", "code_analysis", "debugging"],
                "specializations": ["business_rules", "data_processing", "workflows"],
                "system_prompt": (
                    "You are the Backend Logic Developer agent, responsible for implementing business "
                    "logic and application workflows. You translate business requirements into "
                    "maintainable and testable code that handles complex processes, calculations, "
                    "and data transformations. You design modular components with clear interfaces "
                    "and proper error handling. You optimize for correctness, performance, and "
                    "maintainability."
                )
            },
            
            # DevOps Engineers
            "infrastructure": {
                "description": "Designs and manages cloud infrastructure",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.ADVANCED,
                "permissions": [AgentPermission.ADMIN],
                "tools": ["infrastructure_as_code", "cloud_services", "monitoring"],
                "specializations": ["aws", "gcp", "azure", "kubernetes"],
                "system_prompt": (
                    "You are the Infrastructure Developer agent, responsible for designing and managing "
                    "cloud infrastructure. You create infrastructure-as-code templates, configure "
                    "cloud services, and design scalable and resilient architectures. You optimize "
                    "for performance, cost, and reliability. You implement proper monitoring, logging, "
                    "and alerting solutions. You understand cloud-native design principles and security "
                    "best practices."
                )
            },
            
            "deployment": {
                "description": "Manages CI/CD pipelines and deployment processes",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.EXECUTE],
                "tools": ["ci_cd", "deployment_automation", "release_management"],
                "specializations": ["jenkins", "github_actions", "gitlab_ci", "argocd"],
                "system_prompt": (
                    "You are the Deployment Specialist agent, responsible for managing CI/CD pipelines "
                    "and deployment processes. You automate build, test, and deployment workflows to "
                    "ensure reliable and consistent releases. You implement deployment strategies like "
                    "blue-green or canary deployments. You monitor deployment health and can roll back "
                    "changes when needed. You optimize for deployment speed, reliability, and minimal "
                    "downtime."
                )
            },
            
            "security": {
                "description": "Ensures security best practices and performs security reviews",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": "claude-3-sonnet-20240229",
                "temperature": 0.1,
                "max_tokens": 3000,
                "tier": AgentTier.EXPERT,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["security_scanning", "code_analysis", "vulnerability_assessment"],
                "specializations": ["application_security", "infrastructure_security", "compliance"],
                "system_prompt": (
                    "You are the Security Analyst agent, responsible for ensuring security best practices "
                    "and performing security reviews. You identify and address security vulnerabilities "
                    "in code, infrastructure, and deployment processes. You implement proper "
                    "authentication, authorization, data encryption, and secure communication. You "
                    "stay updated on security threats and compliance requirements. You educate other "
                    "team members on security best practices."
                )
            },
            
            # QA Engineers
            "code_reviewer": {
                "description": "Reviews code for quality, style, and best practices",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.1,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_ONLY],
                "tools": ["code_analysis", "static_analysis", "best_practices"],
                "specializations": ["frontend_review", "backend_review", "performance_review"],
                "system_prompt": (
                    "You are the Code Reviewer agent, responsible for reviewing code for quality, "
                    "style, and best practices. You identify bugs, security issues, and performance "
                    "problems. You ensure code follows project standards and design patterns. You "
                    "provide constructive feedback with clear explanations and improvement suggestions. "
                    "You have deep knowledge of language-specific best practices and common pitfalls."
                )
            },
            
            "test_developer": {
                "description": "Creates and maintains automated tests",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["test_generation", "code_analysis", "test_frameworks"],
                "specializations": ["unit_testing", "integration_testing", "e2e_testing"],
                "system_prompt": (
                    "You are the Test Developer agent, responsible for creating and maintaining "
                    "automated tests. You write comprehensive unit, integration, and end-to-end tests "
                    "that verify system behavior and catch regressions. You design testable code and "
                    "implement test fixtures, mocks, and assertions. You ensure tests are maintainable, "
                    "reliable, and provide good coverage. You implement CI-friendly test suites that "
                    "run efficiently."
                )
            },
            
            "ux_tester": {
                "description": "Evaluates user experience and accessibility",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.3,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_ONLY],
                "tools": ["accessibility_testing", "usability_analysis", "ui_guidelines"],
                "specializations": ["accessibility", "usability", "mobile_ux"],
                "system_prompt": (
                    "You are the UX Tester agent, responsible for evaluating user experience and "
                    "accessibility. You review interfaces for usability, consistency, and accessibility "
                    "compliance (WCAG). You identify potential user experience issues and suggest "
                    "improvements. You understand design principles, user psychology, and diverse user "
                    "needs. You ensure the application is intuitive, efficient, and accessible to all "
                    "users, including those with disabilities."
                )
            },
            
            # Specialized Roles
            "researcher": {
                "description": "Researches solutions, libraries, and best practices",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": "claude-3-sonnet-20240229",
                "temperature": 0.3,
                "max_tokens": 4000,
                "tier": AgentTier.ADVANCED,
                "permissions": [AgentPermission.READ_ONLY],
                "tools": ["information_retrieval", "technical_analysis", "trend_analysis"],
                "specializations": ["libraries", "frameworks", "academic_research"],
                "system_prompt": (
                    "You are the Research Specialist agent, responsible for researching solutions, "
                    "libraries, and best practices. You evaluate technologies, compare alternatives, "
                    "and make informed recommendations. You stay updated on the latest advancements "
                    "and industry trends. You research technical challenges and find innovative solutions. "
                    "You provide comprehensive, objective analysis that helps the team make the best "
                    "technology choices."
                )
            },
            
            "documentation": {
                "description": "Creates and maintains project documentation",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": DEFAULT_MODEL_NAME,
                "temperature": 0.3,
                "max_tokens": 3000,
                "tier": AgentTier.STANDARD,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["markdown", "documentation_generators", "diagramming"],
                "specializations": ["api_docs", "user_guides", "technical_docs"],
                "system_prompt": (
                    "You are the Documentation Writer agent, responsible for creating and maintaining "
                    "project documentation. You create clear, comprehensive, and well-structured "
                    "documentation for APIs, code, and user interfaces. You write both technical "
                    "documentation for developers and user-facing guides. You ensure documentation "
                    "stays updated as the project evolves. You use appropriate formats, examples, and "
                    "diagrams to make information accessible and useful."
                )
            },
            
            "human_interface": {
                "description": "Acts as an interface between the agent system and human team members",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": "claude-3-sonnet-20240229",
                "temperature": 0.5,
                "max_tokens": 4000,
                "tier": AgentTier.EXPERT,
                "permissions": [AgentPermission.ADMIN],
                "tools": ["communication", "task_management", "feedback_collection"],
                "system_prompt": (
                    "You are the Human Interface agent, responsible for facilitating communication "
                    "between the agent system and human team members. You present information in a "
                    "clear, actionable format for human review. You collect and process human feedback "
                    "to improve agent outputs. You translate human instructions into specific tasks "
                    "for other agents. You identify when a task needs human input and how to most "
                    "effectively gather that input."
                )
            },
            
            # Medical domain-specific agents
            "medical_content": {
                "description": "Creates and reviews medical educational content",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": "claude-3-opus-20240229",
                "temperature": 0.2,
                "max_tokens": 4000,
                "tier": AgentTier.EXPERT,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["content_generation", "fact_checking", "medical_research"],
                "specializations": ["anatomy", "physiology", "pharmacology", "pathology"],
                "system_prompt": (
                    "You are the Medical Content agent, responsible for creating and reviewing medical "
                    "educational content. You ensure all medical information is accurate, current, and "
                    "follows evidence-based medicine. You create engaging learning materials suitable "
                    "for medical students at different levels. You structure complex medical concepts "
                    "in clear, pedagogically sound ways. You cite authoritative sources and follow "
                    "medical terminology standards."
                )
            },
            
            "assessment_developer": {
                "description": "Creates medical knowledge assessments and quizzes",
                "model_provider": DEFAULT_MODEL_PROVIDER,
                "model_name": "claude-3-sonnet-20240229",
                "temperature": 0.3,
                "max_tokens": 3000,
                "tier": AgentTier.ADVANCED,
                "permissions": [AgentPermission.READ_WRITE],
                "tools": ["question_generation", "assessment_design", "psychometrics"],
                "specializations": ["mcq", "case_studies", "clinical_reasoning"],
                "system_prompt": (
                    "You are the Assessment Developer agent, responsible for creating medical knowledge "
                    "assessments and quizzes. You design questions at appropriate difficulty levels that "
                    "test understanding rather than mere recall. You create realistic clinical scenarios "
                    "and case studies. You ensure questions have clear, unambiguous answers with "
                    "educational explanations. You balance assessment difficulty and design reliable "
                    "and valid evaluation instruments."
                )
            }
        }
    
    def _load_config_file(self, config_path: str) -> None:
        """Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return
            
            if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                with open(path, 'r') as file:
                    config = yaml.safe_load(file)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as file:
                    config = json.load(file)
            else:
                logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return
            
            # Merge with existing config
            for agent_type, agent_config in config.items():
                if agent_type in self.config:
                    self.config[agent_type].update(agent_config)
                else:
                    self.config[agent_type] = agent_config
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration file {config_path}: {str(e)}")
    
    def _load_env_variables(self) -> None:
        """Load configuration from environment variables."""
        # Format: AGENT_<TYPE>_<PARAM>=<VALUE>
        # e.g., AGENT_PROJECT_MANAGER_MODEL_NAME=claude-3-opus-20240229
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                parts = key[len(self.env_prefix):].lower().split('_', 1)
                if len(parts) < 2:
                    continue
                
                agent_type = parts[0]
                param = parts[1]
                
                # Convert from environment format to config format
                agent_type = agent_type.lower()
                param = param.lower()
                
                # Create agent config if it doesn't exist
                if agent_type not in self.config:
                    self.config[agent_type] = {}
                
                # Try to convert value to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                    value = float(value)
                
                self.config[agent_type][param] = value
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for an agent type.
        
        Args:
            agent_type: Agent type identifier
            
        Returns:
            Configuration dictionary for the agent type
        """
        if agent_type in self.config:
            return self.config[agent_type]
        else:
            logger.warning(f"No configuration found for agent type: {agent_type}")
            return {}
    
    def get_all_agent_types(self) -> List[str]:
        """Get all configured agent types.
        
        Returns:
            List of agent type identifiers
        """
        return list(self.config.keys())
    
    def get_agents_by_tier(self, tier: Union[AgentTier, str]) -> List[str]:
        """Get all agent types of a specific tier.
        
        Args:
            tier: Agent tier to filter by
            
        Returns:
            List of agent type identifiers
        """
        if isinstance(tier, str):
            tier = AgentTier(tier)
        
        return [
            agent_type for agent_type, config in self.config.items()
            if config.get("tier") == tier
        ]
    
    def get_agent_prompt(self, agent_type: str) -> str:
        """Get the system prompt for an agent type.
        
        Args:
            agent_type: Agent type identifier
            
        Returns:
            System prompt for the agent
        """
        config = self.get_agent_config(agent_type)
        return config.get("system_prompt", f"You are a {agent_type} agent.")
    
    def get_model_config(self, agent_type: str) -> Tuple[str, str, float, int]:
        """Get model configuration for an agent type.
        
        Args:
            agent_type: Agent type identifier
            
        Returns:
            Tuple of (model_provider, model_name, temperature, max_tokens)
        """
        config = self.get_agent_config(agent_type)
        return (
            config.get("model_provider", DEFAULT_MODEL_PROVIDER),
            config.get("model_name", DEFAULT_MODEL_NAME),
            config.get("temperature", DEFAULT_TEMPERATURE),
            config.get("max_tokens", DEFAULT_MAX_TOKENS)
        )
    
    def get_agent_tools(self, agent_type: str) -> List[str]:
        """Get the tools available to an agent type.
        
        Args:
            agent_type: Agent type identifier
            
        Returns:
            List of tool identifiers
        """
        config = self.get_agent_config(agent_type)
        return config.get("tools", [])
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save the current configuration to a file.
        
        Args:
            path: Path to save the configuration to (defaults to config_path)
        """
        save_path = path or self.config_path
        if not save_path:
            logger.warning("No path specified for saving configuration")
            return
        
        try:
            path = Path(save_path)
            os.makedirs(path.parent, exist_ok=True)
            
            if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                with open(path, 'w') as file:
                    yaml.dump(self.config, file, default_flow_style=False)
            elif path.suffix.lower() == '.json':
                with open(path, 'w') as file:
                    json.dump(self.config, file, indent=2)
            else:
                logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return
            
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {save_path}: {str(e)}")
    
    def update_agent_config(self, agent_type: str, updates: Dict[str, Any]) -> None:
        """Update configuration for an agent type.
        
        Args:
            agent_type: Agent type identifier
            updates: Dictionary of updates to apply
        """
        if agent_type not in self.config:
            self.config[agent_type] = {}
        
        self.config[agent_type].update(updates)
        logger.info(f"Updated configuration for agent type: {agent_type}")


# Helper function to get agent configuration
def get_default_agent_config(agent_type: str) -> Dict[str, Any]:
    """Get default configuration for an agent type.
    
    Args:
        agent_type: Agent type identifier
        
    Returns:
        Default configuration dictionary
    """
    config = AgentConfig(load_defaults=True)
    return config.get_agent_config(agent_type)
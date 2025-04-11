"""
Overall system configuration for the multi-agent development system.

This module provides system-wide configuration settings, including infrastructure settings,
communication protocols, resource limits, security policies, logging configuration,
and integration points with external services.
"""

import os
import json
import logging
import platform
import socket
import uuid
import yaml
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import multiprocessing

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResourceLimitStrategy(str, Enum):
   """Strategies for handling resource limits."""
   QUEUE = "queue"  # Queue tasks when resources are limited
   REJECT = "reject"  # Reject new tasks when resources are limited
   ADAPT = "adapt"  # Dynamically adapt to available resources
   PRIORITIZE = "prioritize"  # Prioritize tasks based on importance


class SchedulingMode(str, Enum):
   """Task scheduling modes."""
   ROUND_ROBIN = "round_robin"  # Distribute tasks evenly across agents
   PRIORITY_BASED = "priority_based"  # Schedule based on task priorities
   AGENT_SPECIALIZATION = "agent_specialization"  # Match tasks to agent specializations
   LOAD_BALANCED = "load_balanced"  # Balance load across available agents


class CommunicationProtocol(str, Enum):
   """Communication protocols for agent interaction."""
   DIRECT = "direct"  # Direct communication between agents
   ORCHESTRATED = "orchestrated"  # Communication through central orchestrator
   EVENT_BASED = "event_based"  # Event-driven communication
   HYBRID = "hybrid"  # Hybrid approach combining multiple protocols


class SecurityLevel(str, Enum):
   """Security levels for the system."""
   BASIC = "basic"  # Basic security controls
   STANDARD = "standard"  # Standard security controls for most use cases
   ENHANCED = "enhanced"  # Enhanced security for sensitive applications
   STRICT = "strict"  # Strict security for high-security environments


class SystemConfig:
   """System-wide configuration manager."""
   
   def __init__(
       self,
       config_path: Optional[str] = None,
       env_prefix: str = "AGENT_SYSTEM_",
       load_defaults: bool = True,
       instance_id: Optional[str] = None
   ):
       """Initialize the system configuration manager.
       
       Args:
           config_path: Path to configuration file (YAML or JSON)
           env_prefix: Prefix for environment variables
           load_defaults: Whether to load default configurations
           instance_id: Unique identifier for this system instance
       """
       self.config_path = config_path
       self.env_prefix = env_prefix
       self.instance_id = instance_id or str(uuid.uuid4())
       self.config = {}
       
       # Load configurations in order of precedence
       if load_defaults:
           self._load_default_config()
       
       if config_path:
           self._load_config_file(config_path)
       
       self._load_env_variables()
       
       # Determine host system capabilities
       self._detect_system_capabilities()
       
       logger.info(f"System configuration initialized (instance: {self.instance_id})")
   
   def _load_default_config(self) -> None:
       """Load default system configuration."""
       # Determine reasonable defaults based on system 
       cpu_count = multiprocessing.cpu_count()
       memory_gb = self._get_system_memory_gb()
       
       self.config = {
           # System identification
           "system": {
               "name": "Multi-Agent Development System",
               "version": "0.1.0",
               "instance_id": self.instance_id,
               "environment": os.environ.get("DEPLOYMENT_ENVIRONMENT", "development"),
           },
           
           # Resource limits
           "resources": {
               "max_concurrent_agents": max(2, cpu_count - 1),
               "max_concurrent_tasks": cpu_count * 4,
               "max_memory_usage_gb": max(1, memory_gb // 2),
               "max_storage_usage_gb": 10,
               "limit_strategy": ResourceLimitStrategy.QUEUE.value,
               "agent_timeout_seconds": 600,  # 10 minutes
               "task_timeout_seconds": 300,   # 5 minutes
           },
           
           # Task scheduling
           "scheduling": {
               "mode": SchedulingMode.AGENT_SPECIALIZATION.value,
               "priority_levels": 5,
               "default_priority": 2,
               "preemption_enabled": False,
               "max_retries": 3,
               "retry_delay_seconds": 5,
           },
           
           # Communication
           "communication": {
               "protocol": CommunicationProtocol.ORCHESTRATED.value,
               "message_format": "json",
               "max_message_size_mb": 10,
               "compression_enabled": True,
               "encryption_enabled": True,
               "keep_alive_interval_seconds": 30,
           },
           
           # Memory and storage
           "memory": {
               "shared_memory_enabled": True,
               "memory_persistence": True,
               "max_context_items": 1000,
               "context_item_ttl_seconds": 86400,  # 24 hours
               "memory_backends": ["local"],
               "vector_store_enabled": True,
           },
           
           # Security
           "security": {
               "level": SecurityLevel.STANDARD.value,
               "authentication_required": True,
               "authorization_enabled": True,
               "token_expiration_seconds": 3600,  # 1 hour
               "request_validation": True,
               "sensitive_data_masking": True,
               "allowed_model_providers": ["anthropic", "openai", "google"],
           },
           
           # Logging and monitoring
           "logging": {
               "level": "INFO",
               "file_logging_enabled": True,
               "log_directory": "logs",
               "max_log_file_size_mb": 10,
               "max_log_files": 5,
               "performance_tracking": True,
               "request_tracing": True,
           },
           
           # API integration
           "api": {
               "enabled": True,
               "host": "0.0.0.0",
               "port": 8000,
               "base_path": "/api/v1",
               "cors_enabled": True,
               "cors_origins": ["*"],
               "rate_limiting_enabled": True,
               "max_requests_per_minute": 60,
           },
           
           # Model providers
           "model_providers": {
               "anthropic": {
                   "enabled": True,
                   "api_key_env_var": "ANTHROPIC_API_KEY",
                   "default_model": "claude-3-haiku-20240307",
                   "timeout_seconds": 30,
                   "retry_attempts": 2,
               },
               "openai": {
                   "enabled": True,
                   "api_key_env_var": "OPENAI_API_KEY",
                   "default_model": "gpt-4o",
                   "timeout_seconds": 30,
                   "retry_attempts": 2,
               },
               "google": {
                   "enabled": True,
                   "api_key_env_var": "GOOGLE_API_KEY",
                   "default_model": "gemini-1.5-pro",
                   "timeout_seconds": 30,
                   "retry_attempts": 2,
               },
           },
           
           # Human oversight
           "human_oversight": {
               "enabled": True,
               "approval_required_for_critical": True,
               "approval_timeout_seconds": 3600,  # 1 hour
               "notification_channels": ["console"],
               "feedback_collection": True,
           },
           
           # Paths and directories
           "paths": {
               "base_directory": os.path.abspath("."),
               "data_directory": "data",
               "cache_directory": "cache",
               "output_directory": "output",
               "plugin_directory": "plugins",
           },
           
           # Medical domain-specific
           "medical": {
               "content_verification_enabled": True,
               "citation_required": True,
               "terminology_standards": ["SNOMED CT", "ICD-10"],
               "educational_levels": ["preclinical", "clinical", "postgraduate"],
               "default_educational_level": "clinical",
           },
       }
   
   def _get_system_memory_gb(self) -> int:
       """Get the total system memory in GB.
       
       Returns:
           Total memory in GB
       """
       try:
           import psutil
           return psutil.virtual_memory().total // (1024 ** 3)
       except (ImportError, Exception):
           # Fallback to conservative default if psutil isn't available
           return 4
   
   def _detect_system_capabilities(self) -> None:
       """Detect host system capabilities and update configuration."""
       system_info = {
           "os": platform.system(),
           "os_version": platform.version(),
           "architecture": platform.architecture()[0],
           "processor": platform.processor(),
           "cpu_count": multiprocessing.cpu_count(),
           "hostname": socket.gethostname(),
           "python_version": platform.python_version(),
       }
       
       try:
           import psutil
           memory = psutil.virtual_memory()
           system_info["memory_total_gb"] = memory.total // (1024 ** 3)
           system_info["memory_available_gb"] = memory.available // (1024 ** 3)
           
           disk = psutil.disk_usage('/')
           system_info["disk_total_gb"] = disk.total // (1024 ** 3)
           system_info["disk_free_gb"] = disk.free // (1024 ** 3)
       except ImportError:
           pass
       
       # Try to detect if running in a container
       system_info["in_container"] = os.path.exists("/.dockerenv")
       
       # Try to detect GPU availability
       system_info["gpu_available"] = self._check_gpu_availability()
       
       # Update configuration based on detected capabilities
       self.config["system"]["capabilities"] = system_info
       
       # Adjust resource limits based on detected capabilities
       if "memory_available_gb" in system_info:
           # Use at most 75% of available memory
           self.config["resources"]["max_memory_usage_gb"] = max(
               1, int(system_info["memory_available_gb"] * 0.75)
           )
       
       if "in_container" in system_info and system_info["in_container"]:
           # More conservative limits in containerized environments
           self.config["resources"]["max_concurrent_agents"] = max(
               1, self.config["resources"]["max_concurrent_agents"] // 2
           )
   
   def _check_gpu_availability(self) -> bool:
       """Check if GPU is available for machine learning tasks.
       
       Returns:
           True if GPU is available, False otherwise
       """
       try:
           # Try to import pytorch and check for CUDA
           import torch
           return torch.cuda.is_available()
       except ImportError:
           try:
               # Try to check for tensorflow GPU
               import tensorflow as tf
               gpus = tf.config.list_physical_devices('GPU')
               return len(gpus) > 0
           except ImportError:
               # If neither pytorch nor tensorflow is available, check for NVIDIA GPUs directly
               try:
                   nvidia_smi_output = os.popen('nvidia-smi -L').read()
                   return 'GPU' in nvidia_smi_output
               except:
                   return False
   
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
                   file_config = yaml.safe_load(file)
           elif path.suffix.lower() == '.json':
               with open(path, 'r') as file:
                   file_config = json.load(file)
           else:
               logger.warning(f"Unsupported configuration file format: {path.suffix}")
               return
           
           # Merge with existing config (deep merge)
           self._deep_merge(self.config, file_config)
           
           logger.info(f"Loaded configuration from {config_path}")
       except Exception as e:
           logger.error(f"Error loading configuration file {config_path}: {str(e)}")
   
   def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
       """Recursively merge source dict into target dict.
       
       Args:
           target: Target dictionary to merge into
           source: Source dictionary to merge from
       """
       for key, value in source.items():
           if key in target and isinstance(target[key], dict) and isinstance(value, dict):
               self._deep_merge(target[key], value)
           else:
               target[key] = value
   
   def _load_env_variables(self) -> None:
       """Load configuration from environment variables."""
       # Format: AGENT_SYSTEM_SECTION_SUBSECTION_KEY=VALUE
       for key, value in os.environ.items():
           if key.startswith(self.env_prefix):
               parts = key[len(self.env_prefix):].lower().split('_')
               if len(parts) < 2:
                   continue
               
               # Navigate to the correct config section
               config_ref = self.config
               for part in parts[:-1]:
                   if part not in config_ref:
                       config_ref[part] = {}
                   config_ref = config_ref[part]
               
               # Set the value
               param = parts[-1]
               
               # Try to convert value to appropriate type
               if value.lower() in ['true', 'false']:
                   value = value.lower() == 'true'
               elif value.isdigit():
                   value = int(value)
               elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                   value = float(value)
               
               config_ref[param] = value
   
   def get_config(self, *path: str, default: Any = None) -> Any:
       """Get a configuration value by path.
       
       Args:
           *path: Path segments to the configuration value
           default: Default value if path doesn't exist
           
       Returns:
           Configuration value or default
       """
       config_ref = self.config
       for part in path:
           if not isinstance(config_ref, dict) or part not in config_ref:
               return default
           config_ref = config_ref[part]
       return config_ref
   
   def set_config(self, value: Any, *path: str) -> None:
       """Set a configuration value by path.
       
       Args:
           value: Value to set
           *path: Path segments to the configuration value
       """
       if not path:
           logger.warning("Cannot set configuration without a path")
           return
       
       config_ref = self.config
       for part in path[:-1]:
           if part not in config_ref:
               config_ref[part] = {}
           config_ref = config_ref[part]
       
       config_ref[path[-1]] = value
   
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
   
   def setup_logging(self) -> None:
       """Configure logging based on settings."""
       log_level_name = self.get_config("logging", "level", default="INFO")
       log_level = getattr(logging, log_level_name.upper(), logging.INFO)
       
       # Configure root logger
       logging.basicConfig(
           level=log_level,
           format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
       )
       
       # Add file handler if enabled
       if self.get_config("logging", "file_logging_enabled", default=True):
           log_dir = self.get_config("logging", "log_directory", default="logs")
           os.makedirs(log_dir, exist_ok=True)
           
           log_path = os.path.join(log_dir, f"system_{self.instance_id}.log")
           
           # Set up rotating file handler
           from logging.handlers import RotatingFileHandler
           max_bytes = self.get_config("logging", "max_log_file_size_mb", default=10) * 1024 * 1024
           backup_count = self.get_config("logging", "max_log_files", default=5)
           
           file_handler = RotatingFileHandler(
               log_path, 
               maxBytes=max_bytes, 
               backupCount=backup_count
           )
           file_handler.setLevel(log_level)
           file_handler.setFormatter(logging.Formatter(
               "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
           ))
           
           # Add to root logger
           logging.getLogger("").addHandler(file_handler)
           
           logger.info(f"File logging configured to {log_path}")
   
   def get_api_config(self) -> Dict[str, Any]:
       """Get API server configuration.
       
       Returns:
           API configuration dictionary
       """
       return self.get_config("api", default={})
   
   def get_model_config(self, provider: str) -> Dict[str, Any]:
       """Get configuration for a model provider.
       
       Args:
           provider: Name of the model provider
           
       Returns:
           Provider configuration dictionary
       """
       return self.get_config("model_providers", provider, default={})
   
   def get_resource_limits(self) -> Dict[str, Any]:
       """Get resource limit configuration.
       
       Returns:
           Resource limits dictionary
       """
       return self.get_config("resources", default={})
   
   def get_scheduling_config(self) -> Dict[str, Any]:
       """Get task scheduling configuration.
       
       Returns:
           Scheduling configuration dictionary
       """
       return self.get_config("scheduling", default={})
   
   def is_provider_enabled(self, provider: str) -> bool:
       """Check if a model provider is enabled.
       
       Args:
           provider: Name of the model provider
           
       Returns:
           True if provider is enabled, False otherwise
       """
       return self.get_config("model_providers", provider, "enabled", default=False)
   
   def get_api_key(self, provider: str) -> Optional[str]:
       """Get API key for a model provider.
       
       Args:
           provider: Name of the model provider
           
       Returns:
           API key if available, None otherwise
       """
       env_var = self.get_config("model_providers", provider, "api_key_env_var")
       if not env_var:
           return None
       
       return os.environ.get(env_var)
   
   def is_running_in_production(self) -> bool:
       """Check if system is running in production environment.
       
       Returns:
           True if in production, False otherwise
       """
       return self.get_config("system", "environment") == "production"
   
   def should_require_human_approval(self, operation: str) -> bool:
       """Check if an operation requires human approval.
       
       Args:
           operation: Type of operation
           
       Returns:
           True if human approval is required, False otherwise
       """
       if not self.get_config("human_oversight", "enabled", default=True):
           return False
       
       if operation == "critical" or operation == "high_risk":
           return self.get_config("human_oversight", "approval_required_for_critical", default=True)
       
       return False
   
   def get_system_info(self) -> Dict[str, Any]:
       """Get system information.
       
       Returns:
           System information dictionary
       """
       system_info = {
           "name": self.get_config("system", "name", default="Multi-Agent System"),
           "version": self.get_config("system", "version", default="0.1.0"),
           "instance_id": self.instance_id,
           "environment": self.get_config("system", "environment", default="development"),
           "uptime_seconds": 0,  # Would be calculated at runtime
           "capabilities": self.get_config("system", "capabilities", default={})
       }
       return system_info
   
   def get_security_config(self) -> Dict[str, Any]:
       """Get security configuration.
       
       Returns:
           Security configuration dictionary
       """
       return self.get_config("security", default={})
   
   def generate_instance_id(self) -> str:
       """Generate a unique instance ID.
       
       Returns:
           Unique instance ID
       """
       return str(uuid.uuid4())
   
   def get_all_config(self) -> Dict[str, Any]:
       """Get the entire configuration.
       
       Returns:
           Complete configuration dictionary
       """
       return self.config
   
   def is_feature_enabled(self, feature_path: str) -> bool:
       """Check if a feature is enabled in the configuration.
       
       Args:
           feature_path: Dot-separated path to the feature (e.g., "api.enabled")
           
       Returns:
           True if feature is enabled, False otherwise
       """
       path_parts = feature_path.split('.')
       value = self.get_config(*path_parts, default=False)
       
       if isinstance(value, bool):
           return value
       elif isinstance(value, str):
           return value.lower() in ['true', 'yes', 'enabled', '1']
       elif isinstance(value, int):
           return value > 0
       
       return False
   
   def get_paths(self) -> Dict[str, str]:
       """Get system paths.
       
       Returns:
           Dictionary of system paths
       """
       base_dir = self.get_config("paths", "base_directory", default=os.path.abspath("."))
       
       # Ensure all paths are absolute
       paths = {}
       for key, rel_path in self.get_config("paths", default={}).items():
           if key == "base_directory":
               paths[key] = rel_path
           else:
               paths[key] = os.path.join(base_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
       
       return paths


# Global system configuration instance
_system_config_instance = None

def get_system_config(config_path: Optional[str] = None) -> SystemConfig:
   """Get or create the global system configuration instance.
   
   Args:
       config_path: Optional path to configuration file
       
   Returns:
       SystemConfig instance
   """
   global _system_config_instance
   
   if _system_config_instance is None:
       _system_config_instance = SystemConfig(config_path=config_path)
   
   return _system_config_instance


# Utility functions for common configuration needs
def get_model_api_key(provider: str) -> Optional[str]:
   """Get API key for a model provider.
   
   Args:
       provider: Name of the model provider
       
   Returns:
       API key if available, None otherwise
   """
   return get_system_config().get_api_key(provider)


def get_max_concurrent_agents() -> int:
   """Get maximum number of concurrent agents.
   
   Returns:
       Maximum number of concurrent agents
   """
   return get_system_config().get_config("resources", "max_concurrent_agents", default=4)


def get_default_model_for_provider(provider: str) -> str:
   """Get default model for a provider.
   
   Args:
       provider: Name of the model provider
       
   Returns:
       Default model name
   """
   if provider == "anthropic":
       return get_system_config().get_config(
           "model_providers", "anthropic", "default_model", 
           default="claude-3-haiku-20240307"
       )
   elif provider == "openai":
       return get_system_config().get_config(
           "model_providers", "openai", "default_model", 
           default="gpt-4o"
       )
   elif provider == "google":
       return get_system_config().get_config(
           "model_providers", "google", "default_model", 
           default="gemini-1.5-pro"
       )
   else:
       logger.warning(f"Unknown model provider: {provider}")
       return ""


def is_production() -> bool:
   """Check if system is running in production environment.
   
   Returns:
       True if in production, False otherwise
   """
   return get_system_config().is_running_in_production()
"""
Multi-Agent Development System - Main Entry Point

This module serves as the entry point for the multi-agent development system,
responsible for initializing components, connecting services, and starting
the agent network. It handles command-line arguments, configuration loading,
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import time
import uuid
import traceback

# System components
from config.system_config import SystemConfig, get_system_config
from config.agent_config import AgentConfig
from memory.shared_memory import SharedMemory
from memory.context_store import ContextStore, ContextType
from agents.base_agent import BaseAgent, ModelProvider, AgentRole
from tools.code_tools import CodeTool
from human_interface.review_interface import ReviewInterface
from human_interface.feedback_processor import FeedbackProcessor
from orchestrator.workflow_engine import WorkflowEngine, LoggingWorkflowListener
from orchestrator.task_scheduler import TaskScheduler

# Import specific agents
from agents.manager_agent import ProjectManagerAgent
from agents.designer_agent import ArchitectureDesignerAgent
from agents.frontend_agents import (
   UIComponentDeveloper,
   FrontendLogicDeveloper,
   FrontendIntegrationDeveloper
)
from agents.backend_agents import (
   APIDeveloper,
   DatabaseDesigner,
   BackendLogicDeveloper
)
from agents.devops_agents import (
   InfrastructureDeveloper,
   DeploymentSpecialist,
   SecurityAnalyst
)
from agents.qa_agents import (
   CodeReviewer,
   TestDeveloper,
   UXTester
)
from agents.research_agent import ResearchSpecialistAgent
from agents.doc_agent import DocumentationWriterAgent
#from agents.human_interface_agent import HumanInterfaceAgent

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
   handlers=[
       logging.StreamHandler(),
       logging.FileHandler("multi_agent_dev.log")
   ]
)
logger = logging.getLogger(__name__)

# Version information
VERSION = "0.1.0"
BUILD_DATE = "2025-04-10"

# Banner for CLI
BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║  ███╗   ███╗██╗   ██╗██╗  ████████╗██╗      █████╗  ██████╗ ███████╗ ║
║  ████╗ ████║██║   ██║██║  ╚══██╔══╝██║     ██╔══██╗██╔════╝ ██╔════╝ ║
║  ██╔████╔██║██║   ██║██║     ██║   ██║     ███████║██║  ███╗█████╗   ║
║  ██║╚██╔╝██║██║   ██║██║     ██║   ██║     ██╔══██║██║   ██║██╔══╝   ║
║  ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ███████╗██║  ██║╚██████╔╝███████╗ ║
║  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ║
║                                                                       ║
║                     AI Development Team System                        ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


class MultiAgentSystem:
   """Main class for the multi-agent development system."""
   
   def __init__(
       self,
       config_path: Optional[str] = None,
       data_dir: Optional[str] = None,
       debug: bool = False,
       agent_types: Optional[List[str]] = None,
       model_provider: Optional[str] = None
   ):
       """Initialize the multi-agent system.
       
       Args:
           config_path: Path to configuration file
           data_dir: Path to data directory
           debug: Whether to enable debug mode
           agent_types: List of agent types to initialize
           model_provider: Default model provider to use
       """
       self.start_time = time.time()
       self.instance_id = str(uuid.uuid4())
       self.debug = debug
       self.running = False
       self.shutdown_requested = False
       
       # Set log level
       if debug:
           logging.getLogger().setLevel(logging.DEBUG)
           logger.debug("Debug mode enabled")
       
       # Load configuration
       self.system_config = get_system_config(config_path)
       self.agent_config = AgentConfig(config_path)
       
       # Set data directory
       self.data_dir = data_dir or self.system_config.get_config(
           "paths", "data_directory", default=os.path.join(os.getcwd(), "data")
       )
       os.makedirs(self.data_dir, exist_ok=True)
       
       # Set model provider
       self.model_provider = model_provider or self.system_config.get_config(
           "model_providers", "default", default="anthropic"
       )
       
       # Set default agent types
       self.agent_types = agent_types or [
           "project_manager",
           "architecture_designer",
           "ui_developer",
           "frontend_logic",
           "frontend_integration",
           "api_developer",
           "database_designer",
           "backend_logic",
           "infrastructure",
           "deployment",
           "security",
           "code_reviewer",
           "test_developer",
           "ux_tester",
           "researcher",
           "documentation",
           "human_interface"
       ]
       
       # Initialize components
       self.shared_memory = None
       self.context_store = None
       self.task_scheduler = None
       self.workflow_engine = None
       self.review_interface = None
       self.feedback_processor = None
       self.code_tools = None
       self.agent_registry = {}
       
       logger.info(f"Multi-Agent System initialized (instance: {self.instance_id})")
   
   async def initialize(self) -> None:
       """Initialize system components and agents."""
       try:
           # Create component paths
           shared_memory_path = os.path.join(self.data_dir, "shared_memory")
           context_store_path = os.path.join(self.data_dir, "context_store")
           workflow_store_path = os.path.join(self.data_dir, "workflows")
           task_storage_path = os.path.join(self.data_dir, "tasks.json")
           
           os.makedirs(shared_memory_path, exist_ok=True)
           os.makedirs(context_store_path, exist_ok=True)
           os.makedirs(workflow_store_path, exist_ok=True)
           
           # Initialize memory and context components
           logger.info("Initializing memory and context components...")
           
           self.shared_memory = SharedMemory(
               storage_path=shared_memory_path,
               #enable_locking=True
           )
           
           self.context_store = ContextStore(
               storage_backend="filesystem",
               storage_path=context_store_path,
               compression=True
           )
           
           # Initialize tools
           logger.info("Initializing tools...")
           
           self.code_tools = CodeTool(
               working_dir=self.data_dir,
               temp_dir=os.path.join(self.data_dir, "temp")
           )
           
           # Initialize orchestration components
           logger.info("Initializing orchestration components...")
           
           self.task_scheduler = TaskScheduler(
               agent_registry=self.agent_registry,
               shared_memory=self.shared_memory,  
           )
           
           workflow_listeners = [LoggingWorkflowListener()]
           
           self.workflow_engine = WorkflowEngine(
               task_scheduler=self.task_scheduler,
               agent_registry=self.agent_registry,
               shared_memory=self.shared_memory,
               listeners=workflow_listeners,
               workflow_store_path=workflow_store_path,
               #enable_locking=True
           )
           
           # Initialize human interface components
           logger.info("Initializing human interface components...")
           
           self.review_interface = ReviewInterface(
               shared_memory=self.shared_memory
           )
           
           self.feedback_processor = FeedbackProcessor(
               agent_registry=self.agent_registry,
               shared_memory=self.shared_memory,
               task_scheduler=self.task_scheduler
           )
           await self.feedback_processor.load_stored_metrics()
           # Initialize agents
           logger.info("Initializing agents...")
           await self._initialize_agents()
           
           # Start workflow engine
           self.workflow_engine.start()
           
           self.running = True
           logger.info("System initialization complete")
           
       except Exception as e:
           logger.error(f"Error during system initialization: {str(e)}")
           traceback.print_exc()
           raise
   
   async def _initialize_agents(self) -> None:
       """Initialize all agents."""
       # Convert model provider string to enum
       model_provider = ModelProvider(self.model_provider)
       
       for agent_type in self.agent_types:
           try:
               # Get agent configuration
               agent_config = self.agent_config.get_agent_config(agent_type)
               if not agent_config:
                   logger.warning(f"No configuration found for agent type: {agent_type}")
                   continue
               
               # Get model configuration
               provider_str, model_name, temperature, max_tokens = self.agent_config.get_model_config(agent_type)
               # Override with system default if specified
               if self.model_provider:
                   provider_str = self.model_provider
               
               provider = ModelProvider(provider_str)
               
               # Create the appropriate agent based on type
               agent = None
               
               if agent_type == "project_manager":
                   agent = ProjectManagerAgent(
                       name=f"Project Manager",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       task_scheduler=self.task_scheduler,
                       workflow_engine=self.workflow_engine
                   )
               
               elif agent_type == "architecture_designer":
                   agent = ArchitectureDesignerAgent(
                       name=f"Architecture Designer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "ui_developer":
                   agent = UIComponentDeveloper(
                       name=f"UI Component Developer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "frontend_logic":
                   agent = FrontendLogicDeveloper(
                       name=f"Frontend Logic Developer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "frontend_integration":
                   agent = FrontendIntegrationDeveloper(
                       name=f"Frontend Integration Developer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "api_developer":
                   agent = APIDeveloper(
                       name=f"API Developer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "database_designer":
                   agent = DatabaseDesigner(
                       name=f"Database Designer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "backend_logic":
                   agent = BackendLogicDeveloper(
                       name=f"Backend Logic Developer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "infrastructure":
                   agent = InfrastructureDeveloper(
                       name=f"Infrastructure Developer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "deployment":
                   agent = DeploymentSpecialist(
                       name=f"Deployment Specialist",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "security":
                   agent = SecurityAnalyst(
                       name=f"Security Analyst",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "code_reviewer":
                   agent = CodeReviewer(
                       name=f"Code Reviewer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "test_developer":
                   agent = TestDeveloper(
                       name=f"Test Developer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               elif agent_type == "ux_tester":
                   agent = UXTester(
                       name=f"UX Tester",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory
                   )
               
               elif agent_type == "researcher":
                   agent = ResearchSpecialistAgent(
                       name=f"Research Specialist",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory
                   )
               
               elif agent_type == "documentation":
                   agent = DocumentationWriterAgent(
                       name=f"Documentation Writer",
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       code_tools=self.code_tools
                   )
               
               else:
                   # Default to base agent for unknown types
                   agent = BaseAgent(
                       name=f"{agent_type.replace('_', ' ').title()}",
                       agent_type=agent_type,
                       model_provider=provider,
                       model_name=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       shared_memory=self.shared_memory,
                       system_prompt=agent_config.get("system_prompt")
                   )
               
               # Register the agent
               self.agent_registry[agent.state.agent_id] = agent
               
               # Store agent info in shared memory
               if self.shared_memory:
                   await self.shared_memory.store(
                       key=agent.state.agent_id,
                       value={
                           "id": agent.state.agent_id,
                           "name": agent.name,
                           "type": agent.state.agent_type,
                           "model": f"{provider.value}/{model_name}"
                       },
                       category="agents"
                   )
               
               logger.info(f"Initialized agent: {agent.name} ({agent.state.agent_type})")
               
           except Exception as e:
               logger.error(f"Error initializing agent {agent_type}: {str(e)}")
               traceback.print_exc()
   
   async def shutdown(self) -> None:
       """Gracefully shut down the system."""
       if not self.running:
           return
       
       self.running = False
       logger.info("Shutting down the system...")
       
       # Stop workflow engine
       if self.workflow_engine:
           self.workflow_engine.stop()
       
       # Save task state
       if self.task_scheduler:
           # Save any pending tasks
           pass
       
       # Close shared memory
       if self.shared_memory:
           await self.shared_memory.close()
       
       # Close context store
       if self.context_store:
           # Check if context_store has a close method
        if hasattr(self.context_store, 'close'):
            await self.context_store.close()
        if hasattr(self.context_store, 'clear_all'):
            await self.context_store.clear_all()
        else:
            # No close method found, just log it
            logger.info("No close method found for context_store")
       
       # Close code tools
       if self.code_tools:
           self.code_tools.close()
       
       uptime = time.time() - self.start_time
       logger.info(f"System shutdown complete. Uptime: {uptime:.2f} seconds")
   
   async def submit_task(self, task_def: Dict[str, Any]) -> str:
       """Submit a task to the system.
       
       Args:
           task_def: Task definition
           
       Returns:
           Task ID
       """
       if not self.running:
           raise RuntimeError("System is not running")
       
       task_id = await self.task_scheduler.schedule_task(task_def)
       return task_id
   
   async def get_task_status(self, task_id: str) -> Dict[str, Any]:
       """Get the status of a task.
       
       Args:
           task_id: Task ID
           
       Returns:
           Task status information
       """
       if not self.running:
           raise RuntimeError("System is not running")
       
       task = await self.task_scheduler.get_task(task_id)
       if not task:
           raise ValueError(f"Task {task_id} not found")
       
       return task.to_dict()
   
   async def get_agent_info(self, agent_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
       """Get information about an agent or all agents.
       
       Args:
           agent_id: Optional agent ID
           
       Returns:
           Agent information or list of all agent information
       """
       if not self.running:
           raise RuntimeError("System is not running")
       
       if agent_id:
           if agent_id not in self.agent_registry:
               raise ValueError(f"Agent {agent_id} not found")
           
           agent = self.agent_registry[agent_id]
           return {
               "id": agent.state.agent_id,
               "name": agent.name,
               "type": agent.state.agent_type,
               "model": f"{agent.llm.__class__.__name__}",
               "stats": agent.get_performance_stats()
           }
       else:
           return [
               {
                   "id": agent.state.agent_id,
                   "name": agent.name,
                   "type": agent.state.agent_type,
                   "model": f"{agent.llm.__class__.__name__}"
               }
               for agent in self.agent_registry.values()
           ]
   
   async def get_system_status(self) -> Dict[str, Any]:
       """Get system status information.
       
       Returns:
           System status information
       """
       if not self.running:
           return {"status": "stopped"}
       
       uptime = time.time() - self.start_time
       
       status = {
           "status": "running",
           "instance_id": self.instance_id,
           "uptime_seconds": uptime,
           "agent_count": len(self.agent_registry),
           "version": VERSION,
           "debug_mode": self.debug
       }
       
       # Add task stats if available
       if self.task_scheduler:
           task_count = len(getattr(self.task_scheduler, 'tasks', {}))
    
    # Check if the attributes exist, otherwise use empty collections
           if hasattr(self.task_scheduler, 'completed_task_ids'):
                completed_count = len(self.task_scheduler.completed_task_ids)
           else:
                completed_count = 0
                
           if hasattr(self.task_scheduler, 'failed_task_ids'):
                failed_count = len(self.task_scheduler.failed_task_ids)
           else:
                failed_count = 0
           
           status.update({
                "tasks": {
                "total": task_count,
                "completed": completed_count,
                "failed": failed_count,
                "running": 0,  # Would be calculated at runtime
                "pending": 0,  # Would be calculated at runtime
        }
    })
            
            # Add workflow stats if available
           if self.workflow_engine:
                workflow_stats = await self.workflow_engine.get_stats()
                status["workflows"] = workflow_stats
            
           return status
   
   async def create_workflow(self, workflow_def: Dict[str, Any]) -> str:
       """Create a new workflow.
       
       Args:
           workflow_def: Workflow definition
           
       Returns:
           Workflow ID
       """
       if not self.running:
           raise RuntimeError("System is not running")
       
       workflow_id = await self.workflow_engine.register_workflow(workflow_def)
       return workflow_id
   
   async def execute_workflow(self, workflow_id: str, variables: Optional[Dict[str, Any]] = None) -> str:
       """Execute a workflow.
       
       Args:
           workflow_id: Workflow ID
           variables: Initial variables for the workflow
           
       Returns:
           Workflow execution ID
       """
       if not self.running:
           raise RuntimeError("System is not running")
       
       execution_id = await self.workflow_engine.execute_workflow(
           workflow_id=workflow_id,
           input_variables=variables
       )
       return execution_id
   
   async def submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
       """Submit feedback for a task.
       
       Args:
           feedback_data: Feedback data
           
       Returns:
           Result of feedback processing
       """
       if not self.running:
           raise RuntimeError("System is not running")
       
       result = await self.feedback_processor.process_review(feedback_data)
       return {"processed_items": len(result)}


async def interactive_session(system: MultiAgentSystem) -> None:
   """Run an interactive CLI session with the system.
   
   Args:
       system: Multi-agent system instance
   """
   print(BANNER)
   print(f"\nMulti-Agent Development System v{VERSION}")
   print(f"Instance ID: {system.instance_id}")
   print(f"Type 'help' for a list of commands\n")
   
   commands = {
       "help": "Show help",
       "status": "Show system status",
       "agents": "List all agents",
       "agent <id>": "Show agent details",
       "task <definition>": "Submit a new task",
       "task-status <id>": "Check task status",
       "workflow <definition>": "Create a new workflow",
       "run-workflow <id> [variables]": "Execute a workflow",
       "feedback <definition>": "Submit feedback",
       "exit": "Exit the system"
   }
   
   while not system.shutdown_requested:
       try:
           command = input("\n> ").strip()
           
           if not command:
               continue
           
           parts = command.split(' ', 1)
           cmd = parts[0].lower()
           args = parts[1] if len(parts) > 1 else ""
           
           if cmd == "help":
               print("\nAvailable commands:")
               for cmd_name, description in commands.items():
                   print(f"  {cmd_name:<20} - {description}")
           
           elif cmd == "status":
               status = await system.get_system_status()
               print("\nSystem Status:")
               print(json.dumps(status, indent=2))
           
           elif cmd == "agents":
               agents = await system.get_agent_info()
               print("\nRegistered Agents:")
               for agent in agents:
                   print(f"  {agent['name']} ({agent['type']}) - ID: {agent['id']}")
           
           elif cmd == "agent":
               if not args:
                   print("Error: Agent ID is required")
                   continue
               
               try:
                   agent_info = await system.get_agent_info(args)
                   print("\nAgent Details:")
                   print(json.dumps(agent_info, indent=2))
               except ValueError as e:
                   print(f"Error: {str(e)}")
           
           elif cmd == "task":
               if not args:
                   print("Error: Task definition is required")
                   continue
               
               try:
                   task_def = json.loads(args)
                   task_id = await system.submit_task(task_def)
                   print(f"Task submitted successfully. Task ID: {task_id}")
               except json.JSONDecodeError:
                   print("Error: Invalid JSON for task definition")
               except Exception as e:
                   print(f"Error submitting task: {str(e)}")
           
           elif cmd == "task-status":
               if not args:
                   print("Error: Task ID is required")
                   continue
               
               try:
                   status = await system.get_task_status(args)
                   print("\nTask Status:")
                   print(json.dumps(status, indent=2))
               except ValueError as e:
                   print(f"Error: {str(e)}")
               except Exception as e:
                   print(f"Error getting task status: {str(e)}")
           
           elif cmd == "workflow":
               if not args:
                   print("Error: Workflow definition is required")
                   continue
               
               try:
                   workflow_def = json.loads(args)
                   workflow_id = await system.create_workflow(workflow_def)
                   print(f"Workflow created successfully. Workflow ID: {workflow_id}")
               except json.JSONDecodeError:
                   print("Error: Invalid JSON for workflow definition")
               except Exception as e:
                   print(f"Error creating workflow: {str(e)}")
           
           elif cmd == "run-workflow":
               parts = args.split(' ', 1)
               workflow_id = parts[0]
               variables_str = parts[1] if len(parts) > 1 else "{}"
               
               if not workflow_id:
                   print("Error: Workflow ID is required")
                   continue
               
               try:
                   variables = json.loads(variables_str)
                   execution_id = await system.execute_workflow(workflow_id, variables)
                   print(f"Workflow execution started. Execution ID: {execution_id}")
               except json.JSONDecodeError:
                   print("Error: Invalid JSON for variables")
               except Exception as e:
                   print(f"Error executing workflow: {str(e)}")
           
           elif cmd == "feedback":
               if not args:
                   print("Error: Feedback definition is required")
                   continue
               
               try:
                   feedback_data = json.loads(args)
                   result = await system.submit_feedback(feedback_data)
                   print(f"Feedback processed successfully: {result}")
               except json.JSONDecodeError:
                   print("Error: Invalid JSON for feedback data")
               except Exception as e:
                   print(f"Error processing feedback: {str(e)}")
           
           elif cmd == "exit":
               print("Shutting down...")
               system.shutdown_requested = True
               break
           
           else:
               print(f"Unknown command: {cmd}")
               print("Type 'help' for a list of commands")
       
       except KeyboardInterrupt:
           print("\nOperation interrupted. Type 'exit' to quit.")
       except Exception as e:
           print(f"Error: {str(e)}")
   
   # Perform shutdown
   await system.shutdown()


async def run_api_server(system: MultiAgentSystem, host: str, port: int) -> None:
   """Run the API server for the system.
   
   Args:
       system: Multi-agent system instance
       host: Host to bind to
       port: Port to listen on
   """
   try:
       # Dynamically import FastAPI to avoid dependency when not using API
       from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
       from fastapi.middleware.cors import CORSMiddleware
       import uvicorn
       from pydantic import BaseModel as PydanticBaseModel
       
       app = FastAPI(
           title="Multi-Agent Development API",
           description="API for the Multi-Agent Development System",
           version=VERSION
       )
       
       # Add CORS middleware
       app.add_middleware(
           CORSMiddleware,
           allow_origins=["*"],  # Restrict in production
           allow_credentials=True,
           allow_methods=["*"],
           allow_headers=["*"],
       )
       
       # API models
       class TaskDefinition(PydanticBaseModel):
           title: str
           description: str
           agent_id: Optional[str] = None
           agent_type: Optional[str] = None
           action: str
           params: Dict[str, Any] = {}
           priority: Optional[str] = None
           tags: List[str] = []
           category: str = "default"
       
       class WorkflowDefinition(PydanticBaseModel):
           name: str
           description: str
           steps: Dict[str, Any]
           start_step_id: str
           variables: Dict[str, Any] = {}
       class FeedbackSubmission(PydanticBaseModel):
           task_id: str
           reviewer_id: str
           status: str
           feedback_items: List[Dict[str, Any]]
           summary: Optional[str] = None
       
       class WorkflowVariables(PydanticBaseModel):
           variables: Dict[str, Any] = {}
       
       # API routes
       @app.get("/")
       async def root():
           """API root endpoint."""
           return {
               "name": "Multi-Agent Development API",
               "version": VERSION,
               "status": "running",
               "instance_id": system.instance_id
           }
       
       @app.get("/status")
       async def get_status():
           """Get system status."""
           return await system.get_system_status()
       
       @app.get("/agents")
       async def list_agents():
           """List all agents."""
           return await system.get_agent_info()
       
       @app.get("/agents/{agent_id}")
       async def get_agent(agent_id: str):
            """Get agent information."""
            try:
                return await system.get_agent_info(agent_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        
       @app.post("/tasks")
       async def create_task(task: TaskDefinition):
            """Create a new task."""
            try:
                task_id = await system.submit_task(task.dict())
                return {"task_id": task_id, "status": "pending"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
       @app.get("/tasks/{task_id}")
       async def get_task(task_id: str):
            """Get task status."""
            try:
                return await system.get_task_status(task_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        
       @app.post("/workflows")
       async def create_workflow(workflow: WorkflowDefinition):
            """Create a new workflow."""
            try:
                workflow_id = await system.create_workflow(workflow.dict())
                return {"workflow_id": workflow_id}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
       @app.post("/workflows/{workflow_id}/execute")
       async def execute_workflow(workflow_id: str, variables: WorkflowVariables):
            """Execute a workflow."""
            try:
                execution_id = await system.execute_workflow(workflow_id, variables.variables)
                return {"execution_id": execution_id, "status": "running"}
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
       @app.post("/feedback")
       async def submit_feedback(feedback: FeedbackSubmission):
            """Submit feedback for a task."""
            try:
                result = await system.submit_feedback(feedback.dict())
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Handle shutdown
       @app.on_event("shutdown")
       async def shutdown_event():
            await system.shutdown()
        
        # Start the API server
       logger.info(f"Starting API server on {host}:{port}")
       config = uvicorn.Config(app, host=host, port=port)
       server = uvicorn.Server(config)
       await server.serve()
        
   except ImportError:
        logger.error("API server requires FastAPI and uvicorn. Install with 'pip install fastapi uvicorn'")
   except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        traceback.print_exc()


def main() -> None:
    """Main entry point for the multi-agent system."""
    parser = argparse.ArgumentParser(description="Multi-Agent Development System")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", help="Path to data directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--api-host", default="127.0.0.1", help="API server host")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    parser.add_argument("--model-provider", help="Model provider to use (anthropic, openai, google)")
    parser.add_argument("--agents", help="Comma-separated list of agent types to initialize")
    
    args = parser.parse_args()
    
    # Convert comma-separated agents to list if provided
    agent_types = None
    if args.agents:
        agent_types = [agent_type.strip() for agent_type in args.agents.split(',')]
    
    # Create and initialize the system
    system = MultiAgentSystem(
        config_path=args.config,
        data_dir=args.data_dir,
        debug=args.debug,
        agent_types=agent_types,
        model_provider=args.model_provider
    )
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutdown signal received. Gracefully shutting down...")
        system.shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the appropriate mode
    if args.api:
        asyncio.run(async_main(system, run_api=True, api_host=args.api_host, api_port=args.api_port))
    else:
        asyncio.run(async_main(system))


async def async_main(
    system: MultiAgentSystem,
    run_api: bool = False,
    api_host: str = "127.0.0.1",
    api_port: int = 8000
) -> None:
    """Async main entry point.
    
    Args:
        system: Multi-agent system instance
        run_api: Whether to run the API server
        api_host: API server host
        api_port: API server port
    """
    # Initialize the system
    try:
        await system.initialize()
        
        if run_api:
            # Run API server
            await run_api_server(system, api_host, api_port)
        else:
            # Run interactive session
            await interactive_session(system)
            
    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down...")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        traceback.print_exc()
    finally:
        # Ensure system is properly shut down
        if system.running:
            await system.shutdown()


if __name__ == "__main__":
    main()
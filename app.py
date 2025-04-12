"""
Multi-Agent Development System - Streamlit Interface

This module provides a Streamlit-based web interface for the multi-agent development system,
allowing users to:
1. View and manage agents
2. Submit and monitor tasks
3. Create and execute workflows
4. Review and provide feedback on completed tasks
5. Monitor system performance
6. View context and shared memory

Usage:
    streamlit run app.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from streamlit_ace import st_ace

# Add the parent directory to the path to allow importing the system modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components
from config.system_config import SystemConfig, get_system_config
from config.agent_config import AgentConfig
from memory.shared_memory import SharedMemory
from memory.context_store import ContextStore, ContextType
from agents.base_agent import BaseAgent, ModelProvider, AgentRole, TaskStatus, TaskPriority
from human_interface.review_interface import ReviewInterface, ReviewStatus, FeedbackType, FeedbackItem
from human_interface.feedback_processor import FeedbackProcessor
from orchestrator.workflow_engine import WorkflowEngine
from orchestrator.task_scheduler import TaskScheduler
from main import MultiAgentSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Multi-Agent Development System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
REFRESH_INTERVAL = 10  # seconds

# Color schemes
COLORS = {
    "primary": "#1E88E5",
    "success": "#4CAF50",
    "warning": "#FFC107",
    "error": "#F44336",
    "info": "#03A9F4",
    "background": "#F0F2F6",
    "text": "#212121",
    "agent_types": {
        "project_manager": "#D81B60",
        "architecture_designer": "#1E88E5",
        "ui_developer": "#8E24AA",
        "frontend_logic": "#3949AB",
        "frontend_integration": "#00ACC1",
        "api_developer": "#43A047",
        "database_designer": "#E53935",
        "backend_logic": "#FB8C00",
        "infrastructure": "#5E35B1",
        "deployment": "#1E88E5",
        "security": "#E53935",
        "code_reviewer": "#00897B",
        "test_developer": "#7CB342",
        "ux_tester": "#FFB300",
        "researcher": "#039BE5",
        "documentation": "#8D6E63",
        "human_interface": "#F06292"
    }
}

# Utility Functions
def run_async(coroutine):
    """Run an async function in a sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

def format_time(seconds):
    """Format seconds into a readable time string."""
    if seconds is None:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def format_datetime(iso_string):
    """Format ISO datetime string into a readable format."""
    if not iso_string:
        return "N/A"
    
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_string

def truncate_text(text, max_length=100):
    """Truncate text and add ellipsis if needed."""
    if not text:
        return ""
    return text if len(text) <= max_length else text[:max_length] + "..."

def init_session_state():
    """Initialize session state variables."""
    if 'system' not in st.session_state:
        st.session_state.system = None
    
    if 'agent_filter' not in st.session_state:
        st.session_state.agent_filter = "all"
    
    if 'refresh_data' not in st.session_state:
        st.session_state.refresh_data = True
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'selected_task' not in st.session_state:
        st.session_state.selected_task = None
    
    if 'task_results' not in st.session_state:
        st.session_state.task_results = {}
    
    if 'workflow_executions' not in st.session_state:
        st.session_state.workflow_executions = {}
    
    if 'feedback_items' not in st.session_state:
        st.session_state.feedback_items = []
    
    if 'system_metrics' not in st.session_state:
        st.session_state.system_metrics = []
    
    if 'context_entries' not in st.session_state:
        st.session_state.context_entries = []
        
    # Add missing debug_mode
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

def initialize_system():
    """Initialize the multi-agent system."""
    try:
        if 'system' in st.session_state and st.session_state.system:
            # System already initialized
            return st.session_state.system
        
        # Get configuration path
        config_path = st.session_state.get('config_path', None)
        
        # Create and initialize the system
        system = MultiAgentSystem(
            config_path=config_path,
            debug=st.session_state.get('debug_mode', False),
            model_provider=st.session_state.get('model_provider', None)
        )
        
        # Initialize the system
        run_async(system.initialize())
        
        st.session_state.system = system
        logger.info("Multi-agent system initialized")
        
        return system
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.error(traceback.format_exc())
        return None

def shutdown_system():
    """Shut down the multi-agent system."""
    if 'system' in st.session_state and st.session_state.system:
        try:
            run_async(st.session_state.system.shutdown())
            st.session_state.system = None
            logger.info("Multi-agent system shut down")
        except Exception as e:
            st.error(f"Error shutting down system: {str(e)}")

def refresh_data():
    """Refresh all data from the system."""
    if not st.session_state.system:
        return
    
    try:
        # Update last refresh time
        st.session_state.last_refresh = datetime.now()
        
        # Get system status
        status = run_async(st.session_state.system.get_system_status())
        st.session_state.system_status = status
        
        # Get agent information
        agents = run_async(st.session_state.system.get_agent_info())
        st.session_state.agents = agents
        
        # Get task data (from shared memory)
        if st.session_state.system.shared_memory:
            # Fix: Get task keys asynchronously
            task_keys = run_async(st.session_state.system.shared_memory.get_keys(category="tasks"))
            
            # Now task_keys is a list, not a coroutine
            st.session_state.tasks = {}
            for key in task_keys:
                # Also retrieve each item asynchronously
                task_data = run_async(st.session_state.system.shared_memory.retrieve(key, "tasks"))
                st.session_state.tasks[key] = task_data
            
            # Update task results - fix the same issue here
            result_keys = run_async(st.session_state.system.shared_memory.get_keys(category="task_results"))
            
            st.session_state.task_results = {}
            for key in result_keys:
                result_data = run_async(st.session_state.system.shared_memory.retrieve(key, "task_results"))
                st.session_state.task_results[key] = result_data
        
        # Update system metrics history
        if hasattr(st.session_state, 'system_status'):
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'uptime': status.get('uptime_seconds', 0),
                'agents': len(st.session_state.agents),
                'tasks_total': status.get('tasks', {}).get('total', 0),
                'tasks_completed': status.get('tasks', {}).get('completed', 0),
                'tasks_failed': status.get('tasks', {}).get('failed', 0),
            }
            
            if 'system_metrics' not in st.session_state:
                st.session_state.system_metrics = []
            
            st.session_state.system_metrics.append(metrics)
            
            # Keep only the last 100 metrics points
            if len(st.session_state.system_metrics) > 100:
                st.session_state.system_metrics = st.session_state.system_metrics[-100:]
        
        # Schedule next refresh
        st.session_state.refresh_data = True
    except Exception as e:
        st.error(f"Error refreshing data: {str(e)}")
        st.error(traceback.format_exc())

# UI Components
def render_sidebar():
    """Render the sidebar with system controls and navigation."""
    st.sidebar.title("🤖 Multi-Agent Dev System")
    
    # System status indicator
    if 'system' in st.session_state and st.session_state.system:
        status = st.session_state.get('system_status', {})
        status_color = "green" if status.get('status') == "running" else "red"
        
        st.sidebar.markdown(
            f"<div style='display:flex;align-items:center;'>"
            f"<div style='width:12px;height:12px;border-radius:50%;background-color:{status_color};margin-right:8px;'></div>"
            f"<span><strong>Status:</strong> {status.get('status', 'unknown').upper()}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Version and uptime
        st.sidebar.markdown(f"**Version:** {status.get('version', 'unknown')}")
        st.sidebar.markdown(f"**Uptime:** {format_time(status.get('uptime_seconds', 0))}")
        
        # Divider
        st.sidebar.divider()
    
    # Navigation menu
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Agents", "Tasks", "Workflows", "Context Storage", "Feedback", "Settings"]
    )
    
    # Filters for agents page
    if page == "Agents":
        st.sidebar.subheader("Filter Agents")
        agent_types = ["all"] + [agent['type'] for agent in st.session_state.get('agents', [])]
        agent_types = list(set(agent_types))  # Remove duplicates
        st.session_state.agent_filter = st.sidebar.selectbox("Agent Type", agent_types)
    
    # Filters for tasks page
    elif page == "Tasks":
        st.sidebar.subheader("Filter Tasks")
        
        # Task status filter
        task_statuses = ["all", "pending", "in_progress", "completed", "failed"]
        status_filter = st.sidebar.selectbox("Status", task_statuses)
        st.session_state.task_status_filter = status_filter
        
        # Task agent filter
        agent_names = ["all"] + [agent['name'] for agent in st.session_state.get('agents', [])]
        agent_filter = st.sidebar.selectbox("Agent", agent_names)
        st.session_state.task_agent_filter = agent_filter
    
    # Actions section
    st.sidebar.divider()
    st.sidebar.subheader("Actions")
    
    if 'system' not in st.session_state or st.session_state.system is None:
        if st.sidebar.button("Initialize System", use_container_width=True):
            initialize_system()
    else:
        if st.sidebar.button("Shutdown System", use_container_width=True):
            shutdown_system()
        
        # Manual refresh button
        if st.sidebar.button("Refresh Data", use_container_width=True):
            refresh_data()
        
        # Last refresh time
        if 'last_refresh' in st.session_state:
            st.sidebar.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    return page

def render_dashboard():
    """Render the main dashboard with system overview and statistics."""
    st.title("🚀 Multi-Agent Development System Dashboard")
    
    if not st.session_state.system:
        st.info("System not initialized. Please initialize the system from the sidebar.")
        return
    
    # System status overview
    status = st.session_state.get('system_status', {})
    agents = st.session_state.get('agents', [])
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Agents", len(agents))
    
    with col2:
        tasks_total = status.get('tasks', {}).get('total', 0)
        st.metric("Total Tasks", tasks_total)
    
    with col3:
        tasks_completed = status.get('tasks', {}).get('completed', 0)
        completion_rate = f"{(tasks_completed / max(1, tasks_total)) * 100:.1f}%"
        st.metric("Tasks Completed", f"{tasks_completed} ({completion_rate})")
    
    with col4:
        tasks_failed = status.get('tasks', {}).get('failed', 0)
        failure_rate = f"{(tasks_failed / max(1, tasks_total)) * 100:.1f}%"
        st.metric("Tasks Failed", f"{tasks_failed} ({failure_rate})")
    
    # System activity charts
    st.subheader("System Activity")
    
    tab1, tab2, tab3 = st.tabs(["Tasks", "Agent Distribution", "Performance Metrics"])
    
    with tab1:
        if 'system_metrics' in st.session_state and st.session_state.system_metrics:
            # Convert metrics to DataFrame
            df = pd.DataFrame(st.session_state.system_metrics)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create task status chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['tasks_total'],
                mode='lines+markers',
                name='Total',
                line=dict(color=COLORS["primary"])
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['tasks_completed'],
                mode='lines+markers',
                name='Completed',
                line=dict(color=COLORS["success"])
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['tasks_failed'],
                mode='lines+markers',
                name='Failed',
                line=dict(color=COLORS["error"])
            ))
            
            fig.update_layout(
                title="Task Status Over Time",
                xaxis_title="Time",
                yaxis_title="Tasks",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metrics data available yet. Wait for the system to gather more data.")
    
    with tab2:
        # Agent distribution by type
        if agents:
            agent_types = {}
            for agent in agents:
                agent_type = agent['type']
                if agent_type in agent_types:
                    agent_types[agent_type] += 1
                else:
                    agent_types[agent_type] = 1
            
            # Create a bar chart
            fig = px.bar(
                x=list(agent_types.keys()),
                y=list(agent_types.values()),
                labels={'x': 'Agent Type', 'y': 'Count'},
                title="Agent Distribution by Type",
                color=list(agent_types.keys()),
                color_discrete_map={k: COLORS["agent_types"].get(k, COLORS["primary"]) for k in agent_types.keys()}
            )
            
            fig.update_layout(
                xaxis={'categoryorder': 'total descending'},
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agents available. Initialize the system with agents.")
    
    with tab3:
        # Performance metrics
        if 'task_results' in st.session_state and st.session_state.task_results:
            # Get execution times
            execution_times = []
            for result in st.session_state.task_results.values():
                if isinstance(result, dict) and 'execution_time' in result:
                    agent_name = result.get('agent', 'Unknown')
                    execution_times.append({
                        'agent': agent_name,
                        'execution_time': result['execution_time']
                    })
            
            if execution_times:
                df = pd.DataFrame(execution_times)
                
                # Calculate average execution time by agent
                avg_times = df.groupby('agent')['execution_time'].mean().reset_index()
                
                # Create a bar chart
                fig = px.bar(
                    avg_times,
                    x='agent',
                    y='execution_time',
                    labels={'agent': 'Agent', 'execution_time': 'Avg. Execution Time (s)'},
                    title="Average Task Execution Time by Agent",
                    color='agent',
                    color_discrete_sequence=[COLORS["primary"]]
                )
                
                fig.update_layout(
                    xaxis={'categoryorder': 'total descending'},
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available yet.")
        else:
            st.info("No task results available to analyze performance.")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Get recent tasks
    recent_tasks = []
    if 'tasks' in st.session_state and st.session_state.tasks:
        for task_id, task in st.session_state.tasks.items():
            if isinstance(task, dict):
                recent_tasks.append({
                    'id': task_id,
                    'description': task.get('description', 'No description'),
                    'agent': task.get('agent_type', 'Unknown'),
                    'status': task.get('status', 'unknown'),
                    'timestamp': task.get('timestamp', None)
                })
        
        # Sort by timestamp (newest first)
        recent_tasks.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Show only the 10 most recent
        recent_tasks = recent_tasks[:10]
    
    if recent_tasks:
        task_df = pd.DataFrame(recent_tasks)
        st.dataframe(
            task_df,
            column_config={
                "id": "Task ID",
                "description": "Description",
                "agent": "Agent",
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    help="Task status",
                    options=["pending", "in_progress", "completed", "failed"],
                    required=True
                ),
                "timestamp": "Timestamp"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No recent tasks available.")

def render_agents_page():
    """Render the agents management page."""
    st.title("👥 Agent Management")
    
    if not st.session_state.system:
        st.info("System not initialized. Please initialize the system from the sidebar.")
        return
    
    agents = st.session_state.get('agents', [])
    
    if not agents:
        st.warning("No agents found in the system.")
        return
    
    # Filter agents if needed
    if st.session_state.agent_filter != "all":
        agents = [a for a in agents if a['type'] == st.session_state.agent_filter]
    
    # Display agents in an expandable format
    for i, agent in enumerate(agents):
        with st.expander(f"{agent['name']} ({agent['type']})", expanded=i == 0):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Agent Details:**")
                st.write(f"**ID:** {agent['id']}")
                st.write(f"**Type:** {agent['type']}")
                st.write(f"**Model:** {agent.get('model', 'Unknown')}")
            
            # If there are agent stats available
            if 'stats' in agent:
                with col2:
                    st.write("**Performance Stats:**")
                    stats = agent['stats']
                    
                    st.write(f"**Tasks Completed:** {stats.get('tasks_completed', 0)}")
                    st.write(f"**Success Rate:** {stats.get('success_rate', 0):.1f}%")
                    st.write(f"**Avg. Execution Time:** {format_time(stats.get('average_execution_time', 0))}")
            
            # Agent actions
            st.write("**Actions:**")
            
            # Task submission form
            with st.form(key=f"submit_task_form_{agent['id']}"):
                st.subheader("Submit a task to this agent")
                
                task_title = st.text_input("Task Title", key=f"title_{agent['id']}")
                task_desc = st.text_area("Task Description", key=f"desc_{agent['id']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    task_action = st.selectbox(
                        "Action",
                        ["generate_code", "review_code", "analyze", "design", "explain", "summarize", "other"],
                        key=f"action_{agent['id']}"
                    )
                
                with col2:
                    task_priority = st.selectbox(
                        "Priority",
                        ["low", "medium", "high", "critical"],
                        index=1,
                        key=f"priority_{agent['id']}"
                    )
                
                task_params = st.text_area(
                    "Parameters (JSON format)",
                    '{\n  "param1": "value1",\n  "param2": "value2"\n}',
                    key=f"params_{agent['id']}"
                )
                
                submit_button = st.form_submit_button("Submit Task")
                
                if submit_button:
                    try:
                        # Parse parameters
                        params = json.loads(task_params)
                        
                        # Check if agent was selected
                        # Create task definition without workflow reference
                        selected_agent = next((agent for agent in agents if agent['id'] == agent['id']), None)
                        if not selected_agent:
                            raise ValueError("Selected agent not found")
    
                        # Prepare task definition
                        task_def = {
                            "title": task_title,
                            "description": task_desc,
                            "agent_id": selected_agent['id'],
                            "agent_type": selected_agent['type'],
                            "action": task_action,
                            "params": params,
                            "priority": task_priority,
                            "tags": [agent['type'], task_action],
                            "category": "default",
                            "status": "pending",  # Add explicit status
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Submit the task
                        task_id = run_async(st.session_state.system.submit_task(task_def))
                        
                        st.success(f"Task submitted successfully! Task ID: {task_id}")
                        
                        # Refresh data after submission
                        refresh_data()
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format for parameters")
                    except Exception as e:
                        st.error(f"Error submitting task: {str(e)}")
            
            # Recent tasks by this agent
            if 'tasks' in st.session_state and st.session_state.tasks:
                agent_tasks = []
                
                for task_id, task in st.session_state.tasks.items():
                    if isinstance(task, dict) and task.get('agent_id') == agent['id']:
                        agent_tasks.append({
                            'id': task_id,
                            'title': task.get('title', 'No title'),
                            'status': task.get('status', 'unknown'),
                            'timestamp': format_datetime(task.get('timestamp', None))
                        })
                
                if agent_tasks:
                    st.subheader("Recent Tasks")
                    
                    # Sort by timestamp (newest first)
                    agent_tasks.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    
                    # Show only the 5 most recent
                    agent_tasks = agent_tasks[:5]
                    
                    # Display as a table
                    task_df = pd.DataFrame(agent_tasks)
                    st.dataframe(task_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No recent tasks for this agent.")

def render_tasks_page():
    """Render the tasks management page."""
    st.title("📋 Task Management")
    
    if not st.session_state.system:
        st.info("System not initialized. Please initialize the system from the sidebar.")
        return
    if st.session_state.debug_mode:
        st.write("Shared memory status:", "Available" if st.session_state.system.shared_memory else "Not available")
    debug_mode = st.session_state.get('debug_mode', False)
    if debug_mode:
        st.write("Shared memory status:", "Available" if st.session_state.system.shared_memory else "Not available")
    tasks = []
    if 'tasks' in st.session_state and st.session_state.tasks:
        st.write(f"Task data found: {len(st.session_state.tasks)} tasks")
        for task_id, task in st.session_state.tasks.items():
            if st.session_state.debug_mode:
                st.write(f"Task ID: {task_id}, Raw data:", task)
            if isinstance(task, dict):
                tasks.append({
                    'id': task_id,
                    'title': task.get('title', 'No title'),
                    'description': task.get('description', 'No description'),
                    'agent_type': task.get('agent_type', 'Unknown'),
                    'agent_id': task.get('agent_id', ''),
                    'status': task.get('status', 'unknown'),
                    'priority': task.get('priority', 'medium'),
                    'timestamp': task.get('timestamp', None),
                    'formatted_time': format_datetime(task.get('timestamp', None))
                })
    else:
        st.warning("No tasks found in session state.")
    if not tasks:
        st.warning("No tasks found in the system.")
        
        # New task submission form
        with st.form(key="create_new_task_form"):
            st.subheader("Create a New Task")
            
            # Get agent options
            agents = st.session_state.get('agents', [])
            agent_options = [("", "Select an agent...")] + [(a['id'], f"{a['name']} ({a['type']})") for a in agents]
            
            task_title = st.text_input("Task Title")
            task_desc = st.text_area("Task Description")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_agent_id = st.selectbox(
                    "Agent",
                    [a[0] for a in agent_options],
                    format_func=lambda x: next((a[1] for a in agent_options if a[0] == x), x),
                    index=0
                )
            
            with col2:
                task_priority = st.selectbox(
                    "Priority",
                    ["low", "medium", "high", "critical"],
                    index=1
                )
            
            task_action = st.selectbox(
                "Action",
                ["generate_code", "review_code", "analyze", "design", "explain", "summarize", "other"]
            )
            
            task_params = st.text_area(
                "Parameters (JSON format)",
                '{\n  "param1": "value1",\n  "param2": "value2"\n}'
            )
            
            submit_button = st.form_submit_button("Submit Task")
            
            if submit_button:
                if not selected_agent_id:
                    st.error("Please select an agent")
                else:
                    try:
                        # Find selected agent
                        selected_agent = next((a for a in agents if a['id'] == selected_agent_id), None)
                        
                        if not selected_agent:
                            st.error("Invalid agent selection")
                        else:
                            # Parse parameters
                            params = json.loads(task_params)
                            
                            # Prepare task definition
                            task_def = {
                                "title": task_title,
                                "description": task_desc,
                                "agent_id": selected_agent['id'],
                                "agent_type": selected_agent['type'],
                                "action": task_action,
                                "params": params,
                                "priority": task_priority,
                                "tags": [selected_agent['type'], task_action],
                                "category": "default"
                            }
                            
                            # Submit the task
                            task_id = run_async(st.session_state.system.submit_task(task_def))
                            
                            st.success(f"Task submitted successfully! Task ID: {task_id}")
                            
                            # Refresh data after submission
                            refresh_data()
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format for parameters")
                    except Exception as e:
                        st.error(f"Error submitting task: {str(e)}")
        
        return
    
    # Apply filters
    status_filter = st.session_state.get('task_status_filter', 'all')
    agent_filter = st.session_state.get('task_agent_filter', 'all')
    
    filtered_tasks = tasks
    
    if status_filter != 'all':
        filtered_tasks = [t for t in filtered_tasks if t['status'] == status_filter]
    
    if agent_filter != 'all':
        filtered_tasks = [t for t in filtered_tasks if t['agent_type'] == agent_filter]
    
    # Sort tasks by timestamp (newest first)
    filtered_tasks.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Create a dataframe for display
    task_df = pd.DataFrame([
        {
            'ID': t['id'],
            'Title': t['title'],
            'Agent': t['agent_type'],
            'Status': t['status'],
            'Priority': t['priority'],
            'Created': t['formatted_time']
        }
        for t in filtered_tasks
    ])
    
    # Task list
    st.subheader(f"Tasks ({len(filtered_tasks)})")
    
    if len(filtered_tasks) > 0:
        # Use a dataframe with selection
        selected_row = st.dataframe(
            task_df,
            use_container_width=True,
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["pending", "in_progress", "completed", "failed"],
                    required=True
                ),
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["low", "medium", "high", "critical"],
                    required=True
                )
            },
            hide_index=True
        )
        
        # If a task is selected, show details
        task_id = st.selectbox("Select a task to view details:", [t['id'] for t in filtered_tasks])
        if task_id:
            st.session_state.selected_task = task_id
    
    # Show selected task details
    if st.session_state.selected_task:
        selected_task = next((t for t in tasks if t['id'] == st.session_state.selected_task), None)
        
        if selected_task:
            st.subheader(f"Task Details: {selected_task['title']}")
            
            # Display task details and results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"**ID:** {selected_task['id']}")
                st.write(f"**Description:** {selected_task['description']}")
                st.write(f"**Agent:** {selected_task['agent_type']}")
                st.write(f"**Status:** {selected_task['status']}")
                st.write(f"**Priority:** {selected_task['priority']}")
                st.write(f"**Created:** {selected_task['formatted_time']}")
                
                # Display task result if available
                task_result = None
                if 'task_results' in st.session_state:
                    task_result = st.session_state.task_results.get(selected_task['id'])
                
                if task_result:
                    st.subheader("Task Result")
                    
                    if isinstance(task_result, dict) and 'result' in task_result:
                        result = task_result['result']
                        
                        # Check if result contains code
                        if isinstance(result, str) and (
                            result.strip().startswith("```") or
                            "</" in result or
                            "function " in result or
                            "class " in result or
                            "import " in result
                        ):
                            st.code(result)
                        else:
                            st.write(result)
                        
                        # Display execution stats
                        if 'execution_time' in task_result:
                            st.caption(f"Execution time: {format_time(task_result['execution_time'])}")
                    else:
                        st.write(task_result)
                else:
                    st.info("No results available for this task yet.")
            
            with col2:
                # Task actions
                st.subheader("Actions")
                
                # Cancel task button (only for pending/in_progress)
                if selected_task['status'] in ['pending', 'in_progress']:
                    if st.button("Cancel Task", key=f"cancel_{selected_task['id']}"):
                        try:
                            # Not implemented in base system, would need to add this
                            st.warning("Task cancellation not implemented yet.")
                            refresh_data()
                        except Exception as e:
                            st.error(f"Error canceling task: {str(e)}")
                
                # Add feedback (for completed tasks)
                if selected_task['status'] == 'completed':
                    with st.form(key=f"feedback_form_{selected_task['id']}"):
                        st.subheader("Provide Feedback")
                        
                        feedback_status = st.selectbox(
                            "Review Status",
                            ["APPROVED", "PARTIALLY_APPROVED", "NEEDS_CLARIFICATION", "REJECTED"],
                            index=0
                        )
                        
                        feedback_comment = st.text_area("General Feedback")
                        
                        # Specific feedback items
                        st.write("**Feedback Items:**")
                        
                        feedback_type = st.selectbox(
                            "Type",
                            ["CODE_QUALITY", "FUNCTIONALITY", "DESIGN", "DOCUMENTATION", "GENERAL"]
                        )
                        
                        feedback_severity = st.selectbox(
                            "Severity",
                            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                            index=1
                        )
                        
                        specific_comment = st.text_area("Specific Comment")
                        suggested_changes = st.text_area("Suggested Changes")
                        
                        submit_feedback = st.form_submit_button("Submit Feedback")
                        
                        if submit_feedback:
                            try:
                                # Prepare feedback data
                                feedback_data = {
                                    "task_id": selected_task['id'],
                                    "reviewer_id": "human_user",
                                    "status": feedback_status,
                                    "summary": feedback_comment,
                                    "feedback_items": [
                                        {
                                            "id": str(uuid.uuid4()),
                                            "feedback_type": feedback_type,
                                            "comment": specific_comment,
                                            "suggested_changes": suggested_changes,
                                            "severity": feedback_severity
                                        }
                                    ]
                                }
                                
                                # Submit feedback
                                result = run_async(st.session_state.system.submit_feedback(feedback_data))
                                
                                st.success(f"Feedback submitted successfully!")
                                
                                # Refresh data
                                refresh_data()
                            except Exception as e:
                                st.error(f"Error submitting feedback: {str(e)}")

def render_workflows_page():
    """Render the workflows management page."""
    st.title("🔄 Workflow Management")
    
    if not st.session_state.system:
        st.info("System not initialized. Please initialize the system from the sidebar.")
        return
    
    # Workflow tabs
    tab1, tab2 = st.tabs(["Create Workflow", "View Workflows"])
    
    with tab1:
        st.subheader("Define a New Workflow")
        
        with st.form(key="create_workflow_form"):
            workflow_name = st.text_input("Workflow Name")
            workflow_desc = st.text_area("Description")
            
            # Workflow definition - use a code editor
            st.write("**Workflow Definition (JSON):**")
            st.caption("Define the workflow steps and transitions")
            
            default_workflow = {
                "steps": {
                    "step1": {
                        "task": {
                            "agent_type": "architecture_designer",
                            "action": "design",
                            "description": "Design the system architecture"
                        },
                        "next": "step2"
                    },
                    "step2": {
                        "task": {
                            "agent_type": "frontend_logic",
                            "action": "generate_code",
                            "description": "Implement the frontend logic"
                        },
                        "next": "step3"
                    },
                    "step3": {
                        "task": {
                            "agent_type": "code_reviewer",
                            "action": "review_code",
                            "description": "Review the implementation"
                        },
                        "next": None
                    }
                },
                "start_step_id": "step1",
                "variables": {
                    "project_name": "Example Project",
                    "technology_stack": ["React", "Python", "FastAPI"]
                }
            }
            
            workflow_def = st_ace(
                value=json.dumps(default_workflow, indent=2),
                language="json",
                theme="github",
                auto_update=True,
                height=400
            )
            
            submit_button = st.form_submit_button("Create Workflow")
            
            if submit_button:
                try:
                    # Parse workflow definition
                    workflow_json = json.loads(workflow_def)
                    
                    # Prepare workflow data
                    workflow_data = {
                        "name": workflow_name,
                        "description": workflow_desc,
                        **workflow_json
                    }
                    
                    # Create the workflow
                    workflow_id = run_async(st.session_state.system.create_workflow(workflow_data))
                    
                    st.success(f"Workflow created successfully! Workflow ID: {workflow_id}")
                    
                    # Refresh data
                    refresh_data()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for workflow definition")
                except Exception as e:
                    st.error(f"Error creating workflow: {str(e)}")
    
    with tab2:
        st.subheader("Available Workflows")
        
        # Get workflows from shared memory
        workflows = []
        if st.session_state.system.shared_memory:
            workflow_keys = run_async(st.session_state.system.shared_memory.get_keys(category="workflows"))
            
            for key in workflow_keys:
                workflow = run_async(st.session_state.system.shared_memory.retrieve(key, "workflows"))
                if workflow:
                    workflows.append(workflow)
        
        if not workflows:
            st.info("No workflows available. Create a workflow in the 'Create Workflow' tab.")
        else:
            # Sort workflows by name
            workflows.sort(key=lambda w: w.get('name', ''))
            
            # Display workflows in expandable sections
            for workflow in workflows:
                with st.expander(f"{workflow.get('name', 'Unnamed workflow')}"):
                    st.write(f"**ID:** {workflow.get('id', 'Unknown')}")
                    st.write(f"**Description:** {workflow.get('description', 'No description')}")
                    
                    # Display workflow structure
                    st.write("**Workflow Structure:**")
                    steps = workflow.get('steps', {})
                    
                    if steps:
                        step_df = pd.DataFrame([
                            {
                                'Step ID': step_id,
                                'Agent Type': step.get('task', {}).get('agent_type', 'Unknown'),
                                'Action': step.get('task', {}).get('action', 'Unknown'),
                                'Description': step.get('task', {}).get('description', 'No description'),
                                'Next Step': step.get('next', 'None')
                            }
                            for step_id, step in steps.items()
                        ])
                        
                        st.dataframe(step_df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No steps defined in this workflow.")
                    
                    # Display variables
                    variables = workflow.get('variables', {})
                    if variables:
                        st.write("**Variables:**")
                        st.json(variables)
                    
                    # Execute workflow
                    with st.form(key=f"execute_workflow_{workflow.get('id', '')}"):
                        st.subheader("Execute Workflow")
                        
                        variables_json = st.text_area(
                            "Override Variables (JSON)",
                            value=json.dumps(variables, indent=2),
                            height=150
                        )
                        
                        execute_button = st.form_submit_button("Execute Workflow")
                        
                        if execute_button:
                            try:
                                # Parse variables
                                variables_override = json.loads(variables_json)
                                
                                # Execute the workflow
                                execution_id = run_async(
                                    st.session_state.system.execute_workflow(
                                        workflow_id=workflow.get('id', ''),
                                        variables=variables_override
                                    )
                                )
                                
                                st.success(f"Workflow execution started! Execution ID: {execution_id}")
                                
                                # Refresh data
                                refresh_data()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON format for variables")
                            except Exception as e:
                                st.error(f"Error executing workflow: {str(e)}")

def render_context_page():
    """Render the context storage page."""
    st.title("🧠 Context Storage")
    
    if not st.session_state.system:
        st.info("System not initialized. Please initialize the system from the sidebar.")
        return
    
    if not st.session_state.system.context_store:
        st.warning("Context store not available in this system.")
        return
    
    # Create tabs for different context operations
    tab1, tab2, tab3 = st.tabs(["Browse Context", "Search Context", "Add Context"])
    
    with tab1:
        st.subheader("Context Entries")
        
        # Context type filter
        context_types = ["All Types"] + [t.value for t in ContextType]
        selected_type = st.selectbox("Filter by Type", context_types)
        
        # Get context entries
        context_entries = []
        
        query = {
            "types": [selected_type] if selected_type != "All Types" else None,
            "limit": 100
        }
        
        try:
            # Search for context entries
            result = run_async(st.session_state.system.context_store.search(query))
            context_entries = result.entries
            
            st.session_state.context_entries = context_entries
        except Exception as e:
            st.error(f"Error retrieving context entries: {str(e)}")
            context_entries = st.session_state.get('context_entries', [])
        
        if not context_entries:
            st.info("No context entries found with the current filter.")
        else:
            # Display entries in a dataframe
            entries_df = pd.DataFrame([
                {
                    'ID': entry.id,
                    'Type': entry.entry_type,
                    'Title': entry.title,
                    'Created By': entry.created_by,
                    'Created At': format_datetime(entry.created_at),
                    'Is Key': "✓" if entry.is_key_context else ""
                }
                for entry in context_entries
            ])
            
            st.dataframe(entries_df, hide_index=True, use_container_width=True)
            
            # View selected entry
            selected_id = st.selectbox("Select an entry to view:", [e.id for e in context_entries])
            
            if selected_id:
                selected_entry = next((e for e in context_entries if e.id == selected_id), None)
                
                if selected_entry:
                    st.subheader(f"Context: {selected_entry.title}")
                    
                    # Show entry details
                    st.write(f"**Type:** {selected_entry.entry_type}")
                    st.write(f"**Created By:** {selected_entry.created_by}")
                    st.write(f"**Created At:** {format_datetime(selected_entry.created_at)}")
                    
                    if selected_entry.tags:
                        st.write(f"**Tags:** {', '.join(selected_entry.tags)}")
                    
                    # Display content based on type
                    st.subheader("Content")
                    
                    if isinstance(selected_entry.content, str):
                        if selected_entry.entry_type == ContextType.CODE:
                            st.code(selected_entry.content)
                        else:
                            st.write(selected_entry.content)
                    elif isinstance(selected_entry.content, dict):
                        st.json(selected_entry.content)
                    else:
                        st.write(str(selected_entry.content))
                    
                    # Show metadata if available
                    if selected_entry.metadata:
                        st.subheader("Metadata")
                        st.json(selected_entry.metadata)
    
    with tab2:
        st.subheader("Search Context")
        
        with st.form(key="search_context_form"):
            search_text = st.text_input("Search Query")
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_type = st.selectbox(
                    "Context Type",
                    ["All Types"] + [t.value for t in ContextType]
                )
            
            with col2:
                search_tags = st.text_input("Tags (comma separated)")
            
            key_context_only = st.checkbox("Only Key Context")
            
            search_button = st.form_submit_button("Search")
            
            if search_button:
                try:
                    # Prepare search query
                    tags = [t.strip() for t in search_tags.split(',')] if search_tags else None
                    
                    query = {
                        "text": search_text if search_text else None,
                        "types": [search_type] if search_type != "All Types" else None,
                        "tags": tags,
                        "is_key_context": key_context_only if key_context_only else None,
                        "limit": 100,
                        "use_vector_search": True
                    }
                    
                    # Perform search
                    result = run_async(st.session_state.system.context_store.search(query))
                    
                    if not result.entries:
                        st.info("No matching context entries found.")
                    else:
                        st.success(f"Found {result.total_matches} matching entries (showing {len(result.entries)})")
                        
                        # Display in a dataframe
                        entries_df = pd.DataFrame([
                            {
                                'ID': entry.id,
                                'Type': entry.entry_type,
                                'Title': entry.title,
                                'Created By': entry.created_by,
                                'Created At': format_datetime(entry.created_at),
                                'Is Key': "✓" if entry.is_key_context else ""
                            }
                            for entry in result.entries
                        ])
                        
                        st.dataframe(entries_df, hide_index=True, use_container_width=True)
                except Exception as e:
                    st.error(f"Error searching context: {str(e)}")
    
    with tab3:
        st.subheader("Add New Context")
        
        with st.form(key="add_context_form"):
            context_title = st.text_input("Title")
            
            col1, col2 = st.columns(2)
            
            with col1:
                context_type = st.selectbox("Type", [t.value for t in ContextType])
            
            with col2:
                context_tags = st.text_input("Tags (comma separated)")
            
            context_content = st.text_area("Content", height=200)
            
            # Metadata as JSON
            st.write("**Metadata (JSON, optional):**")
            metadata_json = st.text_area("Metadata", value="{}", height=100)
            
            is_key = st.checkbox("Mark as Key Context")
            
            add_button = st.form_submit_button("Add Context")
            
            if add_button:
                try:
                    # Parse metadata
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    # Parse tags
                    tags = [t.strip() for t in context_tags.split(',')] if context_tags else []
                    
                    # Store context
                    entry_id = run_async(
                        st.session_state.system.context_store.store(
                            entry_type=context_type,
                            title=context_title,
                            content=context_content,
                            created_by="human_user",
                            tags=tags,
                            metadata=metadata,
                            is_key_context=is_key
                        )
                    )
                    
                    st.success(f"Context added successfully! Entry ID: {entry_id}")
                    
                    # Refresh entries
                    refresh_data()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for metadata")
                except Exception as e:
                    st.error(f"Error adding context: {str(e)}")

def render_feedback_page():
    """Render the feedback and learning page."""
    st.title("📝 Feedback & Learning")
    
    if not st.session_state.system:
        st.info("System not initialized. Please initialize the system from the sidebar.")
        return
    
    if not st.session_state.system.feedback_processor:
        st.warning("Feedback processor not available in this system.")
        return
    
    # Get feedback metrics
    metrics = st.session_state.system.feedback_processor.get_feedback_metrics()
    
    # Display metrics in columns
    st.subheader("Feedback Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback Items", metrics.total_feedback_items)
    
    with col2:
        st.metric("Approval Rate", f"{metrics.approval_rate:.1f}%")
    
    with col3:
        st.metric("Revision Rate", f"{metrics.revision_rate:.1f}%")
    
    with col4:
        st.metric("Avg. Revisions per Task", f"{metrics.average_revisions_per_task:.2f}")
    
    # Feedback breakdown
    st.subheader("Feedback Breakdown")
    
    tab1, tab2, tab3 = st.tabs(["By Agent", "By Category", "Common Issues"])
    
    with tab1:
        # Feedback by agent
        if metrics.feedback_by_agent:
            agent_feedback = pd.DataFrame({
                'Agent': list(metrics.feedback_by_agent.keys()),
                'Count': list(metrics.feedback_by_agent.values())
            })
            
            fig = px.bar(
                agent_feedback,
                x='Agent',
                y='Count',
                title="Feedback by Agent",
                labels={'x': 'Agent', 'y': 'Feedback Count'},
                color='Agent',
                color_discrete_sequence=[COLORS["primary"]]
            )
            
            fig.update_layout(
                xaxis={'categoryorder': 'total descending'},
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent feedback data available.")
    
    with tab2:
        # Feedback by category
        if metrics.feedback_by_category:
            category_feedback = pd.DataFrame({
                'Category': list(metrics.feedback_by_category.keys()),
                'Count': list(metrics.feedback_by_category.values())
            })
            
            fig = px.pie(
                category_feedback,
                names='Category',
                values='Count',
                title="Feedback by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category feedback data available.")
    
    with tab3:
        # Common issues
        if metrics.common_issues:
            st.write("**Most Common Issues:**")
            
            for issue in metrics.common_issues:
                st.write(f"- **{issue['issue']}**: {issue['count']} occurrences")
        else:
            st.info("No common issues data available.")
    
    # Agent improvement suggestions
    st.subheader("Agent Improvement Suggestions")
    
    agents = st.session_state.get('agents', [])
    
    if not agents:
        st.info("No agents available.")
    else:
        selected_agent_id = st.selectbox(
            "Select Agent",
            [a['id'] for a in agents],
            format_func=lambda x: next((f"{a['name']} ({a['type']})" for a in agents if a['id'] == x), x)
        )
        
        if selected_agent_id:
            try:
                # Get improvement suggestions
                suggestions = run_async(st.session_state.system.feedback_processor.suggest_improvements(selected_agent_id))
                
                if not suggestions or not suggestions.get('suggestions'):
                    st.info("No improvement suggestions available for this agent.")
                else:
                    # Display suggestions
                    for suggestion in suggestions['suggestions']:
                        with st.expander(f"{suggestion['area']} - {suggestion['priority']} Priority"):
                            st.write(f"**Suggestion:** {suggestion['suggestion']}")
                            st.write(f"**Details:** {suggestion['details']}")
                            st.write(f"**Based on:** {suggestion.get('feedback_count', 'N/A')} feedback items")
            except Exception as e:
                st.error(f"Error getting improvement suggestions: {str(e)}")

def render_settings_page():
    """Render the settings page."""
    st.title("⚙️ System Settings")
    
    # System initialization settings
    st.subheader("System Configuration")
    
    config_path = st.text_input(
        "Configuration File Path",
        value=st.session_state.get('config_path', ''),
        help="Path to the system configuration file (optional)"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_provider = st.selectbox(
            "Default Model Provider",
            ["anthropic", "openai", "google"],
            index=0,
            help="Default model provider for all agents"
        )
    
    with col2:
        debug_mode = st.checkbox(
            "Debug Mode",
            value=st.session_state.get('debug_mode', False),
            help="Enable debug logging"
        )
    
    # Save settings
    if st.button("Save Settings"):
        st.session_state.config_path = config_path
        st.session_state.model_provider = model_provider
        st.session_state.debug_mode = debug_mode
        
        st.success("Settings saved. Restart the system to apply changes.")
    
    # System stats
    if 'system' in st.session_state and st.session_state.system:
        st.subheader("System Information")
        
        status = st.session_state.get('system_status', {})
        
        # Display system info
        st.json({
            "instance_id": status.get('instance_id', 'Unknown'),
            "version": status.get('version', 'Unknown'),
            "uptime": format_time(status.get('uptime_seconds', 0)),
            "debug_mode": status.get('debug_mode', False),
            "agent_count": status.get('agent_count', 0),
            "tasks": status.get('tasks', {}),
            "workflows": status.get('workflows', {})
        })
        
        # Advanced actions
        st.subheader("Advanced Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Reset System", use_container_width=True):
                shutdown_system()
                st.session_state.clear()
                st.experimental_rerun()
        
        with col2:
            if st.button("Purge All Data", use_container_width=True):
                if st.session_state.system:
                    # Clear context store
                    if st.session_state.system.context_store:
                        run_async(st.session_state.system.context_store.clear_all())
                    
                    # Clear shared memory
                    if st.session_state.system.shared_memory:
                        run_async(st.session_state.system.shared_memory.clear_all())
                    
                    st.success("All data purged successfully.")
                    
                    # Refresh data
                    refresh_data()

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Check for automatic refresh
    current_time = datetime.now()
    if st.session_state.refresh_data and st.session_state.system:
        if (current_time - st.session_state.last_refresh).total_seconds() > REFRESH_INTERVAL:
            refresh_data()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Agents":
        render_agents_page()
    elif page == "Tasks":
        render_tasks_page()
    elif page == "Workflows":
        render_workflows_page()
    elif page == "Context Storage":
        render_context_page()
    elif page == "Feedback":
        render_feedback_page()
    elif page == "Settings":
        render_settings_page()
    
    # Schedule next refresh
    st.session_state.refresh_data = True

if __name__ == "__main__":
    main()
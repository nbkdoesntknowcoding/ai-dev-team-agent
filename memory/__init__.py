"""
Memory module for the multi-agent development system.

This package provides storage solutions for agent memory, context retention,
and shared state across agents. It enables information persistence, knowledge
sharing, and long-term learning capabilities for the multi-agent system.
"""

# Version information
__version__ = "0.1.0"

# Import primary memory classes
from memory.shared_memory import (
    SharedMemory,
    StorageBackend as SharedStorageBackend,
    MemoryOperation,
    MemoryEventListener,
    MemoryItem,
    MemoryStats,
    MemoryLogger
)

from memory.context_store import (
    ContextStore,
    ContextType,
    StorageBackend as ContextStorageBackend,
    ContextEntry,
    SearchQuery,
    SearchResult
)

# Factory functions for easy instantiation
def create_shared_memory(
    storage_type: str = "memory",
    storage_path: str = None,
    redis_url: str = None,
    cache_size: int = 1000,
    enable_locking: bool = True
) -> SharedMemory:
    """Create a configured shared memory instance.
    
    Args:
        storage_type: Type of storage backend ("memory", "file", "redis", "custom")
        storage_path: Path for file-based storage
        redis_url: URL for Redis connection
        cache_size: Maximum number of items to keep in memory cache
        enable_locking: Whether to enable locking for concurrent access
        
    Returns:
        Configured SharedMemory instance
    """
    return SharedMemory(
        storage_backend=storage_type,
        storage_path=storage_path,
        redis_url=redis_url,
        cache_size=cache_size,
        enable_locking=enable_locking
    )


def create_context_store(
    storage_type: str = "memory",
    storage_path: str = None,
    compression: bool = True,
    max_memory_entries: int = 10000
) -> ContextStore:
    """Create a configured context store instance.
    
    Args:
        storage_type: Type of storage backend ("memory", "filesystem", "json", "pickle", "custom")
        storage_path: Path for file-based storage
        compression: Whether to compress stored data
        max_memory_entries: Maximum entries to keep in memory
        
    Returns:
        Configured ContextStore instance
    """
    return ContextStore(
        storage_backend=storage_type,
        storage_path=storage_path,
        compression=compression,
        max_memory_entries=max_memory_entries
    )


# Constants
DEFAULT_CONTEXT_TYPES = set(ContextType)

MEMORY_CATEGORIES = {
    "agent_state": "Internal agent state information",
    "task_results": "Results of completed tasks",
    "artifacts": "Generated artifacts",
    "decisions": "Decision points and rationales",
    "code": "Code snippets and files",
    "requirements": "Project requirements",
    "architecture": "System architecture",
    "errors": "Error records",
    "metrics": "Performance metrics",
    "conversations": "Agent conversations",
    "user_input": "User inputs and preferences",
    "system": "System configuration and state"
}

# Public API
__all__ = [
    # Shared Memory
    "SharedMemory",
    "SharedStorageBackend",
    "MemoryOperation",
    "MemoryEventListener",
    "MemoryItem",
    "MemoryStats",
    "MemoryLogger",
    
    # Context Store
    "ContextStore",
    "ContextType",
    "ContextStorageBackend",
    "ContextEntry",
    "SearchQuery",
    "SearchResult",
    
    # Factory Functions
    "create_shared_memory",
    "create_context_store",
    
    # Constants
    "DEFAULT_CONTEXT_TYPES",
    "MEMORY_CATEGORIES"
]

# Version compatibility check
import sys
if sys.version_info < (3, 8):
    import warnings
    warnings.warn(
        "The memory package requires Python 3.8 or newer. "
        "Some features may not work as expected.",
        ImportWarning
    )
"""
Shared memory implementation for the multi-agent development system.

This module provides a central memory system that enables different agents to
share information, maintain context, and collaborate effectively. It supports
different storage backends, data structures optimized for agent communication,
and memory management strategies to handle long-running sessions.
"""

import asyncio
import json
import logging
import os
import time
import uuid
import pickle
import threading
from collections import defaultdict
from contextlib import contextmanager
import hashlib
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast, Generator

try:
    import aiofiles
    ASYNC_IO_AVAILABLE = True
except ImportError:
    ASYNC_IO_AVAILABLE = False

try:
    import redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from pydantic import BaseModel, Field, validator

# Set up logging
logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Available storage backends for shared memory."""
    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    CUSTOM = "custom"


class MemoryItem(BaseModel):
    """A single item stored in shared memory."""
    key: str
    value: Any
    category: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    ttl: Optional[int] = None  # Time to live in seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)
    lock_id: Optional[str] = None
    lock_expiry: Optional[float] = None


class MemoryOperation(str, Enum):
    """Types of operations that can be performed on memory."""
    STORE = "store"
    RETRIEVE = "retrieve"
    DELETE = "delete"
    UPDATE = "update"
    LIST = "list"
    CLEAR = "clear"


class MemoryEventListener:
    """Interface for memory event listeners."""
    
    async def on_memory_event(
        self,
        operation: MemoryOperation,
        key: str,
        category: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when a memory event occurs.
        
        Args:
            operation: The operation that was performed
            key: The key that was affected
            category: The category of the item
            value: The value (for STORE and UPDATE operations)
            metadata: Additional metadata about the operation
        """
        pass


class MemoryStats(BaseModel):
    """Statistics about memory usage."""
    total_items: int = 0
    items_by_category: Dict[str, int] = Field(default_factory=dict)
    operations_count: Dict[str, int] = Field(default_factory=dict)
    storage_size_bytes: Optional[int] = None
    cache_hits: int = 0
    cache_misses: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class SharedMemory:
    """Shared memory for communication between agents in the multi-agent system."""
    
    def __init__(
        self,
        storage_backend: Union[StorageBackend, str] = StorageBackend.MEMORY,
        storage_path: Optional[str] = None,
        redis_url: Optional[str] = None,
        cache_size: int = 1000,
        enable_locking: bool = True,
        compress_large_values: bool = True,
        listeners: Optional[List[MemoryEventListener]] = None,
        custom_serializer: Optional[Any] = None,
        custom_deserializer: Optional[Any] = None,
    ):
        """Initialize the shared memory.
        
        Args:
            storage_backend: Backend storage system to use
            storage_path: Path for file-based storage
            redis_url: URL for Redis connection
            cache_size: Maximum number of items to keep in memory cache
            enable_locking: Whether to enable locking for concurrent access
            compress_large_values: Whether to compress large values
            listeners: Optional list of event listeners
            custom_serializer: Optional custom serializer for storage
            custom_deserializer: Optional custom deserializer for retrieval
        """
        # Convert string to enum if needed
        if isinstance(storage_backend, str):
            storage_backend = StorageBackend(storage_backend)
        
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path) if storage_path else None
        self.redis_url = redis_url
        self.cache_size = cache_size
        self.enable_locking = enable_locking
        self.compress_large_values = compress_large_values
        self.listeners = listeners or []
        self.custom_serializer = custom_serializer
        self.custom_deserializer = custom_deserializer
        
        # In-memory storage (serves as cache for persistent backends)
        self._memory: Dict[str, MemoryItem] = {}
        self._category_keys: Dict[str, Set[str]] = defaultdict(set)
        
        # Stats tracking
        self._stats = MemoryStats()
        self._op_counts: Dict[str, int] = defaultdict(int)
        
        # Cache tracking
        self._cache_hits = 0
        self._cache_misses = 0
        
        # For time-based expiry
        self._expiry_times: Dict[str, float] = {}
        
        # Threading lock for concurrent access
        self._lock = threading.RLock()
        
        # Redis client (if Redis backend is used)
        self._redis_client = None
        
        # Initialize the backend
        self._initialize_backend()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Shared memory initialized with {storage_backend} backend")
    
    def _initialize_backend(self) -> None:
        """Initialize the storage backend."""
        if self.storage_backend == StorageBackend.MEMORY:
            # No initialization needed for memory backend
            pass
        
        elif self.storage_backend == StorageBackend.FILE:
            if not self.storage_path:
                raise ValueError("Storage path is required for file backend")
            
            # Create directory if it doesn't exist
            if not self.storage_path.exists():
                self.storage_path.mkdir(parents=True)
                logger.info(f"Created storage directory: {self.storage_path}")
            
            # Create category directories
            categories_dir = self.storage_path / "categories"
            if not categories_dir.exists():
                categories_dir.mkdir(parents=True)
            
            # Create metadata directory
            metadata_dir = self.storage_path / "metadata"
            if not metadata_dir.exists():
                metadata_dir.mkdir(parents=True)
        
        elif self.storage_backend == StorageBackend.REDIS:
            if not REDIS_AVAILABLE:
                raise ImportError("Redis is required for Redis backend. Install with 'pip install redis'")
            
            if not self.redis_url:
                self.redis_url = "redis://localhost:6379/0"
            
            # Initialize Redis client
            self._redis_client = redis.Redis.from_url(self.redis_url)
            
            # Test connection
            try:
                self._redis_client.ping()
                logger.info("Connected to Redis successfully")
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise
        
        elif self.storage_backend == StorageBackend.CUSTOM:
            # Custom initialization would be implemented by the user
            pass
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for maintenance."""
        # Create a daemon thread for expiry checking
        if self.storage_backend == StorageBackend.MEMORY:
            expiry_thread = threading.Thread(
                target=self._check_expirations_loop,
                daemon=True
            )
            expiry_thread.start()
    
    def _check_expirations_loop(self) -> None:
        """Background loop to check for expired items."""
        while True:
            try:
                # Check expirations every second
                time.sleep(1)
                self._check_expirations()
            except Exception as e:
                logger.error(f"Error in expiration check: {str(e)}")
    
    def _check_expirations(self) -> None:
        """Check for and remove expired items."""
        now = time.time()
        expired_keys = []
        
        with self._lock:
            # Find expired keys
            for key, expiry_time in list(self._expiry_times.items()):
                if now > expiry_time:
                    expired_keys.append(key)
            
            # Remove expired items
            for key in expired_keys:
                if key in self._memory:
                    category = self._memory[key].category
                    del self._memory[key]
                    self._category_keys[category].discard(key)
                    del self._expiry_times[key]
        
        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired items from memory")
    
    async def store(
        self,
        key: str,
        value: Any,
        category: str = "default",
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a value in shared memory.
        
        Args:
            key: Unique identifier for the value
            value: The value to store
            category: Category for organizing values
            ttl: Time to live in seconds
            metadata: Optional metadata for the value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the memory item
            item = MemoryItem(
                key=key,
                value=value,
                category=category,
                ttl=ttl,
                metadata=metadata or {}
            )
            
            # Calculate expiry time if TTL is provided
            expiry_time = None
            if ttl is not None:
                expiry_time = time.time() + ttl
            
            # Store in memory
            with self._lock:
                self._memory[key] = item
                self._category_keys[category].add(key)
                
                # Set expiry if needed
                if expiry_time:
                    self._expiry_times[key] = expiry_time
                elif key in self._expiry_times:
                    del self._expiry_times[key]
                
                # Update stats
                self._op_counts[MemoryOperation.STORE.value] += 1
                if category in self._stats.items_by_category:
                    self._stats.items_by_category[category] += 1
                else:
                    self._stats.items_by_category[category] = 1
                
                # Ensure we don't exceed cache size
                self._enforce_cache_limits()
            
            # Store in persistent backend if configured
            if self.storage_backend != StorageBackend.MEMORY:
                await self._store_in_backend(key, item)
            
            # Notify listeners
            await self._notify_listeners(
                MemoryOperation.STORE,
                key,
                category,
                value,
                metadata
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing {key} in {category}: {str(e)}")
            return False
    
    async def _store_in_backend(self, key: str, item: MemoryItem) -> None:
        """Store an item in the backend storage.
        
        Args:
            key: The key for the item
            item: The MemoryItem to store
        """
        if self.storage_backend == StorageBackend.FILE:
            await self._store_in_file(key, item)
        elif self.storage_backend == StorageBackend.REDIS:
            await self._store_in_redis(key, item)
        elif self.storage_backend == StorageBackend.CUSTOM:
            await self._store_in_custom(key, item)
    
    async def _store_in_file(self, key: str, item: MemoryItem) -> None:
        """Store an item in the file backend.
        
        Args:
            key: The key for the item
            item: The MemoryItem to store
        """
        if not self.storage_path:
            raise ValueError("Storage path is required for file backend")
        
        # Create category directory if it doesn't exist
        category_dir = self.storage_path / "categories" / item.category
        if not category_dir.exists():
            category_dir.mkdir(parents=True)
        
        # Prepare the item for storage
        storage_item = item.dict()
        
        # Handle non-serializable values
        try:
            if self.custom_serializer:
                serialized_value = self.custom_serializer(storage_item["value"])
                storage_item["value"] = serialized_value
            else:
                # Test if JSON serializable
                json.dumps(storage_item["value"])
        except (TypeError, OverflowError):
            # If not JSON serializable, pickle it
            if self.custom_serializer:
                storage_item["value"] = self.custom_serializer(storage_item["value"])
            else:
                pickled_value = pickle.dumps(storage_item["value"])
                storage_item["value"] = {"__pickled__": True, "data": pickled_value.hex()}
        
        # Prepare filename
        safe_key = self._safe_filename(key)
        file_path = category_dir / f"{safe_key}.json"
        
        # Write to file
        if ASYNC_IO_AVAILABLE:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(storage_item, indent=2))
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(storage_item, f, indent=2)
        
        # Update category index
        await self._update_category_index(item.category, key)
    
    async def _store_in_redis(self, key: str, item: MemoryItem) -> None:
        """Store an item in Redis.
        
        Args:
            key: The key for the item
            item: The MemoryItem to store
        """
        if not self._redis_client:
            raise ValueError("Redis client not initialized")
        
        # Create Redis key
        redis_key = f"memory:{item.category}:{key}"
        
        # Serialize the item
        if self.custom_serializer:
            serialized = self.custom_serializer(item.dict())
        else:
            try:
                serialized = json.dumps(item.dict())
            except (TypeError, OverflowError):
                # If not JSON serializable, pickle the value
                item_dict = item.dict()
                item_dict["value"] = {"__pickled__": True, "data": pickle.dumps(item_dict["value"]).hex()}
                serialized = json.dumps(item_dict)
        
        # Store in Redis
        await self._redis_client.set(redis_key, serialized)
        
        # Set expiry if needed
        if item.ttl:
            await self._redis_client.expire(redis_key, item.ttl)
        
        # Add to category set
        await self._redis_client.sadd(f"category:{item.category}", key)
    
    async def _store_in_custom(self, key: str, item: MemoryItem) -> None:
        """Store an item using custom backend.
        
        Args:
            key: The key for the item
            item: The MemoryItem to store
        """
        # This would be implemented by the user
        pass
    
    async def _update_category_index(self, category: str, key: str) -> None:
        """Update the category index file.
        
        Args:
            category: The category
            key: The key to add
        """
        if not self.storage_path:
            return
        
        # Create index file path
        index_file = self.storage_path / "metadata" / f"category_{category}.txt"
        
        if ASYNC_IO_AVAILABLE:
            # Check if key already exists in index
            if index_file.exists():
                async with aiofiles.open(index_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if key in content.splitlines():
                        return
            
            # Append key to index
            async with aiofiles.open(index_file, 'a', encoding='utf-8') as f:
                await f.write(f"{key}\n")
        else:
            # Check if key already exists in index
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    if key in f.read().splitlines():
                        return
            
            # Append key to index
            with open(index_file, 'a', encoding='utf-8') as f:
                f.write(f"{key}\n")
    
    async def retrieve(
        self,
        key: str,
        category: str = "default",
        default: Any = None
    ) -> Any:
        """Retrieve a value from shared memory.
        
        Args:
            key: Key of the value to retrieve
            category: Category of the value
            default: Default value if not found
            
        Returns:
            The stored value if found, default otherwise
        """
        try:
            # Check in-memory cache first
            with self._lock:
                if key in self._memory:
                    item = self._memory[key]
                    if item.category == category:
                        self._cache_hits += 1
                        self._op_counts[MemoryOperation.RETRIEVE.value] += 1
                        return item.value
                self._cache_misses += 1
            
            # If not in memory, check backend storage
            if self.storage_backend != StorageBackend.MEMORY:
                item = await self._retrieve_from_backend(key, category)
                if item:
                    # Cache for future use
                    with self._lock:
                        self._memory[key] = item
                        self._category_keys[category].add(key)
                        
                        # Set expiry if needed
                        if item.ttl:
                            self._expiry_times[key] = time.time() + item.ttl
                        
                        # Enforce cache limits
                        self._enforce_cache_limits()
                        
                        self._op_counts[MemoryOperation.RETRIEVE.value] += 1
                    
                    # Notify listeners
                    await self._notify_listeners(
                        MemoryOperation.RETRIEVE,
                        key,
                        category,
                        item.value,
                        item.metadata
                    )
                    
                    return item.value
            
            # Notify listeners for cache miss
            await self._notify_listeners(
                MemoryOperation.RETRIEVE,
                key,
                category,
                None,
                {"found": False}
            )
            
            return default
        
        except Exception as e:
            logger.error(f"Error retrieving {key} from {category}: {str(e)}")
            return default
    
    async def _retrieve_from_backend(self, key: str, category: str) -> Optional[MemoryItem]:
        """Retrieve an item from the backend storage.
        
        Args:
            key: The key to retrieve
            category: The category of the item
            
        Returns:
            MemoryItem if found, None otherwise
        """
        if self.storage_backend == StorageBackend.FILE:
            return await self._retrieve_from_file(key, category)
        elif self.storage_backend == StorageBackend.REDIS:
            return await self._retrieve_from_redis(key, category)
        elif self.storage_backend == StorageBackend.CUSTOM:
            return await self._retrieve_from_custom(key, category)
        
        return None
    
    async def _retrieve_from_file(self, key: str, category: str) -> Optional[MemoryItem]:
        """Retrieve an item from the file backend.
        
        Args:
            key: The key to retrieve
            category: The category of the item
            
        Returns:
            MemoryItem if found, None otherwise
        """
        if not self.storage_path:
            return None
        
        # Construct file path
        safe_key = self._safe_filename(key)
        file_path = self.storage_path / "categories" / category / f"{safe_key}.json"
        
        if not file_path.exists():
            return None
        
        try:
            # Read the file
            if ASYNC_IO_AVAILABLE:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    item_data = json.loads(content)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    item_data = json.load(f)
            
            # Handle pickled values
            if isinstance(item_data["value"], dict) and item_data["value"].get("__pickled__"):
                if self.custom_deserializer:
                    item_data["value"] = self.custom_deserializer(item_data["value"])
                else:
                    pickled_data = bytes.fromhex(item_data["value"]["data"])
                    item_data["value"] = pickle.loads(pickled_data)
            
            # Create MemoryItem
            return MemoryItem(**item_data)
        
        except Exception as e:
            logger.error(f"Error reading {key} from {category}: {str(e)}")
            return None
    
    async def _retrieve_from_redis(self, key: str, category: str) -> Optional[MemoryItem]:
        """Retrieve an item from Redis.
        
        Args:
            key: The key to retrieve
            category: The category of the item
            
        Returns:
            MemoryItem if found, None otherwise
        """
        if not self._redis_client:
            return None
        
        # Construct Redis key
        redis_key = f"memory:{category}:{key}"
        
        try:
            # Get from Redis
            data = await self._redis_client.get(redis_key)
            if not data:
                return None
            
            # Deserialize
            item_data = json.loads(data)
            
            # Handle pickled values
            if isinstance(item_data["value"], dict) and item_data["value"].get("__pickled__"):
                if self.custom_deserializer:
                    item_data["value"] = self.custom_deserializer(item_data["value"])
                else:
                    pickled_data = bytes.fromhex(item_data["value"]["data"])
                    item_data["value"] = pickle.loads(pickled_data)
            
            # Create MemoryItem
            return MemoryItem(**item_data)
        
        except Exception as e:
            logger.error(f"Error reading {key} from Redis: {str(e)}")
            return None
    
    async def _retrieve_from_custom(self, key: str, category: str) -> Optional[MemoryItem]:
        """Retrieve an item using custom backend.
        
        Args:
            key: The key to retrieve
            category: The category of the item
            
        Returns:
            MemoryItem if found, None otherwise
        """
        # This would be implemented by the user
        return None
    
    async def delete(self, key: str, category: str = "default") -> bool:
        """Delete a value from shared memory.
        
        Args:
            key: Key of the value to delete
            category: Category of the value
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            deleted = False
            value = None
            metadata = None
            
            # Delete from memory
            with self._lock:
                if key in self._memory:
                    item = self._memory[key]
                    if item.category == category:
                        value = item.value
                        metadata = item.metadata
                        del self._memory[key]
                        self._category_keys[category].discard(key)
                        if key in self._expiry_times:
                            del self._expiry_times[key]
                        deleted = True
                        
                        # Update stats
                        if category in self._stats.items_by_category and self._stats.items_by_category[category] > 0:
                            self._stats.items_by_category[category] -= 1
                            
                        self._op_counts[MemoryOperation.DELETE.value] += 1
            
            # Delete from backend storage
            if self.storage_backend != StorageBackend.MEMORY:
                backend_deleted = await self._delete_from_backend(key, category)
                deleted = deleted or backend_deleted
            
            # Notify listeners
            if deleted:
                await self._notify_listeners(
                    MemoryOperation.DELETE,
                    key,
                    category,
                    value,
                    metadata
                )
            
            return deleted
        
        except Exception as e:
            logger.error(f"Error deleting {key} from {category}: {str(e)}")
            return False
    
    async def _delete_from_backend(self, key: str, category: str) -> bool:
        """Delete an item from the backend storage.
        
        Args:
            key: The key to delete
            category: The category of the item
            
        Returns:
            True if deleted, False otherwise
        """
        if self.storage_backend == StorageBackend.FILE:
            return await self._delete_from_file(key, category)
        elif self.storage_backend == StorageBackend.REDIS:
            return await self._delete_from_redis(key, category)
        elif self.storage_backend == StorageBackend.CUSTOM:
            return await self._delete_from_custom(key, category)
        
        return False
    
    async def _delete_from_file(self, key: str, category: str) -> bool:
        """Delete an item from the file backend.
        
        Args:
            key: The key to delete
            category: The category of the item
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.storage_path:
            return False
        
        # Construct file path
        safe_key = self._safe_filename(key)
        file_path = self.storage_path / "categories" / category / f"{safe_key}.json"
        
        if not file_path.exists():
            return False
        
        try:
            # Delete the file
            file_path.unlink()
            
            # Update category index
            await self._remove_from_category_index(category, key)
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting {key} from {category}: {str(e)}")
            return False
    
    async def _delete_from_redis(self, key: str, category: str) -> bool:
        """Delete an item from Redis.
        
        Args:
            key: The key to delete
            category: The category of the item
            
        Returns:
            True if deleted, False otherwise
        """
        if not self._redis_client:
            return False
        
        # Construct Redis key
        redis_key = f"memory:{category}:{key}"
        
        try:
            # Delete from Redis
            deleted = await self._redis_client.delete(redis_key)
            
            # Remove from category set
            await self._redis_client.srem(f"category:{category}", key)
            
            return deleted > 0
        
        except Exception as e:
            logger.error(f"Error deleting {key} from Redis: {str(e)}")
            return False
    
    async def _delete_from_custom(self, key: str, category: str) -> bool:
        """Delete an item using custom backend.
        
        Args:
            key: The key to delete
            category: The category of the item
            
        Returns:
            True if deleted, False otherwise
        """
        # This would be implemented by the user
        return False
    
    async def _remove_from_category_index(self, category: str, key: str) -> None:
        """Remove a key from the category index file.
        
        Args:
            category: The category
            key: The key to remove
        """
        if not self.storage_path:
            return
        
        # Create index file path
        index_file = self.storage_path / "metadata" / f"category_{category}.txt"
        
        if not index_file.exists():
            return
        
        try:
            if ASYNC_IO_AVAILABLE:
                # Read existing keys
                async with aiofiles.open(index_file, 'r', encoding='utf-8') as f:
                    keys = [line.strip() for line in await f.readlines()]
                
                # Remove the key
                if key in keys:
                    keys.remove(key)
                    
                    # Write back
                    async with aiofiles.open(index_file, 'w', encoding='utf-8') as f:
                        await f.write('\n'.join(keys) + ('\n' if keys else ''))
            else:
                # Read existing keys
                with open(index_file, 'r', encoding='utf-8') as f:
                    keys = [line.strip() for line in f.readlines()]
                
                # Remove the key
                if key in keys:
                    keys.remove(key)
                    
                    # Write back
                    with open(index_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(keys) + ('\n' if keys else ''))
        
        except Exception as e:
            logger.error(f"Error updating category index for {category}: {str(e)}")
    
    async def get_keys(self, category: str = "default") -> List[str]:
        """Get all keys in a category.
        
        Args:
            category: Category to get keys from
            
        Returns:
            List of keys
        """
        keys = set()
        
        # Get keys from memory
        with self._lock:
            keys.update(self._category_keys.get(category, set()))
            self._op_counts[MemoryOperation.LIST.value] += 1
        
        # Get keys from backend storage
        if self.storage_backend != StorageBackend.MEMORY:
            backend_keys = await self._get_keys_from_backend(category)
            keys.update(backend_keys)
        
        # Notify listeners
        await self._notify_listeners(
            MemoryOperation.LIST,
            "",
            category,
            None,
            {"count": len(keys)}
        )
        
        return list(keys)
    
    async def _get_keys_from_backend(self, category: str) -> Set[str]:
        """Get keys from the backend storage.
        
        Args:
            category: Category to get keys from
            
        Returns:
            Set of keys
        """
        if self.storage_backend == StorageBackend.FILE:
            return await self._get_keys_from_file(category)
        elif self.storage_backend == StorageBackend.REDIS:
            return await self._get_keys_from_redis(category)
        elif self.storage_backend == StorageBackend.CUSTOM:
            return await self._get_keys_from_custom(category)
        
        return set()
    
    async def _get_keys_from_file(self, category: str) -> Set[str]:
        """Get keys from the file backend.
        
        Args:
            category: Category to get keys from
            
        Returns:
            Set of keys
        """
        if not self.storage_path:
            return set()
        
        # Try to read from category index
        index_file = self.storage_path / "metadata" / f"category_{category}.txt"
        if index_file.exists():
            try:
                if ASYNC_IO_AVAILABLE:
                    async with aiofiles.open(index_file, 'r', encoding='utf-8') as f:
                        return {line.strip() for line in await f.readlines() if line.strip()}
                else:
                    with open(index_file, 'r', encoding='utf-8') as f:
                        return {line.strip() for line in f.readlines() if line.strip()}
            except Exception as e:
                logger.error(f"Error reading category index for {category}: {str(e)}")
        
        # If index doesn't exist, scan directory
        category_dir = self.storage_path / "categories" / category
        if not category_dir.exists():
            return set()
        
        keys = set()
        for file_path in category_dir.glob("*.json"):
            # Extract key from filename
            key = file_path.stem
            keys.add(key)
        
        return keys
    
    async def _get_keys_from_redis(self, category: str) -> Set[str]:
        """Get keys from Redis.
        
        Args:
            category: Category to get keys from
            
        Returns:
            Set of keys
        """
        if not self._redis_client:
            return set()
        
        try:
            # Get keys from category set
            redis_keys = await self._redis_client.smembers(f"category:{category}")
            return {key.decode('utf-8') for key in redis_keys}
        except Exception as e:
            logger.error(f"Error getting keys from Redis for {category}: {str(e)}")
            return set()
    
    async def _get_keys_from_custom(self, category: str) -> Set[str]:
        """Get keys using custom backend.
        
        Args:
            category: Category to get keys from
            
        Returns:
            Set of keys
        """
        # This would be implemented by the user
        return set()
    
    async def get_categories(self) -> List[str]:
        """Get all categories.
        
        Returns:
            List of categories
        """
        categories = set()
        
        # Get categories from memory
        with self._lock:
            categories.update(self._category_keys.keys())
        
        # Get categories from backend storage
        if self.storage_backend != StorageBackend.MEMORY:
            backend_categories = await self._get_categories_from_backend()
            categories.update(backend_categories)
        
        return list(categories)
    
    async def _get_categories_from_backend(self) -> Set[str]:
        """Get categories from the backend storage.
        
        Returns:
            Set of categories
        """
        if self.storage_backend == StorageBackend.FILE:
            return await self._get_categories_from_file()
        elif self.storage_backend == StorageBackend.REDIS:
            return await self._get_categories_from_redis()
        elif self.storage_backend == StorageBackend.CUSTOM:
            return await self._get_categories_from_custom()
        
        return set()
    
    async def _get_categories_from_file(self) -> Set[str]:
        """Get categories from the file backend.
        
        Returns:
            Set of categories
        """
        if not self.storage_path:
            return set()
        
        # Scan categories directory
        categories_dir = self.storage_path / "categories"
        if not categories_dir.exists():
            return set()
        
        return {dir_path.name for dir_path in categories_dir.glob("*") if dir_path.is_dir()}
    
    async def _get_categories_from_redis(self) -> Set[str]:
        """Get categories from Redis.
        
        Returns:
            Set of categories
        """
        if not self._redis_client:
            return set()
        
        try:
            # Scan for category keys
            categories = set()
            cursor = b'0'
            pattern = "category:*"
            
            while cursor:
                cursor, keys = await self._redis_client.scan(cursor=cursor, match=pattern)
                
                for key in keys:
                    category = key.decode('utf-8').split(':', 1)[1]
                    categories.add(category)
                
                if cursor == b'0':
                    break
            
            return categories
        except Exception as e:
            logger.error(f"Error getting categories from Redis: {str(e)}")
            return set()
    
    async def _get_categories_from_custom(self) -> Set[str]:
        """Get categories using custom backend.
        
        Returns:
            Set of categories
        """
        # This would be implemented by the user
        return set()
    
    async def clear_category(self, category: str) -> int:
        """Clear all items in a category.
        
        Args:
            category: Category to clear
            
        Returns:
            Number of items cleared
        """
        try:
            # Get keys to clear
            keys = await self.get_keys(category)
            count = 0
            
            # Delete each key
            for key in keys:
                if await self.delete(key, category):
                    count += 1
            
            # Notify listeners
            await self._notify_listeners(
                MemoryOperation.CLEAR,
                "",
                category,
                None,
                {"count": count}
            )
            
            return count
        
        except Exception as e:
            logger.error(f"Error clearing category {category}: {str(e)}")
            return 0
    
    async def clear_all(self) -> int:
        """Clear all items from memory.
        
        Returns:
            Number of items cleared
        """
        try:
            count = 0
            
            # Get all categories
            categories = await self.get_categories()
            
            # Clear each category
            for category in categories:
                count += await self.clear_category(category)
            
            # Update stats
            with self._lock:
                self._stats.total_items = 0
                self._stats.items_by_category = {}
                self._stats.last_updated = datetime.now().isoformat()
                self._op_counts[MemoryOperation.CLEAR.value] += 1
            
            return count
        
        except Exception as e:
            logger.error(f"Error clearing all memory: {str(e)}")
            return 0
    
    async def update(
        self,
        key: str,
        value: Any,
        category: str = "default",
        update_metadata: Optional[Dict[str, Any]] = None,
        reset_ttl: Optional[int] = None
    ) -> bool:
        """Update an existing value in shared memory.
        
        Args:
            key: Key of the value to update
            value: New value
            category: Category of the value
            update_metadata: Metadata to update
            reset_ttl: New TTL to set
            
        Returns:
            True if updated, False otherwise
        """
        try:
            # Get existing item
            item = None
            with self._lock:
                if key in self._memory:
                    existing_item = self._memory[key]
                    if existing_item.category == category:
                        item = existing_item
            
            # If not in memory, try to retrieve from storage
            if not item and self.storage_backend != StorageBackend.MEMORY:
                item = await self._retrieve_from_backend(key, category)
            
            # If item doesn't exist, return False
            if not item:
                return False
            
            # Update the item
            item.value = value
            item.updated_at = datetime.now().isoformat()
            
            # Update metadata if provided
            if update_metadata:
                item.metadata.update(update_metadata)
            
            # Reset TTL if requested
            if reset_ttl is not None:
                item.ttl = reset_ttl
                
                # Update expiry time
                with self._lock:
                    if reset_ttl is None:
                        if key in self._expiry_times:
                            del self._expiry_times[key]
                    else:
                        self._expiry_times[key] = time.time() + reset_ttl
            
            # Store the updated item
            await self._store_entry(key, item)
            
            # Update stats
            with self._lock:
                self._op_counts[MemoryOperation.UPDATE.value] += 1
            
            # Notify listeners
            await self._notify_listeners(
                MemoryOperation.UPDATE,
                key,
                category,
                value,
                update_metadata
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating {key} in {category}: {str(e)}")
            return False
    
    async def _store_entry(self, key: str, item: MemoryItem) -> None:
        """Store a memory item.
        
        Args:
            key: Key for the item
            item: The item to store
        """
        # Store in memory
        with self._lock:
            self._memory[key] = item
            self._category_keys[item.category].add(key)
            
            # Set expiry if needed
            if item.ttl is not None:
                self._expiry_times[key] = time.time() + item.ttl
            elif key in self._expiry_times:
                del self._expiry_times[key]
        
        # Store in backend if needed
        if self.storage_backend != StorageBackend.MEMORY:
            await self._store_in_backend(key, item)
    
    def _enforce_cache_limits(self) -> None:
        """Enforce memory cache size limits."""
        if len(self._memory) <= self.cache_size:
            return
        
        # Calculate how many items to remove
        to_remove = len(self._memory) - self.cache_size
        
        # Sort items by access time (least recently used first)
        # For simplicity, we use updated_at as a proxy for access time
        items_by_age = sorted(
            self._memory.items(),
            key=lambda x: x[1].updated_at
        )
        
        # Remove oldest items
        for i in range(to_remove):
            if i < len(items_by_age):
                key, item = items_by_age[i]
                del self._memory[key]
                self._category_keys[item.category].discard(key)
                if key in self._expiry_times:
                    del self._expiry_times[key]
    
    async def _notify_listeners(
        self,
        operation: MemoryOperation,
        key: str,
        category: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify event listeners of a memory operation.
        
        Args:
            operation: Type of operation
            key: Key affected
            category: Category affected
            value: Value (for STORE and UPDATE operations)
            metadata: Additional metadata
        """
        if not self.listeners:
            return
        
        # Notify each listener
        for listener in self.listeners:
            try:
                await listener.on_memory_event(operation, key, category, value, metadata)
            except Exception as e:
                logger.error(f"Error notifying listener: {str(e)}")
    
    def _safe_filename(self, key: str) -> str:
        """Convert a key to a safe filename.
        
        Args:
            key: The key to convert
            
        Returns:
            Safe filename
        """
        # Hash the key if it's too long
        if len(key) > 100:
            return hashlib.md5(key.encode('utf-8')).hexdigest()
        
        # Replace unsafe characters
        safe_key = key
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
            safe_key = safe_key.replace(char, '_')
        
        return safe_key
    
    async def get_stats(self) -> MemoryStats:
        """Get memory statistics.
        
        Returns:
            Memory statistics
        """
        # Update stats
        with self._lock:
            self._stats.total_items = sum(len(keys) for keys in self._category_keys.values())
            self._stats.items_by_category = {
                category: len(keys) for category, keys in self._category_keys.items()
            }
            self._stats.operations_count = dict(self._op_counts)
            self._stats.cache_hits = self._cache_hits
            self._stats.cache_misses = self._cache_misses
            self._stats.last_updated = datetime.now().isoformat()
            
            # Try to calculate storage size
            if self.storage_backend == StorageBackend.FILE and self.storage_path:
                total_size = 0
                for category_dir in (self.storage_path / "categories").glob("*"):
                    if category_dir.is_dir():
                        for file in category_dir.glob("*.json"):
                            total_size += file.stat().st_size
                
                self._stats.storage_size_bytes = total_size
        
        return self._stats
    
    @contextmanager
    def acquire_lock(self, key: str, category: str = "default", timeout: float = 30.0) -> Generator[bool, None, None]:
        """Acquire a lock on a specific key.
        
        Args:
            key: Key to lock
            category: Category of the key
            timeout: Lock timeout in seconds
            
        Returns:
            True if lock acquired, False otherwise
        """
        if not self.enable_locking:
            yield True
            return
        
        lock_id = str(uuid.uuid4())
        lock_acquired = False
        
        try:
            # Try to acquire lock
            lock_acquired = self._try_acquire_lock(key, category, lock_id, timeout)
            
            if lock_acquired:
                yield True
            else:
                yield False
        
        finally:
            # Release lock if acquired
            if lock_acquired:
                self._release_lock(key, category, lock_id)
    
    def _try_acquire_lock(
        self,
        key: str,
        category: str,
        lock_id: str,
        timeout: float
    ) -> bool:
        """Try to acquire a lock.
        
        Args:
            key: Key to lock
            category: Category of the key
            lock_id: Unique ID for this lock
            timeout: Lock timeout in seconds
            
        Returns:
            True if lock acquired, False otherwise
        """
        with self._lock:
            # Check if key exists
            if key in self._memory:
                item = self._memory[key]
                if item.category == category:
                    # Check if already locked
                    if item.lock_id and item.lock_expiry and time.time() < item.lock_expiry:
                        return False
                    
                    # Set lock
                    item.lock_id = lock_id
                    item.lock_expiry = time.time() + timeout
                    return True
            
            # Key doesn't exist, create a placeholder with lock
            item = MemoryItem(
                key=key,
                value=None,
                category=category,
                lock_id=lock_id,
                lock_expiry=time.time() + timeout
            )
            
            self._memory[key] = item
            self._category_keys[category].add(key)
            return True
    
    def _release_lock(self, key: str, category: str, lock_id: str) -> bool:
        """Release a lock.
        
        Args:
            key: Key to unlock
            category: Category of the key
            lock_id: ID of the lock to release
            
        Returns:
            True if lock released, False otherwise
        """
        with self._lock:
            if key in self._memory:
                item = self._memory[key]
                if item.category == category and item.lock_id == lock_id:
                    item.lock_id = None
                    item.lock_expiry = None
                    return True
        
        return False
    
    async def exists(self, key: str, category: str = "default") -> bool:
        """Check if a key exists in memory.
        
        Args:
            key: Key to check
            category: Category of the key
            
        Returns:
            True if key exists, False otherwise
        """
        # Check in memory
        with self._lock:
            if key in self._memory:
                item = self._memory[key]
                if item.category == category:
                    return True
        
        # Check in backend if not in memory
        if self.storage_backend != StorageBackend.MEMORY:
            item = await self._retrieve_from_backend(key, category)
            return item is not None
        
        return False
    
    async def get_metadata(self, key: str, category: str = "default") -> Optional[Dict[str, Any]]:
        """Get metadata for a key.
        
        Args:
            key: Key to get metadata for
            category: Category of the key
            
        Returns:
            Metadata if key exists, None otherwise
        """
        # Check in memory
        with self._lock:
            if key in self._memory:
                item = self._memory[key]
                if item.category == category:
                    return item.metadata
        
        # Check in backend if not in memory
        if self.storage_backend != StorageBackend.MEMORY:
            item = await self._retrieve_from_backend(key, category)
            if item:
                return item.metadata
        
        return None
    
    async def add_listener(self, listener: MemoryEventListener) -> None:
        """Add an event listener.
        
        Args:
            listener: Listener to add
        """
        if listener not in self.listeners:
            self.listeners.append(listener)
    
    async def remove_listener(self, listener: MemoryEventListener) -> None:
        """Remove an event listener.
        
        Args:
            listener: Listener to remove
        """
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    async def backup(self, backup_path: str) -> Tuple[int, str]:
        """Backup the memory to a file.
        
        Args:
            backup_path: Path to save the backup
            
        Returns:
            Tuple of (number of items backed up, backup file path)
        """
        try:
            # Get all categories and keys
            categories = await self.get_categories()
            backup_data = {}
            
            for category in categories:
                keys = await self.get_keys(category)
                
                category_data = {}
                for key in keys:
                    value = await self.retrieve(key, category)
                    metadata = await self.get_metadata(key, category)
                    
                    # Skip if value not found
                    if value is None:
                        continue
                    
                    # Store in backup data
                    category_data[key] = {
                        "value": value,
                        "metadata": metadata or {}
                    }
                
                if category_data:
                    backup_data[category] = category_data
            
            # Create backup file
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save backup
            with open(backup_file, 'wb') as f:
                pickle.dump(backup_data, f)
            
            # Count total items
            total_items = sum(len(cat_data) for cat_data in backup_data.values())
            
            return (total_items, str(backup_file))
        
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise
    
    async def restore(self, backup_path: str, overwrite: bool = False) -> int:
        """Restore memory from a backup.
        
        Args:
            backup_path: Path to the backup file
            overwrite: Whether to overwrite existing keys
            
        Returns:
            Number of items restored
        """
        try:
            # Check if backup file exists
            backup_file = Path(backup_path)
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Load backup data
            with open(backup_file, 'rb') as f:
                backup_data = pickle.load(f)
            
            # Restore items
            restored_count = 0
            
            for category, category_data in backup_data.items():
                for key, item_data in category_data.items():
                    # Check if key exists
                    exists = await self.exists(key, category)
                    
                    if not exists or overwrite:
                        # Store the item
                        value = item_data["value"]
                        metadata = item_data.get("metadata", {})
                        
                        success = await self.store(key, value, category, metadata=metadata)
                        
                        if success:
                            restored_count += 1
            
            return restored_count
        
        except Exception as e:
            logger.error(f"Error restoring from backup: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close the shared memory and release resources."""
        # Close Redis client if exists
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
        
        # Clear memory
        with self._lock:
            self._memory.clear()
            self._category_keys.clear()
            self._expiry_times.clear()
        
        logger.info("Shared memory closed")


# Example memory event listener
class MemoryLogger(MemoryEventListener):
    """Example memory event listener that logs events."""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize the memory logger.
        
        Args:
            log_level: Logging level to use
        """
        self.logger = logging.getLogger(__name__ + ".MemoryLogger")
        self.logger.setLevel(log_level)
    
    async def on_memory_event(
        self,
        operation: MemoryOperation,
        key: str,
        category: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a memory event.
        
        Args:
            operation: The operation that was performed
            key: The key that was affected
            category: The category of the item
            value: The value (for STORE and UPDATE operations)
            metadata: Additional metadata about the operation
        """
        self.logger.info(f"Memory event: {operation.value} {category}/{key}")
        if metadata:
            self.logger.debug(f"Metadata: {metadata}")


# Example usage
async def example_usage():
    """Example of how to use the shared memory."""
    # Create shared memory
    memory = SharedMemory(storage_backend=StorageBackend.MEMORY)
    
    # Add logger
    logger = MemoryLogger()
    await memory.add_listener(logger)
    
    # Store some values
    await memory.store("greeting", "Hello, world!", "examples")
    await memory.store("counter", 42, "examples", metadata={"type": "integer"})
    
    # Retrieve values
    greeting = await memory.retrieve("greeting", "examples")
    print(f"Greeting: {greeting}")
    
    # Get all keys in a category
    keys = await memory.get_keys("examples")
    print(f"Keys in 'examples': {keys}")
    
    # Get memory stats
    stats = await memory.get_stats()
    print(f"Memory stats: {stats}")
    
    # Close memory
    await memory.close()


if __name__ == "__main__":
    # Run the example
    import asyncio
    asyncio.run(example_usage())
"""
Context storage for the multi-agent development system.

This module provides long-term storage and retrieval of contextual information
generated throughout the development process. It enables agents to maintain
awareness of past decisions, specifications, and interactions across sessions.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import uuid
import hashlib
import pickle
import asyncio
import threading
from enum import Enum
from collections import defaultdict

try:
    import aiofiles
    ASYNC_IO_AVAILABLE = True
except ImportError:
    ASYNC_IO_AVAILABLE = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

from pydantic import BaseModel, Field, validator

# Set up logging
logger = logging.getLogger(__name__)


class ContextType(str, Enum):
    """Types of context entries."""
    CODE = "code"
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    API = "api"
    DATABASE = "database"
    DECISION = "decision"
    DISCUSSION = "discussion"
    USER_FEEDBACK = "user_feedback"
    DESIGN = "design"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    ENVIRONMENT = "environment"
    SECURITY = "security"
    PERFORMANCE = "performance"
    OTHER = "other"


class StorageBackend(str, Enum):
    """Available storage backends."""
    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    JSON = "json"
    PICKLE = "pickle"
    CUSTOM = "custom"


class ContextEntry(BaseModel):
    """A single context entry stored in the system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: ContextType
    title: str
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    created_by: str
    project_id: Optional[str] = None
    expires_at: Optional[str] = None
    is_key_context: bool = False
    embedding: Optional[List[float]] = None
    
    def update_content(self, new_content: Any) -> None:
        """Update the content of this entry.
        
        Args:
            new_content: New content to store
        """
        self.content = new_content
        self.updated_at = datetime.now().isoformat()
    
    def add_tags(self, tags: List[str]) -> None:
        """Add tags to this entry.
        
        Args:
            tags: Tags to add
        """
        # Add only unique tags
        self.tags = list(set(self.tags + tags))
        self.updated_at = datetime.now().isoformat()
    
    def mark_as_key(self) -> None:
        """Mark this entry as key context."""
        self.is_key_context = True
        self.updated_at = datetime.now().isoformat()
    
    def is_expired(self) -> bool:
        """Check if this entry has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if not self.expires_at:
            return False
        
        try:
            expiry_time = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expiry_time
        except (ValueError, TypeError):
            # If we can't parse the date, assume it's not expired
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this entry
        """
        return {
            "id": self.id,
            "entry_type": self.entry_type,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "project_id": self.project_id,
            "expires_at": self.expires_at,
            "is_key_context": self.is_key_context
        }
    
    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary.
        
        Returns:
            JSON-serializable dictionary representation
        """
        result = self.to_dict()
        
        # Convert any non-serializable content to strings
        try:
            json.dumps(result["content"])
        except (TypeError, OverflowError):
            result["content"] = str(result["content"])
        
        # Convert any non-serializable metadata to strings
        for key, value in result["metadata"].items():
            try:
                json.dumps(value)
            except (TypeError, OverflowError):
                result["metadata"][key] = str(value)
        
        # Remove embedding from serialized output
        if "embedding" in result:
            del result["embedding"]
        
        return result


class SearchQuery(BaseModel):
    """A query for searching context entries."""
    text: Optional[str] = None
    types: Optional[List[ContextType]] = None
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None
    project_id: Optional[str] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    is_key_context: Optional[bool] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    limit: int = 10
    use_vector_search: bool = True


class SearchResult(BaseModel):
    """Result of a context search operation."""
    query: SearchQuery
    entries: List[ContextEntry]
    total_matches: int
    search_time_ms: float
    

class ContextStore:
    """Long-term storage for contextual information in the multi-agent system."""
    
    def __init__(
        self,
        storage_backend: Union[StorageBackend, str] = StorageBackend.MEMORY,
        storage_path: Optional[str] = None,
        compression: bool = True,
        encryption_key: Optional[str] = None,
        max_memory_entries: int = 10000,
        initialize: bool = True,
        vector_dimensions: int = 768,
        custom_encoder: Any = None,
        custom_decoder: Any = None,
    ):
        """Initialize the context store.
        
        Args:
            storage_backend: Backend storage system to use
            storage_path: Path for file-based storage
            compression: Whether to compress stored data
            encryption_key: Optional key for encrypting sensitive context
            max_memory_entries: Maximum entries to keep in memory
            initialize: Whether to initialize the storage on creation
            vector_dimensions: Dimensions for context embeddings
            custom_encoder: Optional custom encoder for serialization
            custom_decoder: Optional custom decoder for deserialization
        """
        # Convert string to enum if needed
        if isinstance(storage_backend, str):
            storage_backend = StorageBackend(storage_backend)
        
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path) if storage_path else None
        self.compression = compression
        self.encryption_key = encryption_key
        self.max_memory_entries = max_memory_entries
        self.vector_dimensions = vector_dimensions
        self.custom_encoder = custom_encoder
        self.custom_decoder = custom_decoder
        
        # In-memory storage
        self._entries: Dict[str, ContextEntry] = {}
        self._type_index: Dict[ContextType, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._project_index: Dict[str, Set[str]] = defaultdict(set)
        self._creator_index: Dict[str, Set[str]] = defaultdict(set)
        self._key_context_ids: Set[str] = set()
        
        # Vectorizer for semantic search (initialized lazily)
        self._vectorizer = None
        
        # Thread lock for concurrent access
        self._lock = threading.RLock()
        
        # Initialize storage if needed
        if initialize and self.storage_backend != StorageBackend.MEMORY:
            self._initialize_storage()
            
        logger.info(f"Context store initialized with {storage_backend} backend")
    
    def _initialize_storage(self) -> None:
        """Initialize the storage backend."""
        if self.storage_backend == StorageBackend.MEMORY:
            # No initialization needed for memory backend
            return
        
        if self.storage_backend in [StorageBackend.FILESYSTEM, StorageBackend.JSON, StorageBackend.PICKLE]:
            if not self.storage_path:
                raise ValueError(f"Storage path is required for {self.storage_backend} backend")
            
            # Create directory if it doesn't exist
            if not self.storage_path.exists():
                self.storage_path.mkdir(parents=True)
                logger.info(f"Created storage directory: {self.storage_path}")
            
            # Create type-specific directories
            for context_type in ContextType:
                type_dir = self.storage_path / context_type.value
                if not type_dir.exists():
                    type_dir.mkdir(parents=True)
            
            # Create indexes directory
            index_dir = self.storage_path / "indexes"
            if not index_dir.exists():
                index_dir.mkdir(parents=True)
                
            logger.info(f"Storage initialized at: {self.storage_path}")
    
    async def store(
        self,
        entry_type: Union[ContextType, str],
        title: str,
        content: Any,
        created_by: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        expires_at: Optional[str] = None,
        is_key_context: bool = False,
        entry_id: Optional[str] = None
    ) -> str:
        """Store a context entry.
        
        Args:
            entry_type: Type of context entry
            title: Title of the entry
            content: Content to store
            created_by: ID of the agent or user that created this entry
            tags: Optional tags for categorization
            metadata: Optional additional metadata
            project_id: Optional ID of the associated project
            expires_at: Optional expiration date (ISO format)
            is_key_context: Whether this is key context
            entry_id: Optional specific ID to use
            
        Returns:
            ID of the stored entry
        """
        # Convert string to enum if needed
        if isinstance(entry_type, str):
            entry_type = ContextType(entry_type)
        
        # Create the entry
        entry = ContextEntry(
            id=entry_id or str(uuid.uuid4()),
            entry_type=entry_type,
            title=title,
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            created_by=created_by,
            project_id=project_id,
            expires_at=expires_at,
            is_key_context=is_key_context
        )
        
        # Generate embedding if vector search is available
        if VECTOR_SEARCH_AVAILABLE:
            try:
                entry.embedding = await self._generate_embedding(entry)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {str(e)}")
        
        # Store the entry
        await self._store_entry(entry)
        
        return entry.id
    
    async def _store_entry(self, entry: ContextEntry) -> None:
        """Store an entry in the backend storage.
        
        Args:
            entry: The entry to store
        """
        with self._lock:
            # Update in-memory indexes
            self._entries[entry.id] = entry
            self._type_index[entry.entry_type].add(entry.id)
            for tag in entry.tags:
                self._tag_index[tag].add(entry.id)
            if entry.project_id:
                self._project_index[entry.project_id].add(entry.id)
            self._creator_index[entry.created_by].add(entry.id)
            if entry.is_key_context:
                self._key_context_ids.add(entry.id)
            
            # Ensure we don't exceed memory limits
            self._enforce_memory_limits()
        
        # Store in persistent backend if configured
        if self.storage_backend != StorageBackend.MEMORY:
            if self.storage_backend == StorageBackend.FILESYSTEM:
                await self._store_entry_filesystem(entry)
            elif self.storage_backend == StorageBackend.JSON:
                await self._store_entry_json(entry)
            elif self.storage_backend == StorageBackend.PICKLE:
                await self._store_entry_pickle(entry)
            elif self.storage_backend == StorageBackend.CUSTOM:
                await self._store_entry_custom(entry)
    
    async def _store_entry_filesystem(self, entry: ContextEntry) -> None:
        """Store an entry in the filesystem.
        
        Args:
            entry: The entry to store
        """
        if not self.storage_path:
            raise ValueError("Storage path is required for filesystem backend")
        
        # Create directory for the entry type if it doesn't exist
        type_dir = self.storage_path / entry.entry_type.value
        if not type_dir.exists():
            type_dir.mkdir(parents=True)
        
        # Store the entry in a file
        entry_path = type_dir / f"{entry.id}.json"
        
        # Serialize the entry
        serialized = json.dumps(entry.to_serializable_dict(), indent=2)
        
        # Compress if configured
        if self.compression:
            import gzip
            content = gzip.compress(serialized.encode('utf-8'))
            entry_path = entry_path.with_suffix('.json.gz')
        else:
            content = serialized.encode('utf-8')
        
        # Write to file
        if ASYNC_IO_AVAILABLE:
            async with aiofiles.open(entry_path, 'wb') as f:
                await f.write(content)
        else:
            with open(entry_path, 'wb') as f:
                f.write(content)
        
        # Update indexes
        await self._update_indexes(entry)
    
    async def _store_entry_json(self, entry: ContextEntry) -> None:
        """Store an entry in a JSON file.
        
        Args:
            entry: The entry to store
        """
        if not self.storage_path:
            raise ValueError("Storage path is required for JSON backend")
        
        # Store all entries in a single JSON file per type
        type_file = self.storage_path / f"{entry.entry_type.value}_entries.json"
        
        # Load existing entries
        entries = {}
        if type_file.exists():
            try:
                with open(type_file, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
            except Exception as e:
                logger.error(f"Error loading entries from {type_file}: {str(e)}")
                entries = {}
        
        # Add or update the entry
        entries[entry.id] = entry.to_serializable_dict()
        
        # Write back to the file
        with open(type_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)
        
        # Update indexes
        await self._update_indexes(entry)
    
    async def _store_entry_pickle(self, entry: ContextEntry) -> None:
        """Store an entry using pickle.
        
        Args:
            entry: The entry to store
        """
        if not self.storage_path:
            raise ValueError("Storage path is required for pickle backend")
        
        # Create directory for the entry type if it doesn't exist
        type_dir = self.storage_path / entry.entry_type.value
        if not type_dir.exists():
            type_dir.mkdir(parents=True)
        
        # Store the entry in a pickle file
        entry_path = type_dir / f"{entry.id}.pkl"
        
        # Serialize with pickle
        if self.custom_encoder:
            data = self.custom_encoder(entry)
        else:
            data = pickle.dumps(entry)
        
        # Compress if configured
        if self.compression:
            import gzip
            data = gzip.compress(data)
            entry_path = entry_path.with_suffix('.pkl.gz')
        
        # Write to file
        if ASYNC_IO_AVAILABLE:
            async with aiofiles.open(entry_path, 'wb') as f:
                await f.write(data)
        else:
            with open(entry_path, 'wb') as f:
                f.write(data)
        
        # Update indexes
        await self._update_indexes(entry)
    
    async def _store_entry_custom(self, entry: ContextEntry) -> None:
        """Store an entry using custom encoder.
        
        Args:
            entry: The entry to store
        """
        if not self.custom_encoder:
            raise ValueError("Custom encoder is required for custom backend")
        
        # Use the custom encoder to store the entry
        # This would typically be implemented by the user
        pass
    
    async def _update_indexes(self, entry: ContextEntry) -> None:
        """Update the persistent indexes.
        
        Args:
            entry: The entry to index
        """
        if not self.storage_path:
            return
        
        # Update type index
        type_index_path = self.storage_path / "indexes" / f"type_{entry.entry_type.value}.txt"
        
        if ASYNC_IO_AVAILABLE:
            # Check if the ID is already in the index
            if type_index_path.exists():
                async with aiofiles.open(type_index_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if entry.id not in content.splitlines():
                        async with aiofiles.open(type_index_path, 'a', encoding='utf-8') as f:
                            await f.write(f"{entry.id}\n")
            else:
                async with aiofiles.open(type_index_path, 'w', encoding='utf-8') as f:
                    await f.write(f"{entry.id}\n")
        else:
            # Check if the ID is already in the index
            if type_index_path.exists():
                with open(type_index_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if entry.id not in content.splitlines():
                        with open(type_index_path, 'a', encoding='utf-8') as f:
                            f.write(f"{entry.id}\n")
            else:
                with open(type_index_path, 'w', encoding='utf-8') as f:
                    f.write(f"{entry.id}\n")
        
        # Update tag indexes
        for tag in entry.tags:
            tag_index_path = self.storage_path / "indexes" / f"tag_{tag}.txt"
            
            if ASYNC_IO_AVAILABLE:
                # Check if the ID is already in the index
                if tag_index_path.exists():
                    async with aiofiles.open(tag_index_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        if entry.id not in content.splitlines():
                            async with aiofiles.open(tag_index_path, 'a', encoding='utf-8') as f:
                                await f.write(f"{entry.id}\n")
                else:
                    async with aiofiles.open(tag_index_path, 'w', encoding='utf-8') as f:
                        await f.write(f"{entry.id}\n")
            else:
                # Check if the ID is already in the index
                if tag_index_path.exists():
                    with open(tag_index_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if entry.id not in content.splitlines():
                            with open(tag_index_path, 'a', encoding='utf-8') as f:
                                f.write(f"{entry.id}\n")
                else:
                    with open(tag_index_path, 'w', encoding='utf-8') as f:
                        f.write(f"{entry.id}\n")
        
        # Update project index
        if entry.project_id:
            project_index_path = self.storage_path / "indexes" / f"project_{entry.project_id}.txt"
            
            if ASYNC_IO_AVAILABLE:
                # Check if the ID is already in the index
                if project_index_path.exists():
                    async with aiofiles.open(project_index_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        if entry.id not in content.splitlines():
                            async with aiofiles.open(project_index_path, 'a', encoding='utf-8') as f:
                                await f.write(f"{entry.id}\n")
                else:
                    async with aiofiles.open(project_index_path, 'w', encoding='utf-8') as f:
                        await f.write(f"{entry.id}\n")
            else:
                # Check if the ID is already in the index
                if project_index_path.exists():
                    with open(project_index_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if entry.id not in content.splitlines():
                            with open(project_index_path, 'a', encoding='utf-8') as f:
                                f.write(f"{entry.id}\n")
                else:
                    with open(project_index_path, 'w', encoding='utf-8') as f:
                        f.write(f"{entry.id}\n")
        
        # Update creator index
        creator_index_path = self.storage_path / "indexes" / f"creator_{entry.created_by}.txt"
        
        if ASYNC_IO_AVAILABLE:
            # Check if the ID is already in the index
            if creator_index_path.exists():
                async with aiofiles.open(creator_index_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if entry.id not in content.splitlines():
                        async with aiofiles.open(creator_index_path, 'a', encoding='utf-8') as f:
                            await f.write(f"{entry.id}\n")
            else:
                async with aiofiles.open(creator_index_path, 'w', encoding='utf-8') as f:
                    await f.write(f"{entry.id}\n")
        else:
            # Check if the ID is already in the index
            if creator_index_path.exists():
                with open(creator_index_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if entry.id not in content.splitlines():
                        with open(creator_index_path, 'a', encoding='utf-8') as f:
                            f.write(f"{entry.id}\n")
            else:
                with open(creator_index_path, 'w', encoding='utf-8') as f:
                    f.write(f"{entry.id}\n")
        
        # Update key context index
        if entry.is_key_context:
            key_index_path = self.storage_path / "indexes" / "key_context.txt"
            
            if ASYNC_IO_AVAILABLE:
                # Check if the ID is already in the index
                if key_index_path.exists():
                    async with aiofiles.open(key_index_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        if entry.id not in content.splitlines():
                            async with aiofiles.open(key_index_path, 'a', encoding='utf-8') as f:
                                await f.write(f"{entry.id}\n")
                else:
                    async with aiofiles.open(key_index_path, 'w', encoding='utf-8') as f:
                        await f.write(f"{entry.id}\n")
            else:
                # Check if the ID is already in the index
                if key_index_path.exists():
                    with open(key_index_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if entry.id not in content.splitlines():
                            with open(key_index_path, 'a', encoding='utf-8') as f:
                                f.write(f"{entry.id}\n")
                else:
                    with open(key_index_path, 'w', encoding='utf-8') as f:
                        f.write(f"{entry.id}\n")
    
    async def retrieve(self, entry_id: str) -> Optional[ContextEntry]:
        """Retrieve a context entry by ID.
        
        Args:
            entry_id: ID of the entry to retrieve
            
        Returns:
            ContextEntry if found, None otherwise
        """
        # Check in-memory first
        with self._lock:
            if entry_id in self._entries:
                entry = self._entries[entry_id]
                # Check if expired
                if entry.is_expired():
                    await self.delete(entry_id)
                    return None
                return entry
        
        # If not in memory, check persistent storage
        if self.storage_backend != StorageBackend.MEMORY:
            try:
                entry = await self._retrieve_from_storage(entry_id)
                
                if entry:
                    # Check if expired
                    if entry.is_expired():
                        await self.delete(entry_id)
                        return None
                    
                    # Cache in memory for future use
                    with self._lock:
                        self._entries[entry_id] = entry
                        self._type_index[entry.entry_type].add(entry_id)
                        for tag in entry.tags:
                            self._tag_index[tag].add(entry_id)
                        if entry.project_id:
                            self._project_index[entry.project_id].add(entry_id)
                        self._creator_index[entry.created_by].add(entry_id)
                        if entry.is_key_context:
                            self._key_context_ids.add(entry_id)
                        
                        # Ensure we don't exceed memory limits
                        self._enforce_memory_limits()
                    
                    return entry
            except Exception as e:
                logger.error(f"Error retrieving entry {entry_id}: {str(e)}")
        
        return None
    
    async def _retrieve_from_storage(self, entry_id: str) -> Optional[ContextEntry]:
        """Retrieve an entry from persistent storage.
        
        Args:
            entry_id: ID of the entry to retrieve
            
        Returns:
            ContextEntry if found, None otherwise
        """
        if not self.storage_path:
            return None
        
        # Try each entry type directory
        for entry_type in ContextType:
            # Construct potential file paths
            if self.storage_backend == StorageBackend.FILESYSTEM:
                json_path = self.storage_path / entry_type.value / f"{entry_id}.json"
                gz_path = self.storage_path / entry_type.value / f"{entry_id}.json.gz"
                
                # Check if file exists
                file_path = gz_path if gz_path.exists() else json_path
                if file_path.exists():
                    try:
                        # Read the file
                        if ASYNC_IO_AVAILABLE:
                            async with aiofiles.open(file_path, 'rb') as f:
                                content = await f.read()
                        else:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                        
                        # Decompress if needed
                        if file_path.suffix == '.gz':
                            import gzip
                            content = gzip.decompress(content)
                        
                        # Parse JSON
                        entry_data = json.loads(content.decode('utf-8'))
                        return ContextEntry(**entry_data)
                    except Exception as e:
                        logger.error(f"Error reading entry {entry_id}: {str(e)}")
                        return None
            
            elif self.storage_backend == StorageBackend.JSON:
                type_file = self.storage_path / f"{entry_type.value}_entries.json"
                if type_file.exists():
                    try:
                        # Read the file
                        with open(type_file, 'r', encoding='utf-8') as f:
                            entries = json.load(f)
                        
                        # Check if entry exists
                        if entry_id in entries:
                            return ContextEntry(**entries[entry_id])
                    except Exception as e:
                        logger.error(f"Error reading entry {entry_id} from {type_file}: {str(e)}")
            
            elif self.storage_backend == StorageBackend.PICKLE:
                pkl_path = self.storage_path / entry_type.value / f"{entry_id}.pkl"
                gz_path = self.storage_path / entry_type.value / f"{entry_id}.pkl.gz"
                
                # Check if file exists
                file_path = gz_path if gz_path.exists() else pkl_path
                if file_path.exists():
                    try:
                        # Read the file
                        if ASYNC_IO_AVAILABLE:
                            async with aiofiles.open(file_path, 'rb') as f:
                                content = await f.read()
                        else:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                        
                        # Decompress if needed
                        if file_path.suffix == '.gz':
                            import gzip
                            content = gzip.decompress(content)
                        
                        # Deserialize with pickle or custom decoder
                        if self.custom_decoder:
                            return self.custom_decoder(content)
                        else:
                            return pickle.loads(content)
                    except Exception as e:
                        logger.error(f"Error reading entry {entry_id}: {str(e)}")
                        return None
            
            elif self.storage_backend == StorageBackend.CUSTOM:
                # Custom retrieval would be implemented by the user
                pass
        
        return None
    
    async def update(
        self,
        entry_id: str,
        title: Optional[str] = None,
        content: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_key_context: Optional[bool] = None,
        expires_at: Optional[str] = None
    ) -> bool:
        """Update an existing context entry.
        
        Args:
            entry_id: ID of the entry to update
            title: Optional new title
            content: Optional new content
            tags: Optional new tags
            metadata: Optional new metadata
            is_key_context: Optional new key context status
            expires_at: Optional new expiration date
            
        Returns:
            True if successful, False otherwise
        """
        # Retrieve the entry
        entry = await self.retrieve(entry_id)
        if not entry:
            logger.warning(f"Cannot update entry {entry_id}: Entry not found")
            return False
        
        # Update fields
        if title is not None:
            entry.title = title
        
        if content is not None:
            entry.content = content
            
            # Update embedding if vector search is available
            if VECTOR_SEARCH_AVAILABLE:
                try:
                    entry.embedding = await self._generate_embedding(entry)
                except Exception as e:
                    logger.warning(f"Failed to update embedding: {str(e)}")
        
        if tags is not None:
            # Update tag indexes
            with self._lock:
                for tag in entry.tags:
                    self._tag_index[tag].discard(entry_id)
                
                entry.tags = tags
                
                for tag in tags:
                    self._tag_index[tag].add(entry_id)
        
        if metadata is not None:
            # Merge new metadata with existing
            entry.metadata.update(metadata)
        
        if is_key_context is not None:
            # Update key context status
            with self._lock:
                if is_key_context:
                    self._key_context_ids.add(entry_id)
                    entry.is_key_context = True
                else:
                    self._key_context_ids.discard(entry_id)
                    entry.is_key_context = False
        
        if expires_at is not None:
            entry.expires_at = expires_at
        
        # Update timestamp
        entry.updated_at = datetime.now().isoformat()
        
        # Store the updated entry
        await self._store_entry(entry)
        
        return True
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a context entry.
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Check if entry exists in memory
        entry = None
        with self._lock:
            if entry_id in self._entries:
                entry = self._entries[entry_id]
                
                # Remove from memory indexes
                self._entries.pop(entry_id)
                self._type_index[entry.entry_type].discard(entry_id)
                for tag in entry.tags:
                    self._tag_index[tag].discard(entry_id)
                if entry.project_id:
                    self._project_index[entry.project_id].discard(entry_id)
                self._creator_index[entry.created_by].discard(entry_id)
                self._key_context_ids.discard(entry_id)
        
        # If not in memory, try to retrieve first
        if not entry and self.storage_backend != StorageBackend.MEMORY:
            entry = await self._retrieve_from_storage(entry_id)
        
        # If still not found, nothing to delete
        if not entry:
            logger.warning(f"Cannot delete entry {entry_id}: Entry not found")
            return False
        
        # Delete from persistent storage
        if self.storage_backend != StorageBackend.MEMORY:
            await self._delete_from_storage(entry_id, entry)
        
        return True
    
    async def _delete_from_storage(self, entry_id: str, entry: ContextEntry) -> None:
        """Delete an entry from persistent storage.
        
        Args:
            entry_id: ID of the entry to delete
            entry: The entry object
        """
        if not self.storage_path:
            return
        
        if self.storage_backend == StorageBackend.FILESYSTEM:
            # Delete the file
            json_path = self.storage_path / entry.entry_type.value / f"{entry_id}.json"
            gz_path = self.storage_path / entry.entry_type.value / f"{entry_id}.json.gz"
            
            if json_path.exists():
                json_path.unlink()
            if gz_path.exists():
                gz_path.unlink()
        
        elif self.storage_backend == StorageBackend.JSON:
            # Update the JSON file
            type_file = self.storage_path / f"{entry.entry_type.value}_entries.json"
            if type_file.exists():
                try:
                    with open(type_file, 'r', encoding='utf-8') as f:
                        entries = json.load(f)
                    
                    if entry_id in entries:
                        del entries[entry_id]
                        
                        with open(type_file, 'w', encoding='utf-8') as f:
                            json.dump(entries, f, indent=2)
                except Exception as e:
                    logger.error(f"Error deleting entry {entry_id} from {type_file}: {str(e)}")
        
        elif self.storage_backend == StorageBackend.PICKLE:
            # Delete the file
            pkl_path = self.storage_path / entry.entry_type.value / f"{entry_id}.pkl"
            gz_path = self.storage_path / entry.entry_type.value / f"{entry_id}.pkl.gz"
            
            if pkl_path.exists():
                pkl_path.unlink()
            if gz_path.exists():
                gz_path.unlink()
        
        elif self.storage_backend == StorageBackend.CUSTOM:
            # Custom deletion would be implemented by the user
            pass
        
        # Update indexes
        await self._remove_from_indexes(entry_id, entry)
    
    async def _remove_from_indexes(self, entry_id: str, entry: ContextEntry) -> None:
        """Remove an entry from the persistent indexes.
        
        Args:
            entry_id: ID of the entry to remove
            entry: The entry object
        """
        if not self.storage_path:
            return
        
        # Helper function to remove from an index file
        async def remove_from_index(index_path: Path) -> None:
            if not index_path.exists():
                return
            
            lines = []
            if ASYNC_IO_AVAILABLE:
                async with aiofiles.open(index_path, 'r', encoding='utf-8') as f:
                    async for line in f:
                        if line.strip() != entry_id:
                            lines.append(line.strip())
            else:
                with open(index_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip() != entry_id:
                            lines.append(line.strip())
            
            if lines:
                if ASYNC_IO_AVAILABLE:
                    async with aiofiles.open(index_path, 'w', encoding='utf-8') as f:
                        await f.write('\n'.join(lines) + '\n')
                else:
                    with open(index_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines) + '\n')
            else:
                # If index is empty, delete the file
                index_path.unlink()
        
        # Remove from type index
        type_index_path = self.storage_path / "indexes" / f"type_{entry.entry_type.value}.txt"
        await remove_from_index(type_index_path)
        
        # Remove from tag indexes
        for tag in entry.tags:
            tag_index_path = self.storage_path / "indexes" / f"tag_{tag}.txt"
            await remove_from_index(tag_index_path)
        
        # Remove from project index
        if entry.project_id:
            project_index_path = self.storage_path / "indexes" / f"project_{entry.project_id}.txt"
            await remove_from_index(project_index_path)
        
        # Remove from creator index
        creator_index_path = self.storage_path / "indexes" / f"creator_{entry.created_by}.txt"
        await remove_from_index(creator_index_path)
        
        # Remove from key context index
        if entry.is_key_context:
            key_index_path = self.storage_path / "indexes" / "key_context.txt"
            await remove_from_index(key_index_path)
    
    async def search(self, query: Union[SearchQuery, Dict[str, Any]]) -> SearchResult:
        """Search for context entries.
        
        Args:
            query: Search query parameters
            
        Returns:
            SearchResult containing matching entries
        """
        # Convert dictionary to SearchQuery if needed
        if isinstance(query, dict):
            query = SearchQuery(**query)
        
        start_time = time.time()
        
        # If text query and vector search is enabled, try vector search first
        if query.text and query.use_vector_search and VECTOR_SEARCH_AVAILABLE:
            try:
                vector_results = await self._vector_search(query)
                if vector_results:
                    # Calculate search time
                    search_time_ms = (time.time() - start_time) * 1000
                    return SearchResult(
                        query=query,
                        entries=vector_results[:query.limit],
                        total_matches=len(vector_results),
                        search_time_ms=search_time_ms
                    )
            except Exception as e:
                logger.warning(f"Vector search failed: {str(e)}, falling back to filter search")
        
        # Fall back to filter-based search
        matching_entries = await self._filter_search(query)
        
        # Sort by relevance if text query provided
        if query.text:
            matching_entries = self._sort_by_text_relevance(matching_entries, query.text)
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            query=query,
            entries=matching_entries[:query.limit],
            total_matches=len(matching_entries),
            search_time_ms=search_time_ms
        )
    
    async def _filter_search(self, query: SearchQuery) -> List[ContextEntry]:
        """Perform a filter-based search.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of matching entries
        """
        # Get all candidate IDs
        candidate_ids = set()
        
        with self._lock:
            # Filter by type
            if query.types:
                type_ids = set()
                for entry_type in query.types:
                    if isinstance(entry_type, str):
                        entry_type = ContextType(entry_type)
                    type_ids.update(self._type_index.get(entry_type, set()))
                
                # Initialize candidates if empty
                if not candidate_ids:
                    candidate_ids = type_ids
                else:
                    candidate_ids &= type_ids
            
            # Filter by tags
            if query.tags:
                tag_ids = set()
                for tag in query.tags:
                    tag_ids.update(self._tag_index.get(tag, set()))
                
                # Initialize candidates if empty
                if not candidate_ids:
                    candidate_ids = tag_ids
                else:
                    candidate_ids &= tag_ids
            
            # Filter by creator
            if query.created_by:
                creator_ids = self._creator_index.get(query.created_by, set())
                
                # Initialize candidates if empty
                if not candidate_ids:
                    candidate_ids = creator_ids
                else:
                    candidate_ids &= creator_ids
            
            # Filter by project
            if query.project_id:
                project_ids = self._project_index.get(query.project_id, set())
                
                # Initialize candidates if empty
                if not candidate_ids:
                    candidate_ids = project_ids
                else:
                    candidate_ids &= project_ids
            
            # Filter by key context
            if query.is_key_context:
                # Initialize candidates if empty
                if not candidate_ids:
                    candidate_ids = self._key_context_ids
                else:
                    candidate_ids &= self._key_context_ids
            
            # If no filters applied, use all entries
            if not candidate_ids and not (query.types or query.tags or query.created_by or query.project_id or query.is_key_context):
                candidate_ids = set(self._entries.keys())
        
        # If we have entries in memory, filter by dates and metadata
        matching_entries = []
        for entry_id in candidate_ids:
            # Retrieve the entry (either from memory or storage)
            entry = await self.retrieve(entry_id)
            if not entry or entry.is_expired():
                continue
            
            # Filter by creation date
            if query.created_after and entry.created_at < query.created_after:
                continue
            
            if query.created_before and entry.created_at > query.created_before:
                continue
            
            # Filter by metadata
            if query.metadata_filters:
                match = True
                for key, value in query.metadata_filters.items():
                    if key not in entry.metadata or entry.metadata[key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # If we have a text query, perform text search
            if query.text:
                # Simple text search in title and content
                text = query.text.lower()
                text_match = False
                
                if text in entry.title.lower():
                    text_match = True
                elif isinstance(entry.content, str) and text in entry.content.lower():
                    text_match = True
                elif isinstance(entry.content, dict) and any(
                    isinstance(v, str) and text in v.lower() for v in entry.content.values()
                ):
                    text_match = True
                
                if not text_match:
                    continue
            
            matching_entries.append(entry)
        
        return matching_entries
    
    async def _vector_search(self, query: SearchQuery) -> List[ContextEntry]:
        """Perform a vector-based semantic search.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of matching entries
        """
        if not VECTOR_SEARCH_AVAILABLE:
            raise ValueError("Vector search requires numpy and scikit-learn")
        
        # Generate query embedding
        query_embedding = await self._generate_text_embedding(query.text)
        
        # Filter candidate entries
        candidate_entries = await self._filter_search(query)
        
        # Filter entries with embeddings
        entries_with_embeddings = [
            entry for entry in candidate_entries
            if entry.embedding is not None
        ]
        
        if not entries_with_embeddings:
            return []
        
        # Calculate cosine similarities
        similarities = []
        for entry in entries_with_embeddings:
            if entry.embedding:
                similarity = cosine_similarity(
                    [query_embedding],
                    [entry.embedding]
                )[0][0]
                similarities.append((entry, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Extract entries
        return [entry for entry, _ in similarities]
    
    async def _generate_embedding(self, entry: ContextEntry) -> Optional[List[float]]:
        """Generate an embedding for a context entry.
        
        Args:
            entry: The entry to embed
            
        Returns:
            Embedding vector or None if generation fails
        """
        if not VECTOR_SEARCH_AVAILABLE:
            return None
        
        # Combine title and content
        text = entry.title
        
        if isinstance(entry.content, str):
            text += " " + entry.content
        elif isinstance(entry.content, dict):
            # For dictionary content, concatenate string values
            for key, value in entry.content.items():
                if isinstance(value, str):
                    text += " " + value
        
        return await self._generate_text_embedding(text)
    
    async def _generate_text_embedding(self, text: str) -> List[float]:
        """Generate an embedding for text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector
        """
        if not VECTOR_SEARCH_AVAILABLE:
            raise ValueError("Vector search requires numpy and scikit-learn")
        
        # Initialize vectorizer if needed
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(max_features=self.vector_dimensions)
            
            # Fit on a single document to initialize
            self._vectorizer.fit(["initialize vectorizer"])
        
        # Generate embedding
        try:
            # Transform the text
            vector = self._vectorizer.transform([text])
            
            # Convert to dense array and normalize
            dense_vector = vector.toarray()[0]
            norm = np.linalg.norm(dense_vector)
            if norm > 0:
                dense_vector = dense_vector / norm
            
            return dense_vector.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector
            return [0.0] * self.vector_dimensions
    
    def _sort_by_text_relevance(self, entries: List[ContextEntry], query_text: str) -> List[ContextEntry]:
        """Sort entries by relevance to a text query.
        
        Args:
            entries: Entries to sort
            query_text: Query text
            
        Returns:
            Sorted entries
        """
        query_text = query_text.lower()
        
        # Define a relevance score function
        def relevance_score(entry: ContextEntry) -> float:
            score = 0.0
            
            # Title match is highly relevant
            if query_text in entry.title.lower():
                score += 10.0
                # Exact title match is even better
                if query_text == entry.title.lower():
                    score += 5.0
                # Title starts with query is good too
                elif entry.title.lower().startswith(query_text):
                    score += 3.0
            
            # Content match
            if isinstance(entry.content, str):
                if query_text in entry.content.lower():
                    score += 5.0
                    # Count occurrences
                    score += entry.content.lower().count(query_text) * 0.2
            elif isinstance(entry.content, dict):
                for value in entry.content.values():
                    if isinstance(value, str) and query_text in value.lower():
                        score += 5.0
                        # Count occurrences
                        score += value.lower().count(query_text) * 0.2
            
            # Tag match
            if any(query_text in tag.lower() for tag in entry.tags):
                score += 3.0
            
            # Key context is more relevant
            if entry.is_key_context:
                score *= 1.5
            
            # Recency boost
            try:
                created_time = datetime.fromisoformat(entry.created_at)
                now = datetime.now()
                days_old = (now - created_time).days
                # Boost newer content, but not too aggressively
                recency_boost = max(1.0, 1.2 - (days_old / 30) * 0.1)  # 10% decay per month
                score *= recency_boost
            except (ValueError, TypeError):
                pass
            
            return score
        
        # Sort by score (descending)
        return sorted(entries, key=relevance_score, reverse=True)
    
    def _enforce_memory_limits(self) -> None:
        """Enforce memory limits by removing least recently used entries."""
        if len(self._entries) <= self.max_memory_entries:
            return
        
        # Sort entries by last updated time (oldest first)
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].updated_at
        )
        
        # Calculate how many to remove
        to_remove = len(sorted_entries) - self.max_memory_entries
        
        # Remove oldest entries (but keep key context)
        removed = 0
        for entry_id, entry in sorted_entries:
            if removed >= to_remove:
                break
            
            # Don't remove key context
            if entry.is_key_context:
                continue
            
            # Remove from memory
            del self._entries[entry_id]
            self._type_index[entry.entry_type].discard(entry_id)
            for tag in entry.tags:
                self._tag_index[tag].discard(entry_id)
            if entry.project_id:
                self._project_index[entry.project_id].discard(entry_id)
            self._creator_index[entry.created_by].discard(entry_id)
            
            removed += 1
    
    async def get_key_context(
        self, 
        project_id: Optional[str] = None,
        context_type: Optional[Union[ContextType, str]] = None
    ) -> List[ContextEntry]:
        """Get all key context entries.
        
        Args:
            project_id: Optional project ID to filter by
            context_type: Optional context type to filter by
            
        Returns:
            List of key context entries
        """
        # Convert string to enum if needed
        if isinstance(context_type, str):
            context_type = ContextType(context_type)
        
        key_entries = []
        
        with self._lock:
            for entry_id in self._key_context_ids:
                # Retrieve the entry
                entry = await self.retrieve(entry_id)
                if not entry or entry.is_expired():
                    continue
                
                # Apply filters
                if project_id and entry.project_id != project_id:
                    continue
                
                if context_type and entry.entry_type != context_type:
                    continue
                
                key_entries.append(entry)
        
        return key_entries
    
    async def export_context(
        self, 
        output_path: str,
        project_id: Optional[str] = None,
        context_type: Optional[Union[ContextType, str]] = None,
        format: str = "json"
    ) -> Tuple[int, str]:
        """Export context entries to a file.
        
        Args:
            output_path: Path to export to
            project_id: Optional project ID to filter by
            context_type: Optional context type to filter by
            format: Export format ("json" or "pickle")
            
        Returns:
            Tuple of (number of entries exported, output file path)
        """
        # Convert string to enum if needed
        if isinstance(context_type, str):
            context_type = ContextType(context_type)
        
        # Search for matching entries
        query = SearchQuery(
            project_id=project_id,
            types=[context_type] if context_type else None,
            limit=100000  # Effectively no limit
        )
        
        result = await self.search(query)
        entries = result.entries
        
        # Create output directory if needed
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        
        # Export in the specified format
        if format.lower() == "json":
            # Convert to serializable dictionaries
            serializable_entries = [
                entry.to_serializable_dict() for entry in entries
            ]
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_entries, f, indent=2)
        
        elif format.lower() == "pickle":
            # Write to file
            with open(output_path, 'wb') as f:
                pickle.dump(entries, f)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return len(entries), output_path
    
    async def import_context(
        self, 
        input_path: str,
        format: str = "json",
        replace_existing: bool = False
    ) -> Tuple[int, int]:
        """Import context entries from a file.
        
        Args:
            input_path: Path to import from
            format: Import format ("json" or "pickle")
            replace_existing: Whether to replace existing entries
            
        Returns:
            Tuple of (number of entries imported, number of entries skipped)
        """
        # Check if file exists
        input_file = Path(input_path)
        if not input_file.exists():
            raise ValueError(f"Input file not found: {input_path}")
        
        entries = []
        
        # Import from the specified format
        if format.lower() == "json":
            # Read from file
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to ContextEntry objects
            entries = [ContextEntry(**entry_data) for entry_data in data]
        
        elif format.lower() == "pickle":
            # Read from file
            with open(input_path, 'rb') as f:
                entries = pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        # Import entries
        imported = 0
        skipped = 0
        
        for entry in entries:
            # Check if entry already exists
            existing_entry = await self.retrieve(entry.id)
            
            if existing_entry and not replace_existing:
                skipped += 1
                continue
            
            # Store the entry
            await self._store_entry(entry)
            imported += 1
        
        return imported, skipped
    
    async def purge_expired(self) -> int:
        """Purge all expired entries.
        
        Returns:
            Number of entries purged
        """
        purged = 0
        
        # Get all entries
        with self._lock:
            all_entry_ids = list(self._entries.keys())
        
        # Check each entry
        for entry_id in all_entry_ids:
            entry = await self.retrieve(entry_id)
            if entry and entry.is_expired():
                # Delete the entry
                if await self.delete(entry_id):
                    purged += 1
        
        return purged
    
    async def clear_all(self) -> int:
        """Clear all entries from the store.
        
        Returns:
            Number of entries cleared
        """
        # Get all entries
        with self._lock:
            all_entry_ids = list(self._entries.keys())
            count = len(all_entry_ids)
            
            # Clear in-memory storage
            self._entries.clear()
            self._type_index.clear()
            self._tag_index.clear()
            self._project_index.clear()
            self._creator_index.clear()
            self._key_context_ids.clear()
        
        # Clear persistent storage if configured
        if self.storage_backend != StorageBackend.MEMORY and self.storage_path:
            # Clear each type directory
            for context_type in ContextType:
                type_dir = self.storage_path / context_type.value
                if type_dir.exists():
                    for file in type_dir.glob("*"):
                        file.unlink()
            
            # Clear indexes
            index_dir = self.storage_path / "indexes"
            if index_dir.exists():
                for file in index_dir.glob("*"):
                    file.unlink()
        
        return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the context store.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_entries": 0,
            "entries_by_type": {},
            "entries_by_project": {},
            "key_context_count": 0,
            "unique_tags": 0,
            "storage_backend": self.storage_backend.value,
            "memory_usage": len(self._entries),
            "memory_limit": self.max_memory_entries
        }
        
        # Count entries by type
        with self._lock:
            for entry_type, entry_ids in self._type_index.items():
                stats["entries_by_type"][entry_type.value] = len(entry_ids)
                stats["total_entries"] += len(entry_ids)
            
            # Count entries by project
            for project_id, entry_ids in self._project_index.items():
                stats["entries_by_project"][project_id] = len(entry_ids)
            
            # Count key context
            stats["key_context_count"] = len(self._key_context_ids)
            
            # Count unique tags
            stats["unique_tags"] = len(self._tag_index)
        
        # Add storage statistics if available
        if self.storage_backend != StorageBackend.MEMORY and self.storage_path:
            stats["storage_path"] = str(self.storage_path)
            
            # Calculate storage size
            total_size = 0
            for context_type in ContextType:
                type_dir = self.storage_path / context_type.value
                if type_dir.exists():
                    for file in type_dir.glob("*"):
                        total_size += file.stat().st_size
            
            stats["storage_size_bytes"] = total_size
            stats["storage_size_mb"] = total_size / (1024 * 1024)
        
        return stats
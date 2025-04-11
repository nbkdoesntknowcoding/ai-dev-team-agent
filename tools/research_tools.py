"""
Research and information gathering tools for the multi-agent development system.

This module provides tools for researching solutions, gathering information,
exploring libraries, and staying up-to-date with best practices. These tools
help agents make informed decisions during the development process.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import tempfile
import subprocess
import urllib.parse
from collections import defaultdict

try:
    import requests
    from requests.exceptions import RequestException
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import bs4
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
except ImportError:
    NLTK_AVAILABLE = False

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

from pydantic import BaseModel, Field, validator

# Set up logging
logger = logging.getLogger(__name__)


class ResearchSource(str, Enum):
    """Supported research sources."""
    LOCAL_DOCS = "local_docs"
    OFFLINE_RESOURCES = "offline_resources"
    ONLINE_DOCS = "online_docs"
    CODE_EXAMPLES = "code_examples"
    PACKAGE_METADATA = "package_metadata"
    PUBLIC_REPOS = "public_repos"
    REFERENCE_MATERIAL = "reference_material"
    BEST_PRACTICES = "best_practices"
    STANDARDS_SPECS = "standards_specs"
    CACHED_RESULTS = "cached_results"
    AGENT_KNOWLEDGE = "agent_knowledge"


class SearchResult(BaseModel):
    """A search result from research."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    snippet: str
    content: Optional[str] = None
    url: Optional[str] = None
    source: ResearchSource
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchQuery(BaseModel):
    """A research query."""
    query: str
    topic: Optional[str] = None
    sources: List[ResearchSource] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = 10
    min_relevance: float = 0.0
    language: Optional[str] = None
    include_code_examples: bool = True
    max_age_days: Optional[int] = None
    format: str = "markdown"


class ResearchResult(BaseModel):
    """Results of a research operation."""
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    summary: str = ""
    sources_used: List[ResearchSource] = Field(default_factory=list)
    search_time_ms: float = 0.0
    total_results: int = 0
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class CodeSnippet(BaseModel):
    """A code snippet with metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    code: str
    language: str
    title: str
    source_url: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    stars: Optional[int] = None
    forks: Optional[int] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    relevance_score: float = 0.0


class DocumentationResult(BaseModel):
    """Result of documentation lookup."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    url: Optional[str] = None
    source: str
    api_version: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    examples: List[CodeSnippet] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class PackageInfo(BaseModel):
    """Information about a software package."""
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    documentation: Optional[str] = None
    dependencies: Dict[str, str] = Field(default_factory=dict)
    dev_dependencies: Dict[str, str] = Field(default_factory=dict)
    peer_dependencies: Dict[str, str] = Field(default_factory=dict)
    stars: Optional[int] = None
    forks: Optional[int] = None
    open_issues: Optional[int] = None
    last_updated: Optional[str] = None
    download_count: Optional[int] = None
    keywords: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    alternatives: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BestPracticeItem(BaseModel):
    """A best practice item."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    category: str
    rationale: str
    examples: List[str] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    level: str = "standard"  # beginner, standard, advanced, expert
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ComparisonResult(BaseModel):
    """Result of comparing multiple options."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    options: List[str]
    criteria: List[str]
    scores: Dict[str, Dict[str, float]]  # option -> criterion -> score
    summary: str
    recommendation: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ResearchCache:
    """Cache for research results."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_age_days: int = 30,
        max_size_mb: int = 1024
    ):
        """Initialize the research cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_days: Maximum age of cache entries in days
            max_size_mb: Maximum size of cache in megabytes
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".research_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.max_age_days = max_age_days
        self.max_size_mb = max_size_mb
        self.memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Create index file if it doesn't exist
        self.index_file = self.cache_dir / "index.json"
        if not self.index_file.exists():
            with open(self.index_file, "w") as f:
                json.dump({"entries": {}}, f)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        # Check memory cache first
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            age_days = (datetime.now() - timestamp).days
            if age_days <= self.max_age_days:
                return value
            else:
                # Expired, remove from memory
                del self.memory_cache[key]
        
        # Load index
        try:
            with open(self.index_file, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
        
        # Check if key exists and not expired
        entries = index.get("entries", {})
        if key in entries:
            entry = entries[key]
            file_path = self.cache_dir / entry["file"]
            stored_time = datetime.fromisoformat(entry["timestamp"])
            age_days = (datetime.now() - stored_time).days
            
            if age_days <= self.max_age_days and file_path.exists():
                # Load value from file
                try:
                    with open(file_path, "r") as f:
                        value = json.load(f)
                    
                    # Store in memory for faster access next time
                    self.memory_cache[key] = (value, stored_time)
                    return value
                except Exception as e:
                    logger.warning(f"Error loading cache entry {key}: {str(e)}")
        
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        # Create a unique filename
        filename = f"{key.replace(':', '_').replace('/', '_')}_{int(time.time())}.json"
        file_path = self.cache_dir / filename
        
        # Store in file
        try:
            with open(file_path, "w") as f:
                json.dump(value, f)
            
            # Update index
            try:
                with open(self.index_file, "r") as f:
                    index = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                index = {"entries": {}}
            
            timestamp = datetime.now().isoformat()
            index["entries"][key] = {
                "file": filename,
                "timestamp": timestamp,
                "size_bytes": file_path.stat().st_size
            }
            
            with open(self.index_file, "w") as f:
                json.dump(index, f)
            
            # Store in memory
            self.memory_cache[key] = (value, datetime.now())
            
            # Manage cache size
            self._clean_cache_if_needed()
            
        except Exception as e:
            logger.warning(f"Error storing cache entry {key}: {str(e)}")
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was invalidated, False otherwise
        """
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from index and delete file
        try:
            with open(self.index_file, "r") as f:
                index = json.load(f)
            
            entries = index.get("entries", {})
            if key in entries:
                entry = entries[key]
                file_path = self.cache_dir / entry["file"]
                
                # Delete file if it exists
                if file_path.exists():
                    os.remove(file_path)
                
                # Remove from index
                del entries[key]
                
                with open(self.index_file, "w") as f:
                    json.dump(index, f)
                
                return True
        except Exception as e:
            logger.warning(f"Error invalidating cache entry {key}: {str(e)}")
        
        return False
    
    def clear(self) -> int:
        """Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        # Clear memory cache
        entry_count = len(self.memory_cache)
        self.memory_cache.clear()
        
        # Clear files
        try:
            # Read index first to get entry count
            try:
                with open(self.index_file, "r") as f:
                    index = json.load(f)
                entry_count = len(index.get("entries", {}))
            except:
                pass
            
            # Delete all files except index
            for file_path in self.cache_dir.glob("*.json"):
                if file_path.name != "index.json":
                    os.remove(file_path)
            
            # Reset index
            with open(self.index_file, "w") as f:
                json.dump({"entries": {}}, f)
            
            return entry_count
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")
            return 0
    
    def _clean_cache_if_needed(self) -> None:
        """Clean the cache if it exceeds the size limit."""
        try:
            # Calculate total size
            total_size_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            if total_size_mb <= self.max_size_mb:
                return
            
            # Read index
            with open(self.index_file, "r") as f:
                index = json.load(f)
            
            # Sort entries by timestamp (oldest first)
            entries = index.get("entries", {})
            sorted_entries = sorted(
                entries.items(),
                key=lambda x: datetime.fromisoformat(x[1]["timestamp"])
            )
            
            # Remove oldest entries until under size limit
            removed = 0
            for key, entry in sorted_entries:
                file_path = self.cache_dir / entry["file"]
                
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    os.remove(file_path)
                    total_size_bytes -= file_size
                    total_size_mb = total_size_bytes / (1024 * 1024)
                
                # Remove from index and memory
                if key in self.memory_cache:
                    del self.memory_cache[key]
                del entries[key]
                removed += 1
                
                if total_size_mb <= self.max_size_mb * 0.9:  # Leave some buffer
                    break
            
            # Update index
            with open(self.index_file, "w") as f:
                json.dump(index, f)
            
            logger.info(f"Cleaned cache: removed {removed} entries, current size: {total_size_mb:.2f} MB")
            
        except Exception as e:
            logger.warning(f"Error cleaning cache: {str(e)}")


class ResearchTool:
    """Tools for researching solutions and gathering information."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        offline_docs_dir: Optional[str] = None,
        reference_dir: Optional[str] = None,
        local_repos_dir: Optional[str] = None,
        enable_online_search: bool = False,
        api_keys: Optional[Dict[str, str]] = None,
        http_timeout: int = 10,
        max_cache_age_days: int = 30,
        cache_size_mb: int = 1024,
        knowledge_base: Optional[Any] = None
    ):
        """Initialize the research tool.
        
        Args:
            cache_dir: Directory to store cache files
            offline_docs_dir: Directory containing offline documentation
            reference_dir: Directory containing reference materials
            local_repos_dir: Directory containing local repositories
            enable_online_search: Whether to enable online search
            api_keys: Dictionary of API keys for different services
            http_timeout: Timeout for HTTP requests in seconds
            max_cache_age_days: Maximum age of cache entries in days
            cache_size_mb: Maximum size of cache in megabytes
            knowledge_base: Optional knowledge base to query
        """
        # Initialize directories
        self.offline_docs_dir = Path(offline_docs_dir) if offline_docs_dir else None
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.local_repos_dir = Path(local_repos_dir) if local_repos_dir else None
        
        # Initialize configuration
        self.enable_online_search = enable_online_search
        self.api_keys = api_keys or {}
        self.http_timeout = http_timeout
        self.knowledge_base = knowledge_base
        
        # Initialize cache
        self.cache = ResearchCache(
            cache_dir=cache_dir,
            max_age_days=max_cache_age_days,
            max_size_mb=cache_size_mb
        )
        
        # Initialize HTTP session if online search is enabled
        self.session = None
        if enable_online_search and AIOHTTP_AVAILABLE:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=http_timeout))
        
        logger.info("Research tool initialized")
    
    async def close(self) -> None:
        """Close resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def research(self, query: Union[ResearchQuery, Dict[str, Any]]) -> ResearchResult:
        """Perform research based on a query.
        
        Args:
            query: The research query
            
        Returns:
            Research results
        """
        # Convert dictionary to ResearchQuery if needed
        if isinstance(query, dict):
            query = ResearchQuery(**query)
        
        start_time = time.time()
        
        # If no sources are specified, use default sources
        if not query.sources:
            query.sources = [
                ResearchSource.CACHED_RESULTS,
                ResearchSource.LOCAL_DOCS,
                ResearchSource.OFFLINE_RESOURCES,
                ResearchSource.PACKAGE_METADATA
            ]
            
            # Add online sources if enabled
            if self.enable_online_search:
                query.sources.extend([
                    ResearchSource.ONLINE_DOCS,
                    ResearchSource.CODE_EXAMPLES,
                    ResearchSource.PUBLIC_REPOS
                ])
        
        # Check cache first
        cache_key = f"research:{query.query}:{','.join(sorted([s.value for s in query.sources]))}"
        if ResearchSource.CACHED_RESULTS in query.sources:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                try:
                    # Check if cache is still valid based on max_age_days
                    if query.max_age_days:
                        cache_time = datetime.fromisoformat(cached_result.get("timestamp", ""))
                        age_days = (datetime.now() - cache_time).days
                        if age_days <= query.max_age_days:
                            # Cache is valid, return it
                            result = ResearchResult(**cached_result)
                            result.search_time_ms = 0.0  # Cached lookup time
                            return result
                except Exception:
                    pass
        
        # Search results from all specified sources
        all_results: List[SearchResult] = []
        sources_used: List[ResearchSource] = []
        
        # Check if cached results are available and still valid
        if ResearchSource.CACHED_RESULTS in query.sources:
            cached_results = await self._search_cached_results(query)
            if cached_results:
                all_results.extend(cached_results)
                sources_used.append(ResearchSource.CACHED_RESULTS)
        
        # Search offline resources
        if ResearchSource.LOCAL_DOCS in query.sources and self.offline_docs_dir:
            local_results = await self._search_local_docs(query)
            if local_results:
                all_results.extend(local_results)
                sources_used.append(ResearchSource.LOCAL_DOCS)
        
        if ResearchSource.OFFLINE_RESOURCES in query.sources and self.reference_dir:
            offline_results = await self._search_offline_resources(query)
            if offline_results:
                all_results.extend(offline_results)
                sources_used.append(ResearchSource.OFFLINE_RESOURCES)
        
        # Search package metadata
        if ResearchSource.PACKAGE_METADATA in query.sources:
            package_results = await self._search_package_metadata(query)
            if package_results:
                all_results.extend(package_results)
                sources_used.append(ResearchSource.PACKAGE_METADATA)
        
        # Search online resources if enabled
        if self.enable_online_search:
            if ResearchSource.ONLINE_DOCS in query.sources:
                online_doc_results = await self._search_online_docs(query)
                if online_doc_results:
                    all_results.extend(online_doc_results)
                    sources_used.append(ResearchSource.ONLINE_DOCS)
            
            if ResearchSource.CODE_EXAMPLES in query.sources and query.include_code_examples:
                code_results = await self._search_code_examples(query)
                if code_results:
                    all_results.extend(code_results)
                    sources_used.append(ResearchSource.CODE_EXAMPLES)
            
            if ResearchSource.PUBLIC_REPOS in query.sources:
                repo_results = await self._search_public_repos(query)
                if repo_results:
                    all_results.extend(repo_results)
                    sources_used.append(ResearchSource.PUBLIC_REPOS)
        
        # Search agent knowledge if available
        if ResearchSource.AGENT_KNOWLEDGE in query.sources and self.knowledge_base:
            knowledge_results = await self._search_agent_knowledge(query)
            if knowledge_results:
                all_results.extend(knowledge_results)
                sources_used.append(ResearchSource.AGENT_KNOWLEDGE)
        
        # Filter results by relevance
        if query.min_relevance > 0.0:
            all_results = [r for r in all_results if r.relevance_score >= query.min_relevance]
        
        # Sort by relevance score (highest first)
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit to max results
        all_results = all_results[:query.max_results]
        
        # Create summary if there are results
        summary = ""
        if all_results:
            summary = await self._generate_summary(all_results, query)
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        # Create result
        result = ResearchResult(
            query=query.query,
            results=all_results,
            summary=summary,
            sources_used=sources_used,
            search_time_ms=search_time_ms,
            total_results=len(all_results),
            filters_applied=query.filters
        )
        
        # Cache the result for future use
        if all_results:
            self.cache.put(cache_key, result.dict())
        
        return result
    
    async def _search_cached_results(self, query: ResearchQuery) -> List[SearchResult]:
        """Search cached results.
        
        Args:
            query: The research query
            
        Returns:
            List of search results
        """
        # This is intentionally empty since we check the cache before searching
        # in the main research method. This is included for completeness.
        return []
    
    async def _search_local_docs(self, query: ResearchQuery) -> List[SearchResult]:
        """Search local documentation.
        
        Args:
            query: The research query
            
        Returns:
            List of search results
        """
        results = []
        
        if not self.offline_docs_dir or not self.offline_docs_dir.exists():
            return results
        
        # Create search terms
        search_terms = self._extract_search_terms(query.query)
        search_pattern = r'\b(' + '|'.join(re.escape(term) for term in search_terms) + r')\b'
        
        # Search markdown and text files
        for file_ext in ["*.md", "*.txt", "*.rst", "*.html"]:
            for file_path in self.offline_docs_dir.glob(f"**/{file_ext}"):
                try:
                    # Skip files that are too big
                    if file_path.stat().st_size > 10 * 1024 * 1024:  # 10 MB
                        continue
                    
                    # Read file content
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    
                    # Check if any search term is in the content
                    if not re.search(search_pattern, content, re.IGNORECASE):
                        continue
                    
                    # Calculate relevance score
                    relevance = self._calculate_relevance(content, search_terms)
                    
                    # Extract title
                    title = file_path.stem
                    # For markdown files, try to extract the first heading
                    if file_path.suffix.lower() == ".md":
                        heading_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                        if heading_match:
                            title = heading_match.group(1)
                    
                    # Extract a snippet showing the context of the match
                    snippet = self._extract_snippet(content, search_terms)
                    
                    results.append(SearchResult(
                        title=title,
                        snippet=snippet,
                        content=content if len(content) < 10000 else None,  # Skip very long content
                        url=str(file_path.relative_to(self.offline_docs_dir)),
                        source=ResearchSource.LOCAL_DOCS,
                        relevance_score=relevance,
                        metadata={
                            "file_path": str(file_path),
                            "file_size": file_path.stat().st_size,
                            "file_type": file_path.suffix.lower()[1:]
                        }
                    ))
                except Exception as e:
                    logger.warning(f"Error searching file {file_path}: {str(e)}")
        
        return results
    
    async def _search_offline_resources(self, query: ResearchQuery) -> List[SearchResult]:
        """Search offline resources.
        
        Args:
            query: The research query
            
        Returns:
            List of search results
        """
        results = []
        
        if not self.reference_dir or not self.reference_dir.exists():
            return results
        
        # Create search terms
        search_terms = self._extract_search_terms(query.query)
        search_pattern = r'\b(' + '|'.join(re.escape(term) for term in search_terms) + r')\b'
        
        # Search JSON and YAML files (which might contain structured reference data)
        for file_ext in ["*.json", "*.yaml", "*.yml"]:
            for file_path in self.reference_dir.glob(f"**/{file_ext}"):
                try:
                    # Skip files that are too big
                    if file_path.stat().st_size > 5 * 1024 * 1024:  # 5 MB
                        continue
                    
                    # Read file content
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    
                    # Check if any search term is in the content
                    if not re.search(search_pattern, content, re.IGNORECASE):
                        continue
                    
                    # Calculate relevance score
                    relevance = self._calculate_relevance(content, search_terms)
                    
                    # Extract a snippet showing the context of the match
                    snippet = self._extract_snippet(content, search_terms)
                    
                    results.append(SearchResult(
                        title=file_path.stem,
                        snippet=snippet,
                        content=content if len(content) < 10000 else None,  # Skip very long content
                        url=str(file_path.relative_to(self.reference_dir)),
                        source=ResearchSource.OFFLINE_RESOURCES,
                        relevance_score=relevance,
                        metadata={
                            "file_path": str(file_path),
                            "file_size": file_path.stat().st_size,
                            "file_type": file_path.suffix.lower()[1:]
                        }
                    ))
                except Exception as e:
                    logger.warning(f"Error searching file {file_path}: {str(e)}")
        
        return results
    
    async def _search_package_metadata(self, query: ResearchQuery) -> List[SearchResult]:
        """Search package metadata.
        
        Args:
            query: The research query
            
        Returns:
            List of search results
        """
        results = []
        
        # Extract potential package names from the query
        package_names = self._extract_package_names(query.query)
        if not package_names:
            return results
        
        # Search for package information from local package managers
        for package_name in package_names:
            # Check npm packages (for JavaScript/TypeScript)
            npm_info = await self._get_npm_package_info(package_name)
            if npm_info:
                metadata = npm_info.get("metadata", {})
                content = json.dumps(npm_info, indent=2)
                snippet = npm_info.get("description", "")
                
                results.append(SearchResult(
                    title=f"NPM Package: {package_name}",
                    snippet=snippet,
                    content=content,
                    url=f"https://www.npmjs.com/package/{package_name}",
                    source=ResearchSource.PACKAGE_METADATA,
                    relevance_score=0.9,  # High relevance for exact package match
                    metadata=metadata
                ))
            
            # Check PyPI packages (for Python)
            pypi_info = await self._get_pypi_package_info(package_name)
            if pypi_info:
                metadata = pypi_info.get("metadata", {})
                content = json.dumps(pypi_info, indent=2)
                snippet = pypi_info.get("summary", "")
                
                results.append(SearchResult(
                    title=f"PyPI Package: {package_name}",
                    snippet=snippet,
                    content=content,
                    url=f"https://pypi.org/project/{package_name}/",
                    source=ResearchSource.PACKAGE_METADATA,
                    relevance_score=0.9,  # High relevance for exact package match
                    metadata=metadata
                ))
        
        return results
    
    async def _search_online_docs(self, query: ResearchQuery) -> List[SearchResult]:
        """Search online documentation.
        
        Args:
            query: The research query
            
        Returns:
            List of search results
        """
        if not self.enable_online_search or not self.session:
            return []
        
        results = []
        
        # Create search URL based on topic, if provided
        search_query = query.query
        if query.topic:
            search_query = f"{query.topic} {search_query}"
        
        # Add language if specified
        if query.language:
            search_query = f"{search_query} {query.language}"
        
        # Create a safe URL-encoded query
        encoded_query = urllib.parse.quote(search_query)
        
        # Try to search documentation sites based on the topic or language
        search_urls = []
        
        # Determine which documentation sites to search
        if query.language and query.language.lower() in ["python", "py"]:
            search_urls.append(f"https://docs.python.org/3/search.html?q={encoded_query}")
        elif query.language and query.language.lower() in ["javascript", "js"]:
            search_urls.append(f"https://developer.mozilla.org/en-US/search?q={encoded_query}")
        elif query.language and query.language.lower() in ["typescript", "ts"]:
            search_urls.append(f"https://www.typescriptlang.org/search?q={encoded_query}")
        else:
            # Generic searches
            search_urls.append(f"https://developer.mozilla.org/en-US/search?q={encoded_query}")
            search_urls.append(f"https://docs.python.org/3/search.html?q={encoded_query}")
        
        # Search each documentation site
        for search_url in search_urls[:2]:  # Limit to 2 sites to avoid too many requests
            try:
                # Make the request
                async with self.session.get(search_url, timeout=self.http_timeout) as response:
                    if response.status != 200:
                        continue
                    
                    html_content = await response.text()
                    
                    # Parse the HTML if BeautifulSoup is available
                    if BS4_AVAILABLE:
                        soup = BeautifulSoup(html_content, "html.parser")
                        search_results = self._extract_search_results_from_html(soup, search_url)
                        
                        results.extend(search_results)
            except Exception as e:
                logger.warning(f"Error searching online docs at {search_url}: {str(e)}")
        
        return results
    
    async def _search_code_examples(self, query: ResearchQuery) -> List[SearchResult]:
        """Search for code examples.
        
        Args:
            query: The research query
            
        Returns:
            List of search results
        """
        if not self.enable_online_search or not self.session:
            return []
        
        results = []
        
        # Create search query
        search_query = query.query
        if query.language:
            search_query = f"{search_query} {query.language}"
        
        # Add "example" or "code example" to the query
        search_query = f"{search_query} example code"
        
        # Create a safe URL-encoded query
        encoded_query = urllib.parse.quote(search_query)
        
        # Search GitHub for code examples
        github_search_url = f"https://api.github.com/search/code?q={encoded_query}"
        
        try:
            headers = {}
            if "github" in self.api_keys:
                headers["Authorization"] = f"token {self.api_keys['github']}"
            
            async with self.session.get(github_search_url, headers=headers, timeout=self.http_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get("items", [])
                    
                    for item in items[:5]:  # Limit to 5 results
                        # Get the file content
                        file_url = item.get("html_url", "")
                        raw_url = file_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        
                        try:
                            async with self.session.get(raw_url, timeout=self.http_timeout) as file_response:
                                if file_response.status == 200:
                                    content = await file_response.text()
                                    
                                    # Skip files that are too large
                                    if len(content) > 100000:  # ~100 KB
                                        content = content[:100000] + "\n... (truncated)"
                                    
                                    # Calculate relevance score
                                    search_terms = self._extract_search_terms(query.query)
                                    relevance = self._calculate_relevance(content, search_terms)
                                    
                                    # Extract a snippet showing the context of the match
                                    snippet = self._extract_snippet(content, search_terms)
                                    
                                    results.append(SearchResult(
                                        title=f"Code Example: {item.get('name', 'Unknown')}",
                                        snippet=snippet,
                                        content=content,
                                        url=file_url,
                                        source=ResearchSource.CODE_EXAMPLES,
                                        relevance_score=relevance,
                                        metadata={
                                            "repository": item.get("repository", {}).get("full_name", ""),
                                            "path": item.get("path", ""),
                                            "language": query.language or ""
                                        }
                                    ))
                        except Exception as e:
                            logger.warning(f"Error fetching file content from {raw_url}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error searching GitHub for code examples: {str(e)}")
        
        return results
    
    async def _search_public_repos(self, query: ResearchQuery) -> List[SearchResult]:
        """Search public repositories.
        
        Args:
            query: The research query
            
        Returns:
            List of search results
        """
        if not self.enable_online_search or not self.session:
            return []
        
        results = []
        
        # Create search query
        search_query = query.query
        if query.language:
            search_query = f"{search_query} language:{query.language}"
        
        # Create a safe URL-encoded query
        encoded_query = urllib.parse.quote(search_query)
        
        # Search GitHub for repositories
        github_search_url = f"https://api.github.com/search/repositories?q={encoded_query}&sort=stars&order=desc"
        
        try:
            headers = {}
            if "github" in self.api_keys:
                headers["Authorization"] = f"token {self.api_keys['github']}"
            
            async with self.session.get(github_search_url, headers=headers, timeout=self.http_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get("items", [])
                    
                    for item in items[:5]:  # Limit to 5 results
                        # Get repository information
                        repo_name = item.get("full_name", "")
                        repo_url = item.get("html_url", "")
                        description = item.get("description", "")
                        stars = item.get("stargazers_count", 0)
                        
                        # Get the README content if available
                        readme_url = f"https://raw.githubusercontent.com/{repo_name}/master/README.md"
                        readme_content = ""
                        
                        try:
                            async with self.session.get(readme_url, timeout=self.http_timeout) as readme_response:
                                if readme_response.status == 200:
                                    readme_content = await readme_response.text()
                        except Exception:
                            # Try alternative README locations
                            alternative_urls = [
                                f"https://raw.githubusercontent.com/{repo_name}/main/README.md",
                                f"https://raw.githubusercontent.com/{repo_name}/master/readme.md",
                                f"https://raw.githubusercontent.com/{repo_name}/main/readme.md",
                                f"https://raw.githubusercontent.com/{repo_name}/master/README.markdown",
                                f"https://raw.githubusercontent.com/{repo_name}/main/README.markdown"
                            ]
                            
                            for alt_url in alternative_urls:
                                try:
                                    async with self.session.get(alt_url, timeout=self.http_timeout) as alt_response:
                                        if alt_response.status == 200:
                                            readme_content = await alt_response.text()
                                            break
                                except Exception:
                                    continue
                        
                        # Prepare content for relevance calculation
                        content_for_relevance = description + "\n" + readme_content
                        
                        # Calculate relevance score
                        search_terms = self._extract_search_terms(query.query)
                        relevance = self._calculate_relevance(content_for_relevance, search_terms)
                        
                        # Create result
                        results.append(SearchResult(
                            title=f"Repository: {repo_name}",
                            snippet=description,
                            content=readme_content[:10000] if readme_content else None,  # Limit content size
                            url=repo_url,
                            source=ResearchSource.PUBLIC_REPOS,
                            relevance_score=relevance,
                            metadata={
                                "stars": stars,
                                "language": item.get("language", ""),
                                "topics": item.get("topics", []),
                                "created_at": item.get("created_at", ""),
                                "updated_at": item.get("updated_at", "")
                            }
                        ))
        except Exception as e:
            logger.warning(f"Error searching GitHub for repositories: {str(e)}")
        
        return results
    
    async def _search_agent_knowledge(self, query: ResearchQuery) -> List[SearchResult]:
        """Search agent knowledge base.
        
        Args:
            query: The research query
            
        Returns:
            List of search results
        """
        if not self.knowledge_base:
            return []
        
        results = []
        
        # This is a placeholder implementation
        # In a real implementation, this would query the agent's knowledge base
        # For now, just return an empty list
        return results
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from a query.
        
        Args:
            query: The query to extract terms from
            
        Returns:
            List of search terms
        """
        # Normalize query
        normalized_query = query.lower()
        
        # Remove punctuation and special characters
        normalized_query = re.sub(r'[^\w\s]', ' ', normalized_query)
        
        # Split into words
        words = normalized_query.split()
        
        # Remove common stop words if NLTK is available
        if NLTK_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word not in stop_words]
            except:
                pass
        
        # Remove very short words
        words = [word for word in words if len(word) > 2]
        
        return words
    
    def _calculate_relevance(self, content: str, search_terms: List[str]) -> float:
        """Calculate relevance score for content based on search terms.
        
        Args:
            content: The content to calculate relevance for
            search_terms: The search terms to match
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Normalize content
        normalized_content = content.lower()
        
        # Count occurrences of each search term
        term_counts = {}
        for term in search_terms:
            term_regex = r'\b' + re.escape(term) + r'\b'
            count = len(re.findall(term_regex, normalized_content))
            term_counts[term] = count
        
        # Calculate basic relevance score
        if not term_counts:
            return 0.0
        
        # Calculate term frequency score
        total_occurrences = sum(term_counts.values())
        content_length = len(normalized_content.split())
        term_frequency = total_occurrences / max(1, content_length)
        
        # Calculate term coverage score
        terms_found = sum(1 for count in term_counts.values() if count > 0)
        term_coverage = terms_found / len(search_terms)
        
        # Combine scores (with more weight on term coverage)
        relevance = 0.3 * min(1.0, term_frequency * 100) + 0.7 * term_coverage
        
        return relevance
    
    def _extract_snippet(self, content: str, search_terms: List[str], max_length: int = 200) -> str:
        """Extract a snippet from content showing the context of search term matches.
        
        Args:
            content: The content to extract snippet from
            search_terms: The search terms to match
            max_length: Maximum length of the snippet
            
        Returns:
            Snippet showing search term context
        """
        if not content or not search_terms:
            return ""
        
        # Create regex pattern to find search terms
        pattern = r'\b(' + '|'.join(re.escape(term) for term in search_terms) + r')\b'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if not match:
            # No match found, return the beginning of the content
            return content[:max_length] + ("..." if len(content) > max_length else "")
        
        # Get match position
        start_pos = match.start()
        end_pos = match.end()
        
        # Determine snippet start and end positions
        context_length = (max_length - (end_pos - start_pos)) // 2
        snippet_start = max(0, start_pos - context_length)
        snippet_end = min(len(content), end_pos + context_length)
        
        # Extract snippet
        snippet = content[snippet_start:snippet_end]
        
        # Add ellipses if needed
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _extract_package_names(self, query: str) -> List[str]:
        """Extract potential package names from a query.
        
        Args:
            query: The query to extract package names from
            
        Returns:
            List of potential package names
        """
        # Extract words and combinations from the query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out common words and keep likely package names
        common_words = {'how', 'to', 'use', 'with', 'in', 'for', 'the', 'a', 'an', 'package', 'library', 'module', 'install', 'import'}
        package_candidates = [word for word in words if word not in common_words and len(word) > 2]
        
        # Also consider hyphenated and dot-separated names
        hyphenated = re.findall(r'\b[\w-]+\b', query)
        package_candidates.extend([word for word in hyphenated if '-' in word])
        
        dotted = re.findall(r'\b[\w.]+\b', query)
        package_candidates.extend([word for word in dotted if '.' in word])
        
        # Remove duplicates
        return list(set(package_candidates))
    
    async def _get_npm_package_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an NPM package.
        
        Args:
            package_name: Name of the NPM package
            
        Returns:
            Package information or None if not found
        """
        # Check cache first
        cache_key = f"npm:package:{package_name}"
        cached_info = self.cache.get(cache_key)
        if cached_info:
            return cached_info
        
        # Try running npm view command
        try:
            cmd = ["npm", "view", package_name, "--json"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse the JSON output
                data = json.loads(stdout.decode())
                result = {
                    "name": package_name,
                    "version": data.get("version"),
                    "description": data.get("description"),
                    "author": data.get("author", {}) if isinstance(data.get("author"), dict) else {"name": data.get("author")},
                    "license": data.get("license"),
                    "homepage": data.get("homepage"),
                    "repository": data.get("repository", {}).get("url") if isinstance(data.get("repository"), dict) else data.get("repository"),
                    "dependencies": data.get("dependencies", {}),
                    "keywords": data.get("keywords", []),
                    "metadata": {
                        "maintainers": data.get("maintainers", []),
                        "dist_tags": data.get("dist-tags", {}),
                        "engines": data.get("engines", {})
                    }
                }
                
                # Cache the result
                self.cache.put(cache_key, result)
                
                return result
        except Exception as e:
            logger.warning(f"Error getting NPM package info for {package_name}: {str(e)}")
        
        # If npm command fails, try online API if online search is enabled
        if self.enable_online_search and self.session:
            try:
                npm_api_url = f"https://registry.npmjs.org/{package_name}"
                async with self.session.get(npm_api_url, timeout=self.http_timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        latest_version = data.get("dist-tags", {}).get("latest")
                        latest_data = data.get("versions", {}).get(latest_version, {})
                        
                        result = {
                            "name": package_name,
                            "version": latest_version,
                            "description": data.get("description") or latest_data.get("description"),
                            "author": latest_data.get("author", {}),
                            "license": latest_data.get("license"),
                            "homepage": latest_data.get("homepage"),
                            "repository": latest_data.get("repository", {}).get("url") if isinstance(latest_data.get("repository"), dict) else latest_data.get("repository"),
                            "dependencies": latest_data.get("dependencies", {}),
                            "keywords": latest_data.get("keywords", []),
                            "metadata": {
                                "maintainers": data.get("maintainers", []),
                                "dist_tags": data.get("dist-tags", {}),
                                "engines": latest_data.get("engines", {})
                            }
                        }
                        
                        # Cache the result
                        self.cache.put(cache_key, result)
                        
                        return result
            except Exception as e:
                logger.warning(f"Error getting NPM package info from API for {package_name}: {str(e)}")
        
        return None
    
    async def _get_pypi_package_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a PyPI package.
        
        Args:
            package_name: Name of the PyPI package
            
        Returns:
            Package information or None if not found
        """
        # Check cache first
        cache_key = f"pypi:package:{package_name}"
        cached_info = self.cache.get(cache_key)
        if cached_info:
            return cached_info
        
        # Try running pip show command
        try:
            cmd = ["pip", "show", package_name]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse the output
                output = stdout.decode()
                lines = output.strip().split('\n')
                info = {}
                
                for line in lines:
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        info[key.lower()] = value
                
                # Format the result
                result = {
                    "name": package_name,
                    "version": info.get("version"),
                    "summary": info.get("summary"),
                    "author": info.get("author"),
                    "license": info.get("license"),
                    "homepage": info.get("home-page"),
                    "requires": info.get("requires").split(', ') if info.get("requires") else [],
                    "required_by": info.get("required-by").split(', ') if info.get("required-by") else [],
                    "metadata": {
                        "location": info.get("location"),
                        "requires_python": info.get("requires-python")
                    }
                }
                
                # Cache the result
                self.cache.put(cache_key, result)
                
                return result
        except Exception as e:
            logger.warning(f"Error getting PyPI package info for {package_name}: {str(e)}")
        
        # If pip command fails, try online API if online search is enabled
        if self.enable_online_search and self.session:
            try:
                pypi_api_url = f"https://pypi.org/pypi/{package_name}/json"
                async with self.session.get(pypi_api_url, timeout=self.http_timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        info = data.get("info", {})
                        
                        result = {
                            "name": package_name,
                            "version": info.get("version"),
                            "summary": info.get("summary"),
                            "author": info.get("author"),
                            "license": info.get("license"),
                            "homepage": info.get("home_page") or info.get("project_url"),
                            "requires": info.get("requires_dist", []),
                            "required_by": [],
                            "metadata": {
                                "project_urls": info.get("project_urls", {}),
                                "requires_python": info.get("requires_python"),
                                "keywords": info.get("keywords", "").split() if isinstance(info.get("keywords"), str) else info.get("keywords", []),
                                "classifiers": info.get("classifiers", [])
                            }
                        }
                        
                        # Cache the result
                        self.cache.put(cache_key, result)
                        
                        return result
            except Exception as e:
                logger.warning(f"Error getting PyPI package info from API for {package_name}: {str(e)}")
        
        return None
    
    def _extract_search_results_from_html(self, soup: Any, search_url: str) -> List[SearchResult]:
        """Extract search results from HTML.
        
        Args:
            soup: BeautifulSoup object containing the HTML
            search_url: URL of the search page
            
        Returns:
            List of search results
        """
        results = []
        
        # Different sites have different HTML structures for search results
        if "python.org" in search_url:
            # Python docs search results
            search_results = soup.select("#search-results .search-result")
            
            for result in search_results:
                link = result.select_one("a")
                if not link:
                    continue
                
                title = link.get_text(strip=True)
                url = link.get("href")
                
                # Make URL absolute if it's relative
                if url and not url.startswith("http"):
                    url = f"https://docs.python.org{url}"
                
                snippet = ""
                snippet_div = result.select_one(".search-result-description")
                if snippet_div:
                    snippet = snippet_div.get_text(strip=True)
                
                results.append(SearchResult(
                    title=title,
                    snippet=snippet,
                    url=url,
                    source=ResearchSource.ONLINE_DOCS,
                    relevance_score=0.8,  # Default score for search results
                    metadata={
                        "source_site": "Python Documentation",
                        "search_url": search_url
                    }
                ))
        
        elif "mozilla.org" in search_url:
            # MDN search results
            search_results = soup.select(".search-results-container .result")
            
            for result in search_results:
                link = result.select_one("h3 a")
                if not link:
                    continue
                
                title = link.get_text(strip=True)
                url = link.get("href")
                
                # Make URL absolute if it's relative
                if url and not url.startswith("http"):
                    url = f"https://developer.mozilla.org{url}"
                
                snippet = ""
                snippet_div = result.select_one(".excerpt")
                if snippet_div:
                    snippet = snippet_div.get_text(strip=True)
                
                results.append(SearchResult(
                    title=title,
                    snippet=snippet,
                    url=url,
                    source=ResearchSource.ONLINE_DOCS,
                    relevance_score=0.8,  # Default score for search results
                    metadata={
                        "source_site": "MDN Web Docs",
                        "search_url": search_url
                    }
                ))
        
        return results
    
    async def _generate_summary(self, results: List[SearchResult], query: ResearchQuery) -> str:
        """Generate a summary of search results.
        
        Args:
            results: List of search results
            query: The original query
            
        Returns:
            Summary text
        """
        # Create a basic summary template
        summary_lines = [
            f"# Research Summary: {query.query}",
            "",
            f"Found {len(results)} relevant results from {len(set(r.source for r in results))} sources.",
            ""
        ]
        
        # Group results by source
        sources_grouping = defaultdict(list)
        for result in results:
            sources_grouping[result.source].append(result)
        
        # Add source-specific summaries
        for source, source_results in sources_grouping.items():
            summary_lines.append(f"## {source.value.replace('_', ' ').title()} ({len(source_results)} results)")
            summary_lines.append("")
            
            # Add top results for this source
            for result in source_results[:3]:  # Limit to top 3 per source
                summary_lines.append(f"### {result.title}")
                if result.url:
                    summary_lines.append(f"Source: {result.url}")
                summary_lines.append("")
                summary_lines.append(result.snippet)
                summary_lines.append("")
        
        # Add a conclusion
        summary_lines.append("## Key Findings")
        summary_lines.append("")
        summary_lines.append("The research found relevant information about the query. Review the results for detailed information.")
        
        return "\n".join(summary_lines)
    
    async def lookup_documentation(
        self,
        query: str,
        language: Optional[str] = None,
        library: Optional[str] = None,
        version: Optional[str] = None,
        include_examples: bool = True,
        format: str = "markdown"
    ) -> DocumentationResult:
        """Look up documentation for a specific topic.
        
        Args:
            query: The documentation query
            language: Programming language
            library: Library or framework name
            version: Version of the library
            include_examples: Whether to include code examples
            format: Output format (markdown, html, plain)
            
        Returns:
            Documentation result
        """
        # Construct a research query to find documentation
        research_query = ResearchQuery(
            query=query,
            topic=library or language,
            sources=[
                ResearchSource.CACHED_RESULTS,
                ResearchSource.LOCAL_DOCS,
                ResearchSource.OFFLINE_RESOURCES,
                ResearchSource.ONLINE_DOCS
            ],
            language=language,
            include_code_examples=include_examples,
            format=format
        )
        
        # Execute the research query
        research_result = await self.research(research_query)
        
        # Process the results into documentation format
        examples = []
        if include_examples and research_result.results:
            # Look for code examples in the results
            for result in research_result.results:
                if result.source == ResearchSource.CODE_EXAMPLES:
                    # Extract code blocks from content
                    if result.content:
                        examples.append(CodeSnippet(
                            code=result.content,
                            language=language or "text",
                            title=result.title,
                            source_url=result.url,
                            description=result.snippet,
                            tags=[query, language or "", library or ""],
                            relevance_score=result.relevance_score
                        ))
        
        # Find the most relevant result to use as main content
        main_result = None
        for result in research_result.results:
            if result.content and result.source != ResearchSource.CODE_EXAMPLES:
                if not main_result or result.relevance_score > main_result.relevance_score:
                    main_result = result
        
        # Format the documentation
        if main_result:
            content = main_result.content
            title = main_result.title
            url = main_result.url
            source = main_result.source.value
        else:
            # Create a basic template if no main result found
            content = f"# Documentation: {query}\n\nNo detailed documentation found."
            title = f"Documentation: {query}"
            url = None
            source = "Generated"
        
        # Collect related topics
        related_topics = []
        for result in research_result.results:
            if result != main_result:
                topic = result.title
                if topic and topic not in related_topics and len(related_topics) < 5:
                    related_topics.append(topic)
        
        return DocumentationResult(
            title=title,
            content=content,
            url=url,
            source=source,
            api_version=version,
            examples=examples,
            related_topics=related_topics
        )
    
    async def get_package_info(self, package_name: str, ecosystem: str = "auto") -> Optional[PackageInfo]:
        """Get information about a software package.
        
        Args:
            package_name: Name of the package
            ecosystem: Package ecosystem (npm, pypi, auto)
            
        Returns:
            Package information or None if not found
        """
        # Determine ecosystem if auto
        # Determine ecosystem if auto
        if ecosystem == "auto":
            # Try to infer from package name or context
            if package_name.startswith("@"):
                ecosystem = "npm"
            elif any(char in package_name for char in ["_", "-"]):
                # Could be either, try both
                ecosystems = ["pypi", "npm"]
            else:
                # Default to trying both
                ecosystems = ["pypi", "npm"]
        else:
            # Use specified ecosystem
            ecosystems = [ecosystem.lower()]
        
        # Try each ecosystem
        for eco in ecosystems:
            if eco == "npm":
                npm_info = await self._get_npm_package_info(package_name)
                if npm_info:
                    # Convert to PackageInfo format
                    return PackageInfo(
                        name=npm_info.get("name", package_name),
                        version=npm_info.get("version"),
                        description=npm_info.get("description"),
                        author=npm_info.get("author", {}).get("name") if isinstance(npm_info.get("author"), dict) else npm_info.get("author"),
                        license=npm_info.get("license"),
                        homepage=npm_info.get("homepage"),
                        repository=npm_info.get("repository"),
                        dependencies=npm_info.get("dependencies", {}),
                        keywords=npm_info.get("keywords", []),
                        metadata=npm_info.get("metadata", {})
                    )
            elif eco == "pypi":
                pypi_info = await self._get_pypi_package_info(package_name)
                if pypi_info:
                    # Convert to PackageInfo format
                    return PackageInfo(
                        name=pypi_info.get("name", package_name),
                        version=pypi_info.get("version"),
                        description=pypi_info.get("summary"),
                        author=pypi_info.get("author"),
                        license=pypi_info.get("license"),
                        homepage=pypi_info.get("homepage"),
                        documentation=pypi_info.get("homepage"),  # PyPI often uses homepage for docs
                        dependencies={dep: "*" for dep in pypi_info.get("requires", [])},
                        keywords=pypi_info.get("metadata", {}).get("keywords", []),
                        metadata=pypi_info.get("metadata", {})
                    )
        
        # If we get here, the package wasn't found
        return None
    
    async def compare_options(
        self,
        options: List[str],
        criteria: List[str],
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> ComparisonResult:
        """Compare multiple options based on criteria.
        
        Args:
            options: List of options to compare
            criteria: List of criteria for comparison
            language: Programming language context
            context: Additional context information
            
        Returns:
            Comparison result
        """
        # Gather information about each option
        option_data = {}
        
        for option in options:
            # Research this option
            query = f"{option}"
            if language:
                query += f" {language}"
            if context:
                query += f" {context}"
            
            research_query = ResearchQuery(
                query=query,
                sources=[
                    ResearchSource.CACHED_RESULTS,
                    ResearchSource.LOCAL_DOCS,
                    ResearchSource.PACKAGE_METADATA
                ],
                language=language,
                max_results=5
            )
            
            # Execute the research query
            research_result = await self.research(research_query)
            
            # Store research results for this option
            option_data[option] = {
                "research": research_result
            }
        
        # Evaluate each option against the criteria
        scores = {}
        for option in options:
            scores[option] = {}
            for criterion in criteria:
                # Calculate a score for this option against this criterion
                # This is a simplified implementation
                score = 0.5  # Default neutral score
                
                # Look for evidence in the research results
                research = option_data[option]["research"]
                for result in research.results:
                    # Look for mention of this criterion in the content
                    if result.content and criterion.lower() in result.content.lower():
                        # Simple scoring - could be much more sophisticated
                        mentions = result.content.lower().count(criterion.lower())
                        criterion_score = min(1.0, 0.5 + mentions * 0.1)
                        score = max(score, criterion_score)
                
                # Store the score
                scores[option][criterion] = score
        
        # Generate summary and recommendation
        avg_scores = {option: sum(scores[option].values()) / len(criteria) for option in options}
        best_option = max(avg_scores.items(), key=lambda x: x[1])[0]
        
        summary = (
            f"Compared {len(options)} options ({', '.join(options)}) "
            f"based on {len(criteria)} criteria ({', '.join(criteria)})."
        )
        
        recommendation = f"Based on the analysis, {best_option} appears to be the best option."
        
        # Create the result
        return ComparisonResult(
            title=f"Comparison: {' vs '.join(options)}",
            options=options,
            criteria=criteria,
            scores=scores,
            summary=summary,
            recommendation=recommendation,
            details=option_data
        )
    
    async def get_best_practices(
        self,
        topic: str,
        language: Optional[str] = None,
        level: str = "standard",
        max_results: int = 5
    ) -> List[BestPracticeItem]:
        """Get best practices for a topic.
        
        Args:
            topic: Topic to get best practices for
            language: Programming language
            level: Expertise level (beginner, standard, advanced, expert)
            max_results: Maximum number of best practices to return
            
        Returns:
            List of best practices
        """
        # Construct a research query for best practices
        query = f"best practices {topic}"
        if language:
            query += f" {language}"
        
        research_query = ResearchQuery(
            query=query,
            sources=[
                ResearchSource.CACHED_RESULTS,
                ResearchSource.LOCAL_DOCS,
                ResearchSource.REFERENCE_MATERIAL,
                ResearchSource.BEST_PRACTICES
            ],
            language=language,
            max_results=max_results * 2  # Get more results to filter through
        )
        
        # Execute the research query
        research_result = await self.research(research_query)
        
        # Process results into best practices
        best_practices = []
        
        for result in research_result.results:
            # Skip results that don't seem to be about best practices
            if not any(phrase in result.title.lower() or phrase in result.snippet.lower() 
                      for phrase in ["best practice", "guidelines", "how to", "recommended", "tips"]):
                continue
            
            # Create a best practice item
            practice = BestPracticeItem(
                title=result.title,
                description=result.snippet,
                category=topic,
                rationale="Based on research findings",
                level=level,
                languages=[language] if language else [],
                references=[result.url] if result.url else []
            )
            
            best_practices.append(practice)
            
            # Break if we have enough best practices
            if len(best_practices) >= max_results:
                break
        
        # If we don't have enough best practices, create some generic ones
        while len(best_practices) < max_results:
            practice = BestPracticeItem(
                title=f"Best Practice for {topic}",
                description=f"Follow standard best practices for {topic}.",
                category=topic,
                rationale="General best practice",
                level=level,
                languages=[language] if language else []
            )
            
            best_practices.append(practice)
        
        return best_practices
    
    async def clone_public_repo(
        self,
        repo_url: str,
        target_dir: Optional[str] = None,
        branch: Optional[str] = None
    ) -> Optional[str]:
        """Clone a public repository.
        
        Args:
            repo_url: URL of the repository to clone
            target_dir: Target directory for the clone
            branch: Branch to clone
            
        Returns:
            Path to the cloned repository or None if failed
        """
        if not GIT_AVAILABLE:
            logger.error("Git is not available. Cannot clone repository.")
            return None
        
        # Create a default target directory if not specified
        if not target_dir:
            if not self.local_repos_dir:
                self.local_repos_dir = Path(tempfile.gettempdir()) / "research_repos"
                os.makedirs(self.local_repos_dir, exist_ok=True)
            
            # Extract repo name from URL
            repo_name = repo_url.rstrip("/").split("/")[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
            
            target_dir = self.local_repos_dir / repo_name
        else:
            target_dir = Path(target_dir)
        
        try:
            # Clone the repository
            if branch:
                git.Repo.clone_from(repo_url, target_dir, branch=branch)
            else:
                git.Repo.clone_from(repo_url, target_dir)
            
            return str(target_dir)
        except Exception as e:
            logger.error(f"Error cloning repository {repo_url}: {str(e)}")
            return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages.
        
        Returns:
            List of supported language names
        """
        return [
            "python",
            "javascript",
            "typescript",
            "java",
            "c++",
            "c#",
            "go",
            "rust",
            "php",
            "ruby",
            "swift",
            "kotlin"
        ]
    
    def get_research_metrics(self) -> Dict[str, Any]:
        """Get metrics about the research tool usage.
        
        Returns:
            Dictionary of metrics
        """
        # Get cache size
        cache_size_bytes = 0
        cache_entries = 0
        
        try:
            if self.cache.cache_dir.exists():
                for file_path in self.cache.cache_dir.glob("*.json"):
                    cache_size_bytes += file_path.stat().st_size
                    cache_entries += 1
        except Exception:
            pass
        
        # Return metrics
        return {
            "cache_entries": cache_entries,
            "cache_size_mb": cache_size_bytes / (1024 * 1024),
            "cache_dir": str(self.cache.cache_dir) if self.cache.cache_dir else None,
            "online_search_enabled": self.enable_online_search,
            "sources_available": [s.value for s in ResearchSource],
            "timestamp": datetime.now().isoformat()
        }
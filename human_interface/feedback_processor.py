"""
Feedback processor for the multi-agent development system.

This module processes feedback from human reviewers and routes it to the appropriate
agents. It handles parsing structured feedback, extracting actionable items,
generating task revisions, and tracking feedback metrics across the system.
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union, cast
import uuid

from pydantic import BaseModel, Field, validator

from human_interface.review_interface import (
    ReviewStatus,
    FeedbackType,
    FeedbackItem,
    ReviewSubmission
)

# Set up logging
logger = logging.getLogger(__name__)


class FeedbackSeverity(str, Enum):
    """Severity levels for feedback items."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FeedbackCategory(str, Enum):
    """Categories of feedback for analytics and routing."""
    CORRECTNESS = "correctness"
    STYLE = "style"
    PERFORMANCE = "performance"
    SECURITY = "security"
    FUNCTIONALITY = "functionality"
    UX = "user_experience"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    BUSINESS_LOGIC = "business_logic"
    REQUIREMENTS = "requirements"
    OTHER = "other"


class FeedbackAction(str, Enum):
    """Action to be taken based on feedback."""
    REVISE = "revise"
    DISCUSS = "discuss"
    IGNORE = "ignore"
    ESCALATE = "escalate"
    DOCUMENT = "document"
    REFACTOR = "refactor"
    IMPLEMENT = "implement"


class ProcessedFeedback(BaseModel):
    """Processed feedback with extracted actions and metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_feedback_id: str
    task_id: str
    agent_id: str
    severity: FeedbackSeverity
    category: FeedbackCategory
    action: FeedbackAction
    description: str
    specific_changes: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class TaskRevision(BaseModel):
    """A revision task created in response to feedback."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_task_id: str
    feedback_ids: List[str]
    agent_id: str
    description: str
    requirements: Dict[str, Any]
    priority: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"


class FeedbackMetrics(BaseModel):
    """Metrics for tracking feedback across the system."""
    total_feedback_items: int = 0
    feedback_by_type: Dict[str, int] = Field(default_factory=dict)
    feedback_by_severity: Dict[str, int] = Field(default_factory=dict)
    feedback_by_agent: Dict[str, int] = Field(default_factory=dict)
    feedback_by_category: Dict[str, int] = Field(default_factory=dict)
    common_issues: List[Dict[str, Any]] = Field(default_factory=list)
    approval_rate: float = 0.0
    revision_rate: float = 0.0
    average_revisions_per_task: float = 0.0
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class FeedbackProcessor:
    """Processes feedback from human reviewers and generates appropriate actions."""
    
    def __init__(
        self,
        agent_registry: Optional[Dict[str, Any]] = None,
        shared_memory: Any = None,
        task_scheduler: Any = None,
        feedback_rules: Optional[Dict[str, Dict[str, Any]]] = None,
        learning_enabled: bool = True,
        metrics_update_frequency: int = 10
    ):
        """Initialize the feedback processor.
        
        Args:
            agent_registry: Registry of available agents for routing feedback
            shared_memory: Shared memory interface for system communication
            task_scheduler: Task scheduler for creating revision tasks
            feedback_rules: Rules for categorizing and processing feedback
            learning_enabled: Whether to enable learning from feedback patterns
            metrics_update_frequency: How often to update feedback metrics (in processed items)
        """
        self.agent_registry = agent_registry or {}
        self.shared_memory = shared_memory
        self.task_scheduler = task_scheduler
        self.feedback_rules = feedback_rules or self._default_feedback_rules()
        self.learning_enabled = learning_enabled
        self.metrics_update_frequency = metrics_update_frequency
        
        # Internal storage
        self.processed_feedback: Dict[str, ProcessedFeedback] = {}
        self.task_revisions: Dict[str, TaskRevision] = {}
        self.feedback_count = 0
        self.metrics = FeedbackMetrics()
        
        # Pattern extraction for learning
        self.feedback_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("Feedback processor initialized")
        
        # Initialize metrics if shared memory is available
        if self.shared_memory:
            self.metrics = FeedbackMetrics()
    
    def _default_feedback_rules(self) -> Dict[str, Dict[str, Any]]:
        """Create default rules for processing feedback.
        
        Returns:
            Dictionary of feedback rules
        """
        return {
            # Rules for code quality feedback
            "code_quality": {
                "categories": {
                    "style": {
                        "patterns": [
                            r"(?i)style|format|indent|spacing|naming|convention",
                            r"(?i)pep\s*8|linting",
                            r"(?i)code\s*organization|structure"
                        ],
                        "default_severity": FeedbackSeverity.LOW,
                        "default_action": FeedbackAction.REVISE
                    },
                    "architecture": {
                        "patterns": [
                            r"(?i)architect|structure|organization|design pattern",
                            r"(?i)coupling|cohesion|separation of concerns",
                            r"(?i)dependency|module|component|service"
                        ],
                        "default_severity": FeedbackSeverity.MEDIUM,
                        "default_action": FeedbackAction.REVISE
                    },
                    "performance": {
                        "patterns": [
                            r"(?i)performance|optimization|efficient|slow",
                            r"(?i)time complexity|space complexity|big o",
                            r"(?i)memory|cpu|resource|leak|cache"
                        ],
                        "default_severity": FeedbackSeverity.MEDIUM,
                        "default_action": FeedbackAction.REFACTOR
                    },
                    "security": {
                        "patterns": [
                            r"(?i)security|vulnerability|exploit|attack",
                            r"(?i)injection|xss|csrf|authentication|authorization",
                            r"(?i)sanitize|validate|encrypt|hash|secure"
                        ],
                        "default_severity": FeedbackSeverity.HIGH,
                        "default_action": FeedbackAction.REVISE
                    }
                }
            },
            
            # Rules for functionality feedback
            "functionality": {
                "categories": {
                    "correctness": {
                        "patterns": [
                            r"(?i)incorrect|wrong|error|bug|issue|fix",
                            r"(?i)doesn't work|does not work|failing",
                            r"(?i)expected\s+(?:behavior|result)|actual\s+(?:behavior|result)"
                        ],
                        "default_severity": FeedbackSeverity.HIGH,
                        "default_action": FeedbackAction.REVISE
                    },
                    "completeness": {
                        "patterns": [
                            r"(?i)incomplete|missing|lack|add|implement",
                            r"(?i)not\s+(?:complete|finished|done)",
                            r"(?i)additional|extra|more"
                        ],
                        "default_severity": FeedbackSeverity.MEDIUM,
                        "default_action": FeedbackAction.IMPLEMENT
                    },
                    "edge_cases": {
                        "patterns": [
                            r"(?i)edge case|corner case|boundary|limit",
                            r"(?i)exception|error handling|validation",
                            r"(?i)unexpected input|invalid input"
                        ],
                        "default_severity": FeedbackSeverity.MEDIUM,
                        "default_action": FeedbackAction.REVISE
                    }
                }
            },
            
            # Rules for design feedback
            "design": {
                "categories": {
                    "user_experience": {
                        "patterns": [
                            r"(?i)user experience|ux|ui|interface|layout",
                            r"(?i)confusing|unclear|intuitive|user friendly",
                            r"(?i)accessibility|usability|a11y"
                        ],
                        "default_severity": FeedbackSeverity.MEDIUM,
                        "default_action": FeedbackAction.REVISE
                    },
                    "visual_design": {
                        "patterns": [
                            r"(?i)visual|appearance|look|design|color|style",
                            r"(?i)aesthetic|layout|spacing|alignment",
                            r"(?i)font|typography|icon|image"
                        ],
                        "default_severity": FeedbackSeverity.LOW,
                        "default_action": FeedbackAction.REVISE
                    }
                }
            },
            
            # Rules for documentation feedback
            "documentation": {
                "categories": {
                    "clarity": {
                        "patterns": [
                            r"(?i)unclear|confusing|ambiguous|vague",
                            r"(?i)clarify|explain|detail|elaborate",
                            r"(?i)more information|additional context"
                        ],
                        "default_severity": FeedbackSeverity.MEDIUM,
                        "default_action": FeedbackAction.REVISE
                    },
                    "completeness": {
                        "patterns": [
                            r"(?i)incomplete|missing|lack|documentation",
                            r"(?i)add example|sample|usage",
                            r"(?i)parameters|return value|exception"
                        ],
                        "default_severity": FeedbackSeverity.MEDIUM,
                        "default_action": FeedbackAction.DOCUMENT
                    }
                }
            },
            
            # Rules for general feedback
            "general": {
                "categories": {
                    "requirements": {
                        "patterns": [
                            r"(?i)requirement|spec|specification",
                            r"(?i)not\s+(?:meeting|matching|following)",
                            r"(?i)different from|change from|deviation"
                        ],
                        "default_severity": FeedbackSeverity.HIGH,
                        "default_action": FeedbackAction.REVISE
                    },
                    "approach": {
                        "patterns": [
                            r"(?i)approach|strategy|method|technique",
                            r"(?i)alternative|different way|consider",
                            r"(?i)suggestion|recommend|advise"
                        ],
                        "default_severity": FeedbackSeverity.MEDIUM,
                        "default_action": FeedbackAction.DISCUSS
                    }
                }
            }
        }
    
    async def process_review(self, review: ReviewSubmission) -> List[ProcessedFeedback]:
        """Process a review submission and generate feedback actions.
        
        Args:
            review: The review submission to process
            
        Returns:
            List of processed feedback items
        """
        processed_items: List[ProcessedFeedback] = []
        
        # Get original task context if available
        task_context = None
        if self.shared_memory:
            task_data = self.shared_memory.retrieve(
                key=review.task_id,
                category="tasks"
            )
            if task_data:
                task_context = task_data
        
        # Process each feedback item
        for item in review.feedback_items:
            try:
                processed = await self._process_feedback_item(
                    item, 
                    review.task_id,
                    review.status,
                    task_context
                )
                
                if processed:
                    processed_items.append(processed)
                    
                    # Store processed feedback
                    self.processed_feedback[processed.id] = processed
                    
                    # Store in shared memory if available
                    if self.shared_memory:
                        self.shared_memory.store(
                            key=f"processed_feedback_{processed.id}",
                            value=processed.dict(),
                            category="processed_feedback"
                        )
                    
                    # Update feedback count and potentially metrics
                    self.feedback_count += 1
                    if self.feedback_count % self.metrics_update_frequency == 0:
                        await self._update_metrics()
                    
                    logger.info(
                        f"Processed feedback item {item.id} as {processed.category.value} "
                        f"with {processed.action.value} action"
                    )
            except Exception as e:
                logger.error(f"Error processing feedback item {item.id}: {str(e)}")
        
        # If review was rejected, consider creating task revisions
        if review.status in [ReviewStatus.REJECTED, ReviewStatus.NEEDS_CLARIFICATION, ReviewStatus.PARTIALLY_APPROVED]:
            await self._create_task_revisions(review.task_id, processed_items)
        
        # Update metrics after processing all items
        await self._update_metrics()
        
        return processed_items
    async def load_stored_metrics(self):
        """Load stored metrics from shared memory."""
        if self.shared_memory:
            try:
                stored_metrics = await self.shared_memory.retrieve(
                    key="feedback_metrics",
                    category="metrics"
                )
                if stored_metrics:
                    self.metrics = FeedbackMetrics(**stored_metrics)
                    logger.info("Loaded feedback metrics from shared memory")
            except Exception as e:
                logger.error(f"Error loading stored metrics: {str(e)}")
    async def _process_feedback_item(
        self,
        item: FeedbackItem,
        task_id: str,
        review_status: ReviewStatus,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ProcessedFeedback]:
        """Process an individual feedback item.
        
        Args:
            item: The feedback item to process
            task_id: ID of the task that received feedback
            review_status: Overall status of the review
            task_context: Optional context of the original task
            
        Returns:
            Processed feedback or None if processing failed
        """
        # Get agent ID from task if available
        agent_id = "unknown"
        if task_context and "agent_id" in task_context:
            agent_id = task_context["agent_id"]
        
        # Categorize feedback
        category = self._categorize_feedback(item)
        
        # Determine severity
        severity = self._determine_severity(item, category, review_status)
        
        # Determine action
        action = self._determine_action(item, category, severity, review_status)
        
        # Extract specific changes if available
        specific_changes = self._extract_specific_changes(item)
        
        # Create processed feedback
        processed = ProcessedFeedback(
            original_feedback_id=item.id,
            task_id=task_id,
            agent_id=agent_id,
            severity=severity,
            category=category,
            action=action,
            description=item.comment,
            specific_changes=specific_changes,
            context={
                "feedback_type": item.feedback_type,
                "location": item.location,
                "suggested_changes": item.suggested_changes,
                "review_status": review_status
            }
        )
        
        # If learning is enabled, store patterns for future improvement
        if self.learning_enabled:
            self._learn_from_feedback(processed, item)
        
        return processed
    
    def _categorize_feedback(self, item: FeedbackItem) -> FeedbackCategory:
        """Categorize feedback based on content and type.
        
        Args:
            item: The feedback item to categorize
            
        Returns:
            FeedbackCategory for the item
        """
        # Start with a mapping from feedback type to rule group
        type_to_rule = {
            FeedbackType.CODE_QUALITY: "code_quality",
            FeedbackType.FUNCTIONALITY: "functionality",
            FeedbackType.DESIGN: "design",
            FeedbackType.DOCUMENTATION: "documentation",
            FeedbackType.GENERAL: "general",
            # Map other types to appropriate rule groups
            FeedbackType.PERFORMANCE: "code_quality",
            FeedbackType.SECURITY: "code_quality",
            FeedbackType.USABILITY: "design"
        }
        
        rule_group = type_to_rule.get(item.feedback_type, "general")
        rules = self.feedback_rules.get(rule_group, {}).get("categories", {})
        
        # Check each category's patterns
        for category_name, category_rules in rules.items():
            patterns = category_rules.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, item.comment) or (
                    item.suggested_changes and re.search(pattern, item.suggested_changes)
                ):
                    # Convert category_name to FeedbackCategory
                    try:
                        return FeedbackCategory(category_name)
                    except ValueError:
                        # If not a direct match, find the closest category
                        return self._map_to_feedback_category(category_name)
        
        # Default mapping based on feedback type if no patterns match
        type_to_category = {
            FeedbackType.CODE_QUALITY: FeedbackCategory.STYLE,
            FeedbackType.FUNCTIONALITY: FeedbackCategory.FUNCTIONALITY,
            FeedbackType.DESIGN: FeedbackCategory.UX,
            FeedbackType.DOCUMENTATION: FeedbackCategory.DOCUMENTATION,
            FeedbackType.PERFORMANCE: FeedbackCategory.PERFORMANCE,
            FeedbackType.SECURITY: FeedbackCategory.SECURITY,
            FeedbackType.USABILITY: FeedbackCategory.UX
        }
        
        return type_to_category.get(item.feedback_type, FeedbackCategory.OTHER)
    
    def _map_to_feedback_category(self, category_name: str) -> FeedbackCategory:
        """Map a string category name to a FeedbackCategory enum.
        
        Args:
            category_name: String category name
            
        Returns:
            Matching FeedbackCategory or OTHER if no match
        """
        # Direct mapping where names don't exactly match
        mapping = {
            "style": FeedbackCategory.STYLE,
            "correctness": FeedbackCategory.CORRECTNESS,
            "performance": FeedbackCategory.PERFORMANCE,
            "security": FeedbackCategory.SECURITY,
            "completeness": FeedbackCategory.FUNCTIONALITY,
            "user_experience": FeedbackCategory.UX,
            "visual_design": FeedbackCategory.UX,
            "clarity": FeedbackCategory.DOCUMENTATION,
            "edge_cases": FeedbackCategory.CORRECTNESS,
            "approach": FeedbackCategory.ARCHITECTURE
        }
        
        return mapping.get(category_name, FeedbackCategory.OTHER)
    
    def _determine_severity(
        self, 
        item: FeedbackItem, 
        category: FeedbackCategory,
        review_status: ReviewStatus
    ) -> FeedbackSeverity:
        """Determine the severity of a feedback item.
        
        Args:
            item: The feedback item
            category: The determined category
            review_status: Overall review status
            
        Returns:
            FeedbackSeverity for the item
        """
        # Use explicit severity if provided
        if item.severity:
            try:
                return FeedbackSeverity(item.severity.lower())
            except ValueError:
                pass
        
        # Set severity based on review status
        if review_status == ReviewStatus.REJECTED:
            # Higher baseline severity for rejected reviews
            base_severity = FeedbackSeverity.HIGH
        elif review_status == ReviewStatus.NEEDS_CLARIFICATION:
            base_severity = FeedbackSeverity.MEDIUM
        elif review_status == ReviewStatus.PARTIALLY_APPROVED:
            base_severity = FeedbackSeverity.MEDIUM
        else:
            base_severity = FeedbackSeverity.LOW
        
        # Check for severity indicators in the text
        text = f"{item.comment} {item.suggested_changes or ''}"
        if re.search(r'(?i)critical|urgent|immediate|severe|crucial|blocking', text):
            return FeedbackSeverity.CRITICAL
        elif re.search(r'(?i)important|significant|major|high priority', text):
            return FeedbackSeverity.HIGH
        elif re.search(r'(?i)moderate|medium|average|normal', text):
            return FeedbackSeverity.MEDIUM
        elif re.search(r'(?i)minor|low|trivial|cosmetic|suggestion|nice to have', text):
            return FeedbackSeverity.LOW
        
        # Get default severity from rules if available
        for rule_group in self.feedback_rules.values():
            for category_name, rules in rule_group.get("categories", {}).items():
                if category_name == category.value or self._map_to_feedback_category(category_name) == category:
                    default_severity = rules.get("default_severity")
                    if default_severity:
                        return default_severity
        
        # Fall back to base severity
        return base_severity
    
    def _determine_action(
        self,
        item: FeedbackItem,
        category: FeedbackCategory,
        severity: FeedbackSeverity,
        review_status: ReviewStatus
    ) -> FeedbackAction:
        """Determine the action to take based on the feedback.
        
        Args:
            item: The feedback item
            category: The determined category
            severity: The determined severity
            review_status: Overall review status
            
        Returns:
            FeedbackAction to take
        """
        # Critical severity always requires immediate revision
        if severity == FeedbackSeverity.CRITICAL:
            return FeedbackAction.REVISE
        
        # Check for action indicators in the text
        text = f"{item.comment} {item.suggested_changes or ''}"
        if re.search(r'(?i)revise|update|change|fix|modify|replace|improve', text):
            return FeedbackAction.REVISE
        elif re.search(r'(?i)discuss|clarify|question|unclear|consider|suggest', text):
            return FeedbackAction.DISCUSS
        elif re.search(r'(?i)ignore|minor|not important|don\'t worry|optional', text):
            return FeedbackAction.IGNORE
        elif re.search(r'(?i)escalate|attention|review needed|higher level', text):
            return FeedbackAction.ESCALATE
        elif re.search(r'(?i)document|add documentation|comment|explain', text):
            return FeedbackAction.DOCUMENT
        elif re.search(r'(?i)refactor|restructure|reorganize|clean up', text):
            return FeedbackAction.REFACTOR
        elif re.search(r'(?i)implement|add|create|missing|include', text):
            return FeedbackAction.IMPLEMENT
        
        # Get default action from rules if available
        for rule_group in self.feedback_rules.values():
            for category_name, rules in rule_group.get("categories", {}).items():
                if category_name == category.value or self._map_to_feedback_category(category_name) == category:
                    default_action = rules.get("default_action")
                    if default_action:
                        return default_action
        
        # Default actions based on severity and review status
        if review_status == ReviewStatus.REJECTED:
            return FeedbackAction.REVISE
        elif review_status == ReviewStatus.NEEDS_CLARIFICATION:
            return FeedbackAction.DISCUSS
        elif severity in [FeedbackSeverity.HIGH, FeedbackSeverity.MEDIUM]:
            return FeedbackAction.REVISE
        else:
            return FeedbackAction.DOCUMENT
    
    def _extract_specific_changes(self, item: FeedbackItem) -> Optional[List[Dict[str, Any]]]:
        """Extract specific changes from feedback if available.
        
        Args:
            item: The feedback item
            
        Returns:
            List of specific changes or None if not available
        """
        specific_changes = []
        
        # Check for specific location information
        if item.location:
            # Parse location information (e.g., "file.py:23" or "component_name")
            location_parts = item.location.split(':')
            change = {
                "type": "location",
                "file": location_parts[0]
            }
            if len(location_parts) > 1:
                try:
                    change["line"] = int(location_parts[1])
                except ValueError:
                    change["component"] = location_parts[1]
            
            specific_changes.append(change)
        
        # Check for suggested changes
        if item.suggested_changes:
            # Look for code blocks in the suggested changes
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', item.suggested_changes, re.DOTALL)
            
            if code_blocks:
                for i, block in enumerate(code_blocks):
                    specific_changes.append({
                        "type": "code_change",
                        "index": i,
                        "code": block.strip()
                    })
            
            # Look for "from/to" style suggestions
            from_to = re.findall(
                r'(?:from|change from|replace):\s*`([^`]+)`\s*(?:to|with):\s*`([^`]+)`',
                item.suggested_changes
            )
            
            if from_to:
                for i, (from_text, to_text) in enumerate(from_to):
                    specific_changes.append({
                        "type": "text_replacement",
                        "index": i,
                        "from": from_text,
                        "to": to_text
                    })
        
        return specific_changes if specific_changes else None
    
    def _learn_from_feedback(self, processed: ProcessedFeedback, original: FeedbackItem) -> None:
        """Learn from processed feedback to improve future processing.
        
        Args:
            processed: The processed feedback
            original: The original feedback item
        """
        # Store the mapping from text features to categorization
        key_phrases = self._extract_key_phrases(original.comment)
        
        for phrase in key_phrases:
            self.feedback_patterns[phrase].append({
                "category": processed.category.value,
                "severity": processed.severity.value,
                "action": processed.action.value,
                "feedback_type": original.feedback_type
            })
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for pattern learning.
        
        Args:
            text: The text to extract phrases from
            
        Returns:
            List of key phrases
        """
        # This is a simplified implementation
        # A more advanced version would use NLP techniques for phrase extraction
        phrases = []
        
        # Extract potential key phrases (2-3 word combinations)
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        
        # Get bigrams and trigrams
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
            
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return phrases
    
    async def _create_task_revisions(
        self, 
        task_id: str, 
        feedback_items: List[ProcessedFeedback]
    ) -> Optional[TaskRevision]:
        """Create task revisions based on feedback.
        
        Args:
            task_id: ID of the original task
            feedback_items: List of processed feedback items
            
        Returns:
            Created TaskRevision or None if creation failed
        """
        # Skip if no actionable feedback
        actionable_items = [
            item for item in feedback_items
            if item.action in [FeedbackAction.REVISE, FeedbackAction.IMPLEMENT, FeedbackAction.REFACTOR]
        ]
        
        if not actionable_items:
            logger.info(f"No actionable feedback for task {task_id}, skipping revision")
            return None
        
        # Get the original task context
        task_context = None
        if self.shared_memory:
            task_data = self.shared_memory.retrieve(
                key=task_id,
                category="tasks"
            )
            if task_data:
                task_context = task_data
        
        if not task_context:
            logger.warning(f"Cannot create revision for task {task_id}: original task not found")
            return None
        
        # Extract agent ID from task context
        agent_id = task_context.get("agent_id", "unknown")
        
        # Determine priority based on feedback severity
        priority = "medium"
        if any(item.severity == FeedbackSeverity.CRITICAL for item in actionable_items):
            priority = "critical"
        elif any(item.severity == FeedbackSeverity.HIGH for item in actionable_items):
            priority = "high"
        
        # Create revision requirements
        requirements = {
            "original_task": task_context.get("requirements", {}),
            "feedback": [
                {
                    "id": item.id,
                    "category": item.category.value,
                    "severity": item.severity.value,
                    "description": item.description,
                    "specific_changes": item.specific_changes
                }
                for item in actionable_items
            ]
        }
        
        # Create the revision task
        revision = TaskRevision(
            original_task_id=task_id,
            feedback_ids=[item.id for item in actionable_items],
            agent_id=agent_id,
            description=f"Revise task {task_id} based on feedback",
            requirements=requirements,
            priority=priority
        )
        
        # Store the revision
        self.task_revisions[revision.id] = revision
        
        # Store in shared memory if available
        if self.shared_memory:
            self.shared_memory.store(
                key=f"task_revision_{revision.id}",
                value=revision.dict(),
                category="task_revisions"
            )
        
        # Schedule the revision if task scheduler is available
        if self.task_scheduler:
            try:
                await self.task_scheduler.schedule_revision(revision)
                logger.info(f"Scheduled revision {revision.id} for task {task_id}")
            except Exception as e:
                logger.error(f"Error scheduling revision for task {task_id}: {str(e)}")
        
        return revision
    
    async def _update_metrics(self) -> None:
        """Update feedback metrics based on processed feedback."""
        # Skip if no feedback has been processed
        if not self.processed_feedback:
            return
        
        # Count by type, severity, agent, category
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        agent_counts = defaultdict(int)
        category_counts = defaultdict(int)
        action_counts = defaultdict(int)
        
        # Count revisions by task
        task_revisions = defaultdict(int)
        
        # Track issues for common patterns
        issues = defaultdict(int)
        
        # Process all feedback
        for feedback in self.processed_feedback.values():
            # Update counts
            feedback_type = feedback.context.get("feedback_type", "unknown")
            type_counts[feedback_type] += 1
            severity_counts[feedback.severity.value] += 1
            agent_counts[feedback.agent_id] += 1
            category_counts[feedback.category.value] += 1
            action_counts[feedback.action.value] += 1
            
            # Track task revisions
            if feedback.action in [FeedbackAction.REVISE, FeedbackAction.IMPLEMENT, FeedbackAction.REFACTOR]:
                task_revisions[feedback.task_id] += 1
            
            # Track common issues (category + description)
            issue_key = f"{feedback.category.value}:{feedback.description[:50]}"
            issues[issue_key] += 1
        
        # Calculate approval rate
        total_reviews = 0
        approved_reviews = 0
        
        if self.shared_memory:
            # Get review submissions from shared memory
            submission_keys = self.shared_memory.get_keys(category="review_submissions")
            
            for key in submission_keys:
                submission_data = self.shared_memory.retrieve(key, "review_submissions")
                if submission_data:
                    total_reviews += 1
                    if submission_data.get("status") == ReviewStatus.APPROVED:
                        approved_reviews += 1
        
        approval_rate = (approved_reviews / total_reviews) if total_reviews > 0 else 0.0
        
        # Calculate revision rate
        tasks_with_revisions = len(task_revisions)
        total_tasks = len(set(f.task_id for f in self.processed_feedback.values()))
        revision_rate = (tasks_with_revisions / total_tasks) if total_tasks > 0 else 0.0
        
        # Calculate average revisions per task
        avg_revisions = sum(task_revisions.values()) / total_tasks if total_tasks > 0 else 0.0
        
        # Find common issues (top 10)
        common_issues = [
            {"issue": key, "count": count}
            for key, count in sorted(issues.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Update metrics
        self.metrics = FeedbackMetrics(
            total_feedback_items=len(self.processed_feedback),
            feedback_by_type={k: v for k, v in type_counts.items()},
            feedback_by_severity={k: v for k, v in severity_counts.items()},
            feedback_by_agent={k: v for k, v in agent_counts.items()},
            feedback_by_category={k: v for k, v in category_counts.items()},
            common_issues=common_issues,
            approval_rate=approval_rate,
            revision_rate=revision_rate,
            average_revisions_per_task=avg_revisions
        )
        
        # Store metrics in shared memory if available
        if self.shared_memory:
            self.shared_memory.store(
                key="feedback_metrics",
                value=self.metrics.dict(),
                category="metrics"
            )
        
        logger.info("Updated feedback metrics")
    
    async def get_agent_feedback_report(self, agent_id: str) -> Dict[str, Any]:
        """Generate a feedback report for a specific agent.
        
        Args:
            agent_id: ID of the agent to report on
            
        Returns:
            Dictionary containing the feedback report
        """
        # Get all feedback for this agent
        agent_feedback = [
            f for f in self.processed_feedback.values()
            if f.agent_id == agent_id
        ]
        
        if not agent_feedback:
            return {
                "agent_id": agent_id,
                "total_feedback": 0,
                "message": "No feedback available for this agent"
            }
        
        # Count by category, severity, action
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        action_counts = defaultdict(int)
        
        for feedback in agent_feedback:
            category_counts[feedback.category.value] += 1
            severity_counts[feedback.severity.value] += 1
            action_counts[feedback.action.value] += 1
        
        # Get task revisions for this agent
        task_revisions = [
            r for r in self.task_revisions.values()
            if r.agent_id == agent_id
        ]
        
        # Find common feedback patterns
        feedback_texts = [f.description for f in agent_feedback]
        common_patterns = self._find_common_feedback_patterns(feedback_texts)
        
        # Create the report
        report = {
            "agent_id": agent_id,
            "total_feedback": len(agent_feedback),
            "feedback_by_category": dict(category_counts),
            "feedback_by_severity": dict(severity_counts),
            "feedback_by_action": dict(action_counts),
            "revisions_required": len(task_revisions),
            "common_feedback_patterns": common_patterns,
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _find_common_feedback_patterns(self, texts: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """Find common patterns in feedback texts.
        
        Args:
            texts: List of feedback texts
            top_n: Number of top patterns to return
            
        Returns:
            List of common patterns with counts
        """
        # This is a simplified implementation
        # A more advanced version would use NLP techniques for pattern extraction
        
        # Count frequent phrases
        phrase_counts = defaultdict(int)
        
        for text in texts:
            # Extract phrases (similar to _extract_key_phrases)
            words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
            
            # Get bigrams and trigrams
            for i in range(len(words) - 1):
                phrase_counts[f"{words[i]} {words[i+1]}"] += 1
                
            for i in range(len(words) - 2):
                phrase_counts[f"{words[i]} {words[i+1]} {words[i+2]}"] += 1
        
        # Get top phrases
        top_phrases = [
            {"pattern": phrase, "count": count}
            for phrase, count in sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            if count > 1  # Only include if appears more than once
        ]
        
        return top_phrases
    
    async def get_task_feedback_history(self, task_id: str) -> Dict[str, Any]:
        """Get the feedback history for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary containing the feedback history
        """
        # Get all feedback for this task
        task_feedback = [
            f for f in self.processed_feedback.values()
            if f.task_id == task_id
        ]
        
        # Get all revisions for this task
        task_revisions = [
            r for r in self.task_revisions.values()
            if r.original_task_id == task_id
        ]
        
        # Sort by timestamp
        task_feedback.sort(key=lambda f: f.processed_at)
        task_revisions.sort(key=lambda r: r.created_at)
        
        # Create the history
        history = {
            "task_id": task_id,
            "feedback_items": [f.dict() for f in task_feedback],
            "revisions": [r.dict() for r in task_revisions],
            "total_feedback": len(task_feedback),
            "total_revisions": len(task_revisions),
            "timestamp": datetime.now().isoformat()
        }
        
        return history
    
    async def apply_specific_changes(
        self,
        content: str,
        feedback: ProcessedFeedback
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply specific changes from feedback to content.
        
        Args:
            content: Original content to modify
            feedback: Processed feedback with specific changes
            
        Returns:
            Tuple of (modified content, list of applied changes)
        """
        if not feedback.specific_changes:
            return content, []
        
        modified_content = content
        applied_changes = []
        
        for change in feedback.specific_changes:
            try:
                change_type = change.get("type")
                
                if change_type == "text_replacement" and "from" in change and "to" in change:
                    from_text = change["from"]
                    to_text = change["to"]
                    
                    # Apply the replacement
                    if from_text in modified_content:
                        modified_content = modified_content.replace(from_text, to_text)
                        applied_changes.append({
                            "type": "text_replacement",
                            "from": from_text,
                            "to": to_text,
                            "success": True
                        })
                    else:
                        applied_changes.append({
                            "type": "text_replacement",
                            "from": from_text,
                            "to": to_text,
                            "success": False,
                            "reason": "Text not found"
                        })
                
                elif change_type == "code_change" and "code" in change:
                    # This is a more complex change that would typically
                    # involve pattern matching or context understanding
                    # Simplified implementation just logs it
                    applied_changes.append({
                        "type": "code_change",
                        "success": False,
                        "reason": "Code changes require more context"
                    })
                
                elif change_type == "location" and "line" in change:
                    # Changes that target specific lines
                    # This would typically involve line-by-line parsing
                    applied_changes.append({
                        "type": "location",
                        "line": change["line"],
                        "success": False,
                        "reason": "Line-specific changes require more context"
                    })
                
                else:
                    applied_changes.append({
                        "type": change_type,
                        "success": False,
                        "reason": "Unknown change type or missing information"
                    })
                    
            except Exception as e:
                applied_changes.append({
                    "type": change.get("type", "unknown"),
                    "success": False,
                    "reason": f"Error applying change: {str(e)}"
                })
        
        return modified_content, applied_changes
    
    async def suggest_improvements(self, agent_id: str) -> Dict[str, Any]:
        """Suggest improvements for an agent based on feedback history.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary containing improvement suggestions
        """
        # Get all feedback for this agent
        agent_feedback = [
            f for f in self.processed_feedback.values()
            if f.agent_id == agent_id
        ]
        
        if not agent_feedback:
            return {
                "agent_id": agent_id,
                "message": "No feedback history available for this agent"
            }
        
        # Analyze patterns in feedback
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for feedback in agent_feedback:
            category_counts[feedback.category.value] += 1
            severity_counts[feedback.severity.value] += 1
        
        # Find most common categories
        top_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Find highest severity issues
        high_severity_feedback = [
            f for f in agent_feedback
            if f.severity in [FeedbackSeverity.CRITICAL, FeedbackSeverity.HIGH]
        ]
        
        # Generate improvement suggestions
        suggestions = []
        
        # Suggest improvements based on top categories
        for category, count in top_categories:
            category_feedback = [f for f in agent_feedback if f.category.value == category]
            
            if category == FeedbackCategory.STYLE.value:
                suggestions.append({
                    "area": "Code Style",
                    "suggestion": "Improve adherence to coding style guidelines",
                    "details": "Common style issues include indentation, naming conventions, and code organization",
                    "priority": "Medium",
                    "feedback_count": count
                })
            
            elif category == FeedbackCategory.CORRECTNESS.value:
                suggestions.append({
                    "area": "Correctness",
                    "suggestion": "Focus on correctness and edge case handling",
                    "details": "Ensure all requirements are correctly implemented and edge cases are properly handled",
                    "priority": "High",
                    "feedback_count": count
                })
            
            elif category == FeedbackCategory.PERFORMANCE.value:
                suggestions.append({
                    "area": "Performance",
                    "suggestion": "Optimize code for better performance",
                    "details": "Review algorithms and data structures for inefficiencies",
                    "priority": "Medium",
                    "feedback_count": count
                })
            
            elif category == FeedbackCategory.SECURITY.value:
                suggestions.append({
                    "area": "Security",
                    "suggestion": "Address security vulnerabilities",
                    "details": "Focus on input validation, authentication, and authorization",
                    "priority": "High",
                    "feedback_count": count
                })
            
            elif category == FeedbackCategory.DOCUMENTATION.value:
                suggestions.append({
                    "area": "Documentation",
                    "suggestion": "Improve code documentation",
                    "details": "Add more detailed comments and ensure all functions are properly documented",
                    "priority": "Medium",
                    "feedback_count": count
                })
            
            elif category == FeedbackCategory.UX.value:
                suggestions.append({
                    "area": "User Experience",
                    "suggestion": "Enhance user interface and experience",
                    "details": "Focus on usability, accessibility, and visual design",
                    "priority": "Medium",
                    "feedback_count": count
                })
        
        # Add specific suggestions based on high severity feedback
        for feedback in high_severity_feedback[:3]:  # Top 3 high severity issues
            suggestions.append({
                "area": f"High Priority: {feedback.category.value}",
                "suggestion": f"Address critical feedback: {feedback.description[:100]}...",
                "details": feedback.description,
                "priority": "High",
                "feedback_id": feedback.id
            })
        
        # Create the suggestions report
        report = {
            "agent_id": agent_id,
            "total_feedback_analyzed": len(agent_feedback),
            "suggestions": suggestions,
            "top_improvement_areas": [category for category, _ in top_categories],
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def get_feedback_metrics(self) -> FeedbackMetrics:
        """Get current feedback metrics.
        
        Returns:
            Current feedback metrics
        """
        return self.metrics
    
    def clear_old_feedback(self, days_old: int = 30) -> int:
        """Clear old processed feedback to prevent memory growth.
        
        Args:
            days_old: Age in days for feedback to clear
            
        Returns:
            Number of feedback items cleared
        """
        cutoff = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        to_remove = []
        
        for feedback_id, feedback in self.processed_feedback.items():
            try:
                feedback_time = datetime.fromisoformat(feedback.processed_at).timestamp()
                if feedback_time < cutoff:
                    to_remove.append(feedback_id)
            except (ValueError, TypeError):
                # If we can't parse the date, keep the feedback
                continue
        
        # Remove from local storage
        for feedback_id in to_remove:
            del self.processed_feedback[feedback_id]
            
            # Remove from shared memory if available
            if self.shared_memory:
                self.shared_memory.delete(
                    key=f"processed_feedback_{feedback_id}",
                    category="processed_feedback"
                )
        
        logger.info(f"Cleared {len(to_remove)} feedback items older than {days_old} days")
        return len(to_remove)


# Example usage
async def example_feedback_processing():
    """Example of how to use the feedback processor."""
    # Create processor
    processor = FeedbackProcessor()
    
    # Create a mock review submission
    review = ReviewSubmission(
        task_id="task123",
        reviewer_id="human456",
        status=ReviewStatus.REJECTED,
        feedback_items=[
            FeedbackItem(
                feedback_type=FeedbackType.CODE_QUALITY,
                comment="This code doesn't follow our style guidelines. The indentation is inconsistent.",
                location="main.py:25",
                severity="medium"
            ),
            FeedbackItem(
                feedback_type=FeedbackType.FUNCTIONALITY,
                comment="The function fails when given empty input. Please add validation.",
                location="main.py:32",
                severity="high",
                suggested_changes="Add a check at the beginning: if not input_data: return None"
            )
        ],
        summary="The code needs revision for style issues and proper input validation."
    )
    
    # Process the review
    processed_items = await processor.process_review(review)
    
    print(f"Processed {len(processed_items)} feedback items")
    for item in processed_items:
        print(f"Category: {item.category.value}, Action: {item.action.value}")
    
    # Generate agent report
    report = await processor.get_agent_feedback_report("unknown")
    print(f"Agent report: {len(report['feedback_by_category'])} categories of feedback")


if __name__ == "__main__":
    # Run the example
    import asyncio
    asyncio.run(example_feedback_processing())
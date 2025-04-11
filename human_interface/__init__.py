"""
Human interface module for the multi-agent development system.

This package provides interfaces for human-agent interaction, including
review systems, feedback processing, and collaborative editing. It enables
humans to provide guidance, review artifacts, and steer the multi-agent
system's outputs.
"""

# Version information
__version__ = "0.1.0"

# Import primary interface classes
from human_interface.review_interface import (
    ReviewInterface,
    ReviewRequest,
    ReviewableArtifact,
    ReviewSubmission,
    FeedbackItem,
    ReviewStatus,
    FeedbackType,
    DisplayFormat,
    ReviewCallback,
)

from human_interface.feedback_processor import (
    FeedbackProcessor,
    ProcessedFeedback,
    TaskRevision,
    FeedbackMetrics,
    FeedbackSeverity,
    FeedbackCategory,
    FeedbackAction,
)

# Convenience functions
def create_review_interface(
    storage_dir=None,
    shared_memory=None,
    notification_callback=None
) -> ReviewInterface:
    """Create a configured review interface instance.
    
    Args:
        storage_dir: Directory to store review artifacts
        shared_memory: Shared memory interface
        notification_callback: Callback function for notifications
        
    Returns:
        Configured ReviewInterface instance
    """
    return ReviewInterface(
        storage_dir=storage_dir,
        shared_memory=shared_memory,
        notification_callback=notification_callback
    )


def create_feedback_processor(
    shared_memory=None,
    task_scheduler=None,
    learning_enabled=True
) -> FeedbackProcessor:
    """Create a configured feedback processor instance.
    
    Args:
        shared_memory: Shared memory interface
        task_scheduler: Task scheduler for creating revisions
        learning_enabled: Enable learning from feedback patterns
        
    Returns:
        Configured FeedbackProcessor instance
    """
    return FeedbackProcessor(
        shared_memory=shared_memory,
        task_scheduler=task_scheduler,
        learning_enabled=learning_enabled
    )


# Common notification callbacks
async def email_notification(request):
    """Send an email notification for a review request."""
    # This is a placeholder - actual implementation would be provided elsewhere
    pass


async def slack_notification(request):
    """Send a Slack notification for a review request."""
    # This is a placeholder - actual implementation would be provided elsewhere
    pass


# Constants
DEFAULT_REVIEW_INSTRUCTIONS = {
    "code": (
        "Please review this code for correctness, style, performance, and security. "
        "Provide specific feedback on issues and suggest improvements."
    ),
    "design": (
        "Please review this design for usability, aesthetics, and alignment with requirements. "
        "Suggest specific improvements where appropriate."
    ),
    "document": (
        "Please review this document for clarity, completeness, and accuracy. "
        "Suggest improvements or corrections as needed."
    ),
    "general": (
        "Please review this artifact and provide feedback on any issues or areas for improvement."
    )
}

# Public API
__all__ = [
    # Classes
    "ReviewInterface",
    "ReviewRequest",
    "ReviewableArtifact",
    "ReviewSubmission",
    "FeedbackItem",
    "ReviewStatus",
    "FeedbackType",
    "DisplayFormat",
    "ReviewCallback",
    "FeedbackProcessor",
    "ProcessedFeedback",
    "TaskRevision",
    "FeedbackMetrics",
    "FeedbackSeverity",
    "FeedbackCategory",
    "FeedbackAction",
    
    # Functions
    "create_review_interface",
    "create_feedback_processor",
    "email_notification",
    "slack_notification",
    
    # Constants
    "DEFAULT_REVIEW_INSTRUCTIONS",
]

# Version compatibility check
import sys
if sys.version_info < (3, 8):
    import warnings
    warnings.warn(
        "The human_interface package requires Python 3.8 or newer. "
        "Some features may not work as expected.",
        ImportWarning
    )
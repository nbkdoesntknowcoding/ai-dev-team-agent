"""
Tools for the multi-agent development system.

This package provides a collection of tools for code generation, analysis,
research, and review used by specialized agents in the multi-agent system.
"""

# Version info
__version__ = "0.1.0"

# Import tools for external use
from .code_tools import (
    CodeTool, 
    CodeFragment, 
    CodeAnalysisResult, 
    CodeTransformResult, 
    CodeValidationResult, 
    TestResult,
    CodeLanguage
)

from .research_tools import (
    ResearchTool, 
    ResearchQuery, 
    ResearchResult, 
    SearchResult, 
    DocumentationResult, 
    PackageInfo, 
    ComparisonResult, 
    BestPracticeItem,
    ResearchSource
)

from .review_tools import (
    CodeReviewTool, 
    CodeIssue, 
    QualityMetrics, 
    ReviewResult, 
    TestSuiteResult, 
    CoverageResult,
    IssueType,
    IssueCategory,
    TestStatus
)

# Define what's available for import with "from tools import *"
__all__ = [
    # Code tools
    "CodeTool",
    "CodeFragment",
    "CodeAnalysisResult",
    "CodeTransformResult", 
    "CodeValidationResult",
    "TestResult",
    "CodeLanguage",
    
    # Research tools
    "ResearchTool",
    "ResearchQuery",
    "ResearchResult",
    "SearchResult",
    "DocumentationResult",
    "PackageInfo",
    "ComparisonResult",
    "BestPracticeItem",
    "ResearchSource",
    
    # Review tools
    "CodeReviewTool",
    "CodeIssue",
    "QualityMetrics",
    "ReviewResult",
    "TestSuiteResult",
    "CoverageResult",
    "IssueType",
    "IssueCategory",
    "TestStatus"
]

# Utility function to get all available tools
def get_available_tools():
    """Get information about all available tools.
    
    Returns:
        Dictionary mapping tool names to their availability status
    """
    tools = {}
    
    # Check code tools
    try:
        from .code_tools import CodeTool
        code_tool = CodeTool()
        tools["code_tools"] = {
            "available": True,
            "supported_languages": code_tool.language_features.keys()
        }
    except ImportError:
        tools["code_tools"] = {"available": False}
    
    # Check research tools
    try:
        from .research_tools import ResearchTool
        research_tool = ResearchTool()
        tools["research_tools"] = {
            "available": True,
            "online_search_enabled": research_tool.enable_online_search,
            "supported_languages": research_tool.get_supported_languages()
        }
    except ImportError:
        tools["research_tools"] = {"available": False}
    
    # Check review tools
    try:
        from .review_tools import CodeReviewTool
        review_tool = CodeReviewTool()
        tools["review_tools"] = {
            "available": True,
            "analyzers": review_tool.available_analyzers
        }
    except ImportError:
        tools["review_tools"] = {"available": False}
    
    return tools
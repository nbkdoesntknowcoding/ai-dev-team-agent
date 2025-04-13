"""
QA Agents for the multi-agent development system.

This module contains specialized agents for quality assurance tasks, including
code review, test development, and user experience testing. These agents work
together to ensure high-quality, reliable, and user-friendly software.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple, cast
import uuid

from pydantic import BaseModel, Field, validator

from agents.base_agent import (
    BaseAgent, 
    Task, 
    TaskResult, 
    TaskStatus, 
    TaskPriority,
    TaskContext,
    AgentRole,
    ModelProvider
)

# Set up logging
logger = logging.getLogger(__name__)


class CodeReviewComment(BaseModel):
    """A comment on a specific part of the code during review."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    comment: str
    severity: str  # "critical", "major", "minor", "suggestion"
    category: str  # "security", "performance", "maintainability", "functionality", etc.
    recommended_fix: Optional[str] = None


class CodeReview(BaseModel):
    """A complete code review with comments and summary."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    repository: str
    branch: str
    commit_id: Optional[str] = None
    pull_request_id: Optional[str] = None
    comments: List[CodeReviewComment] = Field(default_factory=list)
    summary: str
    quality_score: float  # 0.0 to 10.0
    recommendations: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class TestCase(BaseModel):
    """A test case definition."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    type: str  # "unit", "integration", "e2e", "performance", etc.
    preconditions: Optional[List[str]] = None
    steps: List[str] = Field(default_factory=list)
    expected_results: List[str] = Field(default_factory=list)
    associated_requirements: Optional[List[str]] = None
    automated: bool = False
    test_code: Optional[str] = None


class TestSuite(BaseModel):
    """A collection of related test cases."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    test_cases: List[TestCase] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class UsabilityTest(BaseModel):
    """A usability test definition."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    user_persona: Dict[str, Any]
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class UsabilityTestResult(BaseModel):
    """Results from a usability test."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    test_id: str
    success_rate: float  # 0.0 to 1.0
    task_results: List[Dict[str, Any]] = Field(default_factory=list)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class CodeReviewer(BaseAgent):
    """Agent specialized in reviewing code for quality and correctness."""
    
    def __init__(self, name, preferred_languages=None, review_checklist=None, model_provider=ModelProvider.ANTHROPIC, model_name="claude-3-sonnet-20240229", **kwargs):
        # Initialize attributes before calling super().__init__
        self.preferred_languages = preferred_languages if isinstance(preferred_languages, list) else ["Python", "JavaScript", "TypeScript"]
        self.review_checklist = review_checklist or self._default_review_checklist()
        
        super().__init__(
            name=name,
            agent_type=AgentRole.CODE_REVIEWER,
            model_provider=model_provider,
            model_name=model_name,
            **kwargs
        )
        """Initialize the Code Reviewer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_languages: List of programming languages the agent specializes in
            review_checklist: Optional checklist of items to review
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.CODE_REVIEWER, 
            **kwargs
        )
        self.preferred_languages = preferred_languages
        self.review_checklist = review_checklist or self._default_review_checklist()
        
        # Track code reviews
        self.code_reviews: Dict[str, CodeReview] = {}
        
        # Track review statistics
        self.review_statistics: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Code Reviewer Agent initialized with {', '.join(preferred_languages)} languages")
    
    def _default_review_checklist(self) -> List[Dict[str, Any]]:
        """Generate a default review checklist.
        
        Returns:
            Default review checklist
        """
        return [
            {
                "category": "code_quality",
                "items": [
                    "Code follows project style guides and conventions",
                    "Functions and methods are appropriately sized",
                    "Variables and functions have descriptive names",
                    "Code is not unnecessarily complex",
                    "No code duplication"
                ]
            },
            {
                "category": "functionality",
                "items": [
                    "Code correctly implements the intended functionality",
                    "Edge cases are handled appropriately",
                    "Input validation is present where needed",
                    "Error handling is implemented properly"
                ]
            },
            {
                "category": "performance",
                "items": [
                    "Algorithms are efficient for the task",
                    "No unnecessary computations or memory usage",
                    "Database queries are optimized",
                    "Resource usage is appropriate"
                ]
            },
            {
                "category": "security",
                "items": [
                    "No obvious security vulnerabilities",
                    "Sensitive data is handled properly",
                    "Authentication and authorization are implemented correctly",
                    "User input is sanitized before use"
                ]
            },
            {
                "category": "maintainability",
                "items": [
                    "Code is well-documented",
                    "Complex logic has explanatory comments",
                    "Code structure is logical and understandable",
                    "Dependencies are clear and minimal"
                ]
            },
            {
                "category": "testing",
                "items": [
                    "Tests are present for new functionality",
                    "Tests cover edge cases",
                    "Tests are meaningful and not just for coverage",
                    "Tests are maintainable"
                ]
            }
        ]
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Code Reviewer."""
        return (
            f"You are {self.name}, a Code Reviewer specialized in reviewing code for "
            f"quality, correctness, and adherence to best practices. "
            f"You are particularly experienced with {', '.join(self.preferred_languages)}. "
            f"Your responsibilities include:\n"
            f"1. Reviewing code for readability, maintainability, and correctness\n"
            f"2. Identifying security vulnerabilities and performance issues\n"
            f"3. Ensuring code follows project standards and best practices\n"
            f"4. Providing constructive feedback and suggestions for improvement\n"
            f"5. Promoting knowledge sharing and code quality across the team\n\n"
            f"When reviewing code, be thorough but constructive. Focus on substantive issues "
            f"rather than stylistic preferences unless they affect readability. Provide specific, "
            f"actionable feedback with examples of better approaches when possible. Remember that "
            f"the goal is to improve the code and help developers grow, not to criticize."
        )
    
    async def review_code(
        self, 
        repository: str,
        branch: str,
        files: List[Dict[str, Any]],
        commit_id: Optional[str] = None,
        pull_request_id: Optional[str] = None,
        review_context: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Review code files for quality, correctness, and best practices.
        
        Args:
            repository: Repository identifier
            branch: Branch name
            files: List of files with content to review
            commit_id: Optional commit ID
            pull_request_id: Optional pull request ID
            review_context: Optional additional context for the review
            
        Returns:
            TaskResult containing the code review
        """
        # Create a task for code review
        task = Task(
            task_id=f"code_review_{repository.replace('/', '_')}_{branch}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            description=f"Review code in {repository} on branch {branch}",
            agent_type=str(AgentRole.CODE_REVIEWER),
            requirements={
                "repository": repository,
                "branch": branch,
                "files": files,
                "commit_id": commit_id,
                "pull_request_id": pull_request_id,
                "review_checklist": self.review_checklist,
                "review_context": review_context or {}
            },
            context=TaskContext(
                notes=(
                    f"Review the provided code files in {repository} on branch {branch}. "
                    f"Focus on code quality, functionality, performance, security, and maintainability. "
                    f"Provide constructive feedback and suggestions for improvement."
                )
            ),
            expected_output=(
                "A comprehensive code review with specific comments on issues found, "
                "a quality score, and recommendations for improvement."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the code review
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the code review from the result
                review_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    review_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            review_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not review_data:
                    logger.warning(f"Could not parse code review as JSON. Attempting to extract from text.")
                    review_data = self._extract_review_from_text(result.result, repository, branch, files)
                
                # Create review comments
                comments = []
                for comment_data in review_data.get("comments", []):
                    comment = CodeReviewComment(
                        file_path=comment_data.get("file_path", ""),
                        line_number=comment_data.get("line_number"),
                        code_snippet=comment_data.get("code_snippet"),
                        comment=comment_data.get("comment", ""),
                        severity=comment_data.get("severity", "minor"),
                        category=comment_data.get("category", "maintainability"),
                        recommended_fix=comment_data.get("recommended_fix")
                    )
                    comments.append(comment)
                
                # Create the code review
                code_review = CodeReview(
                    repository=repository,
                    branch=branch,
                    commit_id=commit_id,
                    pull_request_id=pull_request_id,
                    comments=comments,
                    summary=review_data.get("summary", ""),
                    quality_score=review_data.get("quality_score", 5.0),
                    recommendations=review_data.get("recommendations", [])
                )
                
                # Store the code review
                self.code_reviews[code_review.id] = code_review
                
                # Update review statistics
                self._update_review_statistics(code_review)
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"code_review_{code_review.id}",
                        value=code_review.dict(),
                        category="code_reviews"
                    )
                
                logger.info(f"Created code review for {repository} on branch {branch} with {len(comments)} comments")
                
                # Return the code review as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=code_review.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"review_id": code_review.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing code review: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_review_from_text(
        self, 
        text: str, 
        repository: str, 
        branch: str, 
        files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract structured code review data from unstructured text.
        
        Args:
            text: The text to extract from
            repository: Repository identifier
            branch: Branch name
            files: List of files reviewed
            
        Returns:
            Structured code review data
        """
        review_data = {
            "repository": repository,
            "branch": branch,
            "comments": [],
            "summary": "",
            "quality_score": 5.0,
            "recommendations": []
        }
        
        # Extract summary
        summary_match = re.search(r'(?i)#+\s*(?:Summary|Overview)(?:\n+(.+?))?(?=\n#+|\Z)', text, re.DOTALL)
        if summary_match and summary_match.group(1):
            review_data["summary"] = summary_match.group(1).strip()
        
        # Extract quality score
        score_match = re.search(r'(?i)(?:Quality|Overall) Score:?\s*(\d+(?:\.\d+)?)', text)
        if score_match:
            try:
                score = float(score_match.group(1))
                if 0 <= score <= 10:
                    review_data["quality_score"] = score
            except ValueError:
                pass
        
        # Extract recommendations
        recommendations_section = re.search(
            r'(?i)#+\s*(?:Recommendations|Suggested Improvements)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if recommendations_section and recommendations_section.group(1):
            recommendations_text = recommendations_section.group(1)
            
            # Extract bullet points
            rec_items = re.findall(r'[-*]\s*([^\n]+)', recommendations_text)
            review_data["recommendations"] = [item.strip() for item in rec_items]
        
        # Extract comments by finding file sections
        for file_data in files:
            file_path = file_data.get("path", "")
            
            # Look for a section about this file
            file_section_pattern = re.escape(file_path) if '/' in file_path else fr'\b{re.escape(file_path)}\b'
            file_section = re.search(
                f'(?i)#+\s*(?:File|Review)(?:[:\s]+{file_section_pattern}|\s*{file_section_pattern}[:\s]+)(?:\n+(.+?))?(?=\n#+|\Z)',
                text,
                re.DOTALL
            )
            
            if not file_section:
                # Try a more general pattern if exact path not found
                file_name = file_path.split('/')[-1] if '/' in file_path else file_path
                file_section = re.search(
                    f'(?i)#+\s*(?:File|Review)[:\s]+.*?{re.escape(file_name)}.*?(?:\n+(.+?))?(?=\n#+|\Z)',
                    text,
                    re.DOTALL
                )
            
            if file_section and file_section.group(1):
                file_content = file_section.group(1)
                
                # Look for identified issues/comments in this file section
                comment_sections = re.findall(
                    r'(?:[-*]\s*(?:Line\s+(\d+):|Issue|Comment)|\[Line\s+(\d+)\]|At line\s+(\d+):?)\s*([^\n]+(?:\n+(?!\n|[-*]|\[Line)[^\n]+)*)',
                    file_content
                )
                
                for comment_match in comment_sections:
                    line_number = None
                    comment_text = ""
                    
                    # Extract line number from the various possible formats
                    for group in comment_match[:-1]:
                        if group and group.isdigit():
                            line_number = int(group)
                            break
                    
                    # The last group is always the comment text
                    comment_text = comment_match[-1].strip()
                    
                    # Extract code snippet if present
                    code_snippet = None
                    snippet_match = re.search(r'```(?:[a-z]+)?\n(.*?)\n```', comment_text, re.DOTALL)
                    if snippet_match:
                        code_snippet = snippet_match.group(1).strip()
                        # Remove the snippet from the comment text
                        comment_text = comment_text.replace(snippet_match.group(0), "").strip()
                    
                    # Extract severity if mentioned
                    severity = "minor"  # Default
                    severity_match = re.search(r'(?i)(?:Severity|Priority):\s*(critical|major|minor|suggestion)', comment_text)
                    if severity_match:
                        severity = severity_match.group(1).lower()
                    elif "critical" in comment_text.lower():
                        severity = "critical"
                    elif "major" in comment_text.lower():
                        severity = "major"
                    elif "suggestion" in comment_text.lower():
                        severity = "suggestion"
                    
                    # Extract category if mentioned
                    category = "maintainability"  # Default
                    category_patterns = {
                        "security": r'(?i)security',
                        "performance": r'(?i)performance',
                        "functionality": r'(?i)functional|logic|behavior',
                        "maintainability": r'(?i)maintainability|readability',
                        "style": r'(?i)style|formatting',
                        "testing": r'(?i)test'
                    }
                    
                    for cat, pattern in category_patterns.items():
                        if re.search(pattern, comment_text):
                            category = cat
                            break
                    
                    # Extract recommended fix if provided
                    recommended_fix = None
                    fix_match = re.search(r'(?i)(?:Suggested fix|Recommendation|Better approach):\s*(.*?)(?=\n\n|\Z)', comment_text, re.DOTALL)
                    if fix_match:
                        recommended_fix = fix_match.group(1).strip()
                    
                    # Add the comment
                    review_data["comments"].append({
                        "file_path": file_path,
                        "line_number": line_number,
                        "code_snippet": code_snippet,
                        "comment": comment_text,
                        "severity": severity,
                        "category": category,
                        "recommended_fix": recommended_fix
                    })
                
                # If no structured comments were found, look for general bullet points
                if not review_data["comments"]:
                    general_comments = re.findall(r'[-*]\s*([^\n]+)', file_content)
                    for comment in general_comments:
                        review_data["comments"].append({
                            "file_path": file_path,
                            "comment": comment.strip(),
                            "severity": "minor",
                            "category": "maintainability"
                        })
        
        return review_data
    
    def _update_review_statistics(self, review: CodeReview):
        """Update review statistics based on a new review.
        
        Args:
            review: The code review to include in statistics
        """
        # Initialize stats for this repository if not exists
        if review.repository not in self.review_statistics:
            self.review_statistics[review.repository] = {
                "total_reviews": 0,
                "total_comments": 0,
                "comments_by_severity": {
                    "critical": 0,
                    "major": 0,
                    "minor": 0,
                    "suggestion": 0
                },
                "comments_by_category": {},
                "quality_scores": [],
                "average_quality": 0.0,
                "trend": []
            }
        
        stats = self.review_statistics[review.repository]
        
        # Update basic counts
        stats["total_reviews"] += 1
        stats["total_comments"] += len(review.comments)
        
        # Update comments by severity
        for comment in review.comments:
            stats["comments_by_severity"][comment.severity] += 1
            
            # Update comments by category
            if comment.category not in stats["comments_by_category"]:
                stats["comments_by_category"][comment.category] = 0
            stats["comments_by_category"][comment.category] += 1
        
        # Update quality scores
        stats["quality_scores"].append(review.quality_score)
        stats["average_quality"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
        
        # Update trend
        stats["trend"].append({
            "date": review.created_at,
            "quality_score": review.quality_score,
            "comments_count": len(review.comments)
        })
        
        # Store in shared memory if available
        if self.shared_memory:
            self.shared_memory.store(
                key=f"review_statistics_{review.repository}",
                value=stats,
                category="review_statistics"
            )
    
    async def suggest_improvements(
        self, 
        repository: str,
        code_file: Dict[str, Any],
        focus_areas: Optional[List[str]] = None
    ) -> TaskResult:
        """Suggest improvements for a specific code file.
        
        Args:
            repository: Repository identifier
            code_file: File content and metadata
            focus_areas: Optional specific areas to focus on
            
        Returns:
            TaskResult containing the improvement suggestions
        """
        # Create a task for suggesting improvements
        task = Task(
            task_id=f"suggest_improvements_{code_file.get('path', '').replace('/', '_')}",
            description=f"Suggest improvements for {code_file.get('path', '')}",
            agent_type=str(AgentRole.CODE_REVIEWER),
            requirements={
                "repository": repository,
                "code_file": code_file,
                "focus_areas": focus_areas or ["readability", "maintainability", "performance", "security"],
                "languages": self.preferred_languages
            },
            context=TaskContext(
                notes=(
                    f"Analyze the provided code file and suggest specific improvements. "
                    f"Focus on {', '.join(focus_areas or ['readability', 'maintainability', 'performance', 'security'])}. "
                    f"Provide concrete examples of improved code where possible."
                )
            ),
            expected_output=(
                "Specific improvement suggestions with code examples, explanations of the "
                "benefits, and priority levels for each suggestion."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the suggestions
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                improvements_id = f"improvements_{repository}_{code_file.get('path', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                self.shared_memory.store(
                    key=improvements_id,
                    value={
                        "repository": repository,
                        "file_path": code_file.get("path", ""),
                        "focus_areas": focus_areas or ["readability", "maintainability", "performance", "security"],
                        "suggestions": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="improvement_suggestions"
                )
                
                # Update result metadata
                result.metadata = {
                    "improvements_id": improvements_id,
                    "repository": repository,
                    "file_path": code_file.get("path", "")
                }
            
            logger.info(f"Created improvement suggestions for {code_file.get('path', '')}")
        
        return result
    
    def get_code_review(self, review_id: str) -> Optional[CodeReview]:
        """Get a specific code review.
        
        Args:
            review_id: ID of the code review to retrieve
            
        Returns:
            CodeReview if found, None otherwise
        """
        # Check local storage
        if review_id in self.code_reviews:
            return self.code_reviews[review_id]
        
        # Check shared memory if available
        if self.shared_memory:
            review_data = self.shared_memory.retrieve(
                key=f"code_review_{review_id}",
                category="code_reviews"
            )
            if review_data:
                review = CodeReview(**review_data)
                # Cache locally
                self.code_reviews[review_id] = review
                return review
        
        return None
    
    def get_review_statistics(self, repository: str) -> Optional[Dict[str, Any]]:
        """Get review statistics for a repository.
        
        Args:
            repository: Repository identifier
            
        Returns:
            Review statistics if found, None otherwise
        """
        # Check local storage
        if repository in self.review_statistics:
            return self.review_statistics[repository]
        
        # Check shared memory if available
        if self.shared_memory:
            stats_data = self.shared_memory.retrieve(
                key=f"review_statistics_{repository}",
                category="review_statistics"
            )
            if stats_data:
                # Cache locally
                self.review_statistics[repository] = stats_data
                return stats_data
        
        return None


class TestDeveloper(BaseAgent):
    """Agent specialized in creating and implementing tests."""
    
    def __init__(
        self, 
        name: str = "Test Developer",
        preferred_frameworks: Dict[str, List[str]] = None,
        **kwargs
    ):
        """Initialize the Test Developer agent.
        
        Args:
            name: Human-readable name for this agent
            preferred_frameworks: Dictionary mapping languages to test frameworks
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.TEST_DEVELOPER, 
            **kwargs
        )
        self.preferred_frameworks = preferred_frameworks or {
            "Python": ["pytest", "unittest"],
            "JavaScript": ["jest", "mocha"],
            "TypeScript": ["jest", "mocha"],
            "Java": ["JUnit", "TestNG"],
            "C#": ["NUnit", "xUnit"]
        }
        
        # Track test suites
        self.test_suites: Dict[str, TestSuite] = {}
        
        # Track test coverage
        self.test_coverage: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Test Developer Agent initialized with frameworks for {', '.join(self.preferred_frameworks.keys())} languages")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Test Developer."""
        return (
            f"You are {self.name}, a Test Developer specialized in creating and implementing "
            f"tests for software applications. "
            f"Your responsibilities include:\n"
            f"1. Designing test cases and test suites based on requirements\n"
            f"2. Implementing automated tests using appropriate frameworks\n"
            f"3. Ensuring comprehensive test coverage\n"
            f"4. Writing tests that are maintainable and reliable\n"
            f"5. Creating test documentation and reporting\n\n"
            f"When designing tests, focus on both common and edge cases. Ensure tests are "
            f"isolated, deterministic, and meaningful. Write clear test names and descriptions "
            f"that explain what is being tested and why. Consider performance, usability, "
            f"and security aspects in your testing strategy."
        )
    
    async def create_test_suite(
        self, 
        component_name: str,
        description: str,
        requirements: List[Dict[str, Any]],
        language: str,
        test_level: str = "unit"  # "unit", "integration", "e2e"
    ) -> TaskResult:
        """Create a test suite for a component based on requirements.
        
        Args:
            component_name: Name of the component to test
            description: Description of the component's functionality
            requirements: List of requirements to test against
            language: Programming language for test implementation
            test_level: Level of testing
            
        Returns:
            TaskResult containing the test suite
        """
        # Determine appropriate test frameworks
        frameworks = self.preferred_frameworks.get(language, [])
        if not frameworks:
            frameworks = ["appropriate testing framework for " + language]
        
        # Create a task for designing the test suite
        task = Task(
            task_id=f"create_test_suite_{component_name.lower().replace(' ', '_')}_{test_level}",
            description=f"Create {test_level} test suite for {component_name}",
            agent_type=str(AgentRole.TEST_DEVELOPER),
            requirements={
                "component_name": component_name,
                "description": description,
                "requirements": requirements,
                "language": language,
                "test_level": test_level,
                "frameworks": frameworks
            },
            context=TaskContext(
                notes=(
                    f"Design a comprehensive {test_level} test suite for {component_name} in {language}. "
                    f"Create test cases that verify all requirements are met, including edge cases "
                    f"and error conditions. Consider using {', '.join(frameworks)} for implementation."
                )
            ),
            expected_output=(
                "A complete test suite specification with detailed test cases, "
                "including descriptions, steps, expected results, and implementation guidance."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the test suite
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the test suite from the result
                suite_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    suite_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            suite_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not suite_data:
                    logger.warning(f"Could not parse test suite as JSON. Attempting to extract from text.")
                    suite_data = self._extract_test_suite_from_text(result.result, component_name, description, test_level)
                
                # Create test cases
                test_cases = []
                for case_data in suite_data.get("test_cases", []):
                    test_case = TestCase(
                        name=case_data.get("name", "Unnamed Test"),
                        description=case_data.get("description", ""),
                        type=case_data.get("type", test_level),
                        preconditions=case_data.get("preconditions"),
                        steps=case_data.get("steps", []),
                        expected_results=case_data.get("expected_results", []),
                        associated_requirements=case_data.get("associated_requirements"),
                        automated=case_data.get("automated", False),
                        test_code=case_data.get("test_code")
                    )
                    test_cases.append(test_case)
                
                # Create the test suite
                test_suite = TestSuite(
                    name=f"{component_name} {test_level.capitalize()} Tests",
                    description=f"{test_level.capitalize()} test suite for {component_name}: {description}",
                    test_cases=test_cases,
                    tags=[component_name, language, test_level] + frameworks
                )
                
                # Store the test suite
                self.test_suites[test_suite.id] = test_suite
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"test_suite_{test_suite.id}",
                        value=test_suite.dict(),
                        category="test_suites"
                    )
                
                logger.info(f"Created {test_level} test suite for {component_name} with {len(test_cases)} test cases")
                
                # Return the test suite as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=test_suite.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"test_suite_id": test_suite.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing test suite: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_test_suite_from_text(
        self, 
        text: str, 
        component_name: str, 
        description: str,
        test_level: str
    ) -> Dict[str, Any]:
        """Extract structured test suite data from unstructured text.
        
        Args:
            text: The text to extract from
            component_name: The name of the component being tested
            description: The description of the component
            test_level: The level of testing
            
        Returns:
            Structured test suite data
        """
        suite_data = {
            "name": f"{component_name} {test_level.capitalize()} Tests",
            "description": f"{test_level.capitalize()} test suite for {component_name}: {description}",
            "test_cases": []
        }
        
        # Look for individual test cases
        test_case_sections = re.findall(
            r'(?i)#+\s*(?:Test Case|Test)(?:[:\s]+([^\n]+))?\s*(?:\n+(.+?))?(?=\n+#+\s*(?:Test Case|Test)|\n#+|\Z)',
            text,
            re.DOTALL
        )
        
        for title_match, content in test_case_sections:
            test_case = {
                "name": title_match.strip() if title_match else f"Test for {component_name}",
                "description": "",
                "type": test_level,
                "steps": [],
                "expected_results": [],
                "automated": False
            }
            
            # Extract description
            desc_match = re.search(r'(?i)(?:Description|Purpose|Overview):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
            if desc_match:
                test_case["description"] = desc_match.group(1).strip()
            
            # Extract preconditions
            precond_match = re.search(
                r'(?i)(?:Preconditions|Prerequisites|Setup):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if precond_match:
                precond_text = precond_match.group(1)
                
                # Extract from bullet points
                precond_items = re.findall(r'[-*]\s*([^\n]+)', precond_text)
                test_case["preconditions"] = [item.strip() for item in precond_items]
            
            # Extract steps
            steps_match = re.search(
                r'(?i)(?:Steps|Procedure|Test Steps):\s*\n+((?:(?:\d+\.|\[-\*])\s*[^\n]+\n*)+)',
                content
            )
            if steps_match:
                steps_text = steps_match.group(1)
                
                # Extract from numbered or bullet points
                steps_items = re.findall(r'(?:\d+\.|[-*])\s*([^\n]+)', steps_text)
                test_case["steps"] = [item.strip() for item in steps_items]
            
            # Extract expected results
            expected_match = re.search(
                r'(?i)(?:Expected Results|Expected Outcome|Assertions):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if expected_match:
                expected_text = expected_match.group(1)
                
                # Extract from bullet points
                expected_items = re.findall(r'[-*]\s*([^\n]+)', expected_text)
                test_case["expected_results"] = [item.strip() for item in expected_items]
            
            # Extract associated requirements
            req_match = re.search(
                r'(?i)(?:Requirements|Associated Requirements|Verifies):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                content
            )
            if req_match:
                req_text = req_match.group(1)
                
                # Extract from bullet points
                req_items = re.findall(r'[-*]\s*([^\n]+)', req_text)
                test_case["associated_requirements"] = [item.strip() for item in req_items]
            
            # Extract automation status
            auto_match = re.search(r'(?i)(?:Automation|Automated):\s*([^\n]+)', content)
            if auto_match:
                auto_text = auto_match.group(1).lower()
                test_case["automated"] = "yes" in auto_text or "true" in auto_text
            
            # Extract test code if present
            code_match = re.search(r'```(?:[a-z]+)?\n(.*?)\n```', content, re.DOTALL)
            if code_match:
                test_case["test_code"] = code_match.group(1).strip()
                test_case["automated"] = True  # If code is provided, it's automated
            
            suite_data["test_cases"].append(test_case)
        
        return suite_data
    
    async def implement_tests(
        self, 
        test_suite_id: str,
        code_context: Dict[str, Any],
        framework: Optional[str] = None
    ) -> TaskResult:
        """Implement automated tests based on a test suite.
        
        Args:
            test_suite_id: ID of the test suite to implement
            code_context: Context about the code being tested
            framework: Optional specific test framework to use
            
        Returns:
            TaskResult containing the implemented tests
        """
        # Check if test suite exists
        if test_suite_id not in self.test_suites:
            # Try to load from shared memory if available
            if self.shared_memory:
                suite_data = self.shared_memory.retrieve(
                    key=f"test_suite_{test_suite_id}",
                    category="test_suites"
                )
                if suite_data:
                    self.test_suites[test_suite_id] = TestSuite(**suite_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"implement_tests_{test_suite_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Test suite with ID {test_suite_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"implement_tests_{test_suite_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Test suite with ID {test_suite_id} not found"
                )
        
        # Get the test suite
        test_suite = self.test_suites[test_suite_id]
        
        # Determine language from tags
        language = next((tag for tag in test_suite.tags if tag in self.preferred_frameworks), None)
        
        # Determine framework
        if not framework and language:
            frameworks = self.preferred_frameworks.get(language, [])
            framework = frameworks[0] if frameworks else None
        
        # Create a task for implementing tests
        task = Task(
            task_id=f"implement_tests_{test_suite_id}",
            description=f"Implement tests for {test_suite.name}",
            agent_type=str(AgentRole.TEST_DEVELOPER),
            requirements={
                "test_suite_id": test_suite_id,
                "test_suite": test_suite.dict(),
                "code_context": code_context,
                "language": language,
                "framework": framework
            },
            context=TaskContext(
                notes=(
                    f"Implement automated tests for {test_suite.name} using {framework if framework else 'an appropriate framework'}. "
                    f"The implementation should follow test best practices and include all the test cases "
                    f"defined in the test suite. Use the provided code context to understand the code under test."
                )
            ),
            expected_output=(
                "Complete test implementation code that can be executed to verify the functionality "
                "of the component under test, with appropriate setup, teardown, and assertions."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, update the test suite with implemented tests
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract test implementations from the result
                implementation_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    implementation_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # If the result is code rather than JSON, create a simple structure
                    if "```" in result.result or "class " in result.result or "def test_" in result.result:
                        implementation_data = {
                            "test_code": result.result,
                            "framework": framework,
                            "language": language
                        }
                    else:
                        # Second attempt: Extract JSON from markdown
                        json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                        if json_match:
                            try:
                                implementation_data = json.loads(json_match.group(1))
                            except json.JSONDecodeError:
                                pass
                
                # If we couldn't parse any structured data, use the result as is
                if not implementation_data:
                    implementation_data = {
                        "test_code": result.result,
                        "framework": framework,
                        "language": language
                    }
                
                # Update test cases with implementations
                updated_test_cases = []
                for test_case in test_suite.test_cases:
                    test_case_data = test_case.dict()
                    
                    # Look for specific implementation for this test case
                    if "test_implementations" in implementation_data:
                        for impl in implementation_data["test_implementations"]:
                            if impl.get("test_case_id") == test_case.id or impl.get("test_case_name") == test_case.name:
                                test_case_data["test_code"] = impl.get("test_code", "")
                                test_case_data["automated"] = True
                                break
                    
                    updated_test_cases.append(TestCase(**test_case_data))
                
                # Update the test suite
                test_suite.test_cases = updated_test_cases
                test_suite.updated_at = datetime.now().isoformat()
                
                # Store the updated test suite
                self.test_suites[test_suite_id] = test_suite
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"test_suite_{test_suite_id}",
                        value=test_suite.dict(),
                        category="test_suites"
                    )
                    
                    # Also store the implementation separately
                    self.shared_memory.store(
                        key=f"test_implementation_{test_suite_id}",
                        value={
                            "test_suite_id": test_suite_id,
                            "test_suite_name": test_suite.name,
                            "implementation": implementation_data,
                            "timestamp": datetime.now().isoformat()
                        },
                        category="test_implementations"
                    )
                
                logger.info(f"Implemented tests for {test_suite.name} using {framework}")
                
                # Return the implementation
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=implementation_data,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={
                        "test_suite_id": test_suite_id,
                        "test_suite_name": test_suite.name,
                        "framework": framework
                    }
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing test implementation: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    async def analyze_test_coverage(
        self, 
        component_name: str,
        code_files: List[Dict[str, Any]],
        test_files: List[Dict[str, Any]],
        coverage_requirements: Optional[Dict[str, float]] = None
    ) -> TaskResult:
        """Analyze test coverage for a component.
        
        Args:
            component_name: Name of the component
            code_files: List of code files
            test_files: List of test files
            coverage_requirements: Optional coverage requirements
            
        Returns:
            TaskResult containing the coverage analysis
        """
        # Create a task for analyzing test coverage
        task = Task(
            task_id=f"analyze_coverage_{component_name.lower().replace(' ', '_')}",
            description=f"Analyze test coverage for {component_name}",
            agent_type=str(AgentRole.TEST_DEVELOPER),
            requirements={
                "component_name": component_name,
                "code_files": code_files,
                "test_files": test_files,
                "coverage_requirements": coverage_requirements or {"line": 80.0, "branch": 70.0, "function": 90.0}
            },
            context=TaskContext(
                notes=(
                    f"Analyze the test coverage for {component_name} by examining the provided "
                    f"code files and test files. Identify areas with insufficient coverage and "
                    f"suggest additional tests to improve coverage."
                )
            ),
            expected_output=(
                "A comprehensive test coverage analysis including coverage metrics, "
                "uncovered areas, and recommended additional tests."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the coverage analysis
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the coverage analysis from the result
                coverage_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    coverage_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            coverage_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not coverage_data:
                    coverage_data = self._extract_coverage_from_text(result.result, component_name)
                
                # Store the coverage analysis
                self.test_coverage[component_name] = {
                    "component_name": component_name,
                    "coverage_data": coverage_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"test_coverage_{component_name.lower().replace(' ', '_')}",
                        value=self.test_coverage[component_name],
                        category="test_coverage"
                    )
                
                logger.info(f"Analyzed test coverage for {component_name}")
                
                # Return the coverage analysis
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=coverage_data,
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"component_name": component_name}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing coverage analysis: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_coverage_from_text(self, text: str, component_name: str) -> Dict[str, Any]:
        """Extract structured coverage data from unstructured text.
        
        Args:
            text: The text to extract from
            component_name: The name of the component
            
        Returns:
            Structured coverage data
        """
        coverage_data = {
            "component_name": component_name,
            "overall_coverage": None,
            "coverage_by_type": {},
            "uncovered_code": [],
            "recommendations": []
        }
        
        # Extract overall coverage
        overall_match = re.search(r'(?i)(?:Overall Coverage|Total Coverage):\s*(\d+(?:\.\d+)?)%', text)
        if overall_match:
            try:
                coverage_data["overall_coverage"] = float(overall_match.group(1))
            except ValueError:
                pass
        
        # Extract coverage by type
        coverage_types = ["line", "branch", "function", "statement"]
        for ctype in coverage_types:
            type_match = re.search(rf'(?i){ctype}\s+coverage:\s*(\d+(?:\.\d+)?)%', text)
            if type_match:
                try:
                    coverage_data["coverage_by_type"][ctype] = float(type_match.group(1))
                except ValueError:
                    pass
        
        # Extract uncovered code
        uncovered_section = re.search(
            r'(?i)#+\s*(?:Uncovered Code|Missing Coverage|Coverage Gaps)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if uncovered_section and uncovered_section.group(1):
            uncovered_text = uncovered_section.group(1)
            
            # Extract file sections
            file_sections = re.findall(
                r'(?i)(?:\*\*|[-*])\s*([^:\n]+)(?::\s*|\n+)((?:[-*]\s*[^\n]+\n*)+)',
                uncovered_text
            )
            
            for file_name, details in file_sections:
                # Extract uncovered items
                items = re.findall(r'[-*]\s*([^\n]+)', details)
                
                coverage_data["uncovered_code"].append({
                    "file": file_name.strip(),
                    "items": [item.strip() for item in items]
                })
            
            # If no structured sections found, try bullet points
            if not coverage_data["uncovered_code"]:
                items = re.findall(r'[-*]\s*([^\n]+)', uncovered_text)
                if items:
                    coverage_data["uncovered_code"].append({
                        "file": "general",
                        "items": [item.strip() for item in items]
                    })
        
        # Extract recommendations
        recommendations_section = re.search(
            r'(?i)#+\s*(?:Recommendations|Suggested Tests|Improvements)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if recommendations_section and recommendations_section.group(1):
            recommendations_text = recommendations_section.group(1)
            
            # Extract recommendations
            recommendations = re.findall(r'[-*]\s*([^\n]+)', recommendations_text)
            coverage_data["recommendations"] = [rec.strip() for rec in recommendations]
        
        return coverage_data
    
    def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Get a specific test suite.
        
        Args:
            suite_id: ID of the test suite to retrieve
            
        Returns:
            TestSuite if found, None otherwise
        """
        # Check local storage
        if suite_id in self.test_suites:
            return self.test_suites[suite_id]
        
        # Check shared memory if available
        if self.shared_memory:
            suite_data = self.shared_memory.retrieve(
                key=f"test_suite_{suite_id}",
                category="test_suites"
            )
            if suite_data:
                suite = TestSuite(**suite_data)
                # Cache locally
                self.test_suites[suite_id] = suite
                return suite
        
        return None
    
    def get_test_coverage(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get test coverage for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Test coverage if found, None otherwise
        """
        # Check local storage
        if component_name in self.test_coverage:
            return self.test_coverage[component_name]
        
        # Check shared memory if available
        if self.shared_memory:
            coverage_data = self.shared_memory.retrieve(
                key=f"test_coverage_{component_name.lower().replace(' ', '_')}",
                category="test_coverage"
            )
            if coverage_data:
                # Cache locally
                self.test_coverage[component_name] = coverage_data
                return coverage_data
        
        return None


class UXTester(BaseAgent):
    """Agent specialized in user experience testing."""
    
    def __init__(
        self, 
        name: str = "UX Tester",
        user_personas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Initialize the UX Tester agent.
        
        Args:
            name: Human-readable name for this agent
            user_personas: Optional list of user personas for testing
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.UX_TESTER, 
            **kwargs
        )
        self.user_personas = user_personas or self._default_user_personas()
        
        # Track usability tests
        self.usability_tests: Dict[str, UsabilityTest] = {}
        
        # Track usability test results
        self.usability_test_results: Dict[str, UsabilityTestResult] = {}
        
        logger.info(f"UX Tester Agent initialized with {len(self.user_personas)} user personas")
    
    def _default_user_personas(self) -> List[Dict[str, Any]]:
        """Generate default user personas.
        
        Returns:
            Default user personas
        """
        return [
            {
                "name": "Tech-Savvy User",
                "description": "Highly technical user comfortable with complex interfaces",
                "characteristics": [
                    "Familiar with technical terminology",
                    "Prefers efficiency over hand-holding",
                    "Expects advanced features and customization",
                    "Values keyboard shortcuts and power-user features"
                ],
                "goals": [
                    "Complete tasks quickly",
                    "Access advanced functionality",
                    "Customize the experience to their preferences"
                ]
            },
            {
                "name": "Novice User",
                "description": "New user with limited technical knowledge",
                "characteristics": [
                    "Unfamiliar with technical terminology",
                    "Prefers clear guidance and explanations",
                    "May be uncomfortable with complex interfaces",
                    "Values intuitive design and helpful messages"
                ],
                "goals": [
                    "Understand how to use the application",
                    "Complete basic tasks without confusion",
                    "Learn without being overwhelmed"
                ]
            },
            {
                "name": "Busy Professional",
                "description": "Professional user with limited time",
                "characteristics": [
                    "Values efficiency and time-saving features",
                    "May be interrupted frequently",
                    "Moderate technical knowledge",
                    "Prefers streamlined workflows"
                ],
                "goals": [
                    "Complete tasks with minimal steps",
                    "Find information quickly",
                    "Resume work easily after interruptions"
                ]
            },
            {
                "name": "Accessibility-Focused User",
                "description": "User with accessibility needs",
                "characteristics": [
                    "May use assistive technologies",
                    "Requires proper keyboard navigation",
                    "Needs clear visual hierarchy and contrast",
                    "May have difficulty with fine motor control"
                ],
                "goals": [
                    "Access all features independently",
                    "Navigate efficiently with assistive technologies",
                    "Understand content and feedback clearly"
                ]
            }
        ]
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the UX Tester."""
        return (
            f"You are {self.name}, a UX Tester specialized in evaluating the user experience "
            f"of software applications. "
            f"Your responsibilities include:\n"
            f"1. Designing usability tests with clear tasks and metrics\n"
            f"2. Evaluating interfaces from different user perspectives\n"
            f"3. Identifying usability issues and barriers\n"
            f"4. Making recommendations for UX improvements\n"
            f"5. Ensuring accessibility and inclusive design\n\n"
            f"When evaluating interfaces, think from the perspective of different user personas, "
            f"each with their own goals, technical abilities, and needs. Focus on both objective "
            f"metrics (task completion, error rates) and subjective aspects (satisfaction, comprehension). "
            f"Provide actionable feedback that can be implemented to improve the user experience."
        )
    
    async def create_usability_test(
        self, 
        application_name: str,
        interface_description: str,
        key_user_flows: List[str],
        persona_name: Optional[str] = None
    ) -> TaskResult:
        """Create a usability test for an application interface.
        
        Args:
            application_name: Name of the application
            interface_description: Description of the interface
            key_user_flows: List of key user flows to test
            persona_name: Optional specific user persona to focus on
            
        Returns:
            TaskResult containing the usability test
        """
        # Get the user persona to test with
        persona = None
        if persona_name:
            for p in self.user_personas:
                if p["name"].lower() == persona_name.lower():
                    persona = p
                    break
        
        if not persona:
            # Use the first persona as default
            persona = self.user_personas[0] if self.user_personas else {
                "name": "General User",
                "description": "Average user of the application",
                "characteristics": ["Basic technical knowledge", "Goal-oriented"],
                "goals": ["Complete tasks efficiently", "Find information easily"]
            }
        
        # Create a task for designing the usability test
        task = Task(
            task_id=f"create_usability_test_{application_name.lower().replace(' ', '_')}",
            description=f"Create usability test for {application_name}",
            agent_type=str(AgentRole.UX_TESTER),
            requirements={
                "application_name": application_name,
                "interface_description": interface_description,
                "key_user_flows": key_user_flows,
                "persona": persona
            },
            context=TaskContext(
                notes=(
                    f"Design a usability test for {application_name} focused on the provided "
                    f"user flows. The test should be from the perspective of the '{persona['name']}' "
                    f"user persona and include clear tasks, success criteria, and metrics."
                )
            ),
            expected_output=(
                "A comprehensive usability test design including specific tasks, "
                "success criteria, metrics, and evaluation procedures."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the usability test
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the usability test from the result
                test_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    test_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            test_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not test_data:
                    logger.warning(f"Could not parse usability test as JSON. Attempting to extract from text.")
                    test_data = self._extract_usability_test_from_text(result.result, application_name, interface_description, key_user_flows, persona)
                
                # Create the usability test
                usability_test = UsabilityTest(
                    name=f"{application_name} Usability Test for {persona['name']}",
                    description=test_data.get("description", f"Usability test for {application_name} interface from the perspective of {persona['name']}"),
                    user_persona=persona,
                    tasks=test_data.get("tasks", []),
                    metrics=test_data.get("metrics", []),
                    success_criteria=test_data.get("success_criteria", [])
                )
                
                # Store the usability test
                self.usability_tests[usability_test.id] = usability_test
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"usability_test_{usability_test.id}",
                        value=usability_test.dict(),
                        category="usability_tests"
                    )
                
                logger.info(f"Created usability test for {application_name} with {len(usability_test.tasks)} tasks")
                
                # Return the usability test as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=usability_test.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"test_id": usability_test.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing usability test: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_usability_test_from_text(
        self, 
        text: str, 
        application_name: str, 
        interface_description: str,
        key_user_flows: List[str],
        persona: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract structured usability test data from unstructured text.
        
        Args:
            text: The text to extract from
            application_name: Name of the application
            interface_description: Description of the interface
            key_user_flows: List of key user flows
            persona: User persona for testing
            
        Returns:
            Structured usability test data
        """
        test_data = {
            "name": f"{application_name} Usability Test for {persona['name']}",
            "description": f"Usability test for {application_name} interface from the perspective of {persona['name']}",
            "tasks": [],
            "metrics": [],
            "success_criteria": []
        }
        
        # Extract description if provided
        desc_match = re.search(r'(?i)#+\s*(?:Introduction|Overview|Description)(?:\n+(.+?))?(?=\n#+|\Z)', text, re.DOTALL)
        if desc_match and desc_match.group(1):
            test_data["description"] = desc_match.group(1).strip()
        
        # Extract tasks
        tasks_section = re.search(
            r'(?i)#+\s*(?:Tasks|Test Tasks|User Tasks)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if tasks_section and tasks_section.group(1):
            tasks_text = tasks_section.group(1)
            
            # Look for individual tasks
            task_sections = re.findall(
                r'(?i)(?:#{1,3}|[-*]|(?:\d+\.))(?:\s*Task\s*\d*:?\s*|\s*)\s*([^\n]+)(?:\n+(.+?))?(?=\n+(?:#{1,3}|[-*]|(?:\d+\.))(?:\s*Task|\s*[^\n]+)|\n#+|\Z)',
                tasks_text,
                re.DOTALL
            )
            
            for title, content in task_sections:
                task = {
                    "name": title.strip(),
                    "description": "",
                    "steps": [],
                    "success_criteria": []
                }
                
                # Extract description
                desc_match = re.search(r'(?i)(?:Description|Objective):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', content)
                if desc_match:
                    task["description"] = desc_match.group(1).strip()
                
                # Extract steps
                steps_match = re.search(
                    r'(?i)(?:Steps|Procedure|Instructions):\s*\n+((?:(?:\d+\.|\[-\*])\s*[^\n]+\n*)+)',
                    content
                )
                if steps_match:
                    steps_text = steps_match.group(1)
                    
                    # Extract from numbered or bullet points
                    steps_items = re.findall(r'(?:\d+\.|[-*])\s*([^\n]+)', steps_text)
                    task["steps"] = [item.strip() for item in steps_items]
                
                # Extract success criteria
                criteria_match = re.search(
                    r'(?i)(?:Success Criteria|Completion Criteria):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                    content
                )
                if criteria_match:
                    criteria_text = criteria_match.group(1)
                    
                    # Extract from bullet points
                    criteria_items = re.findall(r'[-*]\s*([^\n]+)', criteria_text)
                    task["success_criteria"] = [item.strip() for item in criteria_items]
                
                test_data["tasks"].append(task)
            
            # If no structured tasks were found, try extracting basic tasks
            if not test_data["tasks"]:
                tasks_items = re.findall(r'[-*]\s*([^\n]+)', tasks_text)
                for item in tasks_items:
                    test_data["tasks"].append({
                        "name": item.strip(),
                        "steps": []
                    })
        
        # Extract metrics
        metrics_section = re.search(
            r'(?i)#+\s*(?:Metrics|Measurements|Evaluation Metrics)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if metrics_section and metrics_section.group(1):
            metrics_text = metrics_section.group(1)
            
            # Extract metrics from bullet points
            metrics_items = re.findall(r'[-*]\s*([^\n]+)', metrics_text)
            test_data["metrics"] = [item.strip() for item in metrics_items]
        
        # Extract success criteria
        criteria_section = re.search(
            r'(?i)#+\s*(?:Success Criteria|Acceptance Criteria|Overall Success)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if criteria_section and criteria_section.group(1):
            criteria_text = criteria_section.group(1)
            
            # Extract criteria from bullet points
            criteria_items = re.findall(r'[-*]\s*([^\n]+)', criteria_text)
            test_data["success_criteria"] = [item.strip() for item in criteria_items]
        
        return test_data
    
    async def evaluate_usability(
        self, 
        test_id: str,
        interface_screenshots: Optional[List[Dict[str, Any]]] = None,
        interface_description: Optional[str] = None
    ) -> TaskResult:
        """Evaluate an interface using a usability test.
        
        Args:
            test_id: ID of the usability test to use
            interface_screenshots: Optional list of interface screenshots
            interface_description: Optional updated interface description
            
        Returns:
            TaskResult containing the usability evaluation
        """
        # Check if test exists
        if test_id not in self.usability_tests:
            # Try to load from shared memory if available
            if self.shared_memory:
                test_data = self.shared_memory.retrieve(
                    key=f"usability_test_{test_id}",
                    category="usability_tests"
                )
                if test_data:
                    self.usability_tests[test_id] = UsabilityTest(**test_data)
                else:
                    return TaskResult(
                        agent_id=self.state.agent_id,
                        agent_name=self.name,
                        task_id=f"evaluate_usability_{test_id}",
                        result=None,
                        status=TaskStatus.FAILED,
                        execution_time=0.0,
                        error=f"Usability test with ID {test_id} not found"
                    )
            else:
                return TaskResult(
                    agent_id=self.state.agent_id,
                    agent_name=self.name,
                    task_id=f"evaluate_usability_{test_id}",
                    result=None,
                    status=TaskStatus.FAILED,
                    execution_time=0.0,
                    error=f"Usability test with ID {test_id} not found"
                )
        
        # Get the usability test
        usability_test = self.usability_tests[test_id]
        
        # Create a task for evaluating usability
        task = Task(
            task_id=f"evaluate_usability_{test_id}",
            description=f"Evaluate usability for {usability_test.name}",
            agent_type=str(AgentRole.UX_TESTER),
            requirements={
                "test_id": test_id,
                "test": usability_test.dict(),
                "has_screenshots": interface_screenshots is not None,
                "interface_screenshots": interface_screenshots or [],
                "interface_description": interface_description
            },
            context=TaskContext(
                notes=(
                    f"Evaluate the usability of the interface using the {usability_test.name} test. "
                    f"Take the perspective of the {usability_test.user_persona['name']} user persona "
                    f"and assess how well the interface supports the specified tasks."
                    + (f" Use the provided screenshots to evaluate the visual design." if interface_screenshots else "")
                )
            ),
            expected_output=(
                "A comprehensive usability evaluation including results for each task, "
                "identified issues, and recommendations for improvement."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, parse and store the usability evaluation
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the usability evaluation from the result
                eval_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    eval_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            eval_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not eval_data:
                    eval_data = self._extract_usability_eval_from_text(result.result, usability_test)
                
                # Calculate success rate
                success_rate = 0.0
                if "task_results" in eval_data and eval_data["task_results"]:
                    successful_tasks = sum(1 for task in eval_data["task_results"] if task.get("success", False))
                    success_rate = successful_tasks / len(eval_data["task_results"])
                
                # Create the usability test result
                test_result = UsabilityTestResult(
                    test_id=test_id,
                    success_rate=success_rate,
                    task_results=eval_data.get("task_results", []),
                    issues=eval_data.get("issues", []),
                    recommendations=eval_data.get("recommendations", [])
                )
                
                # Store the usability test result
                self.usability_test_results[test_result.id] = test_result
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"usability_result_{test_result.id}",
                        value=test_result.dict(),
                        category="usability_results"
                    )
                
                logger.info(f"Evaluated usability for {usability_test.name} with {len(test_result.issues)} issues identified")
                
                # Return the usability test result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=test_result.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"result_id": test_result.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing usability evaluation: {str(e)}")
                # Return the original result if processing failed
                return result
        
        return result
    
    def _extract_usability_eval_from_text(
        self, 
        text: str, 
        usability_test: UsabilityTest
    ) -> Dict[str, Any]:
        """Extract structured usability evaluation data from unstructured text.
        
        Args:
            text: The text to extract from
            usability_test: The usability test being evaluated
            
        Returns:
            Structured usability evaluation data
        """
        eval_data = {
            "task_results": [],
            "issues": [],
            "recommendations": []
        }
        
        # Extract task results
        for task in usability_test.tasks:
            task_name = task["name"]
            
            # Look for a section about this task
            task_section = re.search(
                f'(?i)#+\s*(?:Task|Task Result|Evaluation)(?:[:\s]+{re.escape(task_name)}|\s*{re.escape(task_name)}[:\s]+)(?:\n+(.+?))?(?=\n#+|\Z)',
                text,
                re.DOTALL
            )
            
            task_result = {
                "task_name": task_name,
                "success": False,
                "issues": [],
                "notes": ""
            }
            
            if task_section and task_section.group(1):
                task_content = task_section.group(1)
                
                # Extract success status
                success_match = re.search(r'(?i)(?:Success|Result|Status):\s*([^\n]+)', task_content)
                if success_match:
                    success_text = success_match.group(1).lower()
                    task_result["success"] = "success" in success_text or "pass" in success_text or "yes" in success_text
                
                # Extract issues
                issues_match = re.search(
                    r'(?i)(?:Issues|Problems|Difficulties):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                    task_content
                )
                if issues_match:
                    issues_text = issues_match.group(1)
                    
                    # Extract from bullet points
                    issues_items = re.findall(r'[-*]\s*([^\n]+)', issues_text)
                    task_result["issues"] = [item.strip() for item in issues_items]
                
                # Extract notes
                notes_match = re.search(r'(?i)(?:Notes|Observations):\s*([^\n]+(?:\n+[^#][^\n]+)*?)(?=\n\n|\Z)', task_content)
                if notes_match:
                    task_result["notes"] = notes_match.group(1).strip()
            
            eval_data["task_results"].append(task_result)
        
        # Extract issues
        issues_section = re.search(
            r'(?i)#+\s*(?:Usability Issues|Issues|Problems)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if issues_section and issues_section.group(1):
            issues_text = issues_section.group(1)
            
            # Look for categorized issues
            issue_categories = re.findall(
                r'(?i)(?:#{1,3}|[-*])\s*([^:\n]+):\s*\n+((?:[-*]\s*[^\n]+\n*)+)',
                issues_text
            )
            
            for category, items in issue_categories:
                # Extract issues from bullet points
                issue_items = re.findall(r'[-*]\s*([^\n]+)', items)
                
                for item in issue_items:
                    # Try to extract severity if present
                    severity_match = re.search(r'(?i)\((?:severity|priority):\s*([^)]+)\)', item)
                    severity = severity_match.group(1).strip().lower() if severity_match else "medium"
                    
                    # Remove severity tag from description
                    description = re.sub(r'(?i)\((?:severity|priority):\s*[^)]+\)', '', item).strip()
                    
                    eval_data["issues"].append({
                        "category": category.strip(),
                        "description": description,
                        "severity": severity
                    })
            
            # If no categorized issues found, look for simple bullet points
            if not eval_data["issues"]:
                issue_items = re.findall(r'[-*]\s*([^\n]+)', issues_text)
                
                for item in issue_items:
                    # Try to extract severity if present
                    severity_match = re.search(r'(?i)\((?:severity|priority):\s*([^)]+)\)', item)
                    severity = severity_match.group(1).strip().lower() if severity_match else "medium"
                    
                    # Remove severity tag from description
                    description = re.sub(r'(?i)\((?:severity|priority):\s*[^)]+\)', '', item).strip()
                    
                    eval_data["issues"].append({
                        "description": description,
                        "severity": severity
                    })
        
        # Extract recommendations
        recommendations_section = re.search(
            r'(?i)#+\s*(?:Recommendations|Improvements|Suggested Changes)(?:\n+(.+?))?(?=\n#+|\Z)',
            text,
            re.DOTALL
        )
        if recommendations_section and recommendations_section.group(1):
            recommendations_text = recommendations_section.group(1)
            
            # Extract from bullet points
            rec_items = re.findall(r'[-*]\s*([^\n]+)', recommendations_text)
            eval_data["recommendations"] = [item.strip() for item in rec_items]
        
        return eval_data
    
    async def create_accessibility_report(
        self, 
        application_name: str,
        interface_description: str,
        wcag_level: str = "AA"  # "A", "AA", or "AAA"
    ) -> TaskResult:
        """Create an accessibility compliance report.
        
        Args:
            application_name: Name of the application
            interface_description: Description of the interface
            wcag_level: WCAG compliance level to evaluate against
            
        Returns:
            TaskResult containing the accessibility report
        """
        # Create a task for creating an accessibility report
        task = Task(
            task_id=f"accessibility_report_{application_name.lower().replace(' ', '_')}",
            description=f"Create accessibility report for {application_name}",
            agent_type=str(AgentRole.UX_TESTER),
            requirements={
                "application_name": application_name,
                "interface_description": interface_description,
                "wcag_level": wcag_level
            },
            context=TaskContext(
                notes=(
                    f"Evaluate the accessibility of {application_name} based on the interface description. "
                    f"Check compliance with WCAG {wcag_level} standards and identify accessibility issues "
                    f"and barriers for users with disabilities."
                )
            ),
            expected_output=(
                "A comprehensive accessibility report including compliance status, "
                "identified issues, and recommendations for improvement."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, store the accessibility report
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                report_id = f"accessibility_report_{application_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                self.shared_memory.store(
                    key=report_id,
                    value={
                        "application_name": application_name,
                        "wcag_level": wcag_level,
                        "report": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="accessibility_reports"
                )
                
                # Update result metadata
                result.metadata = {
                    "report_id": report_id,
                    "application_name": application_name,
                    "wcag_level": wcag_level
                }
            
            logger.info(f"Created accessibility report for {application_name} against WCAG {wcag_level}")
        
        return result
    
    def get_usability_test(self, test_id: str) -> Optional[UsabilityTest]:
        """Get a specific usability test.
        
        Args:
            test_id: ID of the usability test to retrieve
            
        Returns:
            UsabilityTest if found, None otherwise
        """
        # Check local storage
        if test_id in self.usability_tests:
            return self.usability_tests[test_id]
        
        # Check shared memory if available
        if self.shared_memory:
            test_data = self.shared_memory.retrieve(
                key=f"usability_test_{test_id}",
                category="usability_tests"
            )
            if test_data:
                test = UsabilityTest(**test_data)
                # Cache locally
                self.usability_tests[test_id] = test
                return test
        
        return None
    
    def get_usability_result(self, result_id: str) -> Optional[UsabilityTestResult]:
        """Get a specific usability test result.
        
        Args:
            result_id: ID of the usability test result to retrieve
            
        Returns:
            UsabilityTestResult if found, None otherwise
        """
        # Check local storage
        if result_id in self.usability_test_results:
            return self.usability_test_results[result_id]
        
        # Check shared memory if available
        if self.shared_memory:
            result_data = self.shared_memory.retrieve(
                key=f"usability_result_{result_id}",
                category="usability_results"
            )
            if result_data:
                result = UsabilityTestResult(**result_data)
                # Cache locally
                self.usability_test_results[result_id] = result
                return result
        
        return None

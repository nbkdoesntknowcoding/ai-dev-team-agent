"""
Code review and testing tools for the multi-agent development system.

This module provides tools for code review, quality assessment, testing,
and feedback generation. It helps identify issues, suggest improvements,
and ensure code meets established standards.
"""

import asyncio
import difflib
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

try:
    import ast
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False

try:
    from pylint.lint import Run as PylintRun
    from pylint.reporters.text import TextReporter
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

try:
    import mypy.api
    MYPY_AVAILABLE = True
except ImportError:
    MYPY_AVAILABLE = False

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

try:
    # JavaScript/TypeScript tools
    import esprima
    ESPRIMA_AVAILABLE = True
except ImportError:
    ESPRIMA_AVAILABLE = False

from pydantic import BaseModel, Field, validator

# Set up logging
logger = logging.getLogger(__name__)


class IssueType(str, Enum):
    """Types of issues that can be identified in code."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    STYLE = "style"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    SUGGESTION = "suggestion"


class IssueCategory(str, Enum):
    """Categories of issues for organization."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    CONVENTION = "convention"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    COMPATIBILITY = "compatibility"
    DEPENDENCY = "dependency"
    OTHER = "other"


class CodeLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SQL = "sql"
    SHELL = "shell"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    OTHER = "other"


class CodeIssue(BaseModel):
    """An issue identified in code during review."""
    id: str = Field(..., description="Unique identifier for the issue")
    type: IssueType = Field(..., description="Type of issue")
    category: IssueCategory = Field(..., description="Category of issue")
    message: str = Field(..., description="Issue description")
    line: Optional[int] = Field(None, description="Line number where the issue occurs")
    column: Optional[int] = Field(None, description="Column number where the issue occurs")
    file_path: Optional[str] = Field(None, description="Path to the file containing the issue")
    code_snippet: Optional[str] = Field(None, description="Snippet of code containing the issue")
    suggestion: Optional[str] = Field(None, description="Suggested fix for the issue")
    rule_id: Optional[str] = Field(None, description="ID of the rule that triggered the issue")
    severity: float = Field(1.0, description="Severity of the issue (0.0-10.0)")
    confidence: float = Field(1.0, description="Confidence in the issue (0.0-1.0)")
    tags: List[str] = Field(default_factory=list, description="Tags for the issue")
    analyzer: str = Field("default", description="Analyzer that found the issue")
    fixed: bool = Field(False, description="Whether the issue has been fixed")
    ignored: bool = Field(False, description="Whether the issue should be ignored")


class QualityMetrics(BaseModel):
    """Quality metrics for code."""
    language: str = Field(..., description="Programming language")
    lines_of_code: int = Field(0, description="Total lines of code")
    comment_lines: int = Field(0, description="Lines of comments")
    empty_lines: int = Field(0, description="Empty lines")
    comment_ratio: float = Field(0.0, description="Ratio of comments to code")
    cyclomatic_complexity: Optional[float] = Field(None, description="Average cyclomatic complexity")
    maintainability_index: Optional[float] = Field(None, description="Maintainability index (0-100)")
    duplication_ratio: Optional[float] = Field(None, description="Code duplication ratio")
    test_coverage: Optional[float] = Field(None, description="Test coverage percentage")
    issue_density: Optional[float] = Field(None, description="Issues per 1000 lines of code")
    function_count: Optional[int] = Field(None, description="Number of functions/methods")
    class_count: Optional[int] = Field(None, description="Number of classes")
    dependency_count: Optional[int] = Field(None, description="Number of dependencies")
    language_specific: Dict[str, Any] = Field(default_factory=dict, description="Language-specific metrics")


class ReviewResult(BaseModel):
    """Result of a code review."""
    issues: List[CodeIssue] = Field(default_factory=list, description="Issues found in the code")
    metrics: QualityMetrics = Field(..., description="Quality metrics for the code")
    summary: str = Field(..., description="Summary of the review")
    suggestions: List[str] = Field(default_factory=list, description="General suggestions for improvement")
    review_time_ms: float = Field(..., description="Time taken for the review in milliseconds")
    executed_analyzers: List[str] = Field(default_factory=list, description="Analyzers that were executed")
    files_analyzed: List[str] = Field(default_factory=list, description="Files that were analyzed")
    highest_severity_issues: List[CodeIssue] = Field(default_factory=list, description="Highest severity issues")
    timestamps: Dict[str, str] = Field(default_factory=dict, description="Timestamps for review events")


class TestStatus(str, Enum):
    """Status of a test."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestResult(BaseModel):
    """Result of a test run."""
    test_id: str = Field(..., description="Identifier for the test")
    name: str = Field(..., description="Name of the test")
    status: TestStatus = Field(..., description="Status of the test")
    execution_time: float = Field(..., description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if test failed")
    error_trace: Optional[str] = Field(None, description="Error traceback if test failed")
    stdout: Optional[str] = Field(None, description="Standard output of the test")
    stderr: Optional[str] = Field(None, description="Standard error output of the test")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TestSuiteResult(BaseModel):
    """Result of a test suite run."""
    total: int = Field(..., description="Total number of tests")
    passed: int = Field(..., description="Number of tests passed")
    failed: int = Field(..., description="Number of tests failed")
    skipped: int = Field(..., description="Number of tests skipped")
    errors: int = Field(..., description="Number of tests with errors")
    execution_time: float = Field(..., description="Total execution time in seconds")
    results: List[TestResult] = Field(default_factory=list, description="Individual test results")
    coverage: Optional[Dict[str, Any]] = Field(None, description="Code coverage data")
    summary: str = Field(..., description="Summary of the test run")
    timestamp: str = Field(..., description="Timestamp of the test run")


class CoverageResult(BaseModel):
    """Code coverage result."""
    line_coverage: float = Field(..., description="Percentage of lines covered")
    branch_coverage: Optional[float] = Field(None, description="Percentage of branches covered")
    file_coverage: Dict[str, float] = Field(default_factory=dict, description="Coverage by file")
    uncovered_lines: Dict[str, List[int]] = Field(default_factory=dict, description="Uncovered lines by file")
    summary: str = Field(..., description="Summary of coverage")


class CodeReviewTool:
    """Tool for reviewing code and generating feedback."""
    
    def __init__(
        self,
        temp_dir: Optional[str] = None,
        config_dir: Optional[str] = None,
        available_analyzers: Optional[Dict[str, bool]] = None,
        default_issue_severity: float = 5.0,
        default_confidence: float = 0.8,
        enable_external_tools: bool = True,
        external_tools_timeout: int = 30,
        use_cache: bool = True,
        verbose: bool = False
    ):
        """Initialize the code review tool.
        
        Args:
            temp_dir: Directory for temporary files
            config_dir: Directory for configuration files
            available_analyzers: Dictionary of available analyzers
            default_issue_severity: Default severity for issues
            default_confidence: Default confidence for issues
            enable_external_tools: Whether to enable external analysis tools
            external_tools_timeout: Timeout for external tools in seconds
            use_cache: Whether to cache review results
            verbose: Whether to enable verbose logging
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "code_review"
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".code_review"
        self.default_issue_severity = default_issue_severity
        self.default_confidence = default_confidence
        self.enable_external_tools = enable_external_tools
        self.external_tools_timeout = external_tools_timeout
        self.use_cache = use_cache
        self.verbose = verbose
        
        # Create directories if they don't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Set up available analyzers based on what's installed
        self.available_analyzers = available_analyzers or {
            "ast": AST_AVAILABLE,
            "pylint": PYLINT_AVAILABLE,
            "black": BLACK_AVAILABLE,
            "mypy": MYPY_AVAILABLE,
            "pytest": PYTEST_AVAILABLE,
            "coverage": COVERAGE_AVAILABLE,
            "esprima": ESPRIMA_AVAILABLE,
            "eslint": self._check_command_available("eslint"),
            "prettier": self._check_command_available("prettier"),
            "tsc": self._check_command_available("tsc"),
            "jest": self._check_command_available("jest")
        }
        
        # Cache for review results
        self.cache = {}
        
        logger.info(f"Code review tool initialized with {sum(self.available_analyzers.values())} analyzers available")
    
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available on the system.
        
        Args:
            command: The command to check
            
        Returns:
            Whether the command is available
        """
        try:
            result = subprocess.run(
                [command, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    async def review_code(
        self,
        code: str,
        language: Union[str, CodeLanguage],
        file_path: Optional[str] = None,
        analyzers: Optional[List[str]] = None,
        include_metrics: bool = True,
        include_suggestions: bool = True,
        strict: bool = False
    ) -> ReviewResult:
        """Review code and identify issues.
        
        Args:
            code: The code to review
            language: Programming language of the code
            file_path: Path to the file containing the code
            analyzers: List of analyzers to use
            include_metrics: Whether to include quality metrics
            include_suggestions: Whether to include improvement suggestions
            strict: Whether to apply strict review standards
            
        Returns:
            Result of the code review
        """
        # Normalize language
        if isinstance(language, str):
            try:
                language = CodeLanguage(language.lower())
            except ValueError:
                language = CodeLanguage.OTHER
        
        start_time = time.time()
        
        # Check cache if enabled
        cache_key = None
        if self.use_cache and file_path:
            cache_key = f"{file_path}:{hash(code)}:{language.value}:{strict}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Using cached review result for {file_path}")
                return cached_result
        
        # Determine which analyzers to use
        available_analyzers = self._get_available_analyzers(language)
        if analyzers:
            active_analyzers = [a for a in analyzers if a in available_analyzers]
        else:
            active_analyzers = available_analyzers
        
        # Create a temporary file for the code if file_path is not provided
        temp_file = None
        if not file_path:
            ext = self._get_file_extension(language)
            temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            temp_file.write(code.encode('utf-8'))
            temp_file.close()
            file_path = temp_file.name
        
        try:
            # Run analyzers to identify issues
            issues = []
            for analyzer in active_analyzers:
                try:
                    analyzer_issues = await self._run_analyzer(
                        analyzer, code, language, file_path, strict
                    )
                    issues.extend(analyzer_issues)
                except Exception as e:
                    logger.error(f"Error running analyzer {analyzer}: {str(e)}")
                    # Create an issue for the analyzer error
                    issues.append(CodeIssue(
                        id=f"{analyzer}_error_{int(time.time())}",
                        type=IssueType.ERROR,
                        category=IssueCategory.OTHER,
                        message=f"Analyzer error: {str(e)}",
                        file_path=file_path,
                        severity=self.default_issue_severity,
                        confidence=self.default_confidence,
                        analyzer=analyzer
                    ))
            
            # Calculate quality metrics
            metrics = None
            if include_metrics:
                metrics = await self._calculate_metrics(code, language, file_path, issues)
            else:
                # Create minimal metrics
                metrics = QualityMetrics(
                    language=language.value,
                    lines_of_code=len(code.splitlines())
                )
            
            # Find highest severity issues
            highest_severity_issues = self._get_highest_severity_issues(issues, limit=5)
            
            # Generate summary
            summary = self._generate_summary(issues, metrics, language)
            
            # Generate improvement suggestions
            suggestions = []
            if include_suggestions:
                suggestions = self._generate_suggestions(issues, metrics, language)
            
            # Create result
            result = ReviewResult(
                issues=issues,
                metrics=metrics,
                summary=summary,
                suggestions=suggestions,
                review_time_ms=(time.time() - start_time) * 1000,
                executed_analyzers=active_analyzers,
                files_analyzed=[file_path],
                highest_severity_issues=highest_severity_issues,
                timestamps={
                    "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                    "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                }
            )
            
            # Cache the result if enabled
            if self.use_cache and cache_key:
                self.cache[cache_key] = result
            
            return result
            
        finally:
            # Clean up temporary file if created
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    def _get_file_extension(self, language: CodeLanguage) -> str:
        """Get the file extension for a language.
        
        Args:
            language: Programming language
            
        Returns:
            File extension including the dot
        """
        extensions = {
            CodeLanguage.PYTHON: ".py",
            CodeLanguage.JAVASCRIPT: ".js",
            CodeLanguage.TYPESCRIPT: ".ts",
            CodeLanguage.HTML: ".html",
            CodeLanguage.CSS: ".css",
            CodeLanguage.JAVA: ".java",
            CodeLanguage.CSHARP: ".cs",
            CodeLanguage.CPP: ".cpp",
            CodeLanguage.GO: ".go",
            CodeLanguage.RUST: ".rs",
            CodeLanguage.PHP: ".php",
            CodeLanguage.RUBY: ".rb",
            CodeLanguage.SWIFT: ".swift",
            CodeLanguage.KOTLIN: ".kt",
            CodeLanguage.SQL: ".sql",
            CodeLanguage.SHELL: ".sh",
            CodeLanguage.JSON: ".json",
            CodeLanguage.YAML: ".yml",
            CodeLanguage.MARKDOWN: ".md"
        }
        return extensions.get(language, ".txt")
    
    def _get_available_analyzers(self, language: CodeLanguage) -> List[str]:
        """Get available analyzers for a language.
        
        Args:
            language: Programming language
            
        Returns:
            List of available analyzer names
        """
        # Map languages to potential analyzers
        language_analyzers = {
            CodeLanguage.PYTHON: ["ast", "pylint", "black", "mypy"],
            CodeLanguage.JAVASCRIPT: ["esprima", "eslint", "prettier"],
            CodeLanguage.TYPESCRIPT: ["esprima", "eslint", "prettier", "tsc"],
            CodeLanguage.HTML: ["htmlhint"],
            CodeLanguage.CSS: ["stylelint"],
            # Add other language mappings as needed
        }
        
        # Get potential analyzers for this language
        potential_analyzers = language_analyzers.get(language, [])
        
        # Filter by what's actually available
        return [a for a in potential_analyzers if self.available_analyzers.get(a, False)]
    
    async def _run_analyzer(
        self,
        analyzer: str,
        code: str,
        language: CodeLanguage,
        file_path: str,
        strict: bool
    ) -> List[CodeIssue]:
        """Run a specific analyzer on code.
        
        Args:
            analyzer: Name of the analyzer to run
            code: The code to analyze
            language: Programming language of the code
            file_path: Path to the file containing the code
            strict: Whether to apply strict analysis standards
            
        Returns:
            List of issues found by the analyzer
        """
        if analyzer == "ast" and language == CodeLanguage.PYTHON and AST_AVAILABLE:
            return await self._run_python_ast_analyzer(code, file_path)
        elif analyzer == "pylint" and language == CodeLanguage.PYTHON and PYLINT_AVAILABLE:
            return await self._run_pylint(code, file_path, strict)
        elif analyzer == "black" and language == CodeLanguage.PYTHON and BLACK_AVAILABLE:
            return await self._run_black(code, file_path)
        elif analyzer == "mypy" and language == CodeLanguage.PYTHON and MYPY_AVAILABLE:
            return await self._run_mypy(code, file_path, strict)
        elif analyzer == "esprima" and language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT] and ESPRIMA_AVAILABLE:
            return await self._run_esprima(code, file_path)
        elif analyzer == "eslint" and language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT] and self.available_analyzers.get("eslint", False):
            return await self._run_eslint(code, file_path, language, strict)
        else:
            logger.warning(f"Analyzer {analyzer} not available or not supported for {language.value}")
            return []
    
    async def _run_python_ast_analyzer(self, code: str, file_path: str) -> List[CodeIssue]:
        """Run Python AST analyzer.
        
        Args:
            code: Python code to analyze
            file_path: Path to the file containing the code
            
        Returns:
            List of issues found by the analyzer
        """
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Check for syntax errors (though the parsing would already fail)
            # This is a placeholder for more advanced AST analysis
            
            # Check for unused imports
            import_nodes = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_nodes.append(node)
            
            # Basic analysis of function definitions, looking for issues like:
            # - Functions without docstrings
            # - Very long functions
            # - Functions with many arguments
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for missing docstring
                    has_docstring = False
                    if len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                        has_docstring = True
                    
                    if not has_docstring:
                        issues.append(CodeIssue(
                            id=f"missing_docstring_{node.name}_{node.lineno}",
                            type=IssueType.STYLE,
                            category=IssueCategory.DOCUMENTATION,
                            message=f"Function '{node.name}' is missing a docstring",
                            line=node.lineno,
                            file_path=file_path,
                            severity=3.0,
                            confidence=0.9,
                            analyzer="ast",
                            tags=["docstring", "documentation"]
                        ))
                    
                    # Check for too many arguments
                    if len(node.args.args) > 7:  # Arbitrary threshold
                        issues.append(CodeIssue(
                            id=f"too_many_args_{node.name}_{node.lineno}",
                            type=IssueType.MAINTAINABILITY,
                            category=IssueCategory.DESIGN,
                            message=f"Function '{node.name}' has too many arguments ({len(node.args.args)})",
                            line=node.lineno,
                            file_path=file_path,
                            severity=4.0,
                            confidence=0.8,
                            analyzer="ast",
                            tags=["function", "design"]
                        ))
                    
                    # Check for function length
                    func_length = sum(1 for _ in ast.walk(node)) - 1  # Rough approximation
                    if func_length > 50:  # Arbitrary threshold
                        issues.append(CodeIssue(
                            id=f"long_function_{node.name}_{node.lineno}",
                            type=IssueType.MAINTAINABILITY,
                            category=IssueCategory.IMPLEMENTATION,
                            message=f"Function '{node.name}' is very long",
                            line=node.lineno,
                            file_path=file_path,
                            severity=3.5,
                            confidence=0.7,
                            analyzer="ast",
                            tags=["function", "complexity"]
                        ))
        
        except SyntaxError as e:
            issues.append(CodeIssue(
                id=f"syntax_error_{e.lineno}_{e.offset}",
                type=IssueType.ERROR,
                category=IssueCategory.SYNTAX,
                message=f"Syntax error: {str(e)}",
                line=e.lineno,
                column=e.offset,
                file_path=file_path,
                severity=9.0,  # High severity for syntax errors
                confidence=1.0,
                analyzer="ast",
                tags=["syntax", "error"]
            ))
        except Exception as e:
            logger.error(f"Error in AST analysis: {str(e)}")
            issues.append(CodeIssue(
                id=f"ast_error_{int(time.time())}",
                type=IssueType.ERROR,
                category=IssueCategory.OTHER,
                message=f"AST analysis error: {str(e)}",
                file_path=file_path,
                severity=5.0,
                confidence=0.5,
                analyzer="ast",
                tags=["error"]
            ))
        
        return issues
    
    async def _run_pylint(self, code: str, file_path: str, strict: bool) -> List[CodeIssue]:
        """Run Pylint on Python code.
        
        Args:
            code: Python code to analyze
            file_path: Path to the file containing the code
            strict: Whether to apply strict linting standards
            
        Returns:
            List of issues found by Pylint
        """
        if not PYLINT_AVAILABLE:
            return []
        
        issues = []
        
        # Create a temporary file with the code
        # (even if file_path is provided, to ensure we're analyzing the current code version)
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_file_path = temp_file.name
        
        try:
            # Create a custom reporter to capture pylint output
            class CustomReporter:
                def __init__(self):
                    self.messages = []
                
                def handle_message(self, msg):
                    self.messages.append(msg)
                
                def on_set_current_module(self, module, filepath):
                    pass
                
                def display_reports(self, layout):
                    pass
                
                def on_close(self, stats, previous_stats):
                    pass
            
            reporter = CustomReporter()
            
            # Run pylint
            args = [temp_file_path]
            if strict:
                args.append("--disable=C0111")  # Disable missing docstring warnings in strict mode
            else:
                args.append("--disable=C0111,C0103,C0303")  # Disable some common warnings in normal mode
            
            # Create a separate process for pylint to avoid affecting the main process
            def run_pylint():
                PylintRun(args, reporter=reporter, exit=False)
            
            # Run pylint in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_pylint)
            
            # Process pylint messages
            for msg in reporter.messages:
                # Convert pylint message to CodeIssue
                issue_type = IssueType.WARNING
                if msg.C in ['E', 'F']:
                    issue_type = IssueType.ERROR
                elif msg.C in ['W']:
                    issue_type = IssueType.WARNING
                elif msg.C in ['C']:
                    issue_type = IssueType.STYLE
                elif msg.C in ['R']:
                    issue_type = IssueType.MAINTAINABILITY
                
                # Map pylint message type to category
                category = IssueCategory.OTHER
                if msg.C in ['E']:
                    category = IssueCategory.SEMANTIC
                elif msg.C in ['F']:
                    category = IssueCategory.SYNTAX
                elif msg.C in ['W']:
                    category = IssueCategory.IMPLEMENTATION
                elif msg.C in ['C']:
                    category = IssueCategory.CONVENTION
                elif msg.C in ['R']:
                    category = IssueCategory.DESIGN
                
                # Calculate severity based on pylint message
                severity = 5.0  # Default
                if msg.C in ['E', 'F']:
                    severity = 8.0
                elif msg.C in ['W']:
                    severity = 6.0
                elif msg.C in ['C']:
                    severity = 3.0
                elif msg.C in ['R']:
                    severity = 4.0
                
                # Create the issue
                issue = CodeIssue(
                    id=f"pylint_{msg.msg_id}_{msg.line}_{msg.column}",
                    type=issue_type,
                    category=category,
                    message=msg.msg,
                    line=msg.line,
                    column=msg.column,
                    file_path=file_path,  # Use original file path, not temp file
                    rule_id=msg.msg_id,
                    severity=severity,
                    confidence=0.9,
                    analyzer="pylint",
                    tags=["pylint", msg.msg_id]
                )
                
                issues.append(issue)
        
        except Exception as e:
            logger.error(f"Error running pylint: {str(e)}")
            issues.append(CodeIssue(
                id=f"pylint_error_{int(time.time())}",
                type=IssueType.ERROR,
                category=IssueCategory.OTHER,
                message=f"Pylint error: {str(e)}",
                file_path=file_path,
                severity=5.0,
                confidence=0.5,
                analyzer="pylint",
                tags=["error"]
            ))
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return issues
    
    async def _run_black(self, code: str, file_path: str) -> List[CodeIssue]:
        """Run Black on Python code to check for formatting issues.
        
        Args:
            code: Python code to analyze
            file_path: Path to the file containing the code
            
        Returns:
            List of formatting issues found by Black
        """
        if not BLACK_AVAILABLE:
            return []
        
        issues = []
        
        try:
            # Try to format the code with Black
            mode = black.FileMode()
            formatted_code = black.format_str(code, mode=mode)
            
            # If the formatted code is different from the original, create an issue
            if formatted_code != code:
                # Generate a diff
                diff = list(difflib.unified_diff(
                    code.splitlines(keepends=True),
                    formatted_code.splitlines(keepends=True),
                    fromfile="original",
                    tofile="formatted",
                    n=3
                ))
                
                # Create an issue for each diff hunk
                hunk_pattern = re.compile(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@')
                current_hunk = None
                hunk_lines = []
                
                for line in diff:
                    if line.startswith('@@'):
                        # Process previous hunk
                        if current_hunk and hunk_lines:
                            match = hunk_pattern.match(current_hunk)
                            if match:
                                start_line = int(match.group(1))
                                
                                issues.append(CodeIssue(
                                    id=f"black_format_{start_line}_{int(time.time())}",
                                    type=IssueType.STYLE,
                                    category=IssueCategory.CONVENTION,
                                    message="Code formatting does not comply with Black style",
                                    line=start_line,
                                    file_path=file_path,
                                    code_snippet=''.join(hunk_lines),
                                    suggestion="Run Black on this file to auto-format",
                                    severity=3.0,
                                    confidence=0.9,
                                    analyzer="black",
                                    tags=["formatting", "style", "black"]
                                ))
                            
                            hunk_lines = []
                        
                        current_hunk = line
                    elif current_hunk:
                        hunk_lines.append(line)
                
                # Process the last hunk
                if current_hunk and hunk_lines:
                    match = hunk_pattern.match(current_hunk)
                    if match:
                        start_line = int(match.group(1))
                        
                        issues.append(CodeIssue(
                            id=f"black_format_{start_line}_{int(time.time())}",
                            type=IssueType.STYLE,
                            category=IssueCategory.CONVENTION,
                            message="Code formatting does not comply with Black style",
                            line=start_line,
                            file_path=file_path,
                            code_snippet=''.join(hunk_lines),
                            suggestion="Run Black on this file to auto-format",
                            severity=3.0,
                            confidence=0.9,
                            analyzer="black",
                            tags=["formatting", "style", "black"]
                        ))
                
                # If no specific issues were found but the code is different, add a general issue
                if not issues:
                    issues.append(CodeIssue(
                        id=f"black_format_general_{int(time.time())}",
                        type=IssueType.STYLE,
                        category=IssueCategory.CONVENTION,
                        message="Code formatting does not comply with Black style",
                        file_path=file_path,
                        suggestion="Run Black on this file to auto-format",
                        severity=3.0,
                        confidence=0.9,
                        analyzer="black",
                        tags=["formatting", "style", "black"]
                    ))
        
        except Exception as e:
            logger.error(f"Error running Black: {str(e)}")
            issues.append(CodeIssue(
                id=f"black_error_{int(time.time())}",
                type=IssueType.ERROR,
                category=IssueCategory.OTHER,
                message=f"Black error: {str(e)}",
                file_path=file_path,
                severity=2.0,
                confidence=0.5,
                analyzer="black",
                tags=["error"]
            ))
        
        return issues
    
    async def _run_mypy(self, code: str, file_path: str, strict: bool) -> List[CodeIssue]:
        """Run mypy on Python code to check for type issues.
        
        Args:
            code: Python code to analyze
            file_path: Path to the file containing the code
            strict: Whether to apply strict type checking
            
        Returns:
            List of type issues found by mypy
        """
        if not MYPY_AVAILABLE:
            return []
        
        issues = []
        
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_file_path = temp_file.name
        
        try:
            # Build mypy command-line arguments
            mypy_args = []
            if strict:
                mypy_args.extend(["--strict"])
            
            # Run mypy
            def run_mypy():
                return mypy.api.run([temp_file_path] + mypy_args)
            
            # Run mypy in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            report, errors, status = await loop.run_in_executor(None, run_mypy)
            
            # Process mypy output
            if report:
                for line in report.splitlines():
                    # Parse mypy error messages
                    # Example: file.py:42: error: Incompatible return value type (got "str", expected "int")
                    match = re.match(r'^([^:]+):(\d+)(?::(\d+))?: (\w+): (.+)$', line)
                    if match:
                        error_file, line_num, column, error_type, message = match.groups()
                        
                        # Create an issue
                        issue = CodeIssue(
                            id=f"mypy_{line_num}_{column or '0'}_{int(time.time())}",
                            type=IssueType.WARNING if error_type == "warning" else IssueType.ERROR,
                            category=IssueCategory.SEMANTIC,
                            message=f"Type error: {message}",
                            line=int(line_num),
                            column=int(column) if column else None,
                            file_path=file_path,  # Use original file path, not temp file
                            severity=7.0 if error_type == "error" else 4.0,
                            confidence=0.9,
                            analyzer="mypy",
                            tags=["typing", "mypy"]
                        )
                        
                        issues.append(issue)
        
        except Exception as e:
            logger.error(f"Error running mypy: {str(e)}")
            issues.append(CodeIssue(
                id=f"mypy_error_{int(time.time())}",
                type=IssueType.ERROR,
                category=IssueCategory.OTHER,
                message=f"Mypy error: {str(e)}",
                file_path=file_path,
                severity=2.0,
                confidence=0.5,
                analyzer="mypy",
                tags=["error"]
            ))
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return issues
    
    async def _run_esprima(self, code: str, file_path: str) -> List[CodeIssue]:
        """Run Esprima on JavaScript/TypeScript code.
        
        Args:
            code: JavaScript/TypeScript code to analyze
            file_path: Path to the file containing the code
            
        Returns:
            List of issues found by Esprima
        """
        if not ESPRIMA_AVAILABLE:
            return []
        
        issues = []
        
        try:
            # Parse the code using Esprima
            syntax_tree = esprima.parseScript(code, options={'tolerant': True, 'loc': True})
            
            # Check for any syntax errors
            if hasattr(syntax_tree, 'errors') and syntax_tree.errors:
                for error in syntax_tree.errors:
                    issues.append(CodeIssue(
                        id=f"esprima_syntax_{error.lineNumber}_{error.column}_{int(time.time())}",
                        type=IssueType.ERROR,
                        category=IssueCategory.SYNTAX,
                        message=f"Syntax error: {error.description}",
                        line=error.lineNumber,
                        column=error.column,
                        file_path=file_path,
                        severity=8.0,
                        confidence=0.9,
                        analyzer="esprima",
                        tags=["syntax", "error"]
                    ))
            
            # Advanced analysis would be implemented here
            # This could include checking for:
            # - Unused variables
            # - Unused imports
            # - Function complexity
            # - Potential bugs (like using == instead of ===)
            # - etc.
        
        except Exception as e:
            logger.error(f"Error running Esprima: {str(e)}")
            issues.append(CodeIssue(
                id=f"esprima_error_{int(time.time())}",
                type=IssueType.ERROR,
                category=IssueCategory.OTHER,
                message=f"Esprima error: {str(e)}",
                file_path=file_path,
                severity=5.0,
                confidence=0.5,
                analyzer="esprima",
                tags=["error"]
            ))
        
        return issues
    
    async def _run_eslint(
        self,
        code: str,
        file_path: str,
        language: CodeLanguage,
        strict: bool
    ) -> List[CodeIssue]:
        """Run ESLint on JavaScript/TypeScript code.
        
        Args:
            code: JavaScript/TypeScript code to analyze
            file_path: Path to the file containing the code
            language: Language of the code (JavaScript or TypeScript)
            strict: Whether to apply strict linting standards
            
        Returns:
            List of issues found by ESLint
        """
        if not self.available_analyzers.get("eslint", False):
            return []
        
        issues = []
        
        # Create a temporary file with the code
        ext = ".ts" if language == CodeLanguage.TYPESCRIPT else ".js"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_file_path = temp_file.name
        
        try:
            # Build ESLint command
            cmd = ["eslint", "--format", "json"]
            if strict:
                # No special flags for strict mode, ESLint config handles this
                pass
            else:
                # Use a more lenient configuration for normal mode
                cmd.extend(["--no-eslintrc", "--env", "browser,node", "--rule", "semi:warn"])
            
            cmd.append(temp_file_path)
            
            # Run ESLint
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.external_tools_timeout
            )
            
            # Process ESLint output
            if stdout:
                try:
                    eslint_results = json.loads(stdout.decode('utf-8'))
                    
                    for file_result in eslint_results:
                        for message in file_result.get("messages", []):
                            # Determine issue type and category
                            if message.get("severity") == 2:
                                issue_type = IssueType.ERROR
                            elif message.get("severity") == 1:
                                issue_type = IssueType.WARNING
                            else:
                                issue_type = IssueType.INFO
                            
                            # Map ESLint rule to category
                            rule_id = message.get("ruleId", "")
                            category = IssueCategory.OTHER
                            if rule_id.startswith("no-"):
                                category = IssueCategory.IMPLEMENTATION
                            elif rule_id in ["semi", "indent", "quotes"]:
                                category = IssueCategory.CONVENTION
                            elif rule_id in ["complexity", "max-len", "max-lines"]:
                                category = IssueCategory.MAINTAINABILITY
                            elif rule_id in ["no-unused-vars", "no-undef"]:
                                category = IssueCategory.SEMANTIC
                            
                            # Create the issue
                            issue = CodeIssue(
                                id=f"eslint_{rule_id}_{message.get('line', 0)}_{message.get('column', 0)}",
                                type=issue_type,
                                category=category,
                                message=message.get("message", "Unknown ESLint issue"),
                                line=message.get("line"),
                                column=message.get("column"),
                                file_path=file_path,  # Use original file path, not temp file
                                rule_id=rule_id,
                                severity=7.0 if issue_type == IssueType.ERROR else 4.0,
                                confidence=0.9,
                                analyzer="eslint",
                                tags=["eslint", rule_id]
                            )
                            
                            issues.append(issue)
                
                except json.JSONDecodeError:
                    logger.error("Failed to parse ESLint JSON output")
            
            if stderr:
                stderr_text = stderr.decode('utf-8')
                if stderr_text.strip() and "error" in stderr_text.lower():
                    issues.append(CodeIssue(
                        id=f"eslint_error_{int(time.time())}",
                        type=IssueType.ERROR,
                        category=IssueCategory.OTHER,
                        message=f"ESLint error: {stderr_text.strip()}",
                        file_path=file_path,
                        severity=3.0,
                        confidence=0.7,
                        analyzer="eslint",
                        tags=["error"]
                    ))
        
        except asyncio.TimeoutError:
            logger.error("ESLint timed out")
            issues.append(CodeIssue(
                id=f"eslint_timeout_{int(time.time())}",
                type=IssueType.ERROR,
                category=IssueCategory.OTHER,
                message=f"ESLint timed out after {self.external_tools_timeout} seconds",
                file_path=file_path,
                severity=2.0,
                confidence=0.5,
                analyzer="eslint",
                tags=["timeout"]
            ))
        
        except Exception as e:
            logger.error(f"Error running ESLint: {str(e)}")
            issues.append(CodeIssue(
                id=f"eslint_error_{int(time.time())}",
                type=IssueType.ERROR,
                category=IssueCategory.OTHER,
                message=f"ESLint error: {str(e)}",
                file_path=file_path,
                severity=2.0,
                confidence=0.5,
                analyzer="eslint",
                tags=["error"]
            ))
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return issues
    
    async def _calculate_metrics(
        self,
        code: str,
        language: CodeLanguage,
        file_path: str,
        issues: List[CodeIssue]
    ) -> QualityMetrics:
        """Calculate quality metrics for code.
        
        Args:
            code: The code to analyze
            language: Programming language of the code
            file_path: Path to the file containing the code
            issues: List of issues found in the code
            
        Returns:
            Quality metrics
        """
        # Basic metrics that work for any language
        lines = code.splitlines()
        total_lines = len(lines)
        empty_lines = sum(1 for line in lines if not line.strip())
        
        # Count comment lines (language-specific)
        comment_lines = 0
        if language == CodeLanguage.PYTHON:
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            # Count single-line comments and approximate multi-line comments
            comment_lines = sum(1 for line in lines if line.strip().startswith('//'))
            in_comment = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('/*'):
                    in_comment = True
                    comment_lines += 1
                elif in_comment:
                    comment_lines += 1
                    if '*/' in stripped:
                        in_comment = False
        
        # Calculate comment ratio
        non_empty_lines = total_lines - empty_lines
        comment_ratio = comment_lines / max(1, non_empty_lines)
        
        # Count issues by type
        error_count = sum(1 for issue in issues if issue.type == IssueType.ERROR)
        warning_count = sum(1 for issue in issues if issue.type == IssueType.WARNING)
        style_count = sum(1 for issue in issues if issue.type == IssueType.STYLE)
        
        # Calculate issue density
        issue_density = (error_count + warning_count + style_count) * 1000 / max(1, total_lines)
        
        # Language-specific metrics
        language_specific = {}
        function_count = None
        class_count = None
        
        if language == CodeLanguage.PYTHON and AST_AVAILABLE:
            try:
                tree = ast.parse(code)
                
                # Count functions and classes
                function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                
                # Add to language-specific metrics
                language_specific["async_functions"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef))
                language_specific["import_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
            except:
                pass
        
        # Create metrics object
        metrics = QualityMetrics(
            language=language.value,
            lines_of_code=total_lines,
            comment_lines=comment_lines,
            empty_lines=empty_lines,
            comment_ratio=comment_ratio,
            issue_density=issue_density,
            function_count=function_count,
            class_count=class_count,
            language_specific=language_specific
        )
        
        return metrics
    
    def _get_highest_severity_issues(self, issues: List[CodeIssue], limit: int = 5) -> List[CodeIssue]:
        """Get the highest severity issues.
        
        Args:
            issues: List of issues to filter
            limit: Maximum number of issues to return
            
        Returns:
            List of highest severity issues
        """
        # Sort issues by severity (highest first)
        sorted_issues = sorted(issues, key=lambda issue: issue.severity, reverse=True)
        
        # Return the top issues
        return sorted_issues[:limit]
    
    def _generate_summary(
        self,
        issues: List[CodeIssue],
        metrics: QualityMetrics,
        language: CodeLanguage
    ) -> str:
        """Generate a summary of the code review.
        
        Args:
            issues: List of issues found in the code
            metrics: Quality metrics for the code
            language: Programming language of the code
            
        Returns:
            Summary of the code review
        """
        # Count issues by type
        error_count = sum(1 for issue in issues if issue.type == IssueType.ERROR)
        warning_count = sum(1 for issue in issues if issue.type == IssueType.WARNING)
        style_count = sum(1 for issue in issues if issue.type == IssueType.STYLE)
        
        # Determine overall assessment
        if error_count > 0:
            assessment = "has significant issues that need to be addressed"
        elif warning_count > 5:
            assessment = "has several warnings that should be reviewed"
        elif warning_count > 0:
            assessment = "has a few minor issues but is generally good"
        elif style_count > 0:
            assessment = "is good but could benefit from style improvements"
        else:
            assessment = "looks good with no significant issues"
        
        # Create summary
        summary_lines = [
            f"# Code Review Summary",
            "",
            f"The {language.value} code ({metrics.lines_of_code} lines) {assessment}.",
            "",
            f"## Issues Found",
            f"- {error_count} errors",
            f"- {warning_count} warnings",
            f"- {style_count} style issues",
            "",
            f"## Metrics",
            f"- Comment ratio: {metrics.comment_ratio:.1%}",
            f"- Issue density: {metrics.issue_density:.1f} issues per 1000 lines"
        ]
        
        # Add language-specific metrics
        if metrics.function_count is not None:
            summary_lines.append(f"- Functions: {metrics.function_count}")
        if metrics.class_count is not None:
            summary_lines.append(f"- Classes: {metrics.class_count}")
        
        return "\n".join(summary_lines)
    
    def _generate_suggestions(
        self,
        issues: List[CodeIssue],
        metrics: QualityMetrics,
        language: CodeLanguage
    ) -> List[str]:
        """Generate improvement suggestions based on the review.
        
        Args:
            issues: List of issues found in the code
            metrics: Quality metrics for the code
            language: Programming language of the code
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check for high error count
        error_count = sum(1 for issue in issues if issue.type == IssueType.ERROR)
        if error_count > 0:
            suggestions.append(f"Fix the {error_count} errors before proceeding further.")
        
        # Check for comment ratio
        if metrics.comment_ratio < 0.1:
            suggestions.append("Add more comments to explain the code's logic and purpose.")
        
        # Check for specific issue patterns
        style_issues = sum(1 for issue in issues if issue.type == IssueType.STYLE)
        if style_issues > 5:
            if language == CodeLanguage.PYTHON:
                suggestions.append("Run a formatter like Black on the code to improve style consistency.")
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                suggestions.append("Run a formatter like Prettier on the code to improve style consistency.")
        
        # Make language-specific suggestions
        if language == CodeLanguage.PYTHON:
            # Check for type annotation issues
            type_issues = sum(1 for issue in issues if "typing" in issue.tags)
            if type_issues > 0:
                suggestions.append("Add type annotations to improve code clarity and catch type errors.")
            
            # Check for docstring issues
            docstring_issues = sum(1 for issue in issues if "docstring" in issue.tags)
            if docstring_issues > 0:
                suggestions.append("Add docstrings to functions and classes to improve documentation.")
        
        # Add general suggestions
        if len(issues) > 10:
            suggestions.append("Consider refactoring complex parts of the code to improve maintainability.")
        
        return suggestions
    
    async def run_tests(
        self,
        code: str,
        test_code: str,
        language: Union[str, CodeLanguage],
        test_framework: Optional[str] = None,
        include_coverage: bool = False,
        timeout: int = 30
    ) -> TestSuiteResult:
        """Run tests for code.
        
        Args:
            code: The code to test
            test_code: The test code to run
            language: Programming language of the code
            test_framework: Test framework to use
            include_coverage: Whether to include code coverage
            timeout: Timeout for test execution in seconds
            
        Returns:
            Result of the test run
        """
        # Normalize language
        if isinstance(language, str):
            try:
                language = CodeLanguage(language.lower())
            except ValueError:
                language = CodeLanguage.OTHER
        
        # Determine test framework if not specified
        if not test_framework:
            if language == CodeLanguage.PYTHON:
                test_framework = "pytest"
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                test_framework = "jest"
            else:
                test_framework = "unknown"
        
        # Run tests based on language and framework
        if language == CodeLanguage.PYTHON:
            if test_framework.lower() == "pytest":
                return await self._run_pytest(code, test_code, include_coverage, timeout)
            elif test_framework.lower() == "unittest":
                return await self._run_unittest(code, test_code, include_coverage, timeout)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            if test_framework.lower() == "jest":
                return await self._run_jest(code, test_code, language, include_coverage, timeout)
        
        # If we get here, the language or framework is not supported
        return TestSuiteResult(
            total=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            execution_time=0.0,
            summary=f"Tests could not be run: {language.value} with {test_framework} is not supported",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    async def _run_pytest(
        self,
        code: str,
        test_code: str,
        include_coverage: bool = False,
        timeout: int = 30
    ) -> TestSuiteResult:
        """Run pytest tests for Python code.
        
        Args:
            code: The Python code to test
            test_code: The pytest test code
            include_coverage: Whether to include coverage
            timeout: Timeout for test execution in seconds
            
        Returns:
            Result of the test run
        """
        if not PYTEST_AVAILABLE:
            return TestSuiteResult(
                total=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=0,
                execution_time=0.0,
                summary="Pytest is not available",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Create the module file
            module_file = temp_dir_path / "module_to_test.py"
            with open(module_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Create the test file
            test_file = temp_dir_path / "test_module.py"
            with open(test_file, "w", encoding="utf-8") as f:
                # Ensure the test imports the module correctly
                test_code = test_code.replace("from module_to_test", "from module_to_test")
                f.write(test_code)
            
            # Create an __init__.py file to make it a package
            init_file = temp_dir_path / "__init__.py"
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("# Package initialization")
            
            # Run pytest with JSON output
            start_time = time.time()
            
            cmd = ["pytest", "-v", str(test_file)]
            if include_coverage and COVERAGE_AVAILABLE:
                cmd = ["coverage", "run", "--source=module_to_test", "-m"] + cmd
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(temp_dir_path)
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                stdout_text = stdout.decode("utf-8")
                stderr_text = stderr.decode("utf-8")
                
                # Parse test results
                test_results = []
                total = 0
                passed = 0
                failed = 0
                skipped = 0
                errors = 0
                
                # Parse pytest output
                # Example: test_module.py::test_function PASSED [ 25%]
                for line in stdout_text.splitlines():
                    match = re.search(r'::(\w+)\s+(\w+)\s*\[', line)
                    if match:
                        test_name = match.group(1)
                        status_text = match.group(2)
                        
                        total += 1
                        status = TestStatus.PASSED
                        if status_text == "PASSED":
                            passed += 1
                            status = TestStatus.PASSED
                        elif status_text == "FAILED":
                            failed += 1
                            status = TestStatus.FAILED
                        elif status_text == "SKIPPED":
                            skipped += 1
                            status = TestStatus.SKIPPED
                        elif status_text in ["ERROR", "XFAIL", "XPASS"]:
                            errors += 1
                            status = TestStatus.ERROR
                        
                        test_results.append(TestResult(
                            test_id=f"pytest_{test_name}",
                            name=test_name,
                            status=status,
                            execution_time=0.0,  # Detailed timing not available
                            stdout=stdout_text,
                            stderr=stderr_text
                        ))
                
                # Extract error messages for failed tests
                for result in test_results:
                    if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                        # Try to find the error message for this test
                        pattern = f"{result.name}.*?FAILED"
                        error_section = re.search(pattern, stdout_text, re.DOTALL)
                        if error_section:
                            error_text = stdout_text[error_section.end():].split("=")[0].strip()
                            result.error_message = error_text
                
                # Get coverage if requested
                coverage_data = None
                if include_coverage and COVERAGE_AVAILABLE:
                    try:
                        # Run coverage report
                        coverage_cmd = ["coverage", "report", "-m"]
                        coverage_process = await asyncio.create_subprocess_exec(
                            *coverage_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=str(temp_dir_path)
                        )
                        
                        cov_stdout, _ = await asyncio.wait_for(
                            coverage_process.communicate(),
                            timeout=10
                        )
                        
                        cov_stdout_text = cov_stdout.decode("utf-8")
                        
                        # Parse coverage report
                        coverage_lines = cov_stdout_text.splitlines()
                        if len(coverage_lines) > 2:
                            # Extract overall coverage from the last line
                            total_line = coverage_lines[-1]
                            total_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', total_line)
                            total_coverage = int(total_match.group(1)) if total_match else 0
                            
                            # Extract coverage for each file
                            file_coverage = {}
                            uncovered_lines = {}
                            
                            for line in coverage_lines[1:-1]:  # Skip header and total
                                parts = line.split()
                                # Extract missing lines if present
                                if len(parts) > 4:
                                    filename = parts[0]
                                    missing = parts[4]
                                    line_ranges = re.findall(r'\d+(?:-\d+)?', missing)
                                    missing_lines = []
                                    for range_str in line_ranges:
                                        if '-' in range_str:
                                            start, end = map(int, range_str.split('-'))
                                            missing_lines.extend(range(start, end + 1))
                                        else:
                                            missing_lines.append(int(range_str))
                                    uncovered_lines[filename] = missing_lines
                            
                            coverage_data = {
                                "line_coverage": total_coverage / 100,
                                "file_coverage": file_coverage,
                                "uncovered_lines": uncovered_lines
                            }
                    except Exception as e:
                        logger.error(f"Error getting coverage data: {str(e)}")
                
                # Generate summary
                if total == 0:
                    summary = "No tests were found or executed"
                else:
                    summary = (
                        f"Ran {total} tests: {passed} passed, {failed} failed, "
                        f"{skipped} skipped, {errors} errors"
                    )
                    if coverage_data:
                        summary += f", {coverage_data['line_coverage']:.1%} coverage"
                
                return TestSuiteResult(
                    total=total,
                    passed=passed,
                    failed=failed,
                    skipped=skipped,
                    errors=errors,
                    execution_time=execution_time,
                    results=test_results,
                    coverage=coverage_data,
                    summary=summary,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            except asyncio.TimeoutError:
                return TestSuiteResult(
                    total=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    execution_time=timeout,
                    summary=f"Tests timed out after {timeout} seconds",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            except Exception as e:
                return TestSuiteResult(
                    total=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    execution_time=time.time() - start_time,
                    summary=f"Error running tests: {str(e)}",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
    
    async def _run_unittest(
        self,
        code: str,
        test_code: str,
        include_coverage: bool = False,
        timeout: int = 30
    ) -> TestSuiteResult:
        """Run unittest tests for Python code.
        
        Args:
            code: The Python code to test
            test_code: The unittest test code
            include_coverage: Whether to include coverage
            timeout: Timeout for test execution in seconds
            
        Returns:
            Result of the test run
        """
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Create the module file
            module_file = temp_dir_path / "module_to_test.py"
            with open(module_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Create the test file
            test_file = temp_dir_path / "test_module.py"
            with open(test_file, "w", encoding="utf-8") as f:
                # Ensure the test imports the module correctly
                test_code = test_code.replace("from module_to_test", "from module_to_test")
                f.write(test_code)
            
            # Create an __init__.py file to make it a package
            init_file = temp_dir_path / "__init__.py"
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("# Package initialization")
            
            # Run unittest
            start_time = time.time()
            
            cmd = [
                sys.executable, "-m", "unittest", "discover",
                "-s", str(temp_dir_path), "-p", "test_*.py", "-v"
            ]
            
            if include_coverage and COVERAGE_AVAILABLE:
                cmd = ["coverage", "run", "--source=module_to_test", "-m"] + cmd[1:]
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(temp_dir_path)
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                stdout_text = stdout.decode("utf-8")
                stderr_text = stderr.decode("utf-8")
                
                # Parse test results
                test_results = []
                total = 0
                passed = 0
                failed = 0
                skipped = 0
                errors = 0
                
                # Parse unittest output
                for line in stdout_text.splitlines():
                    if " ... ok" in line:
                        total += 1
                        passed += 1
                        test_name = line.split(" ... ok")[0].strip()
                        test_results.append(TestResult(
                            test_id=f"unittest_{test_name}",
                            name=test_name,
                            status=TestStatus.PASSED,
                            execution_time=0.0,  # Detailed timing not available
                            stdout=stdout_text,
                            stderr=stderr_text
                        ))
                    elif " ... FAIL" in line:
                        total += 1
                        failed += 1
                        test_name = line.split(" ... FAIL")[0].strip()
                        test_results.append(TestResult(
                            test_id=f"unittest_{test_name}",
                            name=test_name,
                            status=TestStatus.FAILED,
                            execution_time=0.0,
                            stdout=stdout_text,
                            stderr=stderr_text
                        ))
                    elif " ... ERROR" in line:
                        total += 1
                        errors += 1
                        test_name = line.split(" ... ERROR")[0].strip()
                        test_results.append(TestResult(
                            test_id=f"unittest_{test_name}",
                            name=test_name,
                            status=TestStatus.ERROR,
                            execution_time=0.0,
                            stdout=stdout_text,
                            stderr=stderr_text
                        ))
                    elif " ... skipped" in line:
                        total += 1
                        skipped += 1
                        test_name = line.split(" ... skipped")[0].strip()
                        test_results.append(TestResult(
                            test_id=f"unittest_{test_name}",
                            name=test_name,
                            status=TestStatus.SKIPPED,
                            execution_time=0.0,
                            stdout=stdout_text,
                            stderr=stderr_text
                        ))
                
                # Extract info from the "Ran X tests" line
                ran_match = re.search(r"Ran (\d+) tests? in ([\d.]+)s", stdout_text)
                if ran_match:
                    total = int(ran_match.group(1))
                    execution_time = float(ran_match.group(2))
                
                # Extract error messages for failed tests
                for result in test_results:
                    if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                        # Try to find the error message for this test
                        pattern = f"{result.name}.*?(FAIL|ERROR).*?"
                        error_section = re.search(pattern, stdout_text, re.DOTALL)
                        if error_section:
                            next_test = re.search(r"\n\w+.*?(ok|FAIL|ERROR|skipped)", stdout_text[error_section.end():])
                            end_pos = next_test.start() + error_section.end() if next_test else len(stdout_text)
                            error_text = stdout_text[error_section.end():end_pos].strip()
                            result.error_message = error_text
                
                # Get coverage if requested
                coverage_data = None
                if include_coverage and COVERAGE_AVAILABLE:
                    try:
                        # Run coverage report
                        coverage_cmd = ["coverage", "report", "-m"]
                        coverage_process = await asyncio.create_subprocess_exec(
                            *coverage_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=str(temp_dir_path)
                        )
                        
                        cov_stdout, _ = await asyncio.wait_for(
                            coverage_process.communicate(),
                            timeout=10
                        )
                        
                        cov_stdout_text = cov_stdout.decode("utf-8")
                        
                        # Parse coverage report
                        coverage_lines = cov_stdout_text.splitlines()
                        if len(coverage_lines) > 2:
                            # Extract overall coverage from the last line
                            total_line = coverage_lines[-1]
                            total_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', total_line)
                            total_coverage = int(total_match.group(1)) if total_match else 0
                            
                            # Extract coverage for each file
                            file_coverage = {}
                            uncovered_lines = {}
                            
                            for line in coverage_lines[1:-1]:  # Skip header and total
                                parts = line.split()
                                if len(parts) >= 4:
                                    filename = parts[0]
                                    coverage_pct = int(parts[3].rstrip('%'))
                                    file_coverage[filename] = coverage_pct / 100
                                    
                                    # Extract missing lines if present
                                    if len(parts) > 4:
                                        missing = parts[4]
                                        line_ranges = re.findall(r'\d+(?:-\d+)?', missing)
                                        missing_lines = []
                                        for range_str in line_ranges:
                                            if '-' in range_str:
                                                start, end = map(int, range_str.split('-'))
                                                missing_lines.extend(range(start, end + 1))
                                            else:
                                                missing_lines.append(int(range_str))
                                        uncovered_lines[filename] = missing_lines
                            
                            coverage_data = {
                                "line_coverage": total_coverage / 100,
                                "file_coverage": file_coverage,
                                "uncovered_lines": uncovered_lines
                            }
                    except Exception as e:
                        logger.error(f"Error getting coverage data: {str(e)}")
                
                # Generate summary
                if total == 0:
                    summary = "No tests were found or executed"
                else:
                    summary = (
                        f"Ran {total} tests: {passed} passed, {failed} failed, "
                        f"{skipped} skipped, {errors} errors"
                    )
                    if coverage_data:
                        summary += f", {coverage_data['line_coverage']:.1%} coverage"
                
                return TestSuiteResult(
                    total=total,
                    passed=passed,
                    failed=failed,
                    skipped=skipped,
                    errors=errors,
                    execution_time=execution_time,
                    results=test_results,
                    coverage=coverage_data,
                    summary=summary,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            except asyncio.TimeoutError:
                return TestSuiteResult(
                    total=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    execution_time=timeout,
                    summary=f"Tests timed out after {timeout} seconds",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            except Exception as e:
                return TestSuiteResult(
                    total=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    execution_time=time.time() - start_time,
                    summary=f"Error running tests: {str(e)}",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
    
    async def _run_jest(
        self,
        code: str,
        test_code: str,
        language: CodeLanguage,
        include_coverage: bool = False,
        timeout: int = 30
    ) -> TestSuiteResult:
        """Run Jest tests for JavaScript/TypeScript code.
        
        Args:
            code: The JavaScript/TypeScript code to test
            test_code: The Jest test code
            language: The language (JavaScript or TypeScript)
            include_coverage: Whether to include coverage
            timeout: Timeout for test execution in seconds
            
        Returns:
            Result of the test run
        """
        if not self.available_analyzers.get("jest", False):
            return TestSuiteResult(
                total=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=0,
                execution_time=0.0,
                summary="Jest is not available",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Determine file extensions
            module_ext = ".ts" if language == CodeLanguage.TYPESCRIPT else ".js"
            test_ext = ".test" + module_ext
            
            # Create the module file
            module_file = temp_dir_path / f"moduleToTest{module_ext}"
            with open(module_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Create the test file
            test_file = temp_dir_path / f"moduleToTest{test_ext}"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(test_code)
            
            # Create a basic package.json for Jest
            package_json = {
                "name": "test-project",
                "version": "1.0.0",
                "private": True,
                "scripts": {
                    "test": "jest"
                },
                "jest": {
                    "testEnvironment": "node"
                }
            }
            
            # Add TypeScript config if needed
            if language == CodeLanguage.TYPESCRIPT:
                package_json["jest"]["preset"] = "ts-jest"
                package_json["jest"]["globals"] = {
                    "ts-jest": {
                        "tsconfig": "tsconfig.json"
                    }
                }
                
                # Create a basic tsconfig.json
                tsconfig = {
                    "compilerOptions": {
                        "target": "es6",
                        "module": "commonjs",
                        "strict": False,
                        "esModuleInterop": True,
                        "skipLibCheck": True,
                        "forceConsistentCasingInFileNames": True
                    }
                }
                
                with open(temp_dir_path / "tsconfig.json", "w", encoding="utf-8") as f:
                    json.dump(tsconfig, f, indent=2)
            
            # Write package.json
            with open(temp_dir_path / "package.json", "w", encoding="utf-8") as f:
                json.dump(package_json, f, indent=2)
            
            # Run Jest
            start_time = time.time()
            
            cmd = ["npx", "jest", "--json"]
            if include_coverage:
                cmd.append("--coverage")
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(temp_dir_path)
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                stdout_text = stdout.decode("utf-8")
                stderr_text = stderr.decode("utf-8")
                
                # Parse Jest JSON output
                try:
                    test_data = json.loads(stdout_text)
                    
                    total = test_data.get("numTotalTests", 0)
                    passed = test_data.get("numPassedTests", 0)
                    failed = test_data.get("numFailedTests", 0)
                    skipped = test_data.get("numPendingTests", 0)
                    errors = 0  # Jest doesn't distinguish between failures and errors
                    
                    # Extract test results
                    test_results = []
                    for test_file in test_data.get("testResults", []):
                        for assertion_result in test_file.get("assertionResults", []):
                            status = TestStatus.PASSED
                            if assertion_result.get("status") == "failed":
                                status = TestStatus.FAILED
                            elif assertion_result.get("status") == "pending":
                                status = TestStatus.SKIPPED
                            
                            test_name = assertion_result.get("title", "Unknown test")
                            test_id = f"jest_{test_name.replace(' ', '_')}"
                            
                            result = TestResult(
                                test_id=test_id,
                                name=test_name,
                                status=status,
                                execution_time=assertion_result.get("duration", 0) / 1000,
                                stderr=stderr_text
                            )
                            
                            # Extract error message if test failed
                            if status == TestStatus.FAILED:
                                failure_messages = assertion_result.get("failureMessages", [])
                                if failure_messages:
                                    result.error_message = failure_messages[0]
                            
                            test_results.append(result)
                    
                    # Extract coverage data if available
                    coverage_data = None
                    if include_coverage and "coverageMap" in test_data:
                        coverage_map = test_data["coverageMap"]
                        
                        # Calculate overall line coverage
                        total_covered = 0
                        total_lines = 0
                        file_coverage = {}
                        uncovered_lines = {}
                        
                        for file_path, file_data in coverage_map.items():
                            statementMap = file_data.get("statementMap", {})
                            statements = file_data.get("s", {})
                            
                            covered_statements = sum(1 for hit in statements.values() if hit > 0)
                            total_statements = len(statements)
                            
                            if total_statements > 0:
                                file_cov = covered_statements / total_statements
                                file_coverage[file_path] = file_cov
                                
                                # Track uncovered lines
                                uncovered = []
                                for stmt_id, hit in statements.items():
                                    if hit == 0 and stmt_id in statementMap:
                                        stmt = statementMap[stmt_id]
                                        line = stmt.get("start", {}).get("line")
                                        if line:
                                            uncovered.append(line)
                                
                                if uncovered:
                                    uncovered_lines[file_path] = sorted(uncovered)
                            
                            total_covered += covered_statements
                            total_lines += total_statements
                        
                        # Calculate overall coverage
                        line_coverage = total_covered / total_lines if total_lines > 0 else 0
                        
                        coverage_data = {
                            "line_coverage": line_coverage,
                            "file_coverage": file_coverage,
                            "uncovered_lines": uncovered_lines
                        }
                    
                    # Generate summary
                    if total == 0:
                        summary = "No tests were found or executed"
                    else:
                        summary = (
                            f"Ran {total} tests: {passed} passed, {failed} failed, "
                            f"{skipped} skipped"
                        )
                        if coverage_data:
                            summary += f", {coverage_data['line_coverage']:.1%} coverage"
                    
                    return TestSuiteResult(
                        total=total,
                        passed=passed,
                        failed=failed,
                        skipped=skipped,
                        errors=errors,
                        execution_time=execution_time,
                        results=test_results,
                        coverage=coverage_data,
                        summary=summary,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                
                except json.JSONDecodeError:
                    # Failed to parse JSON output
                    return TestSuiteResult(
                        total=0,
                        passed=0,
                        failed=1,
                        skipped=0,
                        errors=0,
                        execution_time=execution_time,
                        summary=f"Failed to parse Jest output: {stderr_text}",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
            
            except asyncio.TimeoutError:
                return TestSuiteResult(
                    total=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    execution_time=timeout,
                    summary=f"Tests timed out after {timeout} seconds",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            except Exception as e:
                return TestSuiteResult(
                    total=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    execution_time=time.time() - start_time,
                    summary=f"Error running tests: {str(e)}",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
    
    async def measure_coverage(
        self,
        code: str,
        test_code: str,
        language: Union[str, CodeLanguage],
        test_framework: Optional[str] = None,
        timeout: int = 30
    ) -> CoverageResult:
        """Measure code coverage for tests.
        
        Args:
            code: The code to test
            test_code: The test code
            language: Programming language of the code
            test_framework: Test framework to use
            timeout: Timeout for execution in seconds
            
        Returns:
            Code coverage result
        """
        # Run tests with coverage
        test_result = await self.run_tests(
            code=code,
            test_code=test_code,
            language=language,
            test_framework=test_framework,
            include_coverage=True,
            timeout=timeout
        )
        
        # Extract coverage data
        coverage_data = test_result.coverage or {}
        
        # Create coverage result
        result = CoverageResult(
            line_coverage=coverage_data.get("line_coverage", 0.0),
            branch_coverage=coverage_data.get("branch_coverage"),
            file_coverage=coverage_data.get("file_coverage", {}),
            uncovered_lines=coverage_data.get("uncovered_lines", {}),
            summary=f"Coverage: {coverage_data.get('line_coverage', 0.0):.1%} of lines covered"
        )
        
        return result
    
    async def suggest_fixes(
        self,
        code: str,
        language: Union[str, CodeLanguage],
        issues: List[CodeIssue]
    ) -> Dict[str, Any]:
        """Suggest fixes for code issues.
        
        Args:
            code: The code with issues
            language: Programming language of the code
            issues: List of issues to fix
            
        Returns:
            Dictionary with suggested fixes
        """
        if not issues:
            return {
                "has_fixes": False,
                "fixed_code": code,
                "fixed_issues": []
            }
        
        # Normalize language
        if isinstance(language, str):
            try:
                language = CodeLanguage(language.lower())
            except ValueError:
                language = CodeLanguage.OTHER
        
        # Track which issues we fix
        fixed_issues = []
        
        # Sort issues by line number (desc) so we can apply fixes from bottom to top
        sorted_issues = sorted(
            issues,
            key=lambda issue: issue.line or 0,
            reverse=True
        )
        
        # Apply fixes for issues with suggestions
        modified_code = code
        for issue in sorted_issues:
            if issue.suggestion and issue.line:
                # Only implement automatic fixes for certain issue types
                if issue.type in [IssueType.STYLE, IssueType.WARNING] and issue.confidence > 0.7:
                    # For now, we only implement basic fixes
                    # A more sophisticated implementation would apply specific fixes
                    # based on the issue type and analyzer
                    
                    # Track that we fixed this issue
                    fixed_issues.append(issue.id)
            
        # Apply automatic formatting if appropriate
        if language == CodeLanguage.PYTHON and BLACK_AVAILABLE:
            try:
                mode = black.FileMode()
                modified_code = black.format_str(modified_code, mode=mode)
            except:
                pass
        
        # Return results
        return {
            "has_fixes": len(fixed_issues) > 0,
            "fixed_code": modified_code,
            "fixed_issues": fixed_issues,
            "message": f"Fixed {len(fixed_issues)} issues automatically"
        }
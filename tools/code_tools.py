"""
Code generation and analysis tools for the multi-agent development system.

This module provides tools for code generation, analysis, transformation, and
validation. It supports various programming languages and frameworks, and
integrates with version control systems and development environments.
"""

import ast
import asyncio
import difflib
import importlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import uuid
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

try:
    import isort
    ISORT_AVAILABLE = True
except ImportError:
    ISORT_AVAILABLE = False

try:
    import pylint.lint
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    import yapf
    YAPF_AVAILABLE = True
except ImportError:
    YAPF_AVAILABLE = False

try:
    from lxml import etree
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

try:
    import libcst as cst
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False

from pydantic import BaseModel, Field, validator

# Set up logging
logger = logging.getLogger(__name__)


class CodeLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    OTHER = "other"


class CodeFragment(BaseModel):
    """A fragment of code with metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    code: str
    language: CodeLanguage
    path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    symbols: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodeAnalysisResult(BaseModel):
    """Results of code analysis."""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    structure: Dict[str, Any] = Field(default_factory=dict)
    imports: List[str] = Field(default_factory=list)
    symbols: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class CodeTransformResult(BaseModel):
    """Results of code transformation."""
    original_code: str
    transformed_code: str
    diff: str
    changes: List[Dict[str, Any]] = Field(default_factory=list)
    language: CodeLanguage
    metrics: Dict[str, Any] = Field(default_factory=dict)


class CodeValidationResult(BaseModel):
    """Results of code validation."""
    is_valid: bool
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class TestResult(BaseModel):
    """Results of code testing."""
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    coverage: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class LanguageFeatures(BaseModel):
    """Features supported by a programming language."""
    language: CodeLanguage
    formatter_available: bool = False
    linter_available: bool = False
    analyzer_available: bool = False
    test_runner_available: bool = False
    imports_analyzer_available: bool = False
    symbolic_execution_available: bool = False
    auto_completion_available: bool = False
    documentation_generator_available: bool = False


class CodeTool:
    """Base class for code tools."""
    
    def __init__(
        self, 
        working_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        max_workers: int = 4,
        timeout: int = 30,
        language_features: Optional[Dict[CodeLanguage, LanguageFeatures]] = None
    ):
        """Initialize the code tool.
        
        Args:
            working_dir: Working directory for code operations
            temp_dir: Temporary directory for code operations
            max_workers: Maximum number of worker threads
            timeout: Default timeout in seconds
            language_features: Features supported by programming languages
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "code_tools"
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Set up language features
        self.language_features = language_features or self._default_language_features()
    
    def _default_language_features(self) -> Dict[CodeLanguage, LanguageFeatures]:
        """Define default language features."""
        features = {}
        
        # Python features
        features[CodeLanguage.PYTHON] = LanguageFeatures(
            language=CodeLanguage.PYTHON,
            formatter_available=BLACK_AVAILABLE or YAPF_AVAILABLE,
            linter_available=PYLINT_AVAILABLE,
            analyzer_available=True,
            test_runner_available=True,
            imports_analyzer_available=True,
            symbolic_execution_available=LIBCST_AVAILABLE,
            auto_completion_available=True,
            documentation_generator_available=True
        )
        
        # JavaScript features
        features[CodeLanguage.JAVASCRIPT] = LanguageFeatures(
            language=CodeLanguage.JAVASCRIPT,
            formatter_available=self._check_command("prettier"),
            linter_available=self._check_command("eslint"),
            analyzer_available=True,
            test_runner_available=self._check_command("jest") or self._check_command("mocha"),
            imports_analyzer_available=True,
            symbolic_execution_available=False,
            auto_completion_available=True,
            documentation_generator_available=self._check_command("jsdoc")
        )
        
        # TypeScript features
        features[CodeLanguage.TYPESCRIPT] = LanguageFeatures(
            language=CodeLanguage.TYPESCRIPT,
            formatter_available=self._check_command("prettier"),
            linter_available=self._check_command("eslint") or self._check_command("tslint"),
            analyzer_available=True,
            test_runner_available=self._check_command("jest") or self._check_command("mocha"),
            imports_analyzer_available=True,
            symbolic_execution_available=False,
            auto_completion_available=True,
            documentation_generator_available=self._check_command("typedoc")
        )
        
        # Add minimal support for other languages
        for lang in CodeLanguage:
            if lang not in features:
                features[lang] = LanguageFeatures(
                    language=lang,
                    formatter_available=False,
                    linter_available=False,
                    analyzer_available=False,
                    test_runner_available=False,
                    imports_analyzer_available=False,
                    symbolic_execution_available=False,
                    auto_completion_available=False,
                    documentation_generator_available=False
                )
        
        return features
    
    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available on the system."""
        try:
            proc = subprocess.Popen(
                [cmd, "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            proc.communicate(timeout=2)
            return proc.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    async def analyze_code(
        self, 
        code: str, 
        language: Union[str, CodeLanguage],
        path: Optional[str] = None,
        metrics: List[str] = ["complexity", "loc", "imports"],
        detailed: bool = False
    ) -> CodeAnalysisResult:
        """Analyze code and gather metrics and insights.
        
        Args:
            code: The code to analyze
            language: Programming language of the code
            path: Optional file path for the code
            metrics: Specific metrics to calculate
            detailed: Whether to perform detailed analysis
            
        Returns:
            Results of the analysis
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        # Select appropriate analyzer based on language
        if language == CodeLanguage.PYTHON:
            return await self._analyze_python(code, path, metrics, detailed)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            return await self._analyze_javascript(code, path, metrics, detailed)
        else:
            # Basic analysis for other languages
            return await self._analyze_generic(code, language, path, metrics)
    
    async def _analyze_python(
        self, 
        code: str, 
        path: Optional[str] = None,
        metrics: List[str] = ["complexity", "loc", "imports"],
        detailed: bool = False
    ) -> CodeAnalysisResult:
        """Analyze Python code.
        
        Args:
            code: The Python code to analyze
            path: Optional file path for the code
            metrics: Specific metrics to calculate
            detailed: Whether to perform detailed analysis
            
        Returns:
            Results of the analysis
        """
        result = CodeAnalysisResult()
        result.metrics["language"] = "python"
        
        # Basic metrics
        lines = code.splitlines()
        result.metrics["total_lines"] = len(lines)
        result.metrics["non_empty_lines"] = sum(1 for line in lines if line.strip())
        result.metrics["comment_lines"] = sum(1 for line in lines if line.strip().startswith("#"))
        
        # Try to parse with ast
        try:
            tree = ast.parse(code)
            
            # Collect imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        imports.append(f"{module}.{name.name}" if module else name.name)
            
            result.imports = imports
            
            # Collect function and class definitions
            functions = {}
            classes = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    functions[func_name] = {
                        "name": func_name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [ast.unparse(d).strip() for d in node.decorator_list],
                        "is_async": isinstance(node, ast.AsyncFunctionDef)
                    }
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append(child.name)
                    
                    classes[class_name] = {
                        "name": class_name,
                        "line": node.lineno,
                        "methods": methods,
                        "bases": [ast.unparse(b).strip() for b in node.bases]
                    }
            
            # Add to structure
            result.structure = {
                "functions": functions,
                "classes": classes
            }
            
            # Add symbols
            result.symbols = {
                "functions": list(functions.keys()),
                "classes": list(classes.keys())
            }
            
            # Calculate complexity metrics if requested
            if "complexity" in metrics and detailed:
                import radon.complexity
                
                try:
                    cc_results = radon.complexity.cc_visit(code)
                    result.metrics["complexity"] = {
                        "average": sum(cc.complexity for cc in cc_results) / len(cc_results) if cc_results else 0,
                        "functions": {
                            cc.name: {
                                "complexity": cc.complexity,
                                "rank": cc.rank
                            } for cc in cc_results
                        }
                    }
                except Exception as e:
                    logger.warning(f"Error calculating complexity metrics: {str(e)}")
            
            # Generate summary
            class_count = len(classes)
            function_count = len(functions)
            result.summary = (
                f"Python code with {class_count} classes and {function_count} functions. "
                f"Uses {len(imports)} imports. "
                f"{result.metrics['total_lines']} total lines."
            )
            
        except SyntaxError as e:
            # Handle syntax errors
            result.issues.append({
                "type": "syntax_error",
                "message": str(e),
                "line": e.lineno,
                "offset": e.offset,
                "severity": "error"
            })
            
            result.summary = "Python code with syntax errors."
        
        # Run linting if available and detailed analysis requested
        if PYLINT_AVAILABLE and detailed:
            try:
                # Create a temporary file for linting
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                
                try:
                    from io import StringIO
                    import pylint.lint
                    
                    # Capture pylint output
                    old_stdout = sys.stdout
                    redirected_output = StringIO()
                    sys.stdout = redirected_output
                    
                    # Run pylint
                    pylint.lint.Run([tmp_path], exit=False)
                    
                    # Restore stdout
                    sys.stdout = old_stdout
                    
                    # Parse output for issues
                    for line in redirected_output.getvalue().splitlines():
                        if ":" in line and len(line.split(":")) >= 3:
                            parts = line.split(":")
                            if parts[0] == tmp_path:
                                try:
                                    line_num = int(parts[1])
                                    message = ":".join(parts[2:]).strip()
                                    
                                    # Determine severity
                                    severity = "warning"
                                    if "[E" in message:
                                        severity = "error"
                                    elif "[W" in message:
                                        severity = "warning"
                                    elif "[C" in message:
                                        severity = "convention"
                                    elif "[R" in message:
                                        severity = "refactor"
                                    
                                    result.issues.append({
                                        "type": "lint",
                                        "message": message,
                                        "line": line_num,
                                        "severity": severity
                                    })
                                except ValueError:
                                    pass
                finally:
                    # Clean up
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Error running pylint: {str(e)}")
        
        return result
    
    async def _analyze_javascript(
        self, 
        code: str, 
        path: Optional[str] = None,
        metrics: List[str] = ["complexity", "loc", "imports"],
        detailed: bool = False
    ) -> CodeAnalysisResult:
        """Analyze JavaScript/TypeScript code.
        
        Args:
            code: The JavaScript/TypeScript code to analyze
            path: Optional file path for the code
            metrics: Specific metrics to calculate
            detailed: Whether to perform detailed analysis
            
        Returns:
            Results of the analysis
        """
        result = CodeAnalysisResult()
        result.metrics["language"] = "javascript/typescript"
        
        # Basic metrics
        lines = code.splitlines()
        result.metrics["total_lines"] = len(lines)
        result.metrics["non_empty_lines"] = sum(1 for line in lines if line.strip())
        result.metrics["comment_lines"] = sum(1 for line in lines if line.strip().startswith("//"))
        
        # Regular expressions for basic analysis
        import_regex = r'import\s+(?:{[^}]*}|[^{;]*)(?:\s+from\s+)?[\'"]([^\'"]+)[\'"]'
        require_regex = r'(?:const|let|var)\s+(?:{[^}]*}|[^{;]*)\s*=\s*require\([\'"]([^\'"]+)[\'"]\)'
        function_regex = r'(?:function|async\s+function)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        class_regex = r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        
        # Find imports
        imports = []
        for match in re.finditer(import_regex, code):
            imports.append(match.group(1))
        for match in re.finditer(require_regex, code):
            imports.append(match.group(1))
        
        result.imports = imports
        
        # Find functions and classes
        functions = {}
        for match in re.finditer(function_regex, code):
            func_name = match.group(1)
            functions[func_name] = {
                "name": func_name,
                "line": len(code[:match.start()].splitlines()) + 1
            }
        
        classes = {}
        for match in re.finditer(class_regex, code):
            class_name = match.group(1)
            classes[class_name] = {
                "name": class_name,
                "line": len(code[:match.start()].splitlines()) + 1
            }
        
        # Add to structure
        result.structure = {
            "functions": functions,
            "classes": classes
        }
        
        # Add symbols
        result.symbols = {
            "functions": list(functions.keys()),
            "classes": list(classes.keys())
        }
        
        # Generate summary
        class_count = len(classes)
        function_count = len(functions)
        result.summary = (
            f"JavaScript/TypeScript code with {class_count} classes and {function_count} functions. "
            f"Uses {len(imports)} imports/requires. "
            f"{result.metrics['total_lines']} total lines."
        )
        
        # Run ESLint if available and requested
        if detailed and self._check_command("eslint"):
            try:
                # Create a temporary file
                ext = ".ts" if path and path.endswith(".ts") else ".js"
                with tempfile.NamedTemporaryFile("w", suffix=ext, delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                
                try:
                    # Run ESLint
                    proc = subprocess.Popen(
                        ["eslint", "--format", "json", tmp_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    stdout, stderr = proc.communicate(timeout=10)
                    
                    if stdout:
                        try:
                            eslint_results = json.loads(stdout)
                            for file_result in eslint_results:
                                for message in file_result.get("messages", []):
                                    result.issues.append({
                                        "type": "lint",
                                        "message": message.get("message", "Unknown issue"),
                                        "line": message.get("line", 0),
                                        "column": message.get("column", 0),
                                        "rule": message.get("ruleId", "unknown-rule"),
                                        "severity": "error" if message.get("severity") == 2 else "warning"
                                    })
                        except json.JSONDecodeError:
                            pass
                finally:
                    # Clean up
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Error running ESLint: {str(e)}")
        
        return result
    
    async def _analyze_generic(
        self, 
        code: str, 
        language: CodeLanguage,
        path: Optional[str] = None,
        metrics: List[str] = ["loc"]
    ) -> CodeAnalysisResult:
        """Analyze code in any language with basic metrics.
        
        Args:
            code: The code to analyze
            language: Programming language of the code
            path: Optional file path for the code
            metrics: Specific metrics to calculate
            
        Returns:
            Results of the analysis
        """
        result = CodeAnalysisResult()
        result.metrics["language"] = language.value
        
        # Basic metrics
        lines = code.splitlines()
        result.metrics["total_lines"] = len(lines)
        result.metrics["non_empty_lines"] = sum(1 for line in lines if line.strip())
        
        # Language-specific comment detection
        comment_lines = 0
        comment_markers = {
            CodeLanguage.PYTHON: ["#"],
            CodeLanguage.JAVASCRIPT: ["//", "/*"],
            CodeLanguage.TYPESCRIPT: ["//", "/*"],
            CodeLanguage.JAVA: ["//", "/*"],
            CodeLanguage.CSHARP: ["//", "/*"],
            CodeLanguage.CPP: ["//", "/*"],
            CodeLanguage.GO: ["//"],
            CodeLanguage.RUST: ["//", "/*"],
            CodeLanguage.PHP: ["//", "#", "/*"],
            CodeLanguage.RUBY: ["#"],
            CodeLanguage.SWIFT: ["//", "/*"],
            CodeLanguage.KOTLIN: ["//", "/*"],
            CodeLanguage.HTML: ["<!--"],
            CodeLanguage.CSS: ["/*"],
            CodeLanguage.SQL: ["--", "/*"],
            CodeLanguage.BASH: ["#"],
            CodeLanguage.MARKDOWN: [],
            CodeLanguage.JSON: [],
            CodeLanguage.YAML: ["#"],
            CodeLanguage.XML: ["<!--"],
        }
        
        markers = comment_markers.get(language, ["#", "//"])
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(marker) for marker in markers):
                comment_lines += 1
        
        result.metrics["comment_lines"] = comment_lines
        
        # Generate summary
        result.summary = (
            f"{language.value.capitalize()} code with {result.metrics['total_lines']} total lines, "
            f"{result.metrics['non_empty_lines']} non-empty lines, and "
            f"{comment_lines} comment lines."
        )
        
        return result
    
    async def generate_code(
        self,
        prompt: str,
        language: Union[str, CodeLanguage],
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None,
        target_framework: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.2
    ) -> CodeFragment:
        """Generate code based on a prompt.
        
        Note: This is a placeholder method. In a real implementation, this would
        integrate with a language model API such as OpenAI's GPT or similar.
        
        Args:
            prompt: The prompt describing the code to generate
            language: Target programming language
            context: Additional context for generation
            constraints: Constraints for the generated code
            target_framework: Target framework if applicable
            examples: Example code for few-shot learning
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (creativity)
            
        Returns:
            Generated code fragment
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        # This is a placeholder for actual code generation
        logger.info(f"Would generate {language.value} code for prompt: {prompt[:100]}...")
        
        # In a real implementation, this would call an LLM API
        # For now, just return a placeholder
        fragment = CodeFragment(
            code=f"# Generated {language.value} code would appear here\n# Based on: {prompt[:50]}...",
            language=language,
            metadata={
                "prompt": prompt,
                "constraints": constraints,
                "target_framework": target_framework,
                "generation_timestamp": time.time()
            }
        )
        
        return fragment
    
    async def format_code(
        self,
        code: str,
        language: Union[str, CodeLanguage],
        style: Optional[str] = None
    ) -> CodeTransformResult:
        """Format code according to language-specific standards.
        
        Args:
            code: The code to format
            language: Programming language of the code
            style: Optional style guide to follow
            
        Returns:
            Formatted code with transformation details
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        # Select appropriate formatter based on language
        if language == CodeLanguage.PYTHON:
            return await self._format_python(code, style)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            return await self._format_javascript(code, style)
        else:
            # For unsupported languages, return unchanged
            return CodeTransformResult(
                original_code=code,
                transformed_code=code,
                diff="",
                language=language,
                metrics={"unchanged": True}
            )
    
    async def _format_python(
        self,
        code: str,
        style: Optional[str] = None
    ) -> CodeTransformResult:
        """Format Python code.
        
        Args:
            code: The Python code to format
            style: Optional style guide to follow
            
        Returns:
            Formatted code
        """
        original_code = code
        transformed_code = code
        
        # Try black first if available
        if BLACK_AVAILABLE:
            try:
                # Use black to format the code
                import black
                
                # Configure black options
                line_length = 88  # black default
                if style == "google":
                    line_length = 80
                elif style == "pep8":
                    line_length = 79
                
                mode = black.FileMode(line_length=line_length)
                transformed_code = black.format_str(code, mode=mode)
            except Exception as e:
                logger.warning(f"Error formatting with black: {str(e)}")
        
        # Try YAPF if black failed or not available
        elif YAPF_AVAILABLE:
            try:
                import yapf.yapflib.yapf_api
                
                # Configure YAPF style
                style_config = "pep8"
                if style:
                    style_config = style
                
                transformed_code, _ = yapf.yapflib.yapf_api.FormatCode(
                    code,
                    style_config=style_config
                )
            except Exception as e:
                logger.warning(f"Error formatting with yapf: {str(e)}")
        
        # Apply isort to organize imports if available
        if ISORT_AVAILABLE and transformed_code != original_code:
            try:
                import isort
                
                isort_config = {"line_length": 88, "profile": "black"}
                if style == "google":
                    isort_config = {"line_length": 80, "profile": "google"}
                elif style == "pep8":
                    isort_config = {"line_length": 79, "profile": "pep8"}
                
                transformed_code = isort.code(transformed_code, **isort_config)
            except Exception as e:
                logger.warning(f"Error organizing imports with isort: {str(e)}")
        
        # Generate diff
        diff = "\n".join(difflib.unified_diff(
            original_code.splitlines(),
            transformed_code.splitlines(),
            fromfile="original",
            tofile="formatted",
            lineterm=""
        ))
        
        # Calculate metrics
        metrics = {
            "original_lines": len(original_code.splitlines()),
            "formatted_lines": len(transformed_code.splitlines()),
            "changed": transformed_code != original_code
        }
        
        # Identify changes
        changes = []
        if transformed_code != original_code:
            try:
                # Parse diffs to identify changes
                current_line = 0
                change_desc = None
                
                for line in diff.splitlines():
                    if line.startswith("@@"):
                        # New hunk
                        match = re.search(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
                        if match:
                            current_line = int(match.group(3))
                            if change_desc:
                                changes.append(change_desc)
                            change_desc = {
                                "type": "formatting",
                                "line_start": current_line,
                                "line_end": current_line + int(match.group(4)),
                                "details": []
                            }
                    elif line.startswith("+") and not line.startswith("+++"):
                        # Added line
                        if change_desc:
                            change_desc["details"].append(f"Added: {line[1:]}")
                        current_line += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        # Removed line
                        if change_desc:
                            change_desc["details"].append(f"Removed: {line[1:]}")
                    else:
                        current_line += 1
                
                if change_desc:
                    changes.append(change_desc)
            except Exception as e:
                logger.warning(f"Error analyzing changes: {str(e)}")
        
        return CodeTransformResult(
            original_code=original_code,
            transformed_code=transformed_code,
            diff=diff,
            changes=changes,
            language=CodeLanguage.PYTHON,
            metrics=metrics
        )
    
    async def _format_javascript(
        self,
        code: str,
        style: Optional[str] = None
    ) -> CodeTransformResult:
        """Format JavaScript/TypeScript code.
        
        Args:
            code: The JavaScript/TypeScript code to format
            style: Optional style guide to follow
            
        Returns:
            Formatted code
        """
        original_code = code
        transformed_code = code
        language = CodeLanguage.JAVASCRIPT
        
        # Use Prettier if available
        if self._check_command("prettier"):
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                
                try:
                    # Configure Prettier options
                    prettier_args = ["prettier", "--write"]
                    
                    if style == "airbnb":
                        prettier_args.extend(["--print-width", "100", "--single-quote"])
                    elif style == "standard":
                        prettier_args.extend(["--print-width", "80", "--single-quote", "--semi", "false"])
                    elif style == "google":
                        prettier_args.extend(["--print-width", "80", "--tab-width", "2"])
                    
                    prettier_args.append(tmp_path)
                    
                    # Run Prettier
                    proc = subprocess.Popen(
                        prettier_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    _, stderr = proc.communicate(timeout=10)
                    
                    if proc.returncode == 0:
                        # Read the formatted code
                        with open(tmp_path, "r") as f:
                            transformed_code = f.read()
                    else:
                        logger.warning(f"Prettier failed: {stderr.decode('utf-8')}")
                finally:
                    # Clean up
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Error formatting with prettier: {str(e)}")
        
        # Generate diff
        diff = "\n".join(difflib.unified_diff(
            original_code.splitlines(),
            transformed_code.splitlines(),
            fromfile="original",
            tofile="formatted",
            lineterm=""
        ))
        
        # Calculate metrics
        metrics = {
            "original_lines": len(original_code.splitlines()),
            "formatted_lines": len(transformed_code.splitlines()),
            "changed": transformed_code != original_code
        }
        
        # Identify changes (simplified)
        changes = []
        if transformed_code != original_code:
            changes.append({
                "type": "formatting",
                "description": "Applied code formatting",
                "details": ["Applied JavaScript/TypeScript formatting standards"]
            })
        
        return CodeTransformResult(
            original_code=original_code,
            transformed_code=transformed_code,
            diff=diff,
            changes=changes,
            language=language,
            metrics=metrics
        )
    
    async def validate_code(
        self,
        code: str,
        language: Union[str, CodeLanguage],
        rules: Optional[List[Dict[str, Any]]] = None,
        strict: bool = False
    ) -> CodeValidationResult:
        """Validate code against rules and best practices.
        
        Args:
            code: The code to validate
            language: Programming language of the code
            rules: Optional custom validation rules
            strict: Whether to apply strict validation
            
        Returns:
            Validation results
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        # Select appropriate validator based on language
        if language == CodeLanguage.PYTHON:
            return await self._validate_python(code, rules, strict)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            return await self._validate_javascript(code, rules, strict)
        else:
            # For unsupported languages, do basic validation
            return await self._validate_generic(code, language, rules, strict)
    
    async def _validate_python(
        self,
        code: str,
        rules: Optional[List[Dict[str, Any]]] = None,
        strict: bool = False
    ) -> CodeValidationResult:
        """Validate Python code.
        
        Args:
            code: The Python code to validate
            rules: Optional custom validation rules
            strict: Whether to apply strict validation
            
        Returns:
            Validation results
        """
        result = CodeValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            suggestions=[],
            metrics={}
        )
        
        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            result.is_valid = False
            result.issues.append({
                "type": "syntax_error",
                "message": str(e),
                "line": e.lineno,
                "offset": e.offset,
                "severity": "error"
            })
        
        # Run pylint if available
        if PYLINT_AVAILABLE:
            try:
                # Create a temporary file for linting
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                
                try:
                    from io import StringIO
                    import pylint.lint
                    
                    # Capture pylint output
                    old_stdout = sys.stdout
                    redirected_output = StringIO()
                    sys.stdout = redirected_output
                    
                    # Run pylint
                    pylint.lint.Run([tmp_path], exit=False)
                    
                    # Restore stdout
                    sys.stdout = old_stdout
                    
                    # Parse output for issues
                    for line in redirected_output.getvalue().splitlines():
                        if ":" in line and len(line.split(":")) >= 3:
                            parts = line.split(":")
                            if parts[0] == tmp_path:
                                try:
                                    line_num = int(parts[1])
                                    message = ":".join(parts[2:]).strip()
                                    
                                    # Determine severity
                                    severity = "warning"
                                    if "[E" in message:
                                        severity = "error"
                                        if strict:  # In strict mode, errors invalidate the code
                                            result.is_valid = False
                                        result.issues.append({
                                            "type": "lint",
                                            "message": message,
                                            "line": line_num,
                                            "severity": severity
                                        })
                                    elif "[W" in message:
                                        severity = "warning"
                                        result.warnings.append({
                                            "type": "lint",
                                            "message": message,
                                            "line": line_num,
                                            "severity": severity
                                        })
                                    elif "[C" in message or "[R" in message:
                                        # Convention and refactoring suggestions
                                        result.suggestions.append({
                                            "type": "lint",
                                            "message": message,
                                            "line": line_num,
                                            "severity": "suggestion"
                                        })
                                except ValueError:
                                    pass
                finally:
                    # Clean up
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Error running pylint: {str(e)}")
        
        # Apply custom rules if provided
        if rules:
            for rule in rules:
                rule_type = rule.get("type", "")
                pattern = rule.get("pattern", "")
                message = rule.get("message", "Custom rule violation")
                severity = rule.get("severity", "warning")
                
                if rule_type == "pattern" and pattern:
                    # Check for pattern in code
                    matches = list(re.finditer(pattern, code))
                    for match in matches:
                        # Calculate line number
                        line_num = len(code[:match.start()].splitlines()) + 1
                        
                        if severity == "error":
                            if strict:
                                result.is_valid = False
                            result.issues.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": severity
                            })
                        elif severity == "warning":
                            result.warnings.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": severity
                            })
                        else:
                            result.suggestions.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": "suggestion"
                            })
        
        # Calculate metrics
        result.metrics = {
            "issues_count": len(result.issues),
            "warnings_count": len(result.warnings),
            "suggestions_count": len(result.suggestions)
        }
        
        return result
    
    async def _validate_javascript(
        self,
        code: str,
        rules: Optional[List[Dict[str, Any]]] = None,
        strict: bool = False
    ) -> CodeValidationResult:
        """Validate JavaScript/TypeScript code.
        
        Args:
            code: The JavaScript/TypeScript code to validate
            rules: Optional custom validation rules
            strict: Whether to apply strict validation
            
        Returns:
            Validation results
        """
        result = CodeValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            suggestions=[],
            metrics={}
        )
        
        # Check for syntax errors
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            try:
                # Run Node.js to check syntax
                proc = subprocess.Popen(
                    ["node", "--check", tmp_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                _, stderr = proc.communicate(timeout=10)
                
                if proc.returncode != 0:
                    # Syntax error found
                    result.is_valid = False
                    
                    error_text = stderr.decode('utf-8')
                    # Parse error message to extract line and column
                    match = re.search(r"(\w+): (.+)\n\s+at\s+.+:(\d+):(\d+)", error_text)
                    if match:
                        error_type, message, line, column = match.groups()
                        result.issues.append({
                            "type": "syntax_error",
                            "message": f"{error_type}: {message}",
                            "line": int(line),
                            "column": int(column),
                            "severity": "error"
                        })
                    else:
                        result.issues.append({
                            "type": "syntax_error",
                            "message": error_text,
                            "severity": "error"
                        })
            finally:
                # Clean up
                os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"Error checking JavaScript syntax: {str(e)}")
            result.warnings.append({
                "type": "process_error",
                "message": f"Could not validate syntax: {str(e)}",
                "severity": "warning"
            })
        
        # Run ESLint if available
        if self._check_command("eslint"):
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                
                try:
                    # Run ESLint
                    proc = subprocess.Popen(
                        ["eslint", "--format", "json", tmp_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    stdout, _ = proc.communicate(timeout=10)
                    
                    if stdout:
                        try:
                            eslint_results = json.loads(stdout)
                            for file_result in eslint_results:
                                for message in file_result.get("messages", []):
                                    issue = {
                                        "type": "lint",
                                        "message": message.get("message", "Unknown issue"),
                                        "line": message.get("line", 0),
                                        "column": message.get("column", 0),
                                        "rule": message.get("ruleId", "unknown-rule")
                                    }
                                    
                                    if message.get("severity") == 2:  # Error
                                        issue["severity"] = "error"
                                        if strict:
                                            result.is_valid = False
                                        result.issues.append(issue)
                                    elif message.get("severity") == 1:  # Warning
                                        issue["severity"] = "warning"
                                        result.warnings.append(issue)
                                    else:
                                        issue["severity"] = "suggestion"
                                        result.suggestions.append(issue)
                        except json.JSONDecodeError:
                            pass
                finally:
                    # Clean up
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Error running ESLint: {str(e)}")
        
        # Apply custom rules if provided
        if rules:
            for rule in rules:
                rule_type = rule.get("type", "")
                pattern = rule.get("pattern", "")
                message = rule.get("message", "Custom rule violation")
                severity = rule.get("severity", "warning")
                
                if rule_type == "pattern" and pattern:
                    # Check for pattern in code
                    matches = list(re.finditer(pattern, code))
                    for match in matches:
                        # Calculate line number
                        line_num = len(code[:match.start()].splitlines()) + 1
                        
                        if severity == "error":
                            if strict:
                                result.is_valid = False
                            result.issues.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": severity
                            })
                        elif severity == "warning":
                            result.warnings.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": severity
                            })
                        else:
                            result.suggestions.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": "suggestion"
                            })
        
        # Calculate metrics
        result.metrics = {
            "issues_count": len(result.issues),
            "warnings_count": len(result.warnings),
            "suggestions_count": len(result.suggestions)
        }
        
        return result
    
    async def _validate_generic(
        self,
        code: str,
        language: CodeLanguage,
        rules: Optional[List[Dict[str, Any]]] = None,
        strict: bool = False
    ) -> CodeValidationResult:
        """Validate code in any language with basic checks.
        
        Args:
            code: The code to validate
            language: Programming language of the code
            rules: Optional custom validation rules
            strict: Whether to apply strict validation
            
        Returns:
            Validation results
        """
        result = CodeValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            suggestions=[],
            metrics={}
        )
        
        # Basic validation - check for common issues
        
        # Check for very long lines
        long_lines = []
        for i, line in enumerate(code.splitlines()):
            if len(line) > 120:  # Arbitrary threshold
                long_lines.append(i + 1)
        
        if long_lines:
            result.warnings.append({
                "type": "style",
                "message": f"Found {len(long_lines)} lines exceeding 120 characters",
                "lines": long_lines,
                "severity": "warning"
            })
        
        # Check for trailing whitespace
        whitespace_lines = []
        for i, line in enumerate(code.splitlines()):
            if line.rstrip() != line:
                whitespace_lines.append(i + 1)
        
        if whitespace_lines:
            result.suggestions.append({
                "type": "style",
                "message": f"Found {len(whitespace_lines)} lines with trailing whitespace",
                "lines": whitespace_lines,
                "severity": "suggestion"
            })
        
        # Check for mixed tabs and spaces
        indentation_type = None
        mixed_indentation_lines = []
        
        for i, line in enumerate(code.splitlines()):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check indentation
            leading_whitespace = line[:len(line) - len(line.lstrip())]
            if leading_whitespace:
                current_type = "tab" if "\t" in leading_whitespace else "space"
                
                if indentation_type is None:
                    indentation_type = current_type
                elif indentation_type != current_type:
                    mixed_indentation_lines.append(i + 1)
        
        if mixed_indentation_lines:
            result.warnings.append({
                "type": "style",
                "message": f"Found {len(mixed_indentation_lines)} lines with mixed indentation (tabs and spaces)",
                "lines": mixed_indentation_lines,
                "severity": "warning"
            })
        
        # Apply custom rules if provided
        if rules:
            for rule in rules:
                rule_type = rule.get("type", "")
                pattern = rule.get("pattern", "")
                message = rule.get("message", "Custom rule violation")
                severity = rule.get("severity", "warning")
                
                if rule_type == "pattern" and pattern:
                    # Check for pattern in code
                    matches = list(re.finditer(pattern, code))
                    for match in matches:
                        # Calculate line number
                        line_num = len(code[:match.start()].splitlines()) + 1
                        
                        if severity == "error":
                            if strict:
                                result.is_valid = False
                            result.issues.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": severity
                            })
                        elif severity == "warning":
                            result.warnings.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": severity
                            })
                        else:
                            result.suggestions.append({
                                "type": "custom_rule",
                                "message": message,
                                "line": line_num,
                                "severity": "suggestion"
                            })
        
        # Calculate metrics
        result.metrics = {
            "issues_count": len(result.issues),
            "warnings_count": len(result.warnings),
            "suggestions_count": len(result.suggestions),
            "long_lines": len(long_lines),
            "trailing_whitespace_lines": len(whitespace_lines),
            "mixed_indentation_lines": len(mixed_indentation_lines)
        }
        
        return result
    
    async def extract_dependencies(
        self,
        code: str,
        language: Union[str, CodeLanguage],
        include_builtin: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract dependencies from code.
        
        Args:
            code: The code to analyze
            language: Programming language of the code
            include_builtin: Whether to include built-in dependencies
            
        Returns:
            List of dependencies with metadata
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        # Select appropriate analyzer based on language
        if language == CodeLanguage.PYTHON:
            return await self._extract_python_dependencies(code, include_builtin)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            return await self._extract_js_dependencies(code, include_builtin)
        else:
            # Basic extraction for other languages
            return await self._extract_generic_dependencies(code, language)
    
    async def _extract_python_dependencies(
        self,
        code: str,
        include_builtin: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract dependencies from Python code.
        
        Args:
            code: The Python code to analyze
            include_builtin: Whether to include built-in dependencies
            
        Returns:
            List of dependencies with metadata
        """
        dependencies = []
        built_in_modules = set(sys.builtin_module_names)
        standard_lib = set(sys.modules)
        
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Collect imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        base_name = name.name.split('.')[0]
                        is_builtin = base_name in built_in_modules
                        is_std_lib = base_name in standard_lib and not is_builtin
                        
                        if include_builtin or (not is_builtin and not is_std_lib):
                            dependencies.append({
                                "name": name.name,
                                "alias": name.asname,
                                "is_builtin": is_builtin,
                                "is_standard_lib": is_std_lib,
                                "line": node.lineno,
                                "type": "import"
                            })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    base_name = module.split('.')[0] if module else ""
                    is_builtin = base_name in built_in_modules
                    is_std_lib = base_name in standard_lib and not is_builtin
                    
                    if include_builtin or (not is_builtin and not is_std_lib):
                        for name in node.names:
                            dependencies.append({
                                "name": f"{module}.{name.name}" if module else name.name,
                                "module": module,
                                "alias": name.asname,
                                "is_builtin": is_builtin,
                                "is_standard_lib": is_std_lib,
                                "line": node.lineno,
                                "type": "from_import"
                            })
        except SyntaxError:
            # Fallback to regex for syntax errors
            import_pattern = r'^import\s+([^;]+)'
            from_pattern = r'^from\s+([^\s]+)\s+import\s+([^;]+)'
            
            for i, line in enumerate(code.splitlines()):
                line = line.strip()
                
                # Check for import statements
                match = re.match(import_pattern, line)
                if match:
                    imports = match.group(1).strip().split(',')
                    for imp in imports:
                        imp = imp.strip()
                        if ' as ' in imp:
                            name, alias = imp.split(' as ')
                        else:
                            name, alias = imp, None
                        
                        base_name = name.split('.')[0]
                        is_builtin = base_name in built_in_modules
                        is_std_lib = base_name in standard_lib and not is_builtin
                        
                        if include_builtin or (not is_builtin and not is_std_lib):
                            dependencies.append({
                                "name": name,
                                "alias": alias,
                                "is_builtin": is_builtin,
                                "is_standard_lib": is_std_lib,
                                "line": i + 1,
                                "type": "import"
                            })
                
                # Check for from ... import statements
                match = re.match(from_pattern, line)
                if match:
                    module = match.group(1).strip()
                    imports = match.group(2).strip().split(',')
                    
                    base_name = module.split('.')[0]
                    is_builtin = base_name in built_in_modules
                    is_std_lib = base_name in standard_lib and not is_builtin
                    
                    if include_builtin or (not is_builtin and not is_std_lib):
                        for imp in imports:
                            imp = imp.strip()
                            if ' as ' in imp:
                                name, alias = imp.split(' as ')
                            else:
                                name, alias = imp, None
                            
                            dependencies.append({
                                "name": f"{module}.{name}",
                                "module": module,
                                "alias": alias,
                                "is_builtin": is_builtin,
                                "is_standard_lib": is_std_lib,
                                "line": i + 1,
                                "type": "from_import"
                            })
        
        return dependencies
    
    async def _extract_js_dependencies(
        self,
        code: str,
        include_builtin: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract dependencies from JavaScript/TypeScript code.
        
        Args:
            code: The JavaScript/TypeScript code to analyze
            include_builtin: Whether to include built-in dependencies
            
        Returns:
            List of dependencies with metadata
        """
        dependencies = []
        
        # Regular expressions for dependency detection
        import_pattern = r'import\s+(?:{[^}]*}|[^{;]*)(?:\s+from\s+)?[\'"]([^\'"]+)[\'"]'
        require_pattern = r'(?:const|let|var)\s+(?:{[^}]*}|[^{;]*)\s*=\s*require\([\'"]([^\'"]+)[\'"]\)'
        dynamic_import_pattern = r'import\([\'"]([^\'"]+)[\'"]\)'
        
        # Node.js built-in modules
        builtin_modules = {
            "assert", "buffer", "child_process", "cluster", "console", "constants", 
            "crypto", "dgram", "dns", "domain", "events", "fs", "http", "https", 
            "module", "net", "os", "path", "punycode", "querystring", "readline", 
            "repl", "stream", "string_decoder", "timers", "tls", "tty", "url", 
            "util", "v8", "vm", "zlib"
        }
        
        # Find ES6 imports
        for match in re.finditer(import_pattern, code):
            module_name = match.group(1)
            is_builtin = module_name in builtin_modules
            is_relative = module_name.startswith('./') or module_name.startswith('../')
            
            line_num = len(code[:match.start()].splitlines()) + 1
            
            if include_builtin or not is_builtin:
                dependencies.append({
                    "name": module_name,
                    "is_builtin": is_builtin,
                    "is_relative": is_relative,
                    "line": line_num,
                    "type": "import"
                })
        
        # Find CommonJS requires
        for match in re.finditer(require_pattern, code):
            module_name = match.group(1)
            is_builtin = module_name in builtin_modules
            is_relative = module_name.startswith('./') or module_name.startswith('../')
            
            line_num = len(code[:match.start()].splitlines()) + 1
            
            if include_builtin or not is_builtin:
                dependencies.append({
                    "name": module_name,
                    "is_builtin": is_builtin,
                    "is_relative": is_relative,
                    "line": line_num,
                    "type": "require"
                })
        
        # Find dynamic imports
        for match in re.finditer(dynamic_import_pattern, code):
            module_name = match.group(1)
            is_builtin = module_name in builtin_modules
            is_relative = module_name.startswith('./') or module_name.startswith('../')
            
            line_num = len(code[:match.start()].splitlines()) + 1
            
            if include_builtin or not is_builtin:
                dependencies.append({
                    "name": module_name,
                    "is_builtin": is_builtin,
                    "is_relative": is_relative,
                    "line": line_num,
                    "type": "dynamic_import"
                })
        
        return dependencies
    
    async def _extract_generic_dependencies(
        self,
        code: str,
        language: CodeLanguage
    ) -> List[Dict[str, Any]]:
        """Extract dependencies from code in any language.
        
        Args:
            code: The code to analyze
            language: Programming language of the code
            
        Returns:
            List of dependencies with metadata
        """
        # Simple patterns for common languages
        patterns = {
            CodeLanguage.JAVA: r'import\s+([^;]+);',
            CodeLanguage.CSHARP: r'using\s+([^;]+);',
            CodeLanguage.RUBY: r'require\s+[\'"]([^\'"]+)[\'"]',
            CodeLanguage.PHP: r'(?:require|include|require_once|include_once)\s*\(?[\'"]([^\'"]+)[\'"]',
            CodeLanguage.GO: r'import\s+(?:\([^)]*"([^"]+)"[^)]*\)|"([^"]+)")',
            CodeLanguage.RUST: r'(?:use|extern crate)\s+([^;]+);',
        }
        
        dependencies = []
        pattern = patterns.get(language)
        
        if pattern:
            for match in re.finditer(pattern, code):
                module_name = match.group(1)
                line_num = len(code[:match.start()].splitlines()) + 1
                
                dependencies.append({
                    "name": module_name,
                    "line": line_num,
                    "type": "import"
                })
        
        return dependencies
    
    async def run_tests(
        self,
        code: str,
        test_code: str,
        language: Union[str, CodeLanguage],
        framework: Optional[str] = None,
        timeout: int = 30
    ) -> TestResult:
        """Run tests for a code fragment.
        
        Args:
            code: The code to test
            test_code: The test code
            language: Programming language of the code
            framework: Optional testing framework
            timeout: Timeout in seconds
            
        Returns:
            Test results
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        # Select appropriate test runner based on language
        if language == CodeLanguage.PYTHON:
            return await self._run_python_tests(code, test_code, framework, timeout)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            return await self._run_js_tests(code, test_code, framework, timeout)
        else:
            # Fallback for unsupported languages
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                errors=[{
                    "message": f"Testing not supported for {language.value}",
                    "type": "unsupported_language"
                }]
            )
    
    async def _run_python_tests(
        self,
        code: str,
        test_code: str,
        framework: Optional[str] = None,
        timeout: int = 30
    ) -> TestResult:
        """Run Python tests.
        
        Args:
            code: The Python code to test
            test_code: The Python test code
            framework: Optional testing framework (pytest, unittest)
            timeout: Timeout in seconds
            
        Returns:
            Test results
        """
        # Default to pytest if not specified
        framework = framework or "pytest"
        
        # Create a temporary directory for the tests
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files for code and tests
            code_file = os.path.join(temp_dir, "module_to_test.py")
            with open(code_file, "w") as f:
                f.write(code)
            
            test_file = os.path.join(temp_dir, "test_module.py")
            with open(test_file, "w") as f:
                # Make sure the test imports the module correctly
                modified_test = test_code.replace("from module_to_test import", "from module_to_test import")
                modified_test = modified_test.replace("import module_to_test", "import module_to_test")
                f.write(modified_test)
            
            # Create __init__.py to make the directory a package
            init_file = os.path.join(temp_dir, "__init__.py")
            with open(init_file, "w") as f:
                f.write("# Package initialization")
            
            # Run the tests
            start_time = time.time()
            
            if framework.lower() == "pytest":
                return await self._run_pytest(temp_dir, test_file, timeout)
            elif framework.lower() == "unittest":
                return await self._run_unittest(temp_dir, test_file, timeout)
            else:
                return TestResult(
                    passed=False,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    errors=[{
                        "message": f"Unsupported Python test framework: {framework}",
                        "type": "unsupported_framework"
                    }],
                    execution_time=time.time() - start_time
                )
    
    async def _run_pytest(
        self,
        temp_dir: str,
        test_file: str,
        timeout: int = 30
    ) -> TestResult:
        """Run tests using pytest.
        
        Args:
            temp_dir: Temporary directory containing the files
            test_file: Path to the test file
            timeout: Timeout in seconds
            
        Returns:
            Test results
        """
        start_time = time.time()
        
        try:
            # Check if pytest is installed
            subprocess.run(["pytest", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Run pytest with JSON output
            cmd = ["pytest", test_file, "-v", "--json-report", "--json-report-file=results.json"]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                
                return TestResult(
                    passed=False,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    errors=[{
                        "message": f"Test execution timed out after {timeout} seconds",
                        "type": "timeout"
                    }],
                    execution_time=timeout
                )
            
            # Parse the results
            results_file = os.path.join(temp_dir, "results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results_data = json.load(f)
                
                summary = results_data.get("summary", {})
                total = summary.get("total", 0)
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0) + summary.get("error", 0)
                
                # Extract failures and errors
                errors = []
                for test_data in results_data.get("tests", []):
                    if test_data.get("outcome") in ["failed", "error"]:
                        errors.append({
                            "message": test_data.get("call", {}).get("longrepr", "Unknown error"),
                            "test": test_data.get("nodeid", "Unknown test"),
                            "type": "test_failure"
                        })
                
                return TestResult(
                    passed=failed == 0,
                    total_tests=total,
                    passed_tests=passed,
                    failed_tests=failed,
                    errors=errors,
                    execution_time=time.time() - start_time
                )
            else:
                # Fallback if results file not found
                return TestResult(
                    passed=proc.returncode == 0,
                    total_tests=0,  # Unknown
                    passed_tests=0,  # Unknown
                    failed_tests=0 if proc.returncode == 0 else 1,  # Unknown if passed
                    errors=[{
                        "message": stderr.decode('utf-8'),
                        "type": "execution_error"
                    }] if proc.returncode != 0 else [],
                    execution_time=time.time() - start_time
                )
                
        except subprocess.CalledProcessError:
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                errors=[{
                    "message": "pytest is not installed",
                    "type": "environment_error"
                }],
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                errors=[{
                    "message": str(e),
                    "type": "execution_error"
                }],
                execution_time=time.time() - start_time
            )
    
    async def _run_unittest(
        self,
        temp_dir: str,
        test_file: str,
        timeout: int = 30
    ) -> TestResult:
        """Run tests using unittest.
        
        Args:
            temp_dir: Temporary directory containing the files
            test_file: Path to the test file
            timeout: Timeout in seconds
            
        Returns:
            Test results
        """
        start_time = time.time()
        
        try:
            # Run unittest
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", temp_dir, "-p", os.path.basename(test_file)]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                
                return TestResult(
                    passed=False,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    errors=[{
                        "message": f"Test execution timed out after {timeout} seconds",
                        "type": "timeout"
                    }],
                    execution_time=timeout
                )
            
            # Parse the results
            output = stdout.decode('utf-8')
            error_output = stderr.decode('utf-8')
            
            # Try to parse the unittest output
            ran_match = re.search(r"Ran (\d+) tests", output)
            total_tests = int(ran_match.group(1)) if ran_match else 0
            
            # Check for failures
            passed = proc.returncode == 0
            failed_tests = 0 if passed else total_tests  # Simplification
            passed_tests = total_tests - failed_tests
            
            # Extract errors
            errors = []
            if error_output or not passed:
                errors.append({
                    "message": error_output or output,
                    "type": "test_failure"
                })
            
            return TestResult(
                passed=passed,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                errors=errors,
                execution_time=time.time() - start_time
            )
                
        except Exception as e:
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                errors=[{
                    "message": str(e),
                    "type": "execution_error"
                }],
                execution_time=time.time() - start_time
            )
    
    async def _run_js_tests(
        self,
        code: str,
        test_code: str,
        framework: Optional[str] = None,
        timeout: int = 30
    ) -> TestResult:
        """Run JavaScript/TypeScript tests.
        
        Args:
            code: The JavaScript/TypeScript code to test
            test_code: The JavaScript/TypeScript test code
            framework: Optional testing framework (jest, mocha)
            timeout: Timeout in seconds
            
        Returns:
            Test results
        """
        # Default to jest if not specified
        framework = framework or "jest"
        
        # Create a temporary directory for the tests
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files for code and tests
            code_file = os.path.join(temp_dir, "moduleToTest.js")
            with open(code_file, "w") as f:
                f.write(code)
            
            test_file = os.path.join(temp_dir, "moduleToTest.test.js")
            with open(test_file, "w") as f:
                # Make sure the test imports the module correctly
                modified_test = test_code.replace("./moduleToTest", "./moduleToTest")
                f.write(modified_test)
            
            # Create a basic package.json
            package_json = os.path.join(temp_dir, "package.json")
            with open(package_json, "w") as f:
                json.dump({
                    "name": "temp-test-project",
                    "version": "1.0.0",
                    "private": True,
                    "scripts": {
                        "test": f"{framework} --no-cache"
                    }
                }, f)
            
            # Run the tests
            start_time = time.time()
            
            if framework.lower() == "jest":
                return await self._run_jest(temp_dir, timeout)
            elif framework.lower() == "mocha":
                return await self._run_mocha(temp_dir, timeout)
            else:
                return TestResult(
                    passed=False,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    errors=[{
                        "message": f"Unsupported JavaScript test framework: {framework}",
                        "type": "unsupported_framework"
                    }],
                    execution_time=time.time() - start_time
                )
    
    async def _run_jest(
        self,
        temp_dir: str,
        timeout: int = 30
    ) -> TestResult:
        """Run tests using Jest.
        
        Args:
            temp_dir: Temporary directory containing the files
            timeout: Timeout in seconds
            
        Returns:
            Test results
        """
        start_time = time.time()
        
        try:
            # Check if jest is installed
            try:
                subprocess.run(["npx", "jest", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError:
                return TestResult(
                    passed=False,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    errors=[{
                        "message": "Jest is not installed",
                        "type": "environment_error"
                    }],
                    execution_time=time.time() - start_time
                )
            
            # Run jest with JSON output
            cmd = ["npx", "jest", "--json"]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                
                return TestResult(
                    passed=False,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    errors=[{
                        "message": f"Test execution timed out after {timeout} seconds",
                        "type": "timeout"
                    }],
                    execution_time=timeout
                )
            
            # Parse the results
            try:
                results_data = json.loads(stdout)
                
                passed = results_data.get("success", False)
                total = results_data.get("numTotalTests", 0)
                failed = results_data.get("numFailedTests", 0)
                passed_tests = total - failed
                
                # Extract failures
                errors = []
                for test_file in results_data.get("testResults", []):
                    for assertion_result in test_file.get("assertionResults", []):
                        if assertion_result.get("status") == "failed":
                            errors.append({
                                "message": "\n".join(assertion_result.get("failureMessages", [])),
                                "test": assertion_result.get("fullName", "Unknown test"),
                                "type": "test_failure"
                            })
                
                return TestResult(
                    passed=passed,
                    total_tests=total,
                    passed_tests=passed_tests,
                    failed_tests=failed,
                    errors=errors,
                    execution_time=time.time() - start_time
                )
            except json.JSONDecodeError:
                return TestResult(
                    passed=proc.returncode == 0,
                    total_tests=0,  # Unknown
                    passed_tests=0,  # Unknown
                    failed_tests=0 if proc.returncode == 0 else 1,  # Unknown if passed
                    errors=[{
                        "message": stderr.decode('utf-8'),
                        "type": "execution_error"
                    }] if proc.returncode != 0 else [],
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                errors=[{
                    "message": str(e),
                    "type": "execution_error"
                }],
                execution_time=time.time() - start_time
            )
    
    async def _run_mocha(
        self,
        temp_dir: str,
        timeout: int = 30
    ) -> TestResult:
        """Run tests using Mocha.
        
        Args:
            temp_dir: Temporary directory containing the files
            timeout: Timeout in seconds
            
        Returns:
            Test results
        """
        start_time = time.time()
        
        try:
            # Check if mocha is installed
            try:
                subprocess.run(["npx", "mocha", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError:
                return TestResult(
                    passed=False,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    errors=[{
                        "message": "Mocha is not installed",
                        "type": "environment_error"
                    }],
                    execution_time=time.time() - start_time
                )
            
            # Run mocha with reporter
            cmd = ["npx", "mocha", "--reporter", "json"]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                
                return TestResult(
                    passed=False,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    errors=[{
                        "message": f"Test execution timed out after {timeout} seconds",
                        "type": "timeout"
                    }],
                    execution_time=timeout
                )
            
            # Parse the results
            try:
                results_data = json.loads(stdout)
                
                passed = results_data.get("stats", {}).get("failures", 0) == 0
                total = results_data.get("stats", {}).get("tests", 0)
                failed = results_data.get("stats", {}).get("failures", 0)
                passed_tests = total - failed
                
                # Extract failures
                errors = []
                for failure in results_data.get("failures", []):
                    errors.append({
                        "message": failure.get("err", {}).get("message", "Unknown error"),
                        "test": failure.get("fullTitle", "Unknown test"),
                        "type": "test_failure"
                    })
                
                return TestResult(
                    passed=passed,
                    total_tests=total,
                    passed_tests=passed_tests,
                    failed_tests=failed,
                    errors=errors,
                    execution_time=time.time() - start_time
                )
            except json.JSONDecodeError:
                return TestResult(
                    passed=proc.returncode == 0,
                    total_tests=0,  # Unknown
                    passed_tests=0,  # Unknown
                    failed_tests=0 if proc.returncode == 0 else 1,  # Unknown if passed
                    errors=[{
                        "message": stderr.decode('utf-8'),
                        "type": "execution_error"
                    }] if proc.returncode != 0 else [],
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                errors=[{
                    "message": str(e),
                    "type": "execution_error"
                }],
                execution_time=time.time() - start_time
            )
    
    async def transform_code(
        self,
        code: str,
        language: Union[str, CodeLanguage],
        transformation: str,
        options: Optional[Dict[str, Any]] = None
    ) -> CodeTransformResult:
        """Transform code according to a specified transformation.
        
        Args:
            code: The code to transform
            language: Programming language of the code
            transformation: Type of transformation to apply
            options: Optional transformation options
            
        Returns:
            Transformed code with transformation details
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        options = options or {}
        
        # Handle different transformation types
        if transformation == "format":
            return await self.format_code(code, language, options.get("style"))
        elif transformation == "modernize":
            return await self._modernize_code(code, language, options)
        elif transformation == "optimize":
            return await self._optimize_code(code, language, options)
        elif transformation == "comment":
            return await self._add_comments(code, language, options)
        elif transformation == "docstring":
            return await self._add_docstrings(code, language, options)
        elif transformation == "type_hints":
            return await self._add_type_hints(code, language, options)
        else:
            # Unsupported transformation
            return CodeTransformResult(
                original_code=code,
                transformed_code=code,
                diff="",
                language=language,
                metrics={"unchanged": True, "reason": f"Unsupported transformation: {transformation}"}
            )
    
    async def _modernize_code(
        self,
        code: str,
        language: CodeLanguage,
        options: Dict[str, Any]
    ) -> CodeTransformResult:
        """Modernize code to use newer language features.
        
        Args:
            code: The code to modernize
            language: Programming language of the code
            options: Modernization options
            
        Returns:
            Modernized code
        """
        # This is a placeholder implementation
        # A real implementation would use language-specific tools
        
        original_code = code
        transformed_code = code
        
        if language == CodeLanguage.PYTHON:
            # Check for Python 2 to 3 modernization
            if options.get("python_version_target") == "3":
                try:
                    # Try to import lib2to3 - it's part of standard library but might be missing in some environments
                    try:
                        from lib2to3.refactor import RefactoringTool, get_fixers_from_package
                        
                        avail_fixes = set(get_fixers_from_package("lib2to3.fixes"))
                        rt = RefactoringTool(avail_fixes)
                        
                        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                            tmp.write(code)
                            tmp_path = tmp.name
                        
                        try:
                            rt.refactor_file(tmp_path, write=True)
                            with open(tmp_path, "r") as f:
                                transformed_code = f.read()
                        finally:
                            os.unlink(tmp_path)
                    except ImportError:
                        logger.info("lib2to3 is not available - this is part of the Python standard library "
                                "but might be missing in this Python installation.")
                except Exception as e:
                    logger.warning(f"Error modernizing Python code: {str(e)}")
        
        # Generate diff
        diff = "\n".join(difflib.unified_diff(
            original_code.splitlines(),
            transformed_code.splitlines(),
            fromfile="original",
            tofile="modernized",
            lineterm=""
        ))
        
        # Calculate metrics
        metrics = {
            "original_lines": len(original_code.splitlines()),
            "transformed_lines": len(transformed_code.splitlines()),
            "changed": transformed_code != original_code
        }
        
        # Identify changes (simplified)
        changes = []
        if transformed_code != original_code:
            changes.append({
                "type": "modernize",
                "description": "Applied code modernization",
                "details": ["Updated code to use newer language features"]
            })
        
        return CodeTransformResult(
            original_code=original_code,
            transformed_code=transformed_code,
            diff=diff,
            changes=changes,
            language=language,
            metrics=metrics
        )
    
    async def _optimize_code(
        self,
        code: str,
        language: CodeLanguage,
        options: Dict[str, Any]
    ) -> CodeTransformResult:
        """Optimize code for performance.
        
        Args:
            code: The code to optimize
            language: Programming language of the code
            options: Optimization options
            
        Returns:
            Optimized code
        """
        # This is a placeholder implementation
        # A real implementation would use language-specific optimization techniques
        
        # For now, just return the original code
        return CodeTransformResult(
            original_code=code,
            transformed_code=code,
            diff="",
            language=language,
            metrics={"unchanged": True, "reason": "Optimization not implemented for this language"}
        )
    
    async def _add_comments(
        self,
        code: str,
        language: CodeLanguage,
        options: Dict[str, Any]
    ) -> CodeTransformResult:
        """Add comments to code.
        
        Args:
            code: The code to add comments to
            language: Programming language of the code
            options: Comment options
            
        Returns:
            Code with added comments
        """
        # This is a placeholder implementation
        # A real implementation would analyze code and add meaningful comments
        
        # For now, just return the original code
        return CodeTransformResult(
            original_code=code,
            transformed_code=code,
            diff="",
            language=language,
            metrics={"unchanged": True, "reason": "Adding comments not implemented"}
        )
    
    async def _add_docstrings(
        self,
        code: str,
        language: CodeLanguage,
        options: Dict[str, Any]
    ) -> CodeTransformResult:
        """Add docstrings to code.
        
        Args:
            code: The code to add docstrings to
            language: Programming language of the code
            options: Docstring options
            
        Returns:
            Code with added docstrings
        """
        # This is a placeholder implementation
        # A real implementation would analyze code and add meaningful docstrings
        
        # For now, just return the original code
        return CodeTransformResult(
            original_code=code,
            transformed_code=code,
            diff="",
            language=language,
            metrics={"unchanged": True, "reason": "Adding docstrings not implemented"}
        )
    
    async def _add_type_hints(
        self,
        code: str,
        language: CodeLanguage,
        options: Dict[str, Any]
    ) -> CodeTransformResult:
        """Add type hints to code.
        
        Args:
            code: The code to add type hints to
            language: Programming language of the code
            options: Type hint options
            
        Returns:
            Code with added type hints
        """
        # This is a placeholder implementation
        # A real implementation would analyze code and add appropriate type hints
        
        # For now, just return the original code
        return CodeTransformResult(
            original_code=code,
            transformed_code=code,
            diff="",
            language=language,
            metrics={"unchanged": True, "reason": "Adding type hints not implemented"}
        )
    
    async def generate_documentation(
        self,
        code: str,
        language: Union[str, CodeLanguage],
        format: str = "markdown",
        include_examples: bool = True,
        include_tests: bool = False
    ) -> str:
        """Generate documentation for code.
        
        Args:
            code: The code to document
            language: Programming language of the code
            format: Output format (markdown, html, rst)
            include_examples: Whether to include examples
            include_tests: Whether to include tests
            
        Returns:
            Generated documentation
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        # Analyze the code
        analysis = await self.analyze_code(code, language, detailed=True)
        
        # Generate documentation based on the analysis
        if format.lower() == "markdown":
            return await self._generate_markdown_docs(code, language, analysis, include_examples, include_tests)
        elif format.lower() == "html":
            return await self._generate_html_docs(code, language, analysis, include_examples, include_tests)
        elif format.lower() == "rst":
            return await self._generate_rst_docs(code, language, analysis, include_examples, include_tests)
        else:
            return f"# Documentation\n\nUnsupported format: {format}. Supported formats are markdown, html, and rst."
    
    async def _generate_markdown_docs(
        self,
        code: str,
        language: CodeLanguage,
        analysis: CodeAnalysisResult,
        include_examples: bool,
        include_tests: bool
    ) -> str:
        """Generate Markdown documentation.
        
        Args:
            code: The code to document
            language: Programming language
            analysis: Code analysis results
            include_examples: Whether to include examples
            include_tests: Whether to include tests
            
        Returns:
            Markdown documentation
        """
        # Start with a title and description
        docs = [f"# Code Documentation\n"]
        
        # Add a summary
        docs.append(f"## Summary\n")
        docs.append(f"{analysis.summary}\n")
        
        # Add metrics
        docs.append(f"## Metrics\n")
        docs.append(f"- Language: {language.value}")
        docs.append(f"- Total lines: {analysis.metrics.get('total_lines', 'Unknown')}")
        docs.append(f"- Non-empty lines: {analysis.metrics.get('non_empty_lines', 'Unknown')}")
        docs.append(f"- Comment lines: {analysis.metrics.get('comment_lines', 'Unknown')}\n")
        
        # Add dependencies
        if analysis.dependencies:
            docs.append(f"## Dependencies\n")
            for dep in analysis.dependencies:
                docs.append(f"- {dep}")
            docs.append("")
        
        # Add imports
        if analysis.imports:
            docs.append(f"## Imports\n")
            for imp in analysis.imports:
                docs.append(f"- {imp}")
            docs.append("")
        
        # Add functions and classes
        if "functions" in analysis.structure:
            docs.append(f"## Functions\n")
            for name, func_info in analysis.structure["functions"].items():
                docs.append(f"### `{name}`\n")
                if "args" in func_info:
                    args_str = ", ".join(func_info["args"])
                    docs.append(f"**Arguments:** `{args_str}`\n")
                if "decorators" in func_info and func_info["decorators"]:
                    docs.append(f"**Decorators:** {', '.join(func_info['decorators'])}\n")
                if "is_async" in func_info and func_info["is_async"]:
                    docs.append(f"**Async:** Yes\n")
                
                # Add a placeholder description
                docs.append(f"No description available.\n")
        
        if "classes" in analysis.structure:
            docs.append(f"## Classes\n")
            for name, class_info in analysis.structure["classes"].items():
                docs.append(f"### `{name}`\n")
                if "bases" in class_info and class_info["bases"]:
                    docs.append(f"**Inherits from:** {', '.join(class_info['bases'])}\n")
                if "methods" in class_info and class_info["methods"]:
                    docs.append(f"**Methods:** {', '.join(class_info['methods'])}\n")
                
                # Add a placeholder description
                docs.append(f"No description available.\n")
        
        # Add examples if requested
        if include_examples:
            docs.append(f"## Examples\n")
            docs.append(f"No examples available.\n")
        
        # Add tests if requested
        # Add tests if requested
        if include_tests:
            docs.append(f"## Tests\n")
            docs.append(f"No tests available.\n")
        
        return "\n".join(docs)
    
    async def _generate_html_docs(
        self,
        code: str,
        language: CodeLanguage,
        analysis: CodeAnalysisResult,
        include_examples: bool,
        include_tests: bool
    ) -> str:
        """Generate HTML documentation.
        
        Args:
            code: The code to document
            language: Programming language
            analysis: Code analysis results
            include_examples: Whether to include examples
            include_tests: Whether to include tests
            
        Returns:
            HTML documentation
        """
        # Generate Markdown first
        markdown = await self._generate_markdown_docs(code, language, analysis, include_examples, include_tests)
        
        # Convert Markdown to HTML
        html_parts = ["<!DOCTYPE html>", "<html>", "<head>", "<title>Code Documentation</title>"]
        
        # Add some basic styling
        html_parts.append("<style>")
        html_parts.append("body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 900px; margin: 0 auto; }")
        html_parts.append("h1 { color: #333; border-bottom: 2px solid #eee; }")
        html_parts.append("h2 { color: #444; margin-top: 20px; }")
        html_parts.append("h3 { color: #555; }")
        html_parts.append("pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }")
        html_parts.append("code { font-family: monospace; background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; }")
        html_parts.append("</style>")
        html_parts.append("</head>")
        html_parts.append("<body>")
        
        # Convert markdown to basic HTML
        lines = markdown.splitlines()
        in_code_block = False
        in_list = False
        
        for line in lines:
            if line.startswith("```"):
                if in_code_block:
                    html_parts.append("</code></pre>")
                    in_code_block = False
                else:
                    html_parts.append("<pre><code>")
                    in_code_block = True
                continue
            
            if in_code_block:
                html_parts.append(line.replace("<", "&lt;").replace(">", "&gt;"))
                continue
            
            # Handle headers
            if line.startswith("# "):
                html_parts.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_parts.append(f"<h3>{line[4:]}</h3>")
                
            # Handle lists
            elif line.startswith("- "):
                if not in_list:
                    html_parts.append("<ul>")
                    in_list = True
                html_parts.append(f"<li>{line[2:]}</li>")
            elif in_list and not line.startswith("-") and line.strip():
                html_parts.append("</ul>")
                in_list = False
                html_parts.append(f"<p>{line}</p>")
            
            # Handle paragraphs
            elif line.strip():
                html_parts.append(f"<p>{line}</p>")
            else:
                if in_list:
                    html_parts.append("</ul>")
                    in_list = False
                html_parts.append("<br>")
        
        if in_list:
            html_parts.append("</ul>")
        
        if in_code_block:
            html_parts.append("</code></pre>")
        
        html_parts.append("</body>")
        html_parts.append("</html>")
        
        return "\n".join(html_parts)
    
    async def _generate_rst_docs(
        self,
        code: str,
        language: CodeLanguage,
        analysis: CodeAnalysisResult,
        include_examples: bool,
        include_tests: bool
    ) -> str:
        """Generate reStructuredText documentation.
        
        Args:
            code: The code to document
            language: Programming language
            analysis: Code analysis results
            include_examples: Whether to include examples
            include_tests: Whether to include tests
            
        Returns:
            reStructuredText documentation
        """
        # Start with a title
        docs = [f"Code Documentation", "===============", ""]
        
        # Add a summary
        docs.append(f"Summary", "-" * 7, "")
        docs.append(f"{analysis.summary}")
        docs.append("")
        
        # Add metrics
        docs.append(f"Metrics", "-" * 7, "")
        docs.append(f"* Language: {language.value}")
        docs.append(f"* Total lines: {analysis.metrics.get('total_lines', 'Unknown')}")
        docs.append(f"* Non-empty lines: {analysis.metrics.get('non_empty_lines', 'Unknown')}")
        docs.append(f"* Comment lines: {analysis.metrics.get('comment_lines', 'Unknown')}")
        docs.append("")
        
        # Add dependencies
        if analysis.dependencies:
            docs.append(f"Dependencies", "-" * 12, "")
            for dep in analysis.dependencies:
                docs.append(f"* {dep}")
            docs.append("")
        
        # Add imports
        if analysis.imports:
            docs.append(f"Imports", "-" * 7, "")
            for imp in analysis.imports:
                docs.append(f"* {imp}")
            docs.append("")
        
        # Add functions and classes
        if "functions" in analysis.structure:
            docs.append(f"Functions", "-" * 9, "")
            for name, func_info in analysis.structure["functions"].items():
                docs.append(f"``{name}``", "~" * (len(name) + 4), "")
                if "args" in func_info:
                    args_str = ", ".join(func_info["args"])
                    docs.append(f"**Arguments:** ``{args_str}``")
                    docs.append("")
                if "decorators" in func_info and func_info["decorators"]:
                    docs.append(f"**Decorators:** {', '.join(func_info['decorators'])}")
                    docs.append("")
                if "is_async" in func_info and func_info["is_async"]:
                    docs.append(f"**Async:** Yes")
                    docs.append("")
                
                # Add a placeholder description
                docs.append(f"No description available.")
                docs.append("")
        
        if "classes" in analysis.structure:
            docs.append(f"Classes", "-" * 7, "")
            for name, class_info in analysis.structure["classes"].items():
                docs.append(f"``{name}``", "~" * (len(name) + 4), "")
                if "bases" in class_info and class_info["bases"]:
                    docs.append(f"**Inherits from:** {', '.join(class_info['bases'])}")
                    docs.append("")
                if "methods" in class_info and class_info["methods"]:
                    docs.append(f"**Methods:** {', '.join(class_info['methods'])}")
                    docs.append("")
                
                # Add a placeholder description
                docs.append(f"No description available.")
                docs.append("")
        
        # Add examples if requested
        if include_examples:
            docs.append(f"Examples", "-" * 8, "")
            docs.append(f"No examples available.")
            docs.append("")
        
        # Add tests if requested
        if include_tests:
            docs.append(f"Tests", "-" * 5, "")
            docs.append(f"No tests available.")
            docs.append("")
        
        return "\n".join(docs)
    
    async def visualize_code(
        self,
        code: str,
        language: Union[str, CodeLanguage],
        visualization_type: str = "flowchart",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate visualization for code.
        
        Args:
            code: The code to visualize
            language: Programming language of the code
            visualization_type: Type of visualization (flowchart, class_diagram, etc.)
            options: Optional visualization options
            
        Returns:
            Visualization data
        """
        # Normalize language enum
        if isinstance(language, str):
            language = CodeLanguage(language)
        
        options = options or {}
        
        # Analyze the code
        analysis = await self.analyze_code(code, language, detailed=True)
        
        # Generate visualization based on the analysis
        if visualization_type.lower() == "flowchart":
            # Generate a simple flowchart using Mermaid syntax
            mermaid_code = self._generate_mermaid_flowchart(analysis, language, options)
            return {
                "type": "mermaid",
                "subtype": "flowchart",
                "code": mermaid_code,
                "language": language.value
            }
        elif visualization_type.lower() == "class_diagram":
            # Generate a class diagram using Mermaid syntax
            mermaid_code = self._generate_mermaid_class_diagram(analysis, language, options)
            return {
                "type": "mermaid",
                "subtype": "classDiagram",
                "code": mermaid_code,
                "language": language.value
            }
        else:
            return {
                "type": "error",
                "message": f"Unsupported visualization type: {visualization_type}",
                "language": language.value
            }
    
    def _generate_mermaid_flowchart(
        self,
        analysis: CodeAnalysisResult,
        language: CodeLanguage,
        options: Dict[str, Any]
    ) -> str:
        """Generate a Mermaid flowchart from code analysis.
        
        Args:
            analysis: Code analysis results
            language: Programming language
            options: Visualization options
            
        Returns:
            Mermaid flowchart code
        """
        # This is a simplified implementation
        flowchart = ["flowchart TD"]
        
        # Add nodes for functions
        for name, func_info in analysis.structure.get("functions", {}).items():
            flowchart.append(f'    func_{name}["{name}()"]')
        
        # Add nodes for classes
        for name, class_info in analysis.structure.get("classes", {}).items():
            flowchart.append(f'    class_{name}["{name}"]')
            
            # Add nodes for methods
            for method in class_info.get("methods", []):
                flowchart.append(f'    method_{name}_{method}["{method}()"]')
                flowchart.append(f'    class_{name} --> method_{name}_{method}')
        
        # Add some basic connections
        # This is very simplistic and doesn't reflect actual code flow
        prev_func = None
        for name in analysis.structure.get("functions", {}).keys():
            if prev_func:
                flowchart.append(f'    func_{prev_func} --> func_{name}')
            prev_func = name
        
        return "\n".join(flowchart)
    
    def _generate_mermaid_class_diagram(
        self,
        analysis: CodeAnalysisResult,
        language: CodeLanguage,
        options: Dict[str, Any]
    ) -> str:
        """Generate a Mermaid class diagram from code analysis.
        
        Args:
            analysis: Code analysis results
            language: Programming language
            options: Visualization options
            
        Returns:
            Mermaid class diagram code
        """
        # This is a simplified implementation
        class_diagram = ["classDiagram"]
        
        # Add classes and their methods
        for name, class_info in analysis.structure.get("classes", {}).items():
            class_diagram.append(f'    class {name}')
            
            # Add methods
            for method in class_info.get("methods", []):
                class_diagram.append(f'    {name} : +{method}()')
            
            # Add inheritance relationships
            for base in class_info.get("bases", []):
                if base != "object" and base in analysis.structure.get("classes", {}):
                    class_diagram.append(f'    {base} <|-- {name}')
        
        return "\n".join(class_diagram)
    
    def close(self) -> None:
        """Close resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
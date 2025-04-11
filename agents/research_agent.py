"""
Research Specialist Agent for the multi-agent development system.

This agent is responsible for researching solutions, best practices, libraries,
frameworks, and technical approaches to support the development team with
up-to-date, accurate technical information.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple
import uuid

from pydantic import BaseModel, Field, validator

from agents.base_agent import (
    BaseAgent,
    Task,
    TaskResult,
    TaskStatus,
    TaskPriority,
    TaskContext,
    AgentRole
)

# Set up logging
logger = logging.getLogger(__name__)


class ResearchQuestion(BaseModel):
    """A specific research question."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    context: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    required_detail_level: str = "comprehensive"  # basic, comprehensive, or expert


class ResearchTopic(BaseModel):
    """A research topic containing multiple related questions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    questions: List[ResearchQuestion]
    related_context: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)


class ResearchFinding(BaseModel):
    """A finding from research."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_id: str
    summary: str
    detailed_findings: str
    sources: List[Dict[str, str]] = Field(default_factory=list)
    confidence_level: float  # 0.0 to 1.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ResearchReport(BaseModel):
    """A comprehensive research report."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic_id: str
    title: str
    summary: str
    findings: List[ResearchFinding]
    recommendations: List[str] = Field(default_factory=list)
    additional_resources: List[Dict[str, str]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = Field(default_factory=list)


class ResearchSpecialistAgent(BaseAgent):
    """Agent specialized in research and information gathering."""
    
    def __init__(
        self, 
        name: str = "Research Specialist",
        web_search_tool: Optional[Any] = None,
        code_search_tool: Optional[Any] = None,
        documentation_access_tool: Optional[Any] = None,
        **kwargs
    ):
        """Initialize the Research Specialist agent.
        
        Args:
            name: Human-readable name for this agent
            web_search_tool: Optional tool for web searches
            code_search_tool: Optional tool for code repository searches
            documentation_access_tool: Optional tool for accessing documentation
            **kwargs: Additional arguments to pass to the BaseAgent constructor
        """
        super().__init__(
            name=name, 
            agent_type=AgentRole.RESEARCHER, 
            **kwargs
        )
        self.web_search_tool = web_search_tool
        self.code_search_tool = code_search_tool
        self.documentation_access_tool = documentation_access_tool
        
        # Track research topics and findings
        self.research_topics: Dict[str, ResearchTopic] = {}
        self.research_findings: Dict[str, Dict[str, ResearchFinding]] = {}  # topic_id -> question_id -> finding
        self.research_reports: Dict[str, ResearchReport] = {}
        
        # Knowledge base of previous research for efficiency
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Research Specialist Agent initialized")
    
    def _get_system_prompt(self) -> str:
        """Get the specialized system prompt for the Research Specialist."""
        return (
            f"You are {self.name}, a Research Specialist in an AI development team. "
            f"Your responsibilities include:\n"
            f"1. Researching technical solutions, libraries, frameworks, and best practices\n"
            f"2. Providing accurate, up-to-date information to support development decisions\n"
            f"3. Summarizing complex technical information in a clear, actionable format\n"
            f"4. Evaluating alternative approaches and making evidence-based recommendations\n"
            f"5. Keeping track of emerging trends and technologies relevant to projects\n\n"
            f"When conducting research, be thorough, accurate, and critical in your evaluation of sources. "
            f"Always cite your sources and provide evidence for your conclusions. "
            f"When there are competing approaches or conflicting information, acknowledge this "
            f"and present a balanced view of the options with their respective trade-offs.\n\n"
            f"Your output should be well-structured, concise, and tailored to the specific "
            f"information needs of the development team. Always consider the project context "
            f"when making recommendations, and format technical information to be readily "
            f"applicable to the development process."
        )
    
    async def research_topic(
        self, 
        topic: Union[ResearchTopic, Dict[str, Any]]
    ) -> TaskResult:
        """Research a comprehensive topic with multiple related questions.
        
        Args:
            topic: The research topic containing multiple questions
            
        Returns:
            TaskResult containing the research report
        """
        # Convert dict to ResearchTopic if necessary
        if isinstance(topic, dict):
            topic = ResearchTopic(**topic)
        
        # Store the topic
        self.research_topics[topic.id] = topic
        
        # Create a task for the overall research
        task = Task(
            task_id=f"research_topic_{topic.id}",
            description=f"Research topic: {topic.title}",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "topic": topic.dict(),
                "required_format": "comprehensive research report"
            },
            context=TaskContext(
                notes=(
                    f"Conduct thorough research on the topic '{topic.title}'. "
                    f"Address all questions within this topic. Consider the provided context "
                    f"and ensure findings are applicable to the project's needs."
                ),
                constraints=[
                    "Be thorough but concise",
                    "Cite sources when possible",
                    "Consider multiple viewpoints",
                    "Prioritize recent and reliable information"
                ]
            ),
            expected_output=(
                "A comprehensive research report addressing all questions, "
                "including findings, recommendations, and potential approaches."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the report
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the research report from the result
                report_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    report_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            report_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not report_data:
                    report_data = self._extract_report_from_text(result.result, topic)
                
                # Create and store the research report
                report = ResearchReport(
                    topic_id=topic.id,
                    title=report_data.get("title", topic.title),
                    summary=report_data.get("summary", ""),
                    findings=[
                        ResearchFinding(
                            question_id=finding.get("question_id", str(uuid.uuid4())),
                            summary=finding.get("summary", ""),
                            detailed_findings=finding.get("detailed_findings", ""),
                            sources=finding.get("sources", []),
                            confidence_level=finding.get("confidence_level", 0.7)
                        )
                        for finding in report_data.get("findings", [])
                    ],
                    recommendations=report_data.get("recommendations", []),
                    additional_resources=report_data.get("additional_resources", []),
                    tags=topic.tags
                )
                
                # Store the report
                self.research_reports[report.id] = report
                
                # Store findings by question
                question_findings = {}
                for finding in report.findings:
                    question_findings[finding.question_id] = finding
                self.research_findings[topic.id] = question_findings
                
                # Update knowledge base
                for finding in report.findings:
                    # Find the corresponding question
                    question = next((q for q in topic.questions if q.id == finding.question_id), None)
                    if question:
                        # Store in knowledge base with relevant tags for future retrieval
                        kb_key = self._normalize_question(question.question)
                        self.knowledge_base[kb_key] = {
                            "question": question.question,
                            "summary": finding.summary,
                            "detailed_findings": finding.detailed_findings,
                            "sources": finding.sources,
                            "confidence": finding.confidence_level,
                            "tags": topic.tags
                        }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"research_report_{report.id}",
                        value=report.dict(),
                        category="research_reports"
                    )
                    
                    # Store the detailed findings as separate items for easier retrieval
                    for finding in report.findings:
                        self.shared_memory.store(
                            key=f"research_finding_{finding.id}",
                            value=finding.dict(),
                            category="research_findings"
                        )
                
                logger.info(f"Created research report '{report.title}' with {len(report.findings)} findings")
                
                # Return the report as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=report.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"report_id": report.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing research report: {str(e)}")
                # Return the original result if processing failed
        
        return result
    
    def _extract_report_from_text(self, text: str, topic: ResearchTopic) -> Dict[str, Any]:
        """Extract structured report data from unstructured text.
        
        Args:
            text: The text to extract from
            topic: The research topic
            
        Returns:
            Structured report data
        """
        report_data = {
            "title": topic.title,
            "summary": "",
            "findings": [],
            "recommendations": [],
            "additional_resources": []
        }
        
        # Extract summary (look for summary section)
        summary_match = re.search(r'(?i)#*\s*Summary\s*\n+(.*?)(?=\n#|\Z)', text, re.DOTALL)
        if summary_match:
            report_data["summary"] = summary_match.group(1).strip()
        else:
            # Try to use the first paragraph
            paragraphs = text.split('\n\n')
            if paragraphs:
                report_data["summary"] = paragraphs[0].strip()
        
        # Extract findings for each question
        for question in topic.questions:
            # Look for sections that might contain answers to this question
            question_keywords = self._extract_keywords(question.question)
            
            # Try to find a section that matches the question
            findings_text = ""
            for keyword in question_keywords:
                pattern = r'(?i)#*\s*.*' + re.escape(keyword) + r'.*\n+(.*?)(?=\n#|\Z)'
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    findings_text = match.group(1).strip()
                    break
            
            # If no specific section found, use keyword search to extract relevant parts
            if not findings_text:
                relevant_parts = []
                paragraphs = re.split(r'\n\n+', text)
                for paragraph in paragraphs:
                    if any(keyword.lower() in paragraph.lower() for keyword in question_keywords):
                        relevant_parts.append(paragraph)
                
                if relevant_parts:
                    findings_text = '\n\n'.join(relevant_parts)
            
            # Extract sources if available
            sources = []
            source_section = re.search(r'(?i)#*\s*Sources|References\s*\n+(.*?)(?=\n#|\Z)', text, re.DOTALL)
            if source_section:
                source_text = source_section.group(1)
                # Extract URLs or formatted references
                urls = re.findall(r'https?://\S+', source_text)
                for url in urls:
                    sources.append({"url": url, "description": "Reference source"})
                
                # Also look for numbered or bulleted references
                ref_matches = re.findall(r'(?:^|\n)[*\-\d.]+\s*(.*?)(?=\n[*\-\d.]+|\Z)', source_text, re.DOTALL)
                for ref in ref_matches:
                    if ref.strip() and not any(ref.strip() in s["description"] for s in sources):
                        sources.append({"description": ref.strip()})
            
            # Create a finding
            if findings_text:
                # Split into summary and details if it's long
                detailed_findings = findings_text
                summary = findings_text[:200] + "..." if len(findings_text) > 200 else findings_text
                
                finding = {
                    "question_id": question.id,
                    "summary": summary,
                    "detailed_findings": detailed_findings,
                    "sources": sources,
                    "confidence_level": 0.7  # Default moderate confidence
                }
                
                report_data["findings"].append(finding)
        
        # Extract recommendations
        rec_match = re.search(r'(?i)#*\s*Recommendations\s*\n+(.*?)(?=\n#|\Z)', text, re.DOTALL)
        if rec_match:
            rec_text = rec_match.group(1)
            # Split by bullet points or numbers
            rec_items = re.findall(r'(?:^|\n)[*\-\d.]+\s*(.*?)(?=\n[*\-\d.]+|\Z)', rec_text, re.DOTALL)
            if rec_items:
                report_data["recommendations"] = [item.strip() for item in rec_items if item.strip()]
            else:
                # Just use the whole text if no clear items
                report_data["recommendations"] = [rec_text.strip()]
        
        # Extract additional resources
        res_match = re.search(r'(?i)#*\s*Additional Resources|Further Reading\s*\n+(.*?)(?=\n#|\Z)', text, re.DOTALL)
        if res_match:
            res_text = res_match.group(1)
            # Extract URLs or names
            urls = re.findall(r'https?://\S+', res_text)
            for url in urls:
                description = ""
                # Try to find a description near the URL
                url_index = res_text.find(url)
                if url_index >= 0:
                    line = res_text[max(0, url_index - 100):min(len(res_text), url_index + len(url) + 100)]
                    # Remove the URL itself
                    line = line.replace(url, "").strip()
                    if line:
                        description = line
                
                report_data["additional_resources"].append({
                    "url": url,
                    "description": description or "Additional resource"
                })
            
            # Also look for named resources without URLs
            if not report_data["additional_resources"]:
                res_items = re.findall(r'(?:^|\n)[*\-\d.]+\s*(.*?)(?=\n[*\-\d.]+|\Z)', res_text, re.DOTALL)
                for item in res_items:
                    if item.strip():
                        report_data["additional_resources"].append({
                            "description": item.strip()
                        })
        
        return report_data
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key words from a text for matching.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Remove common words and punctuation
        common_words = {
            "the", "and", "a", "an", "in", "on", "at", "is", "are", "was", "were",
            "be", "to", "for", "of", "by", "with", "about", "what", "how", "when",
            "where", "why", "which", "who", "can", "could", "would", "should", "did",
            "do", "does", "has", "have", "had"
        }
        
        # Convert to lowercase and split by non-alphanumeric characters
        words = re.findall(r'\w+', text.lower())
        
        # Filter out common words and words too short
        keywords = [word for word in words if word not in common_words and len(word) > 3]
        
        # Add important pairs (adjacent keywords)
        if len(keywords) > 1:
            pairs = [f"{keywords[i]} {keywords[i+1]}" for i in range(len(keywords)-1)]
            keywords.extend(pairs)
        
        return keywords
    
    def _normalize_question(self, question: str) -> str:
        """Normalize a question for knowledge base lookup.
        
        Args:
            question: The question to normalize
            
        Returns:
            Normalized question string
        """
        # Remove punctuation, convert to lowercase, and sort words
        words = re.findall(r'\w+', question.lower())
        words = sorted([w for w in words if len(w) > 3])  # Remove short words
        return " ".join(words)
    
    async def research_question(
        self, 
        question: str,
        context: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        detail_level: str = "comprehensive",
        tags: Optional[List[str]] = None
    ) -> TaskResult:
        """Research a single question.
        
        Args:
            question: The question to research
            context: Optional context for the question
            priority: Priority level for the research
            detail_level: Required level of detail (basic, comprehensive, expert)
            tags: Optional tags for categorization
            
        Returns:
            TaskResult containing the research findings
        """
        # Check if we already have this in knowledge base
        normalized_q = self._normalize_question(question)
        existing_knowledge = self.knowledge_base.get(normalized_q)
        
        if existing_knowledge and existing_knowledge.get("confidence", 0) > 0.7:
            logger.info(f"Found existing research for question: {question[:50]}...")
            
            # Create a finding from existing knowledge
            finding = ResearchFinding(
                question_id=str(uuid.uuid4()),
                summary=existing_knowledge["summary"],
                detailed_findings=existing_knowledge["detailed_findings"],
                sources=existing_knowledge.get("sources", []),
                confidence_level=existing_knowledge["confidence"]
            )
            
            # Return the finding directly
            return TaskResult(
                agent_id=self.state.agent_id,
                agent_name=self.name,
                task_id=f"research_question_{uuid.uuid4()}",
                result=finding.dict(),
                status=TaskStatus.COMPLETED,
                timestamp=datetime.now().isoformat(),
                execution_time=0.1,  # Negligible execution time
                metadata={"from_knowledge_base": True}
            )
        
        # Create a task for the question
        task = Task(
            task_id=f"research_question_{uuid.uuid4()}",
            description=f"Research question: {question}",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "question": question,
                "detail_level": detail_level,
                "required_format": "research finding"
            },
            context=TaskContext(
                notes=context or f"Research the question '{question}' thoroughly.",
                constraints=[
                    "Be thorough but concise",
                    "Cite sources when possible",
                    "Consider multiple viewpoints",
                    "Prioritize recent and reliable information"
                ]
            ),
            expected_output=(
                "A comprehensive research finding addressing the question, "
                "including summary, detailed findings, sources, and confidence level."
            ),
            priority=priority
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the finding
        if result.status == TaskStatus.COMPLETED and result.result:
            try:
                # Extract the finding from the result
                finding_data = None
                
                # Try different parsing strategies
                try:
                    # First attempt: Try to parse as JSON
                    finding_data = json.loads(result.result)
                except json.JSONDecodeError:
                    # Second attempt: Extract JSON from markdown
                    json_match = re.search(r'```json\n(.*?)\n```', result.result, re.DOTALL)
                    if json_match:
                        try:
                            finding_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                # If we couldn't parse JSON, extract structured info from text
                if not finding_data:
                    finding_data = self._extract_finding_from_text(result.result)
                
                # Create a ResearchQuestion
                research_question = ResearchQuestion(
                    question=question,
                    context=context,
                    priority=priority,
                    required_detail_level=detail_level
                )
                
                # Create a ResearchFinding
                finding = ResearchFinding(
                    question_id=research_question.id,
                    summary=finding_data.get("summary", ""),
                    detailed_findings=finding_data.get("detailed_findings", ""),
                    sources=finding_data.get("sources", []),
                    confidence_level=finding_data.get("confidence_level", 0.7)
                )
                
                # Create a simple topic to store everything
                topic_id = str(uuid.uuid4())
                topic = ResearchTopic(
                    id=topic_id,
                    title=f"Research on: {question[:50]}...",
                    description=f"Research on the question: {question}",
                    questions=[research_question],
                    tags=tags or []
                )
                
                # Store everything
                self.research_topics[topic_id] = topic
                self.research_findings[topic_id] = {research_question.id: finding}
                
                # Update knowledge base
                self.knowledge_base[normalized_q] = {
                    "question": question,
                    "summary": finding.summary,
                    "detailed_findings": finding.detailed_findings,
                    "sources": finding.sources,
                    "confidence": finding.confidence_level,
                    "tags": tags or []
                }
                
                # Store in shared memory if available
                if self.shared_memory:
                    self.shared_memory.store(
                        key=f"research_finding_{finding.id}",
                        value=finding.dict(),
                        category="research_findings"
                    )
                
                logger.info(f"Created research finding for question: {question[:50]}...")
                
                # Return the finding as the result
                updated_result = TaskResult(
                    agent_id=result.agent_id,
                    agent_name=result.agent_name,
                    task_id=result.task_id,
                    result=finding.dict(),
                    status=result.status,
                    timestamp=result.timestamp,
                    execution_time=result.execution_time,
                    token_usage=result.token_usage,
                    metadata={"finding_id": finding.id}
                )
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing research finding: {str(e)}")
                # Return the original result if processing failed
        
        return result
    
    def _extract_finding_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured finding data from unstructured text.
        
        Args:
            text: The text to extract from
            
        Returns:
            Structured finding data
        """
        finding_data = {
            "summary": "",
            "detailed_findings": text,
            "sources": [],
            "confidence_level": 0.7  # Default moderate confidence
        }
        
        # Extract summary (first paragraph or summary section)
        summary_match = re.search(r'(?i)#*\s*Summary\s*\n+(.*?)(?=\n#|\Z)', text, re.DOTALL)
        if summary_match:
            finding_data["summary"] = summary_match.group(1).strip()
        else:
            # Use the first paragraph
            paragraphs = text.split('\n\n')
            if paragraphs:
                finding_data["summary"] = paragraphs[0].strip()
        
        # Extract sources if available
        source_section = re.search(r'(?i)#*\s*Sources|References\s*\n+(.*?)(?=\n#|\Z)', text, re.DOTALL)
        if source_section:
            source_text = source_section.group(1)
            # Extract URLs or formatted references
            urls = re.findall(r'https?://\S+', source_text)
            for url in urls:
                finding_data["sources"].append({"url": url, "description": "Reference source"})
            
            # Also look for numbered or bulleted references
            ref_matches = re.findall(r'(?:^|\n)[*\-\d.]+\s*(.*?)(?=\n[*\-\d.]+|\Z)', source_text, re.DOTALL)
            for ref in ref_matches:
                if ref.strip() and not any(ref.strip() in s["description"] for s in finding_data["sources"]):
                    finding_data["sources"].append({"description": ref.strip()})
        
        # Extract confidence level if mentioned
        confidence_pattern = r'(?i)confidence(?:\s*level)?[\s:]+(\d+(?:\.\d+)?%|high|medium|low|very high|very low)'
        confidence_match = re.search(confidence_pattern, text)
        if confidence_match:
            confidence_text = confidence_match.group(1).lower()
            if "%" in confidence_text:
                # Parse percentage
                try:
                    percentage = float(confidence_text.replace("%", "")) / 100.0
                    finding_data["confidence_level"] = min(1.0, max(0.0, percentage))
                except ValueError:
                    pass
            else:
                # Parse textual confidence
                confidence_map = {
                    "very high": 0.9,
                    "high": 0.8,
                    "medium": 0.6,
                    "low": 0.4,
                    "very low": 0.2
                }
                for key, value in confidence_map.items():
                    if key in confidence_text:
                        finding_data["confidence_level"] = value
                        break
        
        return finding_data
    
    async def compare_technologies(
        self, 
        technologies: List[str],
        comparison_criteria: List[str],
        context: Optional[str] = None,
        use_case: Optional[str] = None
    ) -> TaskResult:
        """Compare multiple technologies or approaches based on specific criteria.
        
        Args:
            technologies: List of technologies to compare
            comparison_criteria: Criteria to use for comparison
            context: Optional context for the comparison
            use_case: Optional specific use case
            
        Returns:
            TaskResult containing the technology comparison
        """
        # Create a task for the comparison
        task = Task(
            task_id=f"compare_technologies_{uuid.uuid4()}",
            description=f"Compare technologies: {', '.join(technologies)}",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "technologies": technologies,
                "criteria": comparison_criteria,
                "use_case": use_case or "General development",
                "required_format": "technology comparison"
            },
            context=TaskContext(
                notes=(
                    f"Compare the technologies ({', '.join(technologies)}) "
                    f"based on the provided criteria ({', '.join(comparison_criteria)}). "
                    + (f"Consider the specific use case: {use_case}. " if use_case else "")
                    + (f"Additional context: {context}" if context else "")
                ),
                constraints=[
                    "Be objective and balanced",
                    "Use concrete examples where possible",
                    "Consider both strengths and weaknesses",
                    "Provide a clear recommendation based on the criteria"
                ]
            ),
            expected_output=(
                "A comprehensive comparison of the technologies with ratings for each criterion, "
                "pros and cons, and a final recommendation."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the comparison
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                comparison_id = f"tech_comparison_{uuid.uuid4()}"
                self.shared_memory.store(
                    key=comparison_id,
                    value={
                        "technologies": technologies,
                        "criteria": comparison_criteria,
                        "use_case": use_case,
                        "comparison_result": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="technology_comparisons"
                )
                
                logger.info(f"Created technology comparison for {', '.join(technologies)}")
                
                # Update the result metadata
                result.metadata = {
                    "comparison_id": comparison_id,
                    "technologies": technologies
                }
        
        return result
    
    async def research_best_practices(
        self, 
        topic: str,
        technology_stack: Optional[List[str]] = None,
        industry_standards: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> TaskResult:
        """Research best practices for a specific topic or technology.
        
        Args:
            topic: The topic to research best practices for
            technology_stack: Optional list of relevant technologies
            industry_standards: Optional list of industry standards to consider
            context: Optional context for the research
            
        Returns:
            TaskResult containing the best practices
        """
        # Create a task for researching best practices
        task = Task(
            task_id=f"best_practices_{uuid.uuid4()}",
            description=f"Research best practices for {topic}",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "topic": topic,
                "technology_stack": technology_stack or [],
                "industry_standards": industry_standards or [],
                "required_format": "best practices guide"
            },
            context=TaskContext(
                notes=(
                    f"Research and compile best practices for {topic}. "
                    + (f"Consider the following technology stack: {', '.join(technology_stack)}. " if technology_stack else "")
                    + (f"Adhere to these industry standards: {', '.join(industry_standards)}. " if industry_standards else "")
                    + (f"Additional context: {context}" if context else "")
                ),
                constraints=[
                    "Focus on widely accepted practices",
                    "Provide concrete examples",
                    "Include code samples where relevant",
                    "Consider security, performance, and maintainability"
                ]
            ),
            expected_output=(
                "A comprehensive best practices guide including principles, examples, "
                "code snippets, common pitfalls to avoid, and implementation recommendations."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the best practices
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                practices_id = f"best_practices_{topic.lower().replace(' ', '_')}_{uuid.uuid4()}"
                self.shared_memory.store(
                    key=practices_id,
                    value={
                        "topic": topic,
                        "technology_stack": technology_stack or [],
                        "industry_standards": industry_standards or [],
                        "best_practices": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="best_practices"
                )
                
                logger.info(f"Created best practices guide for {topic}")
                
                # Update the result metadata
                result.metadata = {
                    "practices_id": practices_id,
                    "topic": topic
                }
            
            # Also add to knowledge base
            kb_key = self._normalize_question(f"best practices for {topic}")
            self.knowledge_base[kb_key] = {
                "question": f"What are the best practices for {topic}?",
                "summary": result.result[:500] + "..." if len(result.result) > 500 else result.result,
                "detailed_findings": result.result,
                "confidence": 0.8,
                "tags": (technology_stack or []) + (industry_standards or []) + ["best practices"]
            }
        
        return result
    
    async def evaluate_library(
        self, 
        library_name: str,
        evaluation_criteria: List[str],
        alternatives: Optional[List[str]] = None,
        use_case: Optional[str] = None
    ) -> TaskResult:
        """Evaluate a library or framework based on specific criteria.
        
        Args:
            library_name: Name of the library to evaluate
            evaluation_criteria: Criteria to use for evaluation
            alternatives: Optional list of alternative libraries to consider
            use_case: Optional specific use case
            
        Returns:
            TaskResult containing the library evaluation
        """
        # Create a task for evaluating the library
        task = Task(
            task_id=f"evaluate_library_{library_name.lower().replace(' ', '_')}",
            description=f"Evaluate library: {library_name}",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "library": library_name,
                "criteria": evaluation_criteria,
                "alternatives": alternatives or [],
                "use_case": use_case or "General development",
                "required_format": "library evaluation"
            },
            context=TaskContext(
                notes=(
                    f"Evaluate the library {library_name} based on the provided criteria. "
                    + (f"Compare with alternatives: {', '.join(alternatives)}. " if alternatives else "")
                    + (f"Consider the specific use case: {use_case}. " if use_case else "")
                ),
                constraints=[
                    "Consider the library's maturity and community support",
                    "Evaluate documentation quality",
                    "Assess performance characteristics",
                    "Consider integration complexity",
                    "Evaluate potential limitations and corner cases"
                ]
            ),
            expected_output=(
                "A comprehensive evaluation of the library including strengths, weaknesses, "
                "ratings for each criterion, example usage, and recommendations."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the evaluation
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                evaluation_id = f"library_evaluation_{library_name.lower().replace(' ', '_')}_{uuid.uuid4()}"
                self.shared_memory.store(
                    key=evaluation_id,
                    value={
                        "library": library_name,
                        "criteria": evaluation_criteria,
                        "alternatives": alternatives or [],
                        "use_case": use_case,
                        "evaluation": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="library_evaluations"
                )
                
                logger.info(f"Created evaluation for library {library_name}")
                
                # Update the result metadata
                result.metadata = {
                    "evaluation_id": evaluation_id,
                    "library": library_name
                }
            
            # Also add to knowledge base
            kb_key = self._normalize_question(f"evaluation of {library_name}")
            self.knowledge_base[kb_key] = {
                "question": f"How does {library_name} perform for {use_case or 'general use'}?",
                "summary": result.result[:500] + "..." if len(result.result) > 500 else result.result,
                "detailed_findings": result.result,
                "confidence": 0.8,
                "tags": [library_name] + (alternatives or []) + ["library evaluation"]
            }
        
        return result
    
    async def research_architecture_patterns(
        self, 
        requirements: List[str],
        constraints: List[str],
        domain: Optional[str] = None,
        technology_stack: Optional[List[str]] = None
    ) -> TaskResult:
        """Research architectural patterns that meet specific requirements and constraints.
        
        Args:
            requirements: List of architectural requirements
            constraints: List of constraints to consider
            domain: Optional domain/industry context
            technology_stack: Optional technology stack to consider
            
        Returns:
            TaskResult containing recommended architecture patterns
        """
        # Create a task for researching architecture patterns
        task = Task(
            task_id=f"architecture_patterns_{uuid.uuid4()}",
            description=f"Research architecture patterns for {domain or 'general'} applications",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "requirements": requirements,
                "constraints": constraints,
                "domain": domain or "general",
                "technology_stack": technology_stack or [],
                "required_format": "architecture patterns recommendation"
            },
            context=TaskContext(
                notes=(
                    f"Research and recommend architecture patterns that meet the specified requirements "
                    f"and constraints. " 
                    + (f"Consider the domain context: {domain}. " if domain else "")
                    + (f"Consider the technology stack: {', '.join(technology_stack)}. " if technology_stack else "")
                ),
                constraints=[
                    "Prioritize proven patterns with demonstrated success",
                    "Consider scalability, maintainability, and security",
                    "Provide examples of successful implementations",
                    "Discuss trade-offs of each recommended pattern"
                ]
            ),
            expected_output=(
                "A detailed recommendation of architecture patterns, including pattern descriptions, "
                "diagrams (text-based if needed), trade-offs, implementation considerations, and "
                "specific recommendations for the given requirements."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the patterns
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                patterns_id = f"architecture_patterns_{domain or 'general'}_{uuid.uuid4()}"
                self.shared_memory.store(
                    key=patterns_id,
                    value={
                        "requirements": requirements,
                        "constraints": constraints,
                        "domain": domain or "general",
                        "technology_stack": technology_stack or [],
                        "patterns": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="architecture_patterns"
                )
                
                logger.info(f"Created architecture patterns recommendation for {domain or 'general'} domain")
                
                # Update the result metadata
                result.metadata = {
                    "patterns_id": patterns_id,
                    "domain": domain or "general"
                }
        
        return result
    
    async def find_relevant_examples(
        self, 
        task_description: str,
        technology: Optional[str] = None,
        complexity: str = "medium",
        quantity: int = 3
    ) -> TaskResult:
        """Find relevant code examples for a specific task.
        
        Args:
            task_description: Description of the task
            technology: Optional specific technology to use
            complexity: Desired complexity level (simple, medium, complex)
            quantity: Number of examples to provide
            
        Returns:
            TaskResult containing relevant code examples
        """
        # Create a task for finding examples
        task = Task(
            task_id=f"find_examples_{uuid.uuid4()}",
            description=f"Find {quantity} {complexity} examples for: {task_description}",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "task_description": task_description,
                "technology": technology,
                "complexity": complexity,
                "quantity": quantity,
                "required_format": "code examples"
            },
            context=TaskContext(
                notes=(
                    f"Find {quantity} relevant code examples for the task: {task_description}. "
                    + (f"Use the following technology: {technology}. " if technology else "")
                    + f"Provide {complexity} complexity examples."
                ),
                constraints=[
                    "Examples should be fully functional",
                    "Include explanations for each example",
                    "Highlight best practices",
                    "Consider edge cases"
                ]
            ),
            expected_output=(
                f"{quantity} code examples with explanations, addressing the specified task. "
                "Each example should be complete, well-commented, and follow best practices."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the examples
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                examples_id = f"code_examples_{uuid.uuid4()}"
                self.shared_memory.store(
                    key=examples_id,
                    value={
                        "task_description": task_description,
                        "technology": technology,
                        "complexity": complexity,
                        "examples": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="code_examples"
                )
                
                logger.info(f"Found {quantity} code examples for: {task_description}")
                
                # Update the result metadata
                result.metadata = {
                    "examples_id": examples_id,
                    "technology": technology
                }
        
        return result
    
    async def research_security_considerations(
        self, 
        application_type: str,
        technologies: List[str],
        data_sensitivity: str = "medium",
        compliance_requirements: Optional[List[str]] = None
    ) -> TaskResult:
        """Research security considerations for a specific application type and tech stack.
        
        Args:
            application_type: Type of application (web, mobile, desktop, etc.)
            technologies: List of technologies used
            data_sensitivity: Level of data sensitivity (low, medium, high)
            compliance_requirements: Optional list of compliance requirements
            
        Returns:
            TaskResult containing security considerations
        """
        # Create a task for researching security considerations
        task = Task(
            task_id=f"security_considerations_{application_type.lower().replace(' ', '_')}",
            description=f"Research security considerations for {application_type} applications",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "application_type": application_type,
                "technologies": technologies,
                "data_sensitivity": data_sensitivity,
                "compliance_requirements": compliance_requirements or [],
                "required_format": "security guide"
            },
            context=TaskContext(
                notes=(
                    f"Research security considerations for {application_type} applications "
                    f"using {', '.join(technologies)}. Data sensitivity level: {data_sensitivity}. "
                    + (f"Compliance requirements: {', '.join(compliance_requirements)}. " if compliance_requirements else "")
                ),
                constraints=[
                    "Focus on practical, implementable security measures",
                    "Consider the entire application lifecycle",
                    "Prioritize recommendations based on risk and impact",
                    "Include both prevention and detection measures"
                ]
            ),
            expected_output=(
                "A comprehensive security guide including threat modeling, security best practices, "
                "common vulnerabilities, mitigation strategies, and compliance considerations."
            ),
            priority=TaskPriority.HIGH
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the security considerations
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                security_id = f"security_considerations_{application_type.lower().replace(' ', '_')}_{uuid.uuid4()}"
                self.shared_memory.store(
                    key=security_id,
                    value={
                        "application_type": application_type,
                        "technologies": technologies,
                        "data_sensitivity": data_sensitivity,
                        "compliance_requirements": compliance_requirements or [],
                        "security_guide": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="security_guides"
                )
                
                logger.info(f"Created security guide for {application_type} applications")
                
                # Update the result metadata
                result.metadata = {
                    "security_id": security_id,
                    "application_type": application_type
                }
            
            # Also add to knowledge base
            kb_key = self._normalize_question(f"security considerations {application_type}")
            self.knowledge_base[kb_key] = {
                "question": f"What are the security considerations for {application_type} applications?",
                "summary": result.result[:500] + "..." if len(result.result) > 500 else result.result,
                "detailed_findings": result.result,
                "confidence": 0.85,
                "tags": ["security"] + technologies
            }
        
        return result
    
    async def analyze_technology_trends(
        self, 
        technology_area: str,
        timeframe: str = "current",
        focus_areas: Optional[List[str]] = None
    ) -> TaskResult:
        """Analyze current or emerging trends in a technology area.
        
        Args:
            technology_area: The technology area to analyze
            timeframe: Timeframe to consider (current, emerging, future)
            focus_areas: Optional specific focus areas within the technology
            
        Returns:
            TaskResult containing the trend analysis
        """
        # Create a task for analyzing technology trends
        task = Task(
            task_id=f"technology_trends_{technology_area.lower().replace(' ', '_')}_{timeframe}",
            description=f"Analyze {timeframe} trends in {technology_area}",
            agent_type=str(AgentRole.RESEARCHER),
            requirements={
                "technology_area": technology_area,
                "timeframe": timeframe,
                "focus_areas": focus_areas or [],
                "required_format": "trend analysis"
            },
            context=TaskContext(
                notes=(
                    f"Analyze {timeframe} trends in {technology_area}. "
                    + (f"Focus specifically on: {', '.join(focus_areas)}. " if focus_areas else "")
                ),
                constraints=[
                    "Base analysis on factual information",
                    "Consider industry adoption rates",
                    "Identify key drivers and obstacles",
                    "Consider implications for developers and organizations"
                ]
            ),
            expected_output=(
                "A comprehensive trend analysis including key trends, supporting evidence, "
                "adoption curves, future projections, and strategic recommendations."
            ),
            priority=TaskPriority.MEDIUM
        )
        
        # Execute the task
        result = await self.execute_task(task)
        
        # If successful, process and store the trend analysis
        if result.status == TaskStatus.COMPLETED and result.result:
            # Store in shared memory if available
            if self.shared_memory:
                trends_id = f"technology_trends_{technology_area.lower().replace(' ', '_')}_{timeframe}_{uuid.uuid4()}"
                self.shared_memory.store(
                    key=trends_id,
                    value={
                        "technology_area": technology_area,
                        "timeframe": timeframe,
                        "focus_areas": focus_areas or [],
                        "trend_analysis": result.result,
                        "timestamp": datetime.now().isoformat()
                    },
                    category="technology_trends"
                )
                
                logger.info(f"Created trend analysis for {technology_area} ({timeframe})")
                
                # Update the result metadata
                result.metadata = {
                    "trends_id": trends_id,
                    "technology_area": technology_area,
                    "timeframe": timeframe
                }
        
        return result
    
    def get_knowledge_on_topic(self, topic: str, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve existing knowledge on a specific topic.
        
        Args:
            topic: The topic to retrieve knowledge for
            tags: Optional tags to filter by
            
        Returns:
            List of knowledge items related to the topic
        """
        # Extract keywords from the topic
        keywords = self._extract_keywords(topic)
        
        # Find matching knowledge items
        matching_items = []
        
        for kb_key, knowledge in self.knowledge_base.items():
            # Check if any keyword matches
            if any(keyword.lower() in kb_key.lower() for keyword in keywords):
                # If tags are provided, check tag match
                if tags:
                    knowledge_tags = knowledge.get("tags", [])
                    if not any(tag in knowledge_tags for tag in tags):
                        continue
                
                # Add to matching items
                matching_items.append({
                    "question": knowledge.get("question", ""),
                    "summary": knowledge.get("summary", ""),
                    "confidence": knowledge.get("confidence", 0.0),
                    "tags": knowledge.get("tags", [])
                })
        
        # Sort by confidence
        matching_items.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return matching_items
    
    def clear_knowledge_base(self) -> None:
        """Clear the agent's knowledge base."""
        self.knowledge_base = {}
        logger.info("Knowledge base cleared")
    
    def get_research_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific research report.
        
        Args:
            report_id: ID of the report to retrieve
            
        Returns:
            Report data if found, None otherwise
        """
        # Check local storage
        if report_id in self.research_reports:
            return self.research_reports[report_id].dict()
        
        # Check shared memory
        if self.shared_memory:
            return self.shared_memory.retrieve(
                key=f"research_report_{report_id}",
                category="research_reports"
            )
        
        return None
    
    def get_recent_research(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get summaries of recent research.
        
        Args:
            limit: Maximum number of reports to return
            
        Returns:
            List of recent research summaries
        """
        recent_reports = []
        
        # Collect from local storage
        reports = list(self.research_reports.values())
        reports.sort(key=lambda x: x.timestamp, reverse=True)
        
        for report in reports[:limit]:
            recent_reports.append({
                "id": report.id,
                "title": report.title,
                "summary": report.summary,
                "timestamp": report.timestamp,
                "tags": report.tags
            })
        
        # If we need more and have shared memory, check there
        if len(recent_reports) < limit and self.shared_memory:
            # Get report keys from shared memory
            report_keys = self.shared_memory.list_keys("research_reports")
            
            # Get reports
            for key in report_keys:
                if len(recent_reports) >= limit:
                    break
                    
                report_data = self.shared_memory.retrieve(key, "research_reports")
                if report_data and any(r["id"] != report_data.get("id") for r in recent_reports):
                    recent_reports.append({
                        "id": report_data.get("id", key),
                        "title": report_data.get("title", "Unknown"),
                        "summary": report_data.get("summary", ""),
                        "timestamp": report_data.get("timestamp", ""),
                        "tags": report_data.get("tags", [])
                    })
            
            # Sort by timestamp
            recent_reports.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Limit
            recent_reports = recent_reports[:limit]
        
        return recent_reports
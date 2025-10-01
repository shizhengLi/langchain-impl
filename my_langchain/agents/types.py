# -*- coding: utf-8 -*-
"""
Agent types and data structures
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    Agent configuration
    """
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations for agent execution"
    )
    early_stopping_method: str = Field(
        default="force",
        description="Method for early stopping: 'force', 'generate', 'thorough'"
    )
    return_intermediate_steps: bool = Field(
        default=True,
        description="Whether to return intermediate steps in the result"
    )
    handle_parsing_errors: bool = Field(
        default=True,
        description="Whether to handle parsing errors gracefully"
    )
    handle_tool_errors: bool = Field(
        default=True,
        description="Whether to handle tool execution errors gracefully"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print verbose output"
    )
    agent_type: str = Field(
        default="zero-shot-react-description",
        description="Type of agent to use"
    )
    system_message: Optional[str] = Field(
        default=None,
        description="System message for the agent"
    )


class AgentAction(BaseModel):
    """
    Agent action representation
    """
    tool: str = Field(..., description="Name of the tool to use")
    tool_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters for the tool"
    )
    log: str = Field(..., description="Log of the action")
    thoughts: Optional[str] = Field(
        default=None,
        description="Agent's thoughts behind this action"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class AgentFinish(BaseModel):
    """
    Agent finish representation
    """
    return_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Return values from the agent"
    )
    log: str = Field(..., description="Final log message")
    thoughts: Optional[str] = Field(
        default=None,
        description="Agent's final thoughts"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class AgentStep(BaseModel):
    """
    Single step in agent execution
    """
    action: Optional[AgentAction] = Field(
        default=None,
        description="Action taken in this step"
    )
    observation: Optional[str] = Field(
        default=None,
        description="Observation from the action"
    )
    thoughts: Optional[str] = Field(
        default=None,
        description="Thoughts for this step"
    )
    timestamp: str = Field(
        default_factory=lambda: "",
        description="Timestamp of the step"
    )
    step_number: int = Field(
        default=0,
        description="Step number in the execution"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class AgentResult(BaseModel):
    """
    Result of agent execution
    """
    output: Optional[str] = Field(
        default=None,
        description="Final output of the agent"
    )
    intermediate_steps: List[AgentStep] = Field(
        default_factory=list,
        description="Intermediate steps taken during execution"
    )
    return_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Return values from agent"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    execution_time: float = Field(
        default=0.0,
        description="Total execution time in seconds"
    )
    iterations_used: int = Field(
        default=0,
        description="Number of iterations used"
    )
    finished: bool = Field(
        default=False,
        description="Whether the agent finished successfully"
    )


class ToolInfo(BaseModel):
    """
    Information about a tool
    """
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="Schema for tool input"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class AgentState(BaseModel):
    """
    Current state of the agent
    """
    input: str = Field(..., description="Current input being processed")
    current_step: int = Field(
        default=0,
        description="Current step number"
    )
    thoughts: Optional[str] = Field(
        default=None,
        description="Current thoughts"
    )
    last_action: Optional[AgentAction] = Field(
        default=None,
        description="Last action taken"
    )
    last_observation: Optional[str] = Field(
        default=None,
        description="Last observation received"
    )
    finished: bool = Field(
        default=False,
        description="Whether execution is finished"
    )
    error: Optional[str] = Field(
        default=None,
        description="Current error if any"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional state metadata"
    )


class AgentThought(BaseModel):
    """
    Agent thought process
    """
    text: str = Field(..., description="Thought text")
    category: str = Field(
        default="general",
        description="Category of thought: 'analysis', 'planning', 'reflection', 'general'"
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence level (0.0 to 1.0)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class AgentPlan(BaseModel):
    """
    Agent's plan for solving a problem
    """
    steps: List[str] = Field(
        default_factory=list,
        description="Planned steps"
    )
    reasoning: str = Field(
        default="",
        description="Reasoning behind the plan"
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence in the plan (0.0 to 1.0)"
    )
    alternatives: List[List[str]] = Field(
        default_factory=list,
        description="Alternative plans"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# Error types
class AgentError(Exception):
    """
    Base agent error class
    """
    message: str
    details: Dict[str, Any] = {}

    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class AgentValidationError(AgentError):
    """
    Agent validation error
    """
    def __init__(self, message: str, data: Any = None):
        details = {"validation_error": True}
        if data is not None:
            details["data"] = data
        super().__init__(message, details)


class AgentExecutionError(AgentError):
    """
    Agent execution error
    """
    step: str
    cause: Optional[Exception] = None

    def __init__(self, message: str, step: str, cause: Exception = None, details: Dict[str, Any] = None):
        self.step = step
        self.cause = cause
        execution_details = {
            "step": step,
            "execution_error": True
        }
        if cause:
            execution_details["cause"] = str(cause)
        if details:
            execution_details.update(details)
        super().__init__(message, execution_details)


class AgentOutputParserError(AgentError):
    """
    Agent output parsing error
    """
    output: str
    expected_format: str

    def __init__(self, message: str, output: str, expected_format: str = ""):
        self.output = output
        self.expected_format = expected_format
        parser_details = {
            "output": output,
            "expected_format": expected_format,
            "parser_error": True
        }
        super().__init__(message, parser_details)


class AgentTimeoutError(AgentError):
    """
    Agent timeout error
    """
    timeout_seconds: float

    def __init__(self, message: str, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        timeout_details = {
            "timeout_seconds": timeout_seconds,
            "timeout_error": True
        }
        super().__init__(message, timeout_details)


class AgentToolError(AgentError):
    """
    Agent tool execution error
    """
    tool_name: str
    tool_input: Dict[str, Any]

    def __init__(self, message: str, tool_name: str, tool_input: Dict[str, Any], cause: Exception = None):
        self.tool_name = tool_name
        self.tool_input = tool_input
        tool_details = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_error": True
        }
        if cause:
            tool_details["cause"] = str(cause)
        super().__init__(message, tool_details)


# Enums for agent types and states
class AgentType(str, Enum):
    """Agent type enumeration"""
    ZERO_SHOT_REACT = "zero-shot-react-description"
    REACT_DOCSTORE = "react-docstore"
    SELF_ASK_WITH_SEARCH = "self-ask-with-search"
    CONVERSATIONAL_REACT = "conversational-react-description"
    CHAT_ZERO_SHOT_REACT = "chat-zero-shot-react-description"
    CHAT_CONVERSATIONAL_REACT = "chat-conversational-react-description"
    STRUCTURED_CHAT_ZERO_SHOT_REACT = "structured-chat-zero-shot-react-description"


class AgentState(str, Enum):
    """Agent state enumeration"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"


class AgentStepType(str, Enum):
    """Agent step type enumeration"""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINISH = "finish"
    ERROR = "error"


class ThoughtCategory(str, Enum):
    """Thought category enumeration"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    REFLECTION = "reflection"
    DECISION = "decision"
    GENERAL = "general"
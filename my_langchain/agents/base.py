# -*- coding: utf-8 -*-
"""
Base agent implementation
"""

import time
from typing import Any, Dict, List, Optional, Union

from my_langchain.base.base import BaseComponent
from my_langchain.agents.types import (
    AgentConfig, AgentResult, AgentAction, AgentFinish, AgentStep,
    AgentState, AgentError, AgentValidationError, AgentExecutionError,
    AgentOutputParserError, AgentTimeoutError, AgentToolError
)
from pydantic import ConfigDict, Field


class BaseAgent(BaseComponent):
    """
    Base agent implementation providing common functionality

    This class defines the interface that all agent implementations must follow
    and provides common utility methods for agent execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: AgentConfig = Field(..., description="Agent configuration")
    tools: List[Any] = Field(default_factory=list, description="Available tools")

    def __init__(self, config: Optional[AgentConfig] = None, tools: Optional[List[Any]] = None, **kwargs):
        """
        Initialize agent

        Args:
            config: Agent configuration
            tools: List of available tools
            **kwargs: Additional parameters
        """
        if config is None:
            config = AgentConfig()

        super().__init__(config=config, tools=tools or [], **kwargs)

    def plan(self, input_text: str, **kwargs) -> Union[AgentAction, AgentFinish]:
        """
        Plan next action based on input

        Args:
            input_text: Input text to process
            **kwargs: Additional parameters

        Returns:
            Either an action to take or a finish signal

        Raises:
            AgentValidationError: If input is invalid
            AgentExecutionError: If planning fails
        """
        # Default implementation - just finish with the input as output
        return AgentFinish(
            return_values={"output": input_text},
            log=f"BaseAgent default processing: {input_text}"
        )

    def get_allowed_tools(self) -> List[str]:
        """
        Get list of allowed tool names

        Returns:
            List of tool names this agent can use
        """
        return [tool.config.name if hasattr(tool, 'config') else str(tool) for tool in self.tools]

    def execute(self, input_text: str, **kwargs) -> AgentResult:
        """
        Execute agent with input

        Args:
            input_text: Input to process
            **kwargs: Additional parameters

        Returns:
            Agent execution result

        Raises:
            AgentValidationError: If input is invalid
            AgentExecutionError: If execution fails
            AgentTimeoutError: If execution times out
        """
        start_time = time.time()

        try:
            # Validate input
            self._validate_input(input_text)

            # Initialize execution state
            state = AgentState(input=input_text, current_step=0)
            intermediate_steps: List[AgentStep] = []

            # Execute with iteration limit
            for iteration in range(self.config.max_iterations):
                state.current_step = iteration + 1

                try:
                    # Plan next action
                    result = self.plan(input_text, **kwargs)

                    if isinstance(result, AgentFinish):
                        # Agent finished successfully
                        intermediate_steps.append(AgentStep(
                            action=None,
                            observation=None,
                            thoughts=result.thoughts,
                            step_number=iteration + 1,
                            metadata={"finished": True}
                        ))

                        return AgentResult(
                            output=result.return_values.get("output"),
                            intermediate_steps=intermediate_steps,
                            return_values=result.return_values,
                            execution_time=time.time() - start_time,
                            iterations_used=iteration + 1,
                            finished=True,
                            metadata=result.metadata
                        )

                    elif isinstance(result, AgentAction):
                        # Execute action
                        observation = self._execute_action(result)

                        # Record step
                        step = AgentStep(
                            action=result,
                            observation=observation,
                            thoughts=result.thoughts,
                            step_number=iteration + 1,
                            metadata={"tool_used": result.tool}
                        )
                        intermediate_steps.append(step)

                        # Update input for next iteration
                        input_text = f"Thought: {result.thoughts}\nAction: {result.tool}\nObservation: {observation}"

                        if self.config.verbose:
                            print(f"Step {iteration + 1}:")
                            print(f"  Thought: {result.thoughts}")
                            print(f"  Action: {result.tool}")
                            print(f"  Observation: {observation}")
                            print()

                    else:
                        raise AgentValidationError(
                            f"Invalid planning result type: {type(result)}",
                            data=result
                        )

                except Exception as e:
                    # Handle execution errors
                    if self.config.handle_parsing_errors:
                        error_msg = f"Error in step {iteration + 1}: {str(e)}"
                        intermediate_steps.append(AgentStep(
                            action=None,
                            observation=error_msg,
                            thoughts=f"Encountered an error: {str(e)}",
                            step_number=iteration + 1,
                            metadata={"error": True, "error_type": type(e).__name__}
                        ))

                        if self.config.verbose:
                            print(f"Error in step {iteration + 1}: {error_msg}")

                        # Continue to next iteration based on early stopping method
                        if self.config.early_stopping_method == "force":
                            # Force finish with error
                            return AgentResult(
                                output=f"Execution failed: {error_msg}",
                                intermediate_steps=intermediate_steps,
                                execution_time=time.time() - start_time,
                                iterations_used=iteration + 1,
                                finished=False,
                                metadata={"error": error_msg}
                            )
                        else:
                            # Continue to next iteration
                            continue
                    else:
                        raise AgentExecutionError(
                            f"Execution failed at step {iteration + 1}",
                            step=f"iteration_{iteration + 1}",
                            cause=e
                        )

            # Max iterations reached
            return AgentResult(
                output="Maximum iterations reached without finding a solution",
                intermediate_steps=intermediate_steps,
                execution_time=time.time() - start_time,
                iterations_used=self.config.max_iterations,
                finished=False,
                metadata={"max_iterations_reached": True}
            )

        except Exception as e:
            # Handle top-level errors
            execution_time = time.time() - start_time
            if isinstance(e, AgentError):
                raise

            return AgentResult(
                output=f"Execution failed: {str(e)}",
                intermediate_steps=[],
                execution_time=execution_time,
                iterations_used=0,
                finished=False,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )

    def _validate_input(self, input_text: str) -> None:
        """
        Validate input text

        Args:
            input_text: Input text to validate

        Raises:
            AgentValidationError: If input is invalid
        """
        if not input_text or not input_text.strip():
            raise AgentValidationError("Input text cannot be empty")

        if len(input_text) > 10000:  # Reasonable limit
            raise AgentValidationError("Input text is too long (max 10000 characters)")

    def _execute_action(self, action: AgentAction) -> str:
        """
        Execute an agent action

        Args:
            action: Action to execute

        Returns:
            Observation from executing the action

        Raises:
            AgentToolError: If tool execution fails
            AgentValidationError: If action is invalid
        """
        # Validate tool
        if not action.tool:
            raise AgentValidationError("Action must specify a tool")

        allowed_tools = self.get_allowed_tools()
        if action.tool not in allowed_tools:
            raise AgentValidationError(
                f"Tool '{action.tool}' is not available. Available tools: {allowed_tools}"
            )

        # Find and execute tool
        tool = self._get_tool_by_name(action.tool)
        if not tool:
            raise AgentValidationError(f"Tool '{action.tool}' not found")

        try:
            # Execute tool with input
            result = tool.invoke(action.tool_input)
            return str(result) if result is not None else "Tool returned no result"

        except Exception as e:
            if self.config.handle_tool_errors:
                error_msg = f"Error executing tool '{action.tool}': {str(e)}"
                if self.config.verbose:
                    print(f"Tool error: {error_msg}")
                return error_msg
            else:
                raise AgentToolError(
                    f"Failed to execute tool '{action.tool}'",
                    tool_name=action.tool,
                    tool_input=action.tool_input,
                    cause=e
                )

    def _get_tool_by_name(self, name: str) -> Optional[Any]:
        """
        Get tool by name

        Args:
            name: Tool name to find

        Returns:
            Tool instance or None if not found
        """
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == name:
                return tool
            elif hasattr(tool, '__name__') and tool.__name__ == name:
                return tool
        return None

    def add_tool(self, tool: Any) -> None:
        """
        Add a tool to the agent

        Args:
            tool: Tool to add
        """
        if tool not in self.tools:
            self.tools.append(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool by name

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        for i, tool in enumerate(self.tools):
            tool_name_attr = getattr(tool, 'name', getattr(tool, '__name__', None))
            if tool_name_attr == tool_name:
                self.tools.pop(i)
                return True
        return False

    def get_tool_info(self) -> List[Dict[str, Any]]:
        """
        Get information about available tools

        Returns:
            List of tool information dictionaries
        """
        tool_info = []
        for tool in self.tools:
            info = {
                "name": getattr(tool, 'name', getattr(tool, '__name__', 'Unknown')),
                "description": getattr(tool, 'description', 'No description available'),
            }

            # Add input schema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                info["input_schema"] = tool.args_schema.model_json_schema()

            tool_info.append(info)

        return tool_info

    def run(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> AgentResult:
        """
        Implement BaseComponent run method

        Args:
            inputs: Input string or dictionary with 'input' key
            **kwargs: Additional parameters

        Returns:
            Agent execution result
        """
        if isinstance(inputs, dict):
            input_text = inputs.get("input", str(inputs))
        else:
            input_text = str(inputs)

        return self.execute(input_text, **kwargs)

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent

        Returns:
            Agent information dictionary
        """
        return {
            "agent_type": self.__class__.__name__,
            "config": self.config.model_dump(),
            "tools_count": len(self.tools),
            "allowed_tools": self.get_allowed_tools(),
            "tool_info": self.get_tool_info()
        }

    def reset(self) -> None:
        """
        Reset agent state (if any)
        """
        # Default implementation does nothing
        # Override in subclasses if needed
        pass

    def save_state(self) -> Dict[str, Any]:
        """
        Save current agent state

        Returns:
            Serialized state dictionary
        """
        return {
            "agent_type": self.__class__.__name__,
            "config": self.config.model_dump(),
            "tools_info": self.get_tool_info()
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load agent state

        Args:
            state: Serialized state dictionary
        """
        # Default implementation does nothing
        # Override in subclasses if needed
        pass
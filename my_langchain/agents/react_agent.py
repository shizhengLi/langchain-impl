# -*- coding: utf-8 -*-
"""
ReAct (Reasoning and Acting) Agent implementation
"""

import re
from typing import Any, Dict, List, Optional, Union

from my_langchain.agents.base import BaseAgent
from my_langchain.agents.types import (
    AgentConfig, AgentResult, AgentAction, AgentFinish,
    AgentValidationError, AgentOutputParserError
)
from my_langchain.llms import BaseLLM
from my_langchain.prompts import PromptTemplate


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning and Acting) Agent

    This agent implements the ReAct framework which combines reasoning
    and acting in an interleaved manner. The agent thinks about what
    action to take, executes it, observes the result, and then decides
    whether to continue or finish.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: List[Any],
        config: Optional[AgentConfig] = None,
        **kwargs
    ):
        """
        Initialize ReAct agent

        Args:
            llm: Language model to use for reasoning
            tools: List of available tools
            config: Agent configuration
            **kwargs: Additional parameters
        """
        super().__init__(config=config or AgentConfig(), tools=tools, **kwargs)
        self._llm = llm  # Make private to avoid Pydantic conflicts
        self._prompt_template = self._create_prompt_template()

    @property
    def llm(self):
        """Get the LLM instance (for backward compatibility)"""
        return self._llm

    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create the ReAct prompt template

        Returns:
            Prompt template for ReAct reasoning
        """
        template = """You are a helpful assistant that uses tools to answer questions.

Available tools:
{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "tool_descriptions", "tool_names", "agent_scratchpad"]
        )

    def plan(self, input_text: str, **kwargs) -> Union[AgentAction, AgentFinish]:
        """
        Plan next action using ReAct reasoning

        Args:
            input_text: Input text to process
            **kwargs: Additional parameters (should include 'agent_scratchpad')

        Returns:
            Either an action to take or a finish signal

        Raises:
            AgentValidationError: If input is invalid
            AgentOutputParserError: If LLM output cannot be parsed
        """
        # Prepare tool descriptions
        tool_descriptions = self._get_tool_descriptions()
        tool_names = ", ".join(self.get_allowed_tools())

        # Get scratchpad from kwargs or use empty
        agent_scratchpad = kwargs.get("agent_scratchpad", "")

        # Create prompt
        prompt_inputs = {
            "input": input_text,
            "tool_descriptions": tool_descriptions,
            "tool_names": tool_names,
            "agent_scratchpad": agent_scratchpad
        }

        try:
            # Generate response from LLM
            response = self._llm.invoke(self._prompt_template.format(**prompt_inputs))

            # Parse the response
            return self._parse_response(response)

        except Exception as e:
            if isinstance(e, (AgentValidationError, AgentOutputParserError)):
                raise
            raise AgentValidationError(f"Failed to generate plan: {str(e)}")

    def _get_tool_descriptions(self) -> str:
        """
        Get formatted tool descriptions

        Returns:
            Formatted string describing available tools
        """
        descriptions = []
        for tool in self.tools:
            name = getattr(tool, 'name', getattr(tool, '__name__', 'Unknown'))
            description = getattr(tool, 'description', 'No description available')
            descriptions.append(f"{name}: {description}")

        return "\n".join(descriptions)

    def _parse_response(self, response: str) -> Union[AgentAction, AgentFinish]:
        """
        Parse LLM response into action or finish

        Args:
            response: LLM response string

        Returns:
            Parsed action or finish

        Raises:
            AgentOutputParserError: If response cannot be parsed
        """
        response = response.strip()

        # Check for Final Answer
        final_answer_match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            return AgentFinish(
                return_values={"output": final_answer},
                log=response,
                thoughts=self._extract_thoughts(response)
            )

        # Parse Action and Action Input
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?:\n|\nThought:|\nObservation:|$)", response, re.IGNORECASE | re.DOTALL)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input_str = action_input_match.group(1).strip()

            # Parse action input
            try:
                action_input = self._parse_action_input(action_input_str)
            except Exception as e:
                raise AgentOutputParserError(
                    f"Failed to parse action input '{action_input_str}': {str(e)}",
                    output=response,
                    expected_format="Action Input: <valid JSON or simple string>"
                )

            # Validate action
            allowed_tools = self.get_allowed_tools()
            if action not in allowed_tools:
                raise AgentOutputParserError(
                    f"Action '{action}' is not allowed. Available actions: {allowed_tools}",
                    output=response
                )

            return AgentAction(
                tool=action,
                tool_input=action_input,
                log=response,
                thoughts=self._extract_thoughts(response)
            )

        # If no clear action or final answer, try to extract thoughts and continue
        thoughts = self._extract_thoughts(response)
        if thoughts:
            # Check if response indicates completion without explicit "Final Answer"
            if any(keyword in response.lower() for keyword in ["i now know", "the answer is", "i have found"]):
                # Try to extract an answer
                answer_match = re.search(r"(?:the answer is|i have found|i now know that?)\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
                if answer_match:
                    answer = answer_match.group(1).strip()
                    return AgentFinish(
                        return_values={"output": answer},
                        log=response,
                        thoughts=thoughts
                    )

        # If we get here, parsing failed
        raise AgentOutputParserError(
            "Could not parse LLM response. Expected format with 'Action:' and 'Action Input:' or 'Final Answer:'",
            output=response,
            expected_format="Action: <tool_name>\nAction Input: <tool_input>\nOR\nFinal Answer: <answer>"
        )

    def _parse_action_input(self, action_input_str: str) -> Dict[str, Any]:
        """
        Parse action input string into dictionary

        Args:
            action_input_str: Action input string

        Returns:
            Parsed action input dictionary

        Raises:
            AgentOutputParserError: If parsing fails
        """
        action_input_str = action_input_str.strip()

        # Try to parse as JSON
        if action_input_str.startswith("{") and action_input_str.endswith("}"):
            try:
                import json
                return json.loads(action_input_str)
            except json.JSONDecodeError:
                pass  # Continue to other parsing methods

        # Try to parse as key=value pairs
        if "=" in action_input_str:
            result = {}
            pairs = action_input_str.split(",")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    result[key.strip()] = value.strip().strip('"\'')
            if result:
                return result

        # Treat as simple string input
        return {"input": action_input_str}

    def _extract_thoughts(self, response: str) -> str:
        """
        Extract thoughts from response

        Args:
            response: LLM response

        Returns:
            Extracted thoughts or empty string
        """
        # Look for Thought: content
        thought_matches = re.findall(r"Thought:\s*(.+?)(?:\n(?:Action|Final Answer)|$)", response, re.IGNORECASE | re.DOTALL)
        if thought_matches:
            # Return the last thought
            return thought_matches[-1].strip()

        # If no explicit "Thought:", try to infer from context
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith(("Action:", "Action Input:", "Final Answer:", "Observation:")):
                return line

        return ""

    def get_allowed_tools(self) -> List[str]:
        """
        Get list of allowed tool names

        Returns:
            List of tool names this agent can use
        """
        tool_names = []
        for tool in self.tools:
            name = getattr(tool, 'name', getattr(tool, '__name__', None))
            if name:
                tool_names.append(name)
        return tool_names

    def create_scratchpad(self, intermediate_steps: List) -> str:
        """
        Create scratchpad from intermediate steps

        Args:
            intermediate_steps: List of intermediate steps

        Returns:
            Formatted scratchpad string
        """
        if not intermediate_steps:
            return ""

        scratchpad_parts = []
        for step in intermediate_steps:
            if hasattr(step, 'action') and step.action:
                scratchpad_parts.append(f"Thought: {step.action.thoughts or ''}")
                scratchpad_parts.append(f"Action: {step.action.tool}")
                action_input_str = str(step.action.tool_input)
                scratchpad_parts.append(f"Action Input: {action_input_str}")

            if hasattr(step, 'observation') and step.observation:
                scratchpad_parts.append(f"Observation: {step.observation}")

        return "\n".join(scratchpad_parts)

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the ReAct agent

        Returns:
            Agent information dictionary
        """
        base_info = super().get_agent_info()
        base_info.update({
            "agent_type": "ReActAgent",
            "llm_type": self._llm.__class__.__name__,
            "prompt_template": self._prompt_template.template,
            "reasoning_style": "ReAct (Reasoning and Acting)"
        })
        return base_info
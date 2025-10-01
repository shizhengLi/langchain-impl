# -*- coding: utf-8 -*-
"""
Unit tests for agent module
"""
import pytest
from typing import Dict, Any

from my_langchain.agents import BaseAgent, ReActAgent, ZeroShotAgent
from my_langchain.agents.types import (
    AgentConfig, AgentAction, AgentFinish, AgentResult, AgentStep,
    AgentError, AgentValidationError, AgentExecutionError, AgentOutputParserError,
    AgentState, AgentAction, AgentFinish
)
from my_langchain.llms import MockLLM
from my_langchain.tools import Tool, CalculatorTool, SearchTool


class TestAgentConfig:
    """Test AgentConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = AgentConfig()
        assert config.max_iterations == 10
        assert config.early_stopping_method == "force"
        assert config.return_intermediate_steps is True
        assert config.handle_parsing_errors is True
        assert config.verbose is False
        assert config.agent_type == "zero-shot-react-description"

    def test_custom_config(self):
        """Test custom configuration"""
        config = AgentConfig(
            max_iterations=5,
            early_stopping_method="generate",
            return_intermediate_steps=False,
            verbose=True,
            agent_type="react-docstore"
        )
        assert config.max_iterations == 5
        assert config.early_stopping_method == "generate"
        assert config.return_intermediate_steps is False
        assert config.verbose is True
        assert config.agent_type == "react-docstore"


class TestAgentAction:
    """Test AgentAction class"""

    def test_action_creation(self):
        """Test action creation"""
        action = AgentAction(
            tool="search",
            tool_input={"query": "Python"},
            log="I need to search for Python information",
            thoughts="I should search for information about Python"
        )
        assert action.tool == "search"
        assert action.tool_input["query"] == "Python"
        assert "search for Python information" in action.log
        assert action.thoughts == "I should search for information about Python"

    def test_action_with_defaults(self):
        """Test action with default values"""
        action = AgentAction(
            tool="calculator",
            tool_input={"expression": "2+2"},
            log="Calculate 2+2"
        )
        assert action.tool == "calculator"
        assert action.tool_input["expression"] == "2+2"
        assert action.thoughts is None
        assert action.metadata == {}


class TestAgentFinish:
    """Test AgentFinish class"""

    def test_finish_creation(self):
        """Test finish creation"""
        finish = AgentFinish(
            return_values={"output": "The answer is 4"},
            log="I have calculated the result",
            thoughts="I now know the final answer"
        )
        assert finish.return_values["output"] == "The answer is 4"
        assert "calculated the result" in finish.log
        assert finish.thoughts == "I now know the final answer"

    def test_finish_with_defaults(self):
        """Test finish with default values"""
        finish = AgentFinish(
            return_values={"output": "Done"},
            log="Task completed"
        )
        assert finish.return_values["output"] == "Done"
        assert finish.thoughts is None
        assert finish.metadata == {}


class TestAgentStep:
    """Test AgentStep class"""

    def test_step_creation(self):
        """Test step creation"""
        action = AgentAction(
            tool="search",
            tool_input={"query": "test"},
            log="Searching for test"
        )
        step = AgentStep(
            action=action,
            observation="Found relevant information",
            thoughts="I should search for information",
            step_number=1
        )
        assert step.action == action
        assert step.observation == "Found relevant information"
        assert step.thoughts == "I should search for information"
        assert step.step_number == 1

    def test_step_with_defaults(self):
        """Test step with default values"""
        step = AgentStep(
            observation="No action taken"
        )
        assert step.action is None
        assert step.observation == "No action taken"
        assert step.thoughts is None
        assert step.step_number == 0
        assert step.metadata == {}


class TestAgentResult:
    """Test AgentResult class"""

    def test_result_creation(self):
        """Test result creation"""
        steps = [
            AgentStep(
                action=AgentAction(tool="search", tool_input={}, log=""),
                observation="Found info",
                step_number=1
            )
        ]
        result = AgentResult(
            output="Final answer",
            intermediate_steps=steps,
            return_values={"output": "Final answer"},
            execution_time=1.5,
            iterations_used=3,
            finished=True
        )
        assert result.output == "Final answer"
        assert len(result.intermediate_steps) == 1
        assert result.return_values["output"] == "Final answer"
        assert result.execution_time == 1.5
        assert result.iterations_used == 3
        assert result.finished is True


class TestBaseAgent:
    """Test BaseAgent class"""

    def test_base_agent_creation(self):
        """Test base agent creation"""
        config = AgentConfig(max_iterations=5)
        tools = [CalculatorTool()]
        agent = BaseAgent(config=config, tools=tools)

        assert agent.config.max_iterations == 5
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], CalculatorTool)

    def test_base_agent_validation(self):
        """Test input validation"""
        agent = BaseAgent()

        # Valid input
        try:
            agent._validate_input("Hello world")
        except AgentValidationError:
            pytest.fail("Valid input should not raise ValidationError")

        # Invalid input (empty)
        with pytest.raises(AgentValidationError):
            agent._validate_input("")

        # Invalid input (too long)
        with pytest.raises(AgentValidationError):
            agent._validate_input("x" * 10001)

    def test_tool_management(self):
        """Test tool management"""
        agent = BaseAgent()
        calculator = CalculatorTool()
        search_tool = SearchTool()

        # Add tools
        agent.add_tool(calculator)
        agent.add_tool(search_tool)
        assert len(agent.tools) == 2

        # Get tool by name
        tool = agent._get_tool_by_name("calculator")
        assert tool == calculator

        # Remove tool
        removed = agent.remove_tool("calculator")
        assert removed is True
        assert len(agent.tools) == 1
        assert agent._get_tool_by_name("calculator") is None

        # Try to remove non-existent tool
        removed = agent.remove_tool("non_existent")
        assert removed is False

    def test_get_tool_info(self):
        """Test getting tool information"""
        calculator = CalculatorTool()
        agent = BaseAgent(tools=[calculator])

        tool_info = agent.get_tool_info()
        assert len(tool_info) == 1
        assert tool_info[0]["name"] == "calculator"
        assert "description" in tool_info[0]

    def test_execute_action(self):
        """Test action execution"""
        calculator = CalculatorTool()
        agent = BaseAgent(tools=[calculator])

        action = AgentAction(
            tool="calculator",
            tool_input={"expression": "2+2"},
            log="Calculate 2+2"
        )

        observation = agent._execute_action(action)
        assert "4" in observation  # Calculator should return "4"

    def test_execute_action_invalid_tool(self):
        """Test executing action with invalid tool"""
        agent = BaseAgent()

        action = AgentAction(
            tool="non_existent_tool",
            tool_input={},
            log="Use non-existent tool"
        )

        with pytest.raises(AgentValidationError):
            agent._execute_action(action)

    def test_get_allowed_tools(self):
        """Test getting allowed tools"""
        calculator = CalculatorTool()
        search_tool = SearchTool()
        agent = BaseAgent(tools=[calculator, search_tool])

        allowed_tools = agent.get_allowed_tools()
        assert "calculator" in allowed_tools
        assert "search" in allowed_tools
        assert len(allowed_tools) == 2


class TestReActAgent:
    """Test ReActAgent class"""

    def test_react_agent_creation(self):
        """Test ReAct agent creation"""
        llm = MockLLM(temperature=0.0)
        tools = [CalculatorTool()]
        agent = ReActAgent(llm=llm, tools=tools)

        assert agent.llm == llm
        assert len(agent.tools) == 1
        assert agent.config.max_iterations == 10

    def test_react_agent_custom_config(self):
        """Test ReAct agent with custom config"""
        llm = MockLLM(temperature=0.0)
        config = AgentConfig(max_iterations=5, verbose=True)
        tools = [CalculatorTool()]
        agent = ReActAgent(llm=llm, tools=tools, config=config)

        assert agent.config.max_iterations == 5
        assert agent.config.verbose is True

    def test_react_agent_tool_descriptions(self):
        """Test tool descriptions generation"""
        llm = MockLLM(temperature=0.0)
        calculator = CalculatorTool()
        search_tool = SearchTool()
        agent = ReActAgent(llm=llm, tools=[calculator, search_tool])

        descriptions = agent._get_tool_descriptions()
        assert "calculator:" in descriptions
        assert "search:" in descriptions
        assert "Calculate mathematical expressions" in descriptions
        assert "Search for information" in descriptions

    def test_react_agent_parse_action_response(self):
        """Test parsing action response"""
        llm = MockLLM(temperature=0.0)
        agent = ReActAgent(llm=llm, tools=[CalculatorTool()])

        response = """Thought: I need to calculate 2+2
Action: calculator
Action Input: {"expression": "2+2"}"""

        action = agent._parse_response(response)
        assert isinstance(action, AgentAction)
        assert action.tool == "calculator"
        assert action.tool_input["expression"] == "2+2"

    def test_react_agent_parse_finish_response(self):
        """Test parsing finish response"""
        llm = MockLLM(temperature=0.0)
        agent = ReActAgent(llm=llm, tools=[CalculatorTool()])

        response = """Thought: I have calculated the result
Final Answer: The result is 4"""

        finish = agent._parse_response(response)
        assert isinstance(finish, AgentFinish)
        assert finish.return_values["output"] == "The result is 4"

    def test_react_agent_parse_invalid_response(self):
        """Test parsing invalid response"""
        llm = MockLLM(temperature=0.0)
        agent = ReActAgent(llm=llm, tools=[CalculatorTool()])

        response = "This is not a valid agent response"

        with pytest.raises(AgentOutputParserError):
            agent._parse_response(response)

    def test_react_agent_parse_action_input(self):
        """Test parsing action input"""
        llm = MockLLM(temperature=0.0)
        agent = ReActAgent(llm=llm, tools=[CalculatorTool()])

        # Test JSON input
        json_input = '{"expression": "2+2"}'
        result = agent._parse_action_input(json_input)
        assert result["expression"] == "2+2"

        # Test key=value input
        kv_input = 'expression=2+2'
        result = agent._parse_action_input(kv_input)
        assert result["expression"] == "2+2"

        # Test simple string input
        simple_input = '2+2'
        result = agent._parse_action_input(simple_input)
        assert result["input"] == "2+2"

    def test_react_agent_extract_thoughts(self):
        """Test extracting thoughts from response"""
        llm = MockLLM(temperature=0.0)
        agent = ReActAgent(llm=llm, tools=[CalculatorTool()])

        response = """Thought: I need to calculate something
Action: calculator"""
        thoughts = agent._extract_thoughts(response)
        assert thoughts == "I need to calculate something"

    def test_react_agent_create_scratchpad(self):
        """Test creating scratchpad from steps"""
        llm = MockLLM(temperature=0.0)
        agent = ReActAgent(llm=llm, tools=[CalculatorTool()])

        action = AgentAction(
            tool="calculator",
            tool_input={"expression": "2+2"},
            thoughts="Calculate 2+2",
            log="Using calculator to compute 2+2"
        )
        step = AgentStep(action=action, observation="4")

        scratchpad = agent.create_scratchpad([step])
        assert "Calculate 2+2" in scratchpad
        assert "calculator" in scratchpad
        assert "4" in scratchpad

    def test_react_agent_get_agent_info(self):
        """Test getting agent information"""
        llm = MockLLM(temperature=0.0)
        agent = ReActAgent(llm=llm, tools=[CalculatorTool()])

        info = agent.get_agent_info()
        assert info["agent_type"] == "ReActAgent"
        assert info["llm_type"] == "MockLLM"
        assert "prompt_template" in info
        assert info["reasoning_style"] == "ReAct (Reasoning and Acting)"


class TestZeroShotAgent:
    """Test ZeroShotAgent class"""

    def test_zero_shot_agent_creation(self):
        """Test Zero Shot agent creation"""
        llm = MockLLM(temperature=0.0)
        tools = [CalculatorTool()]
        agent = ZeroShotAgent(llm=llm, tools=tools)

        assert agent.llm == llm
        assert len(agent.tools) == 1
        assert agent.config.max_iterations == 10

    def test_zero_shot_agent_differentiation(self):
        """Test Zero Shot agent is different from ReAct"""
        llm = MockLLM(temperature=0.0)
        tools = [CalculatorTool()]

        react_agent = ReActAgent(llm=llm, tools=tools)
        zero_shot_agent = ZeroShotAgent(llm=llm, tools=tools)

        react_info = react_agent.get_agent_info()
        zero_shot_info = zero_shot_agent.get_agent_info()

        assert react_info["agent_type"] == "ReActAgent"
        assert zero_shot_info["agent_type"] == "ZeroShotAgent"
        assert react_info["reasoning_style"] == "ReAct (Reasoning and Acting)"
        assert zero_shot_info["reasoning_style"] == "Zero-shot reasoning"


class TestAgentErrorHandling:
    """Test agent error handling"""

    def test_agent_validation_error(self):
        """Test AgentValidationError"""
        error = AgentValidationError("Invalid input", data={"input": "bad"})
        assert error.message == "Invalid input"
        assert error.details["data"] == {"input": "bad"}

    def test_agent_execution_error(self):
        """Test AgentExecutionError"""
        error = AgentExecutionError("Execution failed", step="planning")
        assert error.message == "Execution failed"
        assert error.step == "planning"
        assert error.details["execution_error"] is True

    def test_agent_output_parser_error(self):
        """Test AgentOutputParserError"""
        error = AgentOutputParserError(
            "Parse failed",
            output="bad output",
            expected_format="Action: tool\nAction Input: input"
        )
        assert error.message == "Parse failed"
        assert error.output == "bad output"
        assert error.expected_format == "Action: tool\nAction Input: input"


class TestAgentIntegration:
    """Test agent integration scenarios"""

    def test_agent_with_simple_tool(self):
        """Test agent with simple custom tool"""
        def hello_tool(name: str) -> str:
            return f"Hello, {name}!"

        tool = Tool(
            name="hello",
            func=hello_tool,
            description="Say hello to someone"
        )

        llm = MockLLM(temperature=0.0)
        agent = ReActAgent(llm=llm, tools=[tool])

        allowed_tools = agent.get_allowed_tools()
        assert "hello" in allowed_tools

        tool_info = agent.get_tool_info()
        assert tool_info[0]["name"] == "hello"

    def test_agent_configuration_variations(self):
        """Test agent with different configurations"""
        llm = MockLLM(temperature=0.0)
        calculator = CalculatorTool()

        configs = [
            AgentConfig(max_iterations=1, early_stopping_method="force"),
            AgentConfig(max_iterations=5, early_stopping_method="generate"),
            AgentConfig(verbose=True, handle_parsing_errors=False),
            AgentConfig(return_intermediate_steps=False)
        ]

        for config in configs:
            agent = ReActAgent(llm=llm, tools=[calculator], config=config)
            assert agent.config == config

    def test_agent_state_serialization(self):
        """Test agent state save/load"""
        llm = MockLLM(temperature=0.0)
        calculator = CalculatorTool()
        agent = ReActAgent(llm=llm, tools=[calculator])

        state = agent.save_state()
        assert "agent_type" in state
        assert "config" in state
        assert "tools_info" in state

        # Test that we can create a new agent with the same config
        new_config = AgentConfig(**state["config"])
        new_agent = ReActAgent(llm=llm, tools=[calculator], config=new_config)
        assert new_agent.config.max_iterations == agent.config.max_iterations

    def test_agent_tool_validation(self):
        """Test agent tool validation"""
        llm = MockLLM(temperature=0.0)
        calculator = CalculatorTool()

        agent = ReActAgent(llm=llm, tools=[calculator])

        # Test valid action
        valid_action = AgentAction(
            tool="calculator",
            tool_input={"expression": "2+2"},
            log="Calculate 2+2"
        )

        try:
            agent._execute_action(valid_action)
        except Exception:
            pytest.fail("Valid action should not raise exception")

        # Test invalid action
        invalid_action = AgentAction(
            tool="non_existent",
            tool_input={},
            log="Use non-existent tool"
        )

        with pytest.raises(AgentValidationError):
            agent._execute_action(invalid_action)
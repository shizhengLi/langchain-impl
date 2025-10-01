# -*- coding: utf-8 -*-
"""
LLM Chain implementation
"""

from typing import Any, Dict, List, Optional, Union

from my_langchain.chains.base import BaseChain
from my_langchain.chains.types import ChainConfig, ChainResult
from my_langchain.llms import BaseLLM
from my_langchain.prompts import BasePromptTemplate
from pydantic import Field


class LLMChain(BaseChain):
    """
    Chain that combines a prompt template with an LLM

    This is the most fundamental chain that takes a prompt template,
    formats it with inputs, and passes it to an LLM for generation.
    """

    llm: BaseLLM = Field(..., description="Language model to use")
    prompt: BasePromptTemplate = Field(..., description="Prompt template for formatting inputs")
    output_key: str = Field(default="text", description="Key for the output in the result")

    def __init__(
        self,
        llm: BaseLLM,
        prompt: BasePromptTemplate,
        output_key: str = "text",
        config: Optional[ChainConfig] = None,
        **kwargs
    ):
        """
        Initialize LLM chain

        Args:
            llm: Language model to use
            prompt: Prompt template for formatting inputs
            output_key: Key for the output in the result
            config: Chain configuration
            **kwargs: Additional parameters
        """
        if config is None:
            config = ChainConfig(output_key=output_key)

        # Validate prompt and LLM compatibility
        if not prompt.input_variables:
            raise ValueError("Prompt template must have input variables")

        super().__init__(
            llm=llm,
            prompt=prompt,
            output_key=output_key,
            config=config,
            **kwargs
        )

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LLM chain

        Args:
            inputs: Input values for the prompt template

        Returns:
            Dictionary with the generated text
        """
        # Format the prompt with inputs
        formatted_prompt = self.prompt.format(**inputs)

        # Generate response from LLM
        response = self.llm.generate(formatted_prompt)

        # Return result with specified output key
        return {self.output_key: response}

    async def _arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LLM chain asynchronously

        Args:
            inputs: Input values for the prompt template

        Returns:
            Dictionary with the generated text
        """
        # Format the prompt with inputs
        formatted_prompt = self.prompt.format(**inputs)

        # Generate response from LLM asynchronously
        response = await self.llm.agenerate(formatted_prompt)

        # Return result with specified output key
        return {self.output_key: response}

    def run_with_result(self, inputs: Union[Dict[str, Any], str]) -> ChainResult:
        """
        Run the chain and return detailed result

        Args:
            inputs: Input values

        Returns:
            Detailed chain result
        """
        if isinstance(inputs, str):
            # If single string input, use first prompt variable
            if self.prompt.input_variables:
                inputs = {self.prompt.input_variables[0]: inputs}
            else:
                inputs = {"input": inputs}

        return super().run(inputs)

    async def arun_with_result(self, inputs: Union[Dict[str, Any], str]) -> ChainResult:
        """
        Run the chain asynchronously and return detailed result

        Args:
            inputs: Input values

        Returns:
            Detailed chain result
        """
        if isinstance(inputs, str):
            # If single string input, use first prompt variable
            if self.prompt.input_variables:
                inputs = {self.prompt.input_variables[0]: inputs}
            else:
                inputs = {"input": inputs}

        return await super().arun(inputs)

    def apply(self, inputs_list: List[Union[Dict[str, Any], str]]) -> List[Any]:
        """
        Apply the chain to a list of inputs

        Args:
            inputs_list: List of input values

        Returns:
            List of outputs
        """
        results = []
        for inputs in inputs_list:
            result = self.run(inputs)
            results.append(result)
        return results

    async def aapply(self, inputs_list: List[Union[Dict[str, Any], str]]) -> List[Any]:
        """
        Apply the chain asynchronously to a list of inputs

        Args:
            inputs_list: List of input values

        Returns:
            List of outputs
        """
        import asyncio
        tasks = [self.arun(inputs) for inputs in inputs_list]
        return await asyncio.gather(*tasks)

    def _get_input_keys(self) -> List[str]:
        """Get input keys from prompt template"""
        return self.prompt.input_variables

    def _get_output_keys(self) -> List[str]:
        """Get output keys"""
        return [self.output_key]

    def get_prompt(self) -> BasePromptTemplate:
        """Get the prompt template"""
        return self.prompt

    def update_prompt(self, prompt: BasePromptTemplate) -> None:
        """
        Update the prompt template

        Args:
            prompt: New prompt template
        """
        self.prompt = prompt

    def update_llm(self, llm: BaseLLM) -> None:
        """
        Update the language model

        Args:
            llm: New language model
        """
        self.llm = llm
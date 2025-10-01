# -*- coding: utf-8 -*-
"""
Base chain implementation
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from my_langchain.base.base import BaseChain as BaseChainComponent
from my_langchain.chains.types import (
    ChainConfig, ChainResult, ChainInput, ChainError,
    ChainValidationError, ChainExecutionError
)
from pydantic import ConfigDict, Field


class BaseChain(BaseChainComponent):
    """
    Base implementation for chains

    Provides common functionality for chain execution, validation,
    and error handling.
    """

    config: ChainConfig = Field(..., description="Chain configuration")

    def __init__(self, config: Optional[ChainConfig] = None, **kwargs):
        """
        Initialize chain

        Args:
            config: Chain configuration
            **kwargs: Additional parameters
        """
        if config is None:
            config = ChainConfig()

        super().__init__(config=config, **kwargs)

    @abstractmethod
    def _run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chain logic

        Args:
            inputs: Input values for the chain

        Returns:
            Chain execution result as dictionary
        """
        pass

    @abstractmethod
    async def _arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chain logic asynchronously

        Args:
            inputs: Input values for the chain

        Returns:
            Chain execution result as dictionary
        """
        pass

    def run(self, inputs: Union[Dict[str, Any], str], config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run the chain with inputs

        Args:
            inputs: Input values (can be dict or single value)
            config: Runtime configuration overrides

        Returns:
            Chain output
        """
        start_time = time.time()
        intermediate_steps = []

        try:
            # Normalize inputs to dictionary
            if isinstance(inputs, dict):
                input_dict = inputs
            else:
                # Single value, use input_key or default
                input_key = self.config.input_key or "input"
                input_dict = {input_key: inputs}

            # Validate inputs
            self._validate_inputs(input_dict)

            # Run the chain
            result = self._run_with_handling(input_dict, config, intermediate_steps)

            # Create result object
            execution_time = time.time() - start_time
            chain_result = ChainResult(
                output=result,
                intermediate_steps=intermediate_steps,
                metadata={"execution_time": execution_time},
                execution_time=execution_time
            )

            # Return based on configuration
            if self.config.return_intermediate_steps:
                return chain_result
            elif self.config.output_key:
                return result.get(self.config.output_key, result)
            else:
                return result

        except Exception as e:
            execution_time = time.time() - start_time
            raise ChainExecutionError(
                f"Chain execution failed: {str(e)}",
                step="run",
                cause=e
            )

    async def arun(self, inputs: Union[Dict[str, Any], str], config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run the chain asynchronously with inputs

        Args:
            inputs: Input values (can be dict or single value)
            config: Runtime configuration overrides

        Returns:
            Chain output
        """
        import asyncio
        start_time = time.time()
        intermediate_steps = []

        try:
            # Normalize inputs to dictionary
            if isinstance(inputs, dict):
                input_dict = inputs
            else:
                input_key = self.config.input_key or "input"
                input_dict = {input_key: inputs}

            # Validate inputs
            self._validate_inputs(input_dict)

            # Run the chain asynchronously
            result = await self._arun_with_handling(input_dict, config, intermediate_steps)

            # Create result object
            execution_time = time.time() - start_time
            chain_result = ChainResult(
                output=result,
                intermediate_steps=intermediate_steps,
                metadata={"execution_time": execution_time},
                execution_time=execution_time
            )

            # Return based on configuration
            if self.config.return_intermediate_steps:
                return chain_result
            elif self.config.output_key:
                return result.get(self.config.output_key, result)
            else:
                return result

        except Exception as e:
            execution_time = time.time() - start_time
            raise ChainExecutionError(
                f"Chain execution failed: {str(e)}",
                step="arun",
                cause=e
            )

    def _run_with_handling(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]],
                          intermediate_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run chain with error handling and step tracking

        Args:
            inputs: Input values
            config: Runtime configuration
            intermediate_steps: List to record intermediate steps

        Returns:
            Chain execution result
        """
        step_name = f"{self.__class__.__name__}_run"
        if self.config.verbose:
            intermediate_steps.append({
                "step": step_name,
                "action": "start",
                "inputs": inputs
            })

        try:
            result = self._run(inputs)

            if self.config.verbose:
                intermediate_steps.append({
                    "step": step_name,
                    "action": "complete",
                    "output": result
                })

            return result

        except Exception as e:
            if self.config.verbose:
                intermediate_steps.append({
                    "step": step_name,
                    "action": "error",
                    "error": str(e)
                })
            raise

    async def _arun_with_handling(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]],
                                intermediate_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run chain asynchronously with error handling and step tracking

        Args:
            inputs: Input values
            config: Runtime configuration
            intermediate_steps: List to record intermediate steps

        Returns:
            Chain execution result
        """
        step_name = f"{self.__class__.__name__}_arun"
        if self.config.verbose:
            intermediate_steps.append({
                "step": step_name,
                "action": "start",
                "inputs": inputs
            })

        try:
            result = await self._arun(inputs)

            if self.config.verbose:
                intermediate_steps.append({
                    "step": step_name,
                    "action": "complete",
                    "output": result
                })

            return result

        except Exception as e:
            if self.config.verbose:
                intermediate_steps.append({
                    "step": step_name,
                    "action": "error",
                    "error": str(e)
                })
            raise

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Validate input values

        Args:
            inputs: Input values to validate

        Raises:
            ChainValidationError: If inputs are invalid
        """
        if not inputs:
            raise ChainValidationError("Inputs cannot be empty", inputs)

        # Check for required input key if specified
        if self.config.input_key and self.config.input_key not in inputs:
            raise ChainValidationError(
                f"Required input key '{self.config.input_key}' not found in inputs",
                inputs
            )

    def invoke(self, input: Any) -> Any:
        """
        Invoke the chain with input (unified interface)

        Args:
            input: Input data

        Returns:
            Chain output
        """
        return self.run(input)

    def get_chain_info(self) -> Dict[str, Any]:
        """
        Get chain information

        Returns:
            Chain information dictionary
        """
        return {
            "chain_type": self.__class__.__name__,
            "config": self.config.model_dump(),
            "input_keys": self._get_input_keys(),
            "output_keys": self._get_output_keys()
        }

    def _get_input_keys(self) -> List[str]:
        """
        Get expected input keys

        Returns:
            List of input key names
        """
        if self.config.input_key:
            return [self.config.input_key]
        return ["input"]  # Default

    def _get_output_keys(self) -> List[str]:
        """
        Get expected output keys

        Returns:
            List of output key names
        """
        if self.config.output_key:
            return [self.config.output_key]
        return ["output"]  # Default
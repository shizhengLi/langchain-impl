# -*- coding: utf-8 -*-
"""
Simple Chain implementation
"""

from typing import Any, Callable, Dict, List, Optional, Union

from my_langchain.chains.base import BaseChain
from my_langchain.chains.types import ChainConfig, ChainExecutionError
from pydantic import Field


class SimpleChain(BaseChain):
    """
    Simple chain that executes a function on inputs

    Useful for wrapping custom functions or simple transformations.
    """

    func: Callable = Field(..., description="Function to execute")
    input_keys: List[str] = Field(default_factory=list, description="Expected input keys")
    output_keys: List[str] = Field(default_factory=list, description="Output keys")

    def __init__(
        self,
        func: Callable,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        config: Optional[ChainConfig] = None,
        **kwargs
    ):
        """
        Initialize simple chain

        Args:
            func: Function to execute
            input_keys: Expected input keys
            output_keys: Output keys
            config: Chain configuration
            **kwargs: Additional parameters
        """
        if config is None:
            config = ChainConfig()

        super().__init__(
            func=func,
            input_keys=input_keys or [],
            output_keys=output_keys or [],
            config=config,
            **kwargs
        )

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the function with inputs

        Args:
            inputs: Input values

        Returns:
            Function output
        """
        try:
            # Prepare function arguments
            if self.input_keys:
                # Use specified input keys - try as keyword arguments first
                try:
                    func_args = {key: inputs[key] for key in self.input_keys if key in inputs}
                    result = self.func(**func_args)
                except TypeError:
                    # If keyword arguments fail, try positional arguments
                    args = [inputs[key] for key in self.input_keys if key in inputs]
                    result = self.func(*args)
            else:
                # Use all inputs as arguments
                try:
                    result = self.func(**inputs)
                except TypeError:
                    # Try positional if keyword fails
                    args = list(inputs.values())
                    result = self.func(*args)

            # Prepare output
            if self.output_keys:
                if len(self.output_keys) == 1:
                    return {self.output_keys[0]: result}
                else:
                    # If function returns dict with multiple keys
                    if isinstance(result, dict):
                        return {key: result.get(key) for key in self.output_keys}
                    else:
                        # If function returns tuple/list, map to output keys
                        if isinstance(result, (list, tuple)) and len(result) == len(self.output_keys):
                            return dict(zip(self.output_keys, result))
                        else:
                            return {self.output_keys[0]: result}
            else:
                # Default output key
                return {"output": result}

        except Exception as e:
            raise ChainExecutionError(
                f"Function execution failed: {str(e)}",
                step="simple_chain_run",
                cause=e
            )

    async def _arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the function asynchronously with inputs

        Args:
            inputs: Input values

        Returns:
            Function output
        """
        try:
            # Check if function is async
            import inspect
            if inspect.iscoroutinefunction(self.func):
                # Prepare function arguments for async function
                if self.input_keys:
                    # Try keyword arguments first
                    try:
                        func_args = {key: inputs[key] for key in self.input_keys if key in inputs}
                        result = await self.func(**func_args)
                    except TypeError:
                        # Try positional arguments
                        args = [inputs[key] for key in self.input_keys if key in inputs]
                        result = await self.func(*args)
                else:
                    # Use all inputs as arguments
                    try:
                        result = await self.func(**inputs)
                    except TypeError:
                        # Try positional if keyword fails
                        args = list(inputs.values())
                        result = await self.func(*args)
            else:
                # Run sync function in thread pool
                import asyncio

                # Prepare arguments the same way as sync version
                if self.input_keys:
                    try:
                        func_args = {key: inputs[key] for key in self.input_keys if key in inputs}
                        result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.func(**func_args))
                    except TypeError:
                        args = [inputs[key] for key in self.input_keys if key in inputs]
                        result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.func(*args))
                else:
                    try:
                        result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.func(**inputs))
                    except TypeError:
                        args = list(inputs.values())
                        result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.func(*args))

            # Prepare output
            if self.output_keys:
                if len(self.output_keys) == 1:
                    return {self.output_keys[0]: result}
                else:
                    # If function returns dict with multiple keys
                    if isinstance(result, dict):
                        return {key: result.get(key) for key in self.output_keys}
                    else:
                        # If function returns tuple/list, map to output keys
                        if isinstance(result, (list, tuple)) and len(result) == len(self.output_keys):
                            return dict(zip(self.output_keys, result))
                        else:
                            return {self.output_keys[0]: result}
            else:
                # Default output key
                return {"output": result}

        except Exception as e:
            raise ChainExecutionError(
                f"Function execution failed: {str(e)}",
                step="simple_chain_arun",
                cause=e
            )

    def run(self, inputs: Union[Dict[str, Any], str], config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run the simple chain with inputs

        Args:
            inputs: Input values (can be dict or single value)
            config: Runtime configuration overrides

        Returns:
            Simple chain output (returns simplified output for single output)
        """
        # Use parent run method
        result = super().run(inputs, config)

        # If we have a single output key and result is a dict, return the value
        if self.output_keys and len(self.output_keys) == 1 and isinstance(result, dict):
            output_key = self.output_keys[0]
            return result.get(output_key, result)

        return result

    async def arun(self, inputs: Union[Dict[str, Any], str], config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run the simple chain asynchronously with inputs

        Args:
            inputs: Input values (can be dict or single value)
            config: Runtime configuration overrides

        Returns:
            Simple chain output (returns simplified output for single output)
        """
        # Use parent arun method
        result = await super().arun(inputs, config)

        # If we have a single output key and result is a dict, return the value
        if self.output_keys and len(self.output_keys) == 1 and isinstance(result, dict):
            output_key = self.output_keys[0]
            return result.get(output_key, result)

        return result

    def _get_input_keys(self) -> List[str]:
        """Get input keys"""
        return self.input_keys or ["input"]

    def _get_output_keys(self) -> List[str]:
        """Get output keys"""
        return self.output_keys or ["output"]

    def set_input_keys(self, input_keys: List[str]) -> None:
        """
        Set input keys

        Args:
            input_keys: List of input keys
        """
        self.input_keys = input_keys

    def set_output_keys(self, output_keys: List[str]) -> None:
        """
        Set output keys

        Args:
            output_keys: List of output keys
        """
        self.output_keys = output_keys

    @classmethod
    def from_function(
        cls,
        func: Callable,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None
    ) -> 'SimpleChain':
        """
        Create SimpleChain from function

        Args:
            func: Function to wrap
            input_keys: Expected input keys
            output_keys: Output keys

        Returns:
            SimpleChain instance
        """
        return cls(
            func=func,
            input_keys=input_keys,
            output_keys=output_keys
        )
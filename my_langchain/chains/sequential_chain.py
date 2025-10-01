# -*- coding: utf-8 -*-
"""
Sequential Chain implementation
"""

from typing import Any, Dict, List, Optional, Union

from my_langchain.chains.base import BaseChain
from my_langchain.chains.types import ChainConfig, ChainError
from pydantic import Field


class SequentialChain(BaseChain):
    """
    Chain that executes multiple chains in sequence

    The output of each chain becomes input for the next chain.
    """

    chains: List[BaseChain] = Field(default_factory=list, description="Chains to execute in sequence")
    input_variables: List[str] = Field(default_factory=list, description="Expected input variables")
    output_variables: List[str] = Field(default_factory=list, description="Output variables")
    return_all: bool = Field(default=False, description="Return all intermediate outputs")

    def __init__(
        self,
        chains: List[BaseChain],
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
        return_all: bool = False,
        config: Optional[ChainConfig] = None,
        **kwargs
    ):
        """
        Initialize sequential chain

        Args:
            chains: List of chains to execute in sequence
            input_variables: Expected input variables
            output_variables: Output variables to return
            return_all: Whether to return all intermediate outputs
            config: Chain configuration
            **kwargs: Additional parameters
        """
        if config is None:
            config = ChainConfig()

        if not chains:
            raise ValueError("At least one chain must be provided")

        super().__init__(
            chains=chains,
            input_variables=input_variables or [],
            output_variables=output_variables or [],
            return_all=return_all,
            config=config,
            **kwargs
        )

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all chains in sequence

        Args:
            inputs: Input values for the first chain

        Returns:
            Dictionary with final output (and intermediate outputs if return_all=True)
        """
        current_inputs = inputs.copy()
        all_outputs = {}

        # Execute each chain in sequence
        for i, chain in enumerate(self.chains):
            try:
                # Run the chain
                chain_output = chain.run(current_inputs)

                # Handle different output types
                if isinstance(chain_output, dict):
                    # Chain returned dictionary
                    current_inputs.update(chain_output)
                    all_outputs.update(chain_output)
                else:
                    # Chain returned single value
                    chain_info = chain.get_chain_info()
                    output_keys = chain_info.get("output_keys", ["output"])
                    if output_keys:
                        output_key = output_keys[0]
                        current_inputs[output_key] = chain_output
                        all_outputs[output_key] = chain_output
                    else:
                        # Default output key
                        current_inputs[f"step_{i}_output"] = chain_output
                        all_outputs[f"step_{i}_output"] = chain_output

            except Exception as e:
                raise ChainError(
                    f"Error in chain {i} ({chain.__class__.__name__}): {str(e)}",
                    error_type="SEQUENTIAL_CHAIN_ERROR",
                    details={"chain_index": i, "chain_type": chain.__class__.__name__}
                )

        # Prepare final output
        if self.return_all:
            return all_outputs
        elif self.output_variables:
            # Return only specified output variables
            return {var: all_outputs.get(var) for var in self.output_variables}
        else:
            # Return the last chain's output
            return {self.output_variables[0] if self.output_variables else "output": list(all_outputs.values())[-1]}

    async def _arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all chains in sequence asynchronously

        Args:
            inputs: Input values for the first chain

        Returns:
            Dictionary with final output
        """
        current_inputs = inputs.copy()
        all_outputs = {}

        # Execute each chain in sequence
        for i, chain in enumerate(self.chains):
            try:
                # Run the chain asynchronously
                chain_output = await chain.arun(current_inputs)

                # Handle different output types
                if isinstance(chain_output, dict):
                    # Chain returned dictionary
                    current_inputs.update(chain_output)
                    all_outputs.update(chain_output)
                else:
                    # Chain returned single value
                    chain_info = chain.get_chain_info()
                    output_keys = chain_info.get("output_keys", ["output"])
                    if output_keys:
                        output_key = output_keys[0]
                        current_inputs[output_key] = chain_output
                        all_outputs[output_key] = chain_output
                    else:
                        # Default output key
                        current_inputs[f"step_{i}_output"] = chain_output
                        all_outputs[f"step_{i}_output"] = chain_output

            except Exception as e:
                raise ChainError(
                    f"Error in chain {i} ({chain.__class__.__name__}): {str(e)}",
                    error_type="SEQUENTIAL_CHAIN_ERROR",
                    details={"chain_index": i, "chain_type": chain.__class__.__name__}
                )

        # Prepare final output
        if self.return_all:
            return all_outputs
        elif self.output_variables:
            # Return only specified output variables
            return {var: all_outputs.get(var) for var in self.output_variables}
        else:
            # Return the last chain's output
            return {self.output_variables[0] if self.output_variables else "output": list(all_outputs.values())[-1]}

    def add_chain(self, chain: BaseChain) -> None:
        """
        Add a chain to the sequence

        Args:
            chain: Chain to add
        """
        self.chains.append(chain)

    def remove_chain(self, index: int) -> None:
        """
        Remove a chain from the sequence

        Args:
            index: Index of the chain to remove
        """
        if 0 <= index < len(self.chains):
            self.chains.pop(index)

    def get_chain_count(self) -> int:
        """
        Get the number of chains in the sequence

        Returns:
            Number of chains
        """
        return len(self.chains)

    def get_chain_at(self, index: int) -> Optional[BaseChain]:
        """
        Get chain at specific index

        Args:
            index: Index of the chain

        Returns:
            Chain at the index, or None if index is invalid
        """
        if 0 <= index < len(self.chains):
            return self.chains[index]
        return None

    def _get_input_keys(self) -> List[str]:
        """Get input keys from the first chain"""
        if self.input_variables:
            return self.input_variables
        elif self.chains:
            first_chain = self.chains[0]
            return first_chain._get_input_keys()
        return ["input"]

    def _get_output_keys(self) -> List[str]:
        """Get output keys from the last chain"""
        if self.output_variables:
            return self.output_variables
        elif self.chains:
            last_chain = self.chains[-1]
            return last_chain._get_output_keys()
        return ["output"]

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate inputs against first chain requirements"""
        super()._validate_inputs(inputs)

        if self.chains:
            first_chain = self.chains[0]
            required_inputs = first_chain._get_input_keys()
            missing_inputs = [key for key in required_inputs if key not in inputs]

            if missing_inputs:
                from my_langchain.chains.types import ChainValidationError
                raise ChainValidationError(
                    f"Missing required inputs for sequential chain: {missing_inputs}",
                    inputs
                )